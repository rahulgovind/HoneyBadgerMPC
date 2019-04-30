import logging
import asyncio
import traceback
import time
from honeybadgermpc.mpc import PreProcessedElements, TaskProgramRunner
from honeybadgermpc.mixins import BeaverTriple, MixinOpName
from honeybadgermpc.elliptic_curve import Subgroup
from honeybadgermpc.field import GF, GFElement
import random
import numpy as np
import threading
from sklearn.model_selection import train_test_split
import pandas as pd
import gc

import traceback

_ppe = None

F = 32
KAPPA = 32
K = 64
p = modulus = Subgroup.BLS12_381
Field = GF.get(p)
USE_RANDOM_BIT_PPE = True
random_lock = threading.Lock()

open_array_counter = 0
TIME = False
DEBUG = False


async def _deterministic_gather(*args):
    result = []
    for arg in args:
        result.append(await arg)
    return result


def deterministic_gather(*args):
    return asyncio.ensure_future(_deterministic_gather(*args))


async def open_single(ctx, x, t=None):
    return await ctx.Share(x, t).open()


async def open_array(ctx, arr, t=None):
    global open_array_counter
    open_array_counter += 1
    if DEBUG:
        print(f"open_array {open_array_counter}: {len(list(arr))}")
    return await ctx.ShareArray(list(arr), t).open()


async def open_nd_array(ctx, arr, t=None):
    return np.array(await open_array(ctx, arr.flatten().tolist(), t)).reshape(arr.shape)


async def reduce_single(ctx, x):
    x = ctx.Share(x, 2 * ctx.t)
    r_t, r_2t = _ppe.get_double_share(ctx)
    diff = await (x - r_2t).open()
    return diff + r_t.v


async def reduce_array(ctx, arr):
    r_t, r_2t = [], []
    for _ in range(len(arr)):
        r_t_, r_2t_ = _ppe.get_double_share(ctx)
        r_t.append(r_t_.v)
        r_2t.append(r_2t_.v)
    diff = await open_array(ctx, list(map(lambda x, y: x - y, arr, r_2t)), 2 * ctx.t)

    return list(map(lambda x, y: x + y, diff, r_t))


async def reduce_nd_array(ctx, arr):
    return np.array(await reduce_array(ctx, arr.flatten().tolist())).reshape(arr.shape)


async def trunc_pr_single(ctx, x, k, m):
    """
    k: Maximum number of bits
    m: Truncation bits
    """
    assert k > m
    r1, _ = await random2m(ctx, m)
    r2, _ = await random2m(ctx, k + KAPPA - m)
    r2 = ctx.Share(r2.v * Field(2) ** m)

    c = await ctx.Share(x + Field(2 ** (k - 1)) + r1.v + r2.v).open()
    c2 = c.value % (2 ** m)
    d = (x - Field(c2) + r1.v) * ~(Field(2) ** m)
    return d


async def random_bit_share(ctx):
    if USE_RANDOM_BIT_PPE:
        return _ppe.get_random_bit(ctx)
    else:
        r = _ppe.get_rand(ctx)
        r_square = await (r * r)
        r_sq = await r_square.open()

        if pow(r_sq, (modulus - 1) // 2) != Field(1) or r_sq == 0:
            return await random_bit_share(ctx)

        root = r_sq.sqrt()
        return (~Field(2)) * ((~root) * r + Field(1))


async def random_bit_shares(ctx, n):
    if USE_RANDOM_BIT_PPE:
        return _ppe.get_random_bits(ctx, n)
    else:
        raise NotImplementedError


async def randoms2m(ctx, m, n):
    all_bits = await random_bit_shares(ctx, m * n)

    # Split bits into n parts
    # For each chunk compute the sum (2 ** i * chunk[i])
    bits = []
    powsums = []

    for i in range(0, m * n, m):
        chunk = all_bits[i:i + m]
        powsum = sum(Field(2) ** j * chunk[j] for j in range(m))
        bits.append(chunk)
        powsums.append(powsum)
    return powsums, bits


async def trunc_pr_array(ctx, arr, k, m):
    assert k > m
    batch_size = len(arr)
    r1, _ = await randoms2m(ctx, m, batch_size)
    r2, _ = await randoms2m(ctx, k + KAPPA - m, batch_size)
    r2 = [x * Field(2) ** m for x in r2]
    c_shares = list(map(lambda xi, r1i, r2i: xi + Field(2 ** (k - 1)) + r1i + r2i,
                        arr, r1, r2))
    c = await open_array(ctx, c_shares)
    c = list(map(lambda x: ctx.field(x.value % (2 ** m)), c))
    d = list(map(lambda x, ci, r1i: (x - ci + r1i) * ~(Field(2) ** m),
                 arr, c, r1))
    return d


async def trunc_pr_nd_array(ctx, arr, k, m):
    res = await trunc_pr_array(ctx, arr.flatten().tolist(), k, m)
    return np.array(res).reshape(arr.shape)


async def random2m(ctx, m):
    result = ctx.Share(0)
    bits = []
    for i in range(m):
        bits.append(await random_bit_share(ctx))
        result = result + Field(2) ** i * bits[-1]

    return result, bits


async def trunc_pr(ctx, x, k, m):
    """
    k: Maximum number of bits
    m: Truncation bits
    """
    assert k > m
    r1, _ = await random2m(ctx, m)
    r2, _ = await random2m(ctx, k + KAPPA - m)
    r2 = ctx.Share(r2.v * Field(2) ** m)
    c = await (x + Field(2 ** (k - 1)) + r1.v + r2.v).open()
    c2 = c.value % (2 ** m)
    d = ctx.Share((x.v - Field(c2) + r1.v) * ~(Field(2) ** m))
    return d


def binary_repr(x, k):
    """
    Convert x to a k-bit representation
    Least significant bit first
    """

    def _binary_repr(v):
        res = []
        v = int(v)
        for i in range(k):
            res.append(v % 2)
            v //= 2
        return res

    if type(x) is int:
        return _binary_repr(x)
    else:
        assert type(x) is np.ndarray
        assert x.ndim == 1
        return np.array([_binary_repr(xi) for xi in x])


async def binary_addition(ctx, x, y):
    """
    :param ctx:
    :param x:
    :param y:
    :return:
    """
    assert x.ndim == 2
    assert x.shape == y.shape

    c = x * y
    c = await reduce_nd_array(ctx, c)
    d = x + y - 2 * c

    for i in range(int(np.ceil(np.log2(x.shape[1])))):
        t1, t2 = c[:, 2 ** i:] + (d[:, 2 ** i:] * c[:, :-2 ** i]), \
                 d[:, 2 ** i:] * d[:, :-2 ** i]
        t1 = await reduce_nd_array(ctx, t1)
        t2 = await reduce_nd_array(ctx, t2)
        c[:, 2 ** i:], d[:, 2 ** i:] = t1, t2

    res = x + y - 2 * c
    res[:, 1:] += c[:, :-1]
    # print("c: ", await open_nd_array(ctx, c))
    return res


async def to_bits(ctx, x, k, m):
    assert x.ndim == 1
    batch_size = x.shape[0]
    r1, r1_bits = await randoms2m(ctx, m, batch_size)
    r1, r1_bits = np.array(r1), np.array(r1_bits)
    r2, _ = await randoms2m(ctx, k + KAPPA - m, batch_size)
    r2 = np.array(r2)
    c = await open_nd_array(ctx, x - (r1 + r2 * 2 ** m) + 2 ** k + 2 ** (k + KAPPA))
    return await binary_addition(ctx, binary_repr(c, m), r1_bits)


def from_bits(arr):
    assert arr.ndim == 2
    assert arr.shape[1] > 0
    result = arr[:, 0]
    for i in range(1, arr.shape[1]):
        result = result + arr[:, i] * 2 ** i
    return result


async def get_carry_bit(ctx, a_bits, b_bits, low_carry_bit=1):
    a_bits.reverse()
    b_bits.reverse()

    async def _bit_ltl_reduce(x):
        if len(x) == 1:
            return x[0]
        carry1, all_one1 = await _bit_ltl_reduce(x[:len(x) // 2])
        carry2, all_one2 = await _bit_ltl_reduce(x[len(x) // 2:])
        return carry1 + (await (all_one1 * carry2)), (await (all_one1 * all_one2))

    carry_bits = [(await (ai * bi)) for ai, bi in zip(a_bits, b_bits)]
    all_one_bits = [ctx.Share(ai.v + bi.v - 2 * carryi.v) for ai, bi, carryi in
                    zip(a_bits, b_bits, carry_bits)]
    carry_bits.append(ctx.Share(low_carry_bit))
    all_one_bits.append(ctx.Share(0))
    return (await _bit_ltl_reduce(list(zip(carry_bits, all_one_bits))))[0]


async def get_carry_bits(ctx, a_bits, b_bits, low_carry_bit=1):
    assert a_bits.ndim <= 2 and b_bits.ndim <= 2
    if a_bits.ndim == 1:
        a_bits = a_bits.reshape(-1, 1)
    if b_bits.ndim == 1:
        b_bits = b_bits.reshape(-1, 1)
    assert a_bits.shape == b_bits.shape

    # Reverse bits
    a_bits = a_bits[::-1]
    b_bits = b_bits[::-1]

    # carry_bits = np.array(await reduce_nd_array(ctx, a_bits * b_bits))
    carry_bits = a_bits * b_bits
    # Reduction not necessary since either a_bits or b_bits were public
    all_one_bits = a_bits + b_bits - 2 * carry_bits

    # Effectively batch size
    m = 1 if len(a_bits.shape) == 1 else a_bits.shape[1]

    # Append previous carry bit
    new_row_shape = (1,) + carry_bits.shape[1:]

    carry_bits = np.r_[carry_bits,
                       np.array([ctx.field(low_carry_bit)
                                 for _ in range(m)]).reshape(new_row_shape)]
    all_one_bits = np.r_[all_one_bits,
                         np.array([ctx.field(0)
                                   for _ in range(m)]).reshape(new_row_shape)]

    # Pad to make number of bits a power of 2
    n = len(carry_bits)

    if n & (n - 1) != 0:
        new_n = 2 ** n.bit_length()
        carry_bits = np.r_[carry_bits,
                           np.array([[ctx.field(0) for _ in range(m)]
                                     for _ in range(new_n - n)])]
        all_one_bits = np.r_[all_one_bits,
                             np.array([[ctx.field(0) for _ in range(m)]
                                       for _ in range(new_n - n)])]

    iter = 0
    while carry_bits.shape[0] > 1:
        start_time = time.time()
        temp1 = all_one_bits[0::2] * carry_bits[1::2]
        temp1 = temp1 + carry_bits[0::2]
        # Based on the observation that the [0::2] values can be used as 2t-shares
        # in the next iteration
        if temp1.shape[0] > 1:
            temp1[1::2] = await reduce_nd_array(ctx, temp1[1::2])

        temp2 = all_one_bits[0::2] * all_one_bits[1::2]
        temp2 = await reduce_nd_array(ctx, temp2)
        carry_bits, all_one_bits = temp1, temp2
        end_time = time.time()
        if TIME:
            print(f"get_carry_bits: Iteration {iter} took {end_time - start_time}s")
        iter += 1
    carry_bits = await reduce_nd_array(ctx, carry_bits)
    return carry_bits.flatten()


async def bit_ltl(ctx, a, b_bits):
    """
    a: Public
    b: List of private bit shares. Least significant digit first
    """
    b_bits = [ctx.Share(Field(1) - bi.v) for bi in b_bits]
    a_bits = [ctx.Share(ai) for ai in binary_repr(int(a), len(b_bits))]

    carry = await get_carry_bit(ctx, a_bits, b_bits)
    return ctx.Share(Field(1) - carry.v)


async def bit_ltl2(ctx, a, b_bits):
    """
    a: Public
    b: List of private bit shares. Least significant digit first
    """
    b_bits = [ctx.Share(Field(1) - bi.v) for bi in b_bits]
    a_bits = [ctx.Share(ai) for ai in binary_repr(int(a), len(b_bits))]

    a_bits_opened = [(await a_bit.open()) for a_bit in a_bits]
    # print("a: ", a_bits_opened)

    b_bits_opened = [(await b_bit.open()) for b_bit in b_bits]
    # print("b: ", b_bits_opened)

    carry = (await get_carry_bits(ctx,
                                  np.array([x.v for x in a_bits]),
                                  np.array([x.v for x in b_bits])))[0]

    return ctx.Share(Field(1) - carry)


async def bit_ltl_array(ctx, a, b):
    start_time = time.time()
    assert (a.ndim == 1 and b.ndim == 2) or (a.ndim == 2 and b.ndim == 1)
    assert a.shape[0] == b.shape[0]

    def ones_complement(x):
        return ctx.field(1) - x

    nbits = a.shape[1] if a.ndim == 2 else b.shape[1]

    def to_bits_field(x):
        def _to_bits_field(x):
            return [ctx.field(xi) for xi in binary_repr(int(x), nbits)]

        return np.array([_to_bits_field(x[i]) for i in range(len(x))])

    a_bits = to_bits_field(a) if a.ndim == 1 else a
    b_bits = to_bits_field(b) if b.ndim == 1 else b
    b_bits = np.vectorize(ones_complement)(b_bits)

    carry_bits = await get_carry_bits(ctx, a_bits.T, b_bits.T)
    end_time = time.time()
    if TIME:
        print(f"bit_ltl_array took {end_time - start_time}s")
    return np.array(ctx.field(1)) - carry_bits


async def mod2m(ctx, x, k, m):
    r1, r1_bits = await random2m(ctx, m)
    r2, _ = await random2m(ctx, k + KAPPA - m)
    r2 = ctx.Share(r2.v * Field(2) ** m)

    c = await (x + r2 + r1 + Field(2) ** (k - 1)).open()
    c2 = int(c) % (2 ** m)
    u = await bit_ltl(ctx, c2, r1_bits)
    a2 = ctx.Share(Field(c2) - r1.v + (2 ** m) * u.v)
    return a2


async def mod2m_array(ctx, arr, k, m):
    assert k > m
    assert arr.ndim == 1
    batch_size = len(arr)

    def field_mod_2m(x):
        return ctx.field(int(x) % (2 ** m))

    r1, r1_bits = await randoms2m(ctx, m, batch_size)
    r1, r1_bits = np.array(r1), np.array(r1_bits)
    r2, _ = await randoms2m(ctx, k + KAPPA - m, batch_size)
    r2 = np.array(r2)
    r2 = Field(2) ** m * r2
    c_shares = arr + Field(2 ** (k - 1)) + r1 + r2
    c = await open_nd_array(ctx, c_shares)
    c2 = np.vectorize(field_mod_2m)(c)
    u = await bit_ltl_array(ctx, c2, r1_bits)
    a2 = c2 - r1 + (2 ** m) * u
    return a2


async def ltz_array(ctx, arr, k):
    assert arr.ndim == 1
    # print("ltz: ", await open_nd_array(ctx, arr))
    return -(await trunc_array(ctx, arr, k, k - 1))


async def trunc(ctx, x, k, m):
    a2 = await mod2m(ctx, x, k, m)
    d = ctx.Share((x.v - a2.v) / (Field(2)) ** m)
    return d


async def trunc_array(ctx, arr, k, m):
    assert arr.ndim == 1
    assert k > m
    a2 = await mod2m_array(ctx, arr, k, m)
    d = (arr - a2) / (Field(2) ** m)
    return d


async def pre_or(ctx, arr, reverse=False):
    """
    :param ctx:
    :param arr:
    :param reverse:
    :return:

    Only makes sense if the shares correspond to shares of bits
    """
    assert arr.ndim == 2
    if reverse:
        arr = arr[:, ::-1]

    res = np.array(arr)
    for i in range(int(np.ceil(np.log2(arr.shape[1])))):
        res[:, 2 ** i:] = res[:, 2 ** i:] + res[:, :-2 ** i] - \
                          res[:, 2 ** i:] * res[:, :-2 ** i]
        res[:, 2 ** i:] = await reduce_nd_array(ctx, res[:, 2 ** i:])

    if reverse:
        res = res[:, ::-1]
    return res


async def norm(ctx, arr, k, f):
    """
    :param ctx:
    :param arr:
    :param k:
    :param f:
    :return:

    Source: Catrina et al, Secure Computation With Fixed-Point Numbers
    """
    assert arr.ndim == 1

    s = 1 - 2 * (await ltz_array(ctx, arr, k))

    x = await reduce_nd_array(ctx, s * arr)
    x_bits = await to_bits(ctx, x, k, k)

    y_bits = await pre_or(ctx, x_bits, reverse=True)

    z = y_bits[:, :-1] - y_bits[:, 1:]
    pow2_reverse = np.array([[ctx.field(2) ** (k - i - 1) for i in range(k - 1)]])

    # Unsigned normalization factor
    v = np.sum(z * pow2_reverse, axis=1)

    # Signed normalization factor
    v2 = await reduce_nd_array(ctx, s * v)

    # Unsigned normalized quantity c \in [0.5, 1) with 2^-k precision <- IMPORTANT!
    # c <- Q^+_{k,k}
    c = await reduce_nd_array(ctx, x * v)

    return c, v2


async def _approx_reciprocal(ctx, arr, k, f):
    alpha = to_fixed_point_repr(2.9142, k)  # 2^{-k} precision, ~k+2 bits used
    c, v = await norm(ctx, arr, k, f)
    d = alpha - 2 * c  # 2^{-k} precision, ~k+1 bits used
    w = await reduce_nd_array(ctx, d * v)  # 2^{-k-f} precision. 2k+1 bits used
    w = await trunc_pr_nd_array(ctx, w, 2 * k + 1, k)
    return w


async def fpdiv(ctx, a, b, k, f):
    theta = int(np.ceil(np.log2(k / 3.5)))
    alpha = ctx.field(to_fixed_point_repr(1, 2 * f))
    w = await _approx_reciprocal(ctx, b, k, f)
    x = alpha - (await reduce_nd_array(ctx, b * w))
    y = await reduce_nd_array(ctx, a * w)
    y = await trunc_pr_nd_array(ctx, y, 2 * k, f)
    for i in range(theta):
        y = await reduce_nd_array(ctx, y * (alpha + x))
        x = await reduce_nd_array(ctx, x * x)
        y = await trunc_pr_nd_array(ctx, y, 2 * k, 2 * f)
        x = await trunc_pr_nd_array(ctx, x, 2 * k, 2 * f)
    y = await reduce_nd_array(ctx, y * (alpha + x))
    y = await trunc_pr_nd_array(ctx, y, 2 * k, 2 * f)
    return y


async def reciprocal(ctx, arr, k, f):
    """
    Combination of
    - https://en.wikipedia.org/wiki/Division_algorithm#Pseudocode
    - Secure Computation With Fixed-Point Numbers
    """
    assert arr.ndim == 1
    f2 = f
    theta = int(np.ceil(np.log2(k / 4.087)))

    arr_normalized_k, v = await norm(ctx, arr, k, f)
    arr_normalized = await trunc_pr_nd_array(ctx, arr_normalized_k, k, k - f2)

    # 1's representation in Q_{., 2f}
    one_2f = ctx.field(to_fixed_point_repr(1, 2 * f2))

    # Initial approximation: x0 = 48 / 17 - 32 / 17 * d
    x = ctx.field(to_fixed_point_repr(48 / 17, 2 * f2)) - \
        ctx.field(to_fixed_point_repr(32 / 17, f2)) * arr_normalized

    x = await trunc_pr_nd_array(ctx, x, 2 * k, f2)

    # print("Theta: ", theta)
    for _ in range(theta):
        dx = await reduce_nd_array(ctx, (one_2f - arr_normalized * x))
        dx = await trunc_pr_nd_array(ctx, dx, 2 * k, f2)
        dx = await reduce_nd_array(ctx, x * dx)
        dx = await trunc_pr_nd_array(ctx, dx, 2 * k, f2)
        x = x + dx
    res = await reduce_nd_array(ctx, x * v)
    return await trunc_pr_nd_array(ctx, res, 2 * k, f2)


async def approx_sqrt(ctx, x, k, f):
    assert x.ndim == 1
    x_bits = await to_bits(ctx, x, k, k)
    sqrt2_powers = np.array([2.0 ** ((i - f) / 2) for i in range(x_bits.shape[1])])
    sqrt2_powers = np.vectorize(to_fixed_point_repr)(sqrt2_powers, f)
    sqrt2_powers = sqrt2_powers.reshape(1, -1)

    return np.sum(x_bits * sqrt2_powers, axis=1)


class FixedPoint(object):
    def __init__(self, ctx, x):
        self.ctx = ctx
        if type(x) in [int, float]:
            self.share = _ppe.get_zero(ctx) + ctx.Share(int(x * 2 ** F))
        elif type(x) is ctx.Share:
            self.share = x
        else:
            raise NotImplementedError

    def add(self, x):
        if type(x) is FixedPoint:
            return FixedPoint(self.ctx, self.share + x.share)

    def sub(self, x):
        if type(x) is FixedPoint:
            return FixedPoint(self.ctx, self.share - x.share)
        raise NotImplementedError

    async def mul(self, x):
        if type(x) is FixedPoint:
            start_time = time.time()
            res_share = await (self.share * x.share)
            end_time = time.time()
            # print("Multiplication time: ", end_time - start_time)
            start_time = time.time()
            res_share = await trunc_pr(self.ctx, res_share, 2 * K, F)
            end_time = time.time()
            # print("Trunc time: ", end_time - start_time)
            return FixedPoint(self.ctx, res_share)
        raise NotImplementedError

    async def open(self):
        x = (await self.share.open()).value
        if x >= 2 ** (K - 1):
            x = -(p - x)
        return float(x) / 2 ** F

    def neg(self):
        return FixedPoint(self.ctx, Field(-1) * self.share)

    async def ltz(self):
        t = await trunc(self.ctx, self.share, K, K - 1)
        return self.ctx.Share(-t.v)

    async def lt(self, x):
        return await self.sub(x).ltz()

    async def div(self, x):
        if type(x) in [float, int]:
            return await self.mul(FixedPoint(self.ctx, 1. / x))
        raise NotImplementedError


def to_fixed_point_repr(x, f=F):
    return int(x * 2 ** f)


def from_fixed_point_repr(x, k=K, f=F, signed=True):
    x = x.value
    if x >= 2 ** (k - 1) and signed:
        x = -(p - x)

    return float(x) / 2 ** f


class SuperFixedPoint(object):
    def __init__(self, ctx, x):
        self.ctx = ctx
        if type(x) in [int, float]:
            self.data = asyncio.Future()
            self.data.set_result(_ppe.get_zero(ctx).v + to_fixed_point_repr(x))
            self.is_future = False
        elif type(x) is ctx.Share:
            self.data = asyncio.Future()
            self.data.set_result(x.v)
            asyncio.ensure_future(self.data)
            self.is_future = False
        elif type(x) is asyncio.Future:
            self.is_future = True
            self.data = x
        else:
            raise NotImplementedError

    def __add__(self, other):
        if type(other) is not SuperFixedPoint:
            other = SuperFixedPoint(self.ctx, other)
        res = SuperFixedPoint(self.ctx, asyncio.Future())

        def callback(_):
            res.data.set_result(self.data.result() + other.data.result())

        deterministic_gather(self.data, other.data).add_done_callback(callback)
        return res

    __radd__ = __add__

    def __sub__(self, other):
        if type(other) is not SuperFixedPoint:
            other = SuperFixedPoint(self.ctx, other)

        res = SuperFixedPoint(self.ctx, asyncio.Future())

        def callback(_):
            res.data.set_result(self.data.result() - other.data.result())

        deterministic_gather(self.data, other.data).add_done_callback(callback)
        return res

    def __rsub__(self, other):
        return -(other.__sub__(self))

    def __neg__(self):
        res = SuperFixedPoint(self.ctx, asyncio.Future())

        def callback(_):
            res.data.set_result(-self.data.result())

        self.data.add_done_callback(callback)
        return res

    def __mul__(self, other):
        assert type(other) in [int, float, SuperFixedPoint]
        if type(other) is not SuperFixedPoint:
            other = SuperFixedPoint(self.ctx, other)
        res = SuperFixedPoint(self.ctx, asyncio.Future())

        def reduce_callback(_):
            fut = asyncio.ensure_future(reduce_single(self.ctx,
                                                      self.data.result() *
                                                      other.data.result()))
            fut.add_done_callback(trunc_callback)

        def trunc_callback(val):
            fut = asyncio.ensure_future(trunc_pr_single(self.ctx, val.result(),
                                                        2 * K, F))
            fut.add_done_callback(result_callback)

        def result_callback(val):
            res.data.set_result(np.array(val.result()))

        deterministic_gather(self.data, other.data).add_done_callback(reduce_callback)
        return res

    __rmul__ = __mul__

    def __truediv__(self, other):
        assert type(other) in [int, float]
        inv = SuperFixedPoint(self.ctx, 1.0 / other)
        return self * inv

    def open(self):
        res = asyncio.Future()

        def data_callback(_):
            fut = asyncio.ensure_future(open_single(self.ctx, self.data.result()))
            return fut.add_done_callback(opening_callback)

        def opening_callback(val):
            x = val.result().value
            if x >= 2 ** (K - 1):
                x = -(p - x)

            return res.set_result(float(x) / 2 ** F)

        self.data.add_done_callback(data_callback)
        return res


def get_broadcast_shape(shape1, shape2):
    # Super inefficient. Needs to be improved
    return np.broadcast(np.empty(shape1), np.empty(shape2)).shape


class FixedPointNDArray(object):
    def __init__(self, ctx, x, shape=None):
        self.ctx = ctx
        if type(x) in [int, float]:
            self.data = asyncio.Future()
            self.data.set_result(_ppe.get_zero(ctx).v + to_fixed_point_repr(x))
            self.is_future = False
        if type(x) is asyncio.Future:
            assert shape is not None
            self.is_future = True
            self.data = x
            self.shape = shape
        else:
            def convert(z):
                return _ppe.get_zero(ctx).v + ctx.field(to_fixed_point_repr(z))

            convert = np.vectorize(convert)

            self.data = asyncio.Future()
            x = np.array(x)
            if x.dtype not in [np.float64, np.int64]:
                # print("Dtype is ", x.dtype)
                raise NotImplementedError

            self.data.set_result(convert(x))
            self.shape = x.shape

    def __add__(self, other):
        if type(other) is not FixedPointNDArray:
            other = FixedPointNDArray(self.ctx, other)

        res = FixedPointNDArray(self.ctx, asyncio.Future(),
                                get_broadcast_shape(self.shape, other.shape))

        def callback(_):
            # print("Add callback done")
            res.data.set_result(np.array(self.data.result() + other.data.result()))

        deterministic_gather(self.data, other.data).add_done_callback(callback)
        return res

    __radd__ = __add__

    def __sub__(self, other):
        if type(other) is not FixedPointNDArray:
            other = FixedPointNDArray(self.ctx, other)

        res = FixedPointNDArray(self.ctx, asyncio.Future(),
                                get_broadcast_shape(self.shape, other.shape))

        def callback(_):
            # print("Add callback done")
            res.data.set_result(np.array(self.data.result() - other.data.result()))

        deterministic_gather(self.data, other.data).add_done_callback(callback)
        return res

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __mul__(self, other):
        if type(other) is not FixedPointNDArray:
            other = FixedPointNDArray(self.ctx, other)

        res = FixedPointNDArray(self.ctx, asyncio.Future(),
                                get_broadcast_shape(self.shape, other.shape))

        def reduce_callback(_):
            raw_result = np.array(self.data.result() * other.data.result())
            fut = asyncio.ensure_future(reduce_array(self.ctx,
                                                     raw_result.flatten().tolist()))
            fut.add_done_callback(trunc_callback)

        def trunc_callback(val):
            fut = asyncio.ensure_future(trunc_pr_array(self.ctx, val.result(),
                                                       2 * K, F))
            fut.add_done_callback(result_callback)

        def result_callback(val):
            res.data.set_result(np.array(val.result()).reshape(res.shape))

        deterministic_gather(self.data, other.data).add_done_callback(reduce_callback)
        return res

    __rmul__ = __mul__

    def __matmul__(self, other):
        if type(other) is not FixedPointNDArray:
            other = FixedPointNDArray(self.ctx, other)

        res = FixedPointNDArray(self.ctx, asyncio.Future(),
                                self.shape[:-1] + other.shape[1:])

        if not (1 <= len(self.shape) <= 2 and 1 <= len(other.shape) <= 2):
            # print("Invalid dimensions")
            raise ValueError("Invalid dimensions")

        if self.shape[-1] != other.shape[0]:
            raise ValueError("Invalid dimensions for matmul")

        def reduce_callback(v):
            # print("reduce_callback", v)
            raw_result = self.data.result().dot(other.data.result())
            fut = asyncio.ensure_future(reduce_array(self.ctx,
                                                     raw_result.flatten().tolist()))
            fut.add_done_callback(trunc_callback)

        def trunc_callback(val):
            # print("turnc_callback", val)
            fut = asyncio.ensure_future(trunc_pr_array(self.ctx, val.result(),
                                                       2 * K, F))
            fut.add_done_callback(result_callback)

        def result_callback(val):
            # print("result_callback", val)
            res.data.set_result(np.array(val.result()).reshape(res.shape))

        deterministic_gather(self.data, other.data).add_done_callback(reduce_callback)
        return res

    def __neg__(self):
        res = FixedPointNDArray(self.ctx, asyncio.Future(), self.shape)

        def callback(_):
            res.data.set_result(-self.data.result())

        self.data.add_done_callback(callback)
        return res

    def transpose(self):
        res = FixedPointNDArray(self.ctx, asyncio.Future(), self.shape[::-1])

        def callback(_):
            res.data.set_result(self.data.result().T)

        self.data.add_done_callback(callback)
        return res

    def ltz(self):
        res = FixedPointNDArray(self.ctx, asyncio.Future(), self.shape)

        def trunc_callback(_):
            # print(_)
            fut = asyncio.ensure_future(trunc_array(self.ctx,
                                                    self.data.result().flatten(),
                                                    K, K - 1))
            fut.add_done_callback(result_callback)

        def result_callback(val):
            # print(val)
            res.data.set_result(-2 ** F * val.result().reshape(res.shape))

        self.data.add_done_callback(trunc_callback)
        return res

    def __lt__(self, other):
        if type(other) is not FixedPointNDArray:
            other = FixedPointNDArray(self.ctx, other)

        return (self - other).ltz()

    def __gt__(self, other):
        if type(other) is not FixedPointNDArray:
            other = FixedPointNDArray(self.ctx, other)
        return other.__lt__(self)

    def _reciprocal(self):
        res = FixedPointNDArray(self.ctx, asyncio.Future(), self.shape)

        def reciprocal_callback(_):
            # Time to evaluate reciprocal!
            # print(_)
            inv = asyncio.ensure_future(reciprocal(self.ctx,
                                                   self.data.result().flatten(),
                                                   K, F))
            inv.add_done_callback(set_data_callback)

        def set_data_callback(x):
            res.data.set_result(x.result().reshape(res.shape))

        self.data.add_done_callback(reciprocal_callback)
        return res

    def __truediv__(self, other):
        if type(other) is FixedPointNDArray:
            other_inv = other._reciprocal()
        else:
            other = np.array(other)
            assert other.dtype in [np.float64, np.int64]
            other_inv = FixedPointNDArray(self.ctx, 1.0 / other)
        return self * other_inv

    @property
    def T(self):
        return self.transpose()

    def open(self):
        res = asyncio.Future()

        def data_callback(_):
            fut = asyncio.ensure_future(open_array(self.ctx,
                                                   self.data.result().flatten().tolist())
                                        )
            return fut.add_done_callback(opening_callback)

        def opening_callback(reduced_array):
            def convert(val):
                x = val.value
                if x >= 2 ** (K - 1):
                    x = -(p - x)

                return float(x) / 2 ** F

            convert = np.vectorize(convert)
            flattened_result = convert(np.array(reduced_array.result()))
            return res.set_result(flattened_result.reshape(self.shape))

        self.data.add_done_callback(data_callback)
        return res

    async def resolve(self):
        await self.data

    def sqrt(self, theta=3):
        res = FixedPointNDArray(self.ctx, asyncio.Future(), self.shape)

        def approx_sqrt_callback(_):
            fut = asyncio.ensure_future(approx_sqrt(self.ctx,
                                                    self.data.result().flatten(),
                                                    K, F))
            fut.add_done_callback(iterative_set_callback)

        def iterative_set_callback(x0_data):
            x = FixedPointNDArray(self.ctx, asyncio.Future(), self.shape)
            x.data.set_result(x0_data.result().reshape(self.shape))

            for i in range(theta):
                try:
                    x = 0.5 * (x + self / x)
                except Exception as e:
                    print(e)
            x.data.add_done_callback(set_data_callback)

        def set_data_callback(x):
            res.data.set_result(x.result())

        self.data.add_done_callback(approx_sqrt_callback)
        return res

    def variance(self, axis=None):
        pass

    @property
    def size(self):
        res = 1
        for s in self.shape:
            res *= s
        return res

    def sum(self, axis=None):
        if axis is None:
            new_shape = tuple()
        else:
            new_shape = self.shape[:axis] + self.shape[axis + 1:]

        res = FixedPointNDArray(self.ctx, asyncio.Future(), new_shape)

        def sum_callback(_):
            res.data.set_result(np.array(np.sum(self.data.result(), axis=axis)))

        self.data.add_done_callback(sum_callback)
        return res

    def mean(self, axis=None):
        sum = self.sum(axis=axis)
        if axis is None:
            return sum / self.size
        else:
            return sum / self.shape[axis]

    def reshape(self, new_shape):
        res = FixedPointNDArray(self.ctx, asyncio.Future(), new_shape)

        def reshape_callback(_):
            res.data.set_result(self.data.result().reshape(new_shape))

        self.data.add_done_callback(reshape_callback)
        return res

    def var(self, axis=None):
        if axis is None:
            diff = (self - self.mean())
            return (diff * diff).mean()
        else:
            broadcast_shape = self.shape[:axis] + (1,) + self.shape[axis + 1:]
            diff = (self - self.mean(axis=axis).reshape(broadcast_shape))
            return (diff * diff).mean(axis=axis)

    def __getitem__(self, item):
        new_shape = np.zeros(self.shape)[item].shape

        res = FixedPointNDArray(self.ctx, asyncio.Future(), new_shape)

        def set_result_callback(_):
            res.data.set_result(self.data.result()[item])

        self.data.add_done_callback(set_result_callback)
        return res


def concatenate(ctx, arr, axis=0):
    # Make copy
    arr = list(arr)
    assert len(arr) > 0

    for i in range(len(arr)):
        if type(arr[i]) is not FixedPointNDArray:
            arr[i] = FixedPointNDArray(ctx, arr[i])

    # Take first shape as base shape
    sh = arr[0].shape[:axis] + arr[0].shape[axis + 1:]

    if not all((arr[i].shape[:axis] + arr[i].shape[axis + 1:]) == sh
               for i in range(len(arr))):
        raise ValueError("Invalid shapes")
    # print([arr[i].shape for i in range(len(arr))])
    new_axis_sum = sum(arr[i].shape[axis] for i in range(len(arr)))

    res = FixedPointNDArray(ctx, asyncio.Future(),
                            sh[:axis] + (new_axis_sum,) + sh[axis:])

    def callback(_):
        res.data.set_result(np.concatenate([arr[i].data.result()
                                            for i in range(len(arr))],
                                           axis=axis))

    deterministic_gather(*[x.data for x in arr]).add_done_callback(callback)
    return res


def uniform_random(low, high, shape, seed=0):
    """
    :param low: Lower boundary of interval
    :param high: Upper boundary of interval
    :param shape: Output shape
    :param seed: Random seed
    :return:
    """
    # Lock is required if multiple threads are being used
    # This ensures that the random numbers generated by each thread
    # are the same for the same seed
    random_lock.acquire()
    np.random.seed(seed)
    res = np.random.uniform(low, high, size=shape)
    random_lock.release()
    return res


async def sigmoid(x):
    """
    :type x: array-like
    :return: approximate sigmoid of x

    An approximation of the sigmoid function is found using the expansion
    y = 1/2 + x / 4 - x ^ 3 / 48 + x ^ 5 / 480

    We then cap this to be 0 if x < -2 and 1 if x > 2, since the expansion is not
    very accurate after this point
    """
    less = (x > -2)
    await less.resolve()
    gt = (x < 2)
    await gt.resolve()
    x2 = x * x
    x3 = x * x2
    x5 = x3 * x2
    return (0.5 + 0.25 * x - x3 / 48 + x5 / 480) * less * gt + (1 - gt)


def sigmoid_deriv(y):
    """
    :param y: sigma(x)
    :return: sigma'(x)
    """
    return y * (1 - y)


def relu(x):
    """
    relu(x) = max(0, x)
    """
    return (-x).ltz() * x


async def linear_regression_mpc(ctx, X, y, epochs=1, learning_rate=0.05):
    X = concatenate(ctx, [X, np.ones(X.shape[0]).reshape(-1, 1)], axis=1)
    theta = FixedPointNDArray(ctx,
                              uniform_random(-1, 1, X.shape[1], seed=0).reshape(-1, 1))

    N = X.shape[0]
    learning_rate = learning_rate / N
    for epoch in range(epochs):
        print(f"Starting epoch {epoch}")
        dtheta = (X.T @ ((X @ theta) - y))
        await dtheta.resolve()
        theta = theta - learning_rate * dtheta
        err = X @ theta - y
        print("Error: ", np.linalg.norm(await err.open()))
        print(f"EPOCH {epoch}", "theta: ", (await theta.open()).flatten())
    return theta


async def linear_regression_numpy(ctx, X, y, epochs=1, learning_rate=0.05):
    X = np.concatenate([X, np.ones(X.shape[0]).reshape(-1, 1)], axis=1)
    theta = np.zeros(X.shape[1]).reshape(-1, 1) + 0.5
    theta = uniform_random(-1, 1, theta.shape, seed=0)

    N = X.shape[0]
    learning_rate = learning_rate / N
    for epoch in range(epochs):
        dtheta = (X.T @ ((X @ theta) - y))
        theta = theta - learning_rate * dtheta
        print(f"EPOCH {epoch}", "theta: ", theta.flatten())
    return theta


class NeuralNetwork(object):
    def __init__(self, hidden_size, learning_rate, epochs):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs

    async def fit(self, ctx, X, y):
        self.input_size = X.shape[1]
        self.output_size = y.shape[0]

        self.w1 = FixedPointNDArray(ctx, uniform_random(0, 1, (self.input_size,
                                                               self.hidden_size),
                                                        seed=0))
        self.w2 = FixedPointNDArray(ctx, uniform_random(0, 1,
                                                        (self.hidden_size, 1),
                                                        seed=1))

        for epoch in range(self.epochs):
            start_time = time.time()
            # Forward
            print(f"-------------- Starting epoch {epoch} ------------------")
            print(f"-------------- Evaluating input layer ------------------")
            start_time2 = time.time()
            y1 = X @ self.w1
            # print("y1 done")
            await y1.resolve()
            end_time2 = time.time()
            print(f"First layer linear operations took {end_time2 - start_time2}s")

            start_time2 = time.time()
            l1 = await sigmoid(y1)
            await l1.resolve()
            end_time2 = time.time()
            print(f"First layer non-linear ops took {end_time2 - start_time2}s")
            gc.collect()

            print(f"-------------- Evaluating hidden layer -----------------")
            y2 = l1 @ self.w2
            await y2.resolve()

            l2 = await sigmoid(y2)

            await l2.resolve()
            gc.collect()

            print(f"--------------- Starting error evaluation --------------")
            #             print("l2: ", l2)

            err = y - l2
            print("Error: ", np.linalg.norm(await err.open()))

            print(f"--------------- Starting back-propagation ---------------")
            l2_delta = err * sigmoid_deriv(l2)
            await l2_delta.resolve()
            l1_delta = (l2_delta @ self.w2.T) * sigmoid_deriv(l1)
            await l1_delta.open()
            gc.collect()

            self.w2 = self.w2 + (l1.T @ l2_delta) * self.learning_rate
            self.w1 = self.w1 + (X.T @ l1_delta) * self.learning_rate
            await self.w1.resolve()
            await self.w2.resolve()
            print(f"--------------- Back-propagation complete ---------------")

            gc.collect()
            end_time = time.time()
            print(f"Epoch took {end_time - start_time}")

    async def evaluate(self, x):
        l1 = await sigmoid(x @ self.w1)
        l2 = await sigmoid(l1 @ self.w2)
        return (l2 > 0.5)


def accuracy(y_pred, y_actual):
    return np.sum(y_pred.reshape(-1, 1) == y_actual.reshape(-1, 1)) / len(y_actual)


async def _neural_network_mpc_program(ctx, x_train, y_train, x_test, y_test):
    # Currently, normalization is done in a non-MPC fashion. However, this shouldn't be
    # the case since normalization is a global operation.
    nn = NeuralNetwork(12, 0.24, 2)
    x_train = FixedPointNDArray(ctx, x_train)
    y_train = FixedPointNDArray(ctx, y_train.reshape(-1, 1))
    await nn.fit(ctx, x_train, y_train)
    x_test = FixedPointNDArray(ctx, x_test)
    evaluations = await nn.evaluate(x_test)
    print("Accuracy: ", accuracy((await evaluations.open()), y_test))


async def _linear_regression_mpc_program(ctx, X, y):
    """
    Given data y = 9 * x_ 1 + 4 * x_2 + 7 * x_3 + 2
    Find the coefficients of the best fit line
    Should be (9, 4, 7, 2)
    """
    theta = await linear_regression_mpc(ctx, X, y.reshape(-1, 1), epochs=50,
                                        learning_rate=0.05)


def set_ppe(v):
    global _ppe
    _ppe = v


async def _prog(ctx):
    # Testing spot
    def to_share(x):
        return ctx.field(to_fixed_point_repr(x))

    print("Starting _prog")
    # a = np.vectorize(to_share)([5.0])
    # b = await approx_sqrt(ctx, a, K, F)
    # b_opened = await open_nd_array(ctx, b)
    # print(np.vectorize(from_fixed_point_repr)(b_opened))
    a = FixedPointNDArray(ctx, np.array([4.0, 4.0]))
    b = FixedPointNDArray(ctx, np.array([4.0, 4.0]))
    c = a * b
    print(await c.open())
    # print(await b.open())
    # print(await b.open())

#
# if __name__ == "__main__":
#     n = 5
#     t = 1
#     multiprocess = True
#     ppe = PreProcessedElements()
#     set_ppe(ppe)
#     logging.info("Generating zeros in sharedata/")
#     _ppe.generate_zeros(1000, n, t)
#     logging.info("Generating random shares of bits in sharedata/")
#     _ppe.generate_random_bits(1000, n, t)
#     logging.info('Generating random shares in sharedata/')
#     _ppe.generate_rands(1000, n, t)
#     logging.info('Generating random shares of triples in sharedata/')
#     _ppe.generate_triples(1000, n, t)
#     logging.info("Generating random doubles in sharedata/")
#     _ppe.generate_double_shares(1000, n, t)
#
#     # logging.info('Generating random shares of bits in sharedata/')
#     # ppe.generate_bits(1000, n, t)
#
#     start_time = time.time()
#     asyncio.set_event_loop(asyncio.new_event_loop())
#     loop = asyncio.get_event_loop()
#     try:
#         config = {MixinOpName.MultiplyShare: BeaverTriple.multiply_shares}
#         if multiprocess:
#             from honeybadgermpc.config import HbmpcConfig
#             from honeybadgermpc.ipc import ProcessProgramRunner
#
#
#             async def _process_prog(peers, n, t, my_id):
#                 program_runner = ProcessProgramRunner(peers, n, t, my_id,
#                                                       config)
#                 await program_runner.start()
#                 # X = uniform_random(0, 10, (5, 3), seed=0)
#                 # y = 9 * X[:, 0] + 4 * X[:, 1] + 7 * X[:, 2] + 2
#
#                 program_runner.add(0, _prog)
#
#                 await program_runner.join()
#                 await program_runner.close()
#
#
#             loop.run_until_complete(_process_prog(
#                 HbmpcConfig.peers, HbmpcConfig.N, HbmpcConfig.t,
#                 HbmpcConfig.my_id))
#         else:
#             program_runner = TaskProgramRunner(n, t, config)
#             X = np.random.uniform(0, 10, (15, 3))
#             y = 9 * X[:, 0] + 4 * X[:, 1] + 7 * X[:, 2] + 2
#             X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
#             program_runner.add(_linear_regression_mpc_program, X=X, y=y)
#             loop.run_until_complete(program_runner.join())
#     finally:
#         loop.close()
#
#     end_time = time.time()
#     print(end_time - start_time)


# if __name__ == "__main__":
#     n = 5
#     t = 1
#     multiprocess = True
#     _ppe = PreProcessedElements()
#     logging.info("Generating zeros in sharedata/")
#     _ppe.generate_zeros(1000, n, t)
#     logging.info("Generating random shares of bits in sharedata/")
#     _ppe.generate_random_bits(1000, n, t)
#     logging.info('Generating random shares in sharedata/')
#     _ppe.generate_rands(1000, n, t)
#     logging.info('Generating random shares of triples in sharedata/')
#     _ppe.generate_triples(1000, n, t)
#     logging.info("Generating random doubles in sharedata/")
#     _ppe.generate_double_shares(1000, n, t)
#
#     # logging.info('Generating random shares of bits in sharedata/')
#     # ppe.generate_bits(1000, n, t)
#
#     start_time = time.time()
#     asyncio.set_event_loop(asyncio.new_event_loop())
#     loop = asyncio.get_event_loop()
#     try:
#         config = {MixinOpName.MultiplyShare: BeaverTriple.multiply_shares}
#         if multiprocess:
#             from honeybadgermpc.config import HbmpcConfig
#             from honeybadgermpc.ipc import ProcessProgramRunner
#
#
#             async def _process_prog(peers, n, t, my_id):
#                 program_runner = ProcessProgramRunner(peers, n, t, my_id,
#                                                       config)
#                 await program_runner.start()
#                 df = pd.read_csv('data.csv')
#                 del df['Unnamed: 32']
#
#                 X = df.iloc[:, 2:].values
#                 y = np.vectorize(lambda x: 1 if x == 'M' else 0)(df.iloc[:, 1].values)
#                 x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
#                                                                     random_state=0)
#
#                 # Normalize values
#                 x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
#                 x_test = (x_test - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)
#
#                 program_runner.add(0, _neural_network_mpc_program,
#                                    x_train=x_train[:64], y_train=y_train[:64],
#                                    x_test=x_test[:32], y_test=y_test[:32])
#                 await program_runner.join()
#                 await program_runner.close()
#
#
#             loop.run_until_complete(_process_prog(
#                 HbmpcConfig.peers, HbmpcConfig.N, HbmpcConfig.t,
#                 HbmpcConfig.my_id))
#         else:
#             program_runner = TaskProgramRunner(n, t, config)
#             #
#             # df = pd.read_csv('data.csv')
#             # del df['Unnamed: 32']
#             #
#             # X = df.iloc[:, 2:].values
#             # y = np.vectorize(lambda x: 1 if x == 'M' else 0)(df.iloc[:, 1].values)
#             # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
#             #                                                     random_state=0)
#             #
#             # # Normalize values
#             # x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
#             # x_test = (x_test - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)
#             #
#             # program_runner.add(_neural_network_mpc_program,
#             #                    x_train=x_train[:32], y_train=y_train[:32],
#             #                    x_test=x_test[:32], y_test=y_test[:32])
#             X = np.random.uniform(0, 10, (15, 3))
#             y = 9 * X[:, 0] + 4 * X[:, 1] + 7 * X[:, 2] + 2
#             X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
#             program_runner.add(_linear_regression_mpc_program, X=X, y=y)
#             # program_runner.add(_prog)
#             loop.run_until_complete(program_runner.join())
#     finally:
#         loop.close()
#
#     end_time = time.time()
#     print(end_time - start_time)
