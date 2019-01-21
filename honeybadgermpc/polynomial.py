import operator
import random
from functools import reduce
from .field import GF, GFElement
from itertools import zip_longest
from honeybadgermpc.betterpairing import ZR, bls12_381_r


def strip_trailing_zeros(a):
    if len(a) == 0:
        return []
    for i in range(len(a), 0, -1):
        if a[i - 1] != 0:
            break
    return a[:i]


_poly_cache = {}


def polynomialsOver(field):
    if field in _poly_cache:
        return _poly_cache[field]

    USE_RUST = False
    if field.modulus == bls12_381_r:
        USE_RUST = False
        print('using bls12_381_r')

    class Polynomial(object):
        def __init__(self, coeffs):
            self.coeffs = list(strip_trailing_zeros(coeffs))
            for i in range(len(self.coeffs)):
                if type(self.coeffs[i]) is int:
                    self.coeffs[i] = field(self.coeffs[i])
            if USE_RUST:
                self._zrcoeffs = [ZR(c.value) for c in self.coeffs]
            self.field = field

        def isZero(self):
            return self.coeffs == [] or (len(self.coeffs) == 1 and self.coeffs[0] == 0)

        def __repr__(self):
            if self.isZero():
                return '0'
            return ' + '.join(['%s x^%d' % (a, i) if i > 0 else '%s' % a
                               for i, a in enumerate(self.coeffs)])

        def __call__(self, x):
            if USE_RUST:
                assert type(x) is int
                x = ZR(x)
                k = len(self.coeffs) - 1
                y = ZR(0)
                for i in range(k):
                    y *= x
                    y += self._zrcoeffs[k - i]
                return field(int(y))
            else:
                y = 0
                xx = 1
                for coeff in self.coeffs:
                    y += coeff * xx
                    xx *= x
                return y

        @classmethod
        def interpolate_at(cls, shares, x_recomb=field(0)):
            # shares are in the form (x, y=f(x))
            if type(x_recomb) is int:
                x_recomb = field(x_recomb)
            assert type(x_recomb) is GFElement
            xs, ys = zip(*shares)
            vector = []
            for i, x_i in enumerate(xs):
                factors = [(x_k - x_recomb) / (x_k - x_i)
                           for k, x_k in enumerate(xs) if k != i]
                vector.append(reduce(operator.mul, factors))
            return sum(map(operator.mul, ys, vector))

        _lagrangeCache = {}  # Cache lagrange polynomials

        @classmethod
        def interpolate(cls, shares):
            X = cls([field(0), field(1)])  # This is the polynomial f(x) = x
            ONE = cls([field(1)])  # This is the polynomial f(x) = 1
            xs, ys = zip(*shares)

            def lagrange(xi):
                # Let's cache lagrange values
                if (xs, xi) in cls._lagrangeCache:
                    return cls._lagrangeCache[(xs, xi)]

                def mul(a, b): return a * b

                num = reduce(mul, [X - cls([xj])
                                   for xj in xs if xj != xi], ONE)
                den = reduce(mul, [xi - xj for xj in xs if xj != xi], field(1))
                p = num * cls([1 / den])
                cls._lagrangeCache[(xs, xi)] = p
                return p

            f = cls([0])
            for xi, yi in zip(xs, ys):
                pi = lagrange(xi)
                f += cls([yi]) * pi
            return f

        @classmethod
        def interpolate_fft(cls, ys, omega):
            """
            Returns a polynoial f of given degree,
            such that f(omega^i) == ys[i]
            """
            n = len(ys)
            assert n & (n - 1) == 0, "n must be power of two"
            assert type(omega) is GFElement
            assert omega ** n == 1, "must be an n'th root of unity"
            assert omega ** (n //
                             2) != 1, "must be a primitive n'th root of unity"
            coeffs = [b / n for b in fft_helper(ys, 1 / omega, field)]
            return cls(coeffs)

        def evaluate_fft(self, omega, n):
            assert n & (n - 1) == 0, "n must be power of two"
            assert type(omega) is GFElement
            assert omega ** n == 1, "must be an n'th root of unity"
            assert omega ** (n //
                             2) != 1, "must be a primitive n'th root of unity"
            return fft(self, omega, n)

        @classmethod
        def random(cls, degree, y0=None):
            coeffs = [field(random.randint(0, field.modulus - 1))
                      for _ in range(degree + 1)]
            if y0 is not None:
                coeffs[0] = y0
            return cls(coeffs)

        @classmethod
        def interp_extrap(cls, xs, omega):
            """
            Interpolates the polynomial based on the even points omega^2i
            then evaluates at all points omega^i
            """
            n = len(xs)
            assert n & (n - 1) == 0, "n must be power of 2"
            assert pow(omega, 2 * n) == 1, "omega must be 2n'th root of unity"
            assert pow(
                omega, n) != 1, "omega must be primitive 2n'th root of unity"

            # Interpolate the polynomial up to degree n
            poly = cls.interpolate_fft(xs, omega ** 2)

            # Evaluate the polynomial
            xs2 = poly.evaluate_fft(omega, 2 * n)

            return xs2

        # the valuation only gives 0 to the zero polynomial, i.e. 1+degree
        def __abs__(self):
            return len(self.coeffs)

        def __iter__(self):
            return iter(self.coeffs)

        def __sub__(self, other):
            return self + (-other)

        def __neg__(self):
            return Polynomial([-a for a in self])

        def __len__(self):
            return len(self.coeffs)

        def __add__(self, other):
            newCoefficients = [sum(x) for x in zip_longest(
                self, other, fillvalue=self.field(0))]
            return Polynomial(newCoefficients)

        def __mul__(self, other):
            if self.isZero() or other.isZero():
                return Zero()

            newCoeffs = [self.field(0)
                         for _ in range(len(self) + len(other) - 1)]

            for i, a in enumerate(self):
                for j, b in enumerate(other):
                    newCoeffs[i + j] += a * b
            return Polynomial(newCoeffs)

        def degree(self):
            return abs(self) - 1

        def leadingCoefficient(self):
            return self.coeffs[-1]

        def derivative(self):
            new_coeffs = self.coeffs[1:]
            new_coeffs = [new_coeffs[i] * (i + 1) for i in range(len(new_coeffs))]
            return Polynomial(new_coeffs)

        @staticmethod
        def build_from_roots(roots):
            """O(N) method to create polynomial from its roots

            :param roots: Roots of polynomial
            :return: Polynomial of degree len(roots)
            """
            if len(roots) == 0:
                return Polynomial([1])
            elif len(roots) == 1:
                return Polynomial([-roots[0], 1])

            n = len(roots)
            return Polynomial.build_from_roots(roots[:n // 2]) * \
                   Polynomial.build_from_roots(roots[n // 2:])

        def __divmod__(self, divisor):
            quotient, remainder = Zero(), self
            divisorDeg = divisor.degree()
            divisorLC = divisor.leadingCoefficient()

            while remainder.degree() >= divisorDeg:
                monomialExponent = remainder.degree() - divisorDeg
                monomialZeros = [self.field(0)
                                 for _ in range(monomialExponent)]
                monomialDivisor = Polynomial(
                    monomialZeros + [remainder.leadingCoefficient() / divisorLC])

                quotient += monomialDivisor
                remainder -= monomialDivisor * divisor

            return quotient, remainder

        def __truediv__(self, divisor):
            if divisor.isZero():
                raise ZeroDivisionError
            return divmod(self, divisor)[0]

        def __mod__(self, divisor):
            if divisor.isZero():
                raise ZeroDivisionError
            return divmod(self, divisor)[1]

    def Zero():
        return Polynomial([])

    _poly_cache[field] = Polynomial
    return Polynomial


def get_omega(field, n, seed=None):
    """
    Given a field, this method returns an n^th root of unity.
    If the seed is not None then this method will return the
    same n'th root of unity for every run with the same seed

    This only makes sense if n is a power of 2!
    """
    assert n & n - 1 == 0, "n must be a power of 2"

    if seed is not None:
        random.seed(seed)
    x = field(random.randint(0, field.modulus - 1))
    y = pow(x, (field.modulus - 1) // n)
    if y == 1 or pow(y, n // 2) == 1:
        return get_omega(field, n)
    assert pow(y, n) == 1, "omega must be 2n'th root of unity."
    assert pow(y, n // 2) != 1, "omega must be primitive 2n'th root of unity"

    return y


def fft_helper(A, omega, field):
    """
    Given coefficients A of polynomial this method does FFT and returns
    the evaluation of the polynomial at [omega^0, omega^(n-1)]

    If the polynomial is a0*x^0 + a1*x^1 + ... + an*x^n then the coefficients
    list is of the form [a0, a1, ... , an].
    """
    n = len(A)
    assert not (n & (n - 1)), "n must be a power of 2"

    if n == 1:
        return A

    B, C = A[0::2], A[1::2]
    B_bar = fft_helper(B, pow(omega, 2), field)
    C_bar = fft_helper(C, pow(omega, 2), field)
    A_bar = [field(1)] * (n)
    for j in range(n):
        k = (j % (n // 2))
        A_bar[j] = B_bar[k] + pow(omega, j) * C_bar[k]
    return A_bar


def fft(poly, omega, n, seed=None):
    assert n & n - 1 == 0, "n must be a power of 2"
    assert len(poly.coeffs) <= n
    assert pow(omega, n) == 1
    assert pow(omega, n // 2) != 1

    paddedCoeffs = poly.coeffs + ([poly.field(0)] * (n - len(poly.coeffs)))
    return fft_helper(paddedCoeffs, omega, poly.field)


def multiply_fft(Poly, p, q, omega, n):
    """Multiply polynomials p and q using FFT"""

    if n & (n - 1) != 0:
        raise ValueError(f"n must be a power of 2. n={n}")

    if p.field != q.field:
        raise ValueError(f"Fields of polynomials must match")

    if omega ** n != 1:
        raise ValueError("omega must be the n'th root of unity")

    if omega ** (n // 2) == 1:
        raise ValueError("omega must be a primitive n'th root of unity")

    a = p.evaluate_fft(omega, n)
    b = q.evaluate_fft(omega, n)
    c = [ai * bi for (ai, bi) in zip(a, b)]
    return Poly.interpolate_fft(c, omega)


def fnt_decode(Poly, zs, ys, omega2, n):
    """
    :param Poly: Polynomial generator
    :param zs: Subset of [0, n)
    :param ys:
    :param omega2: (2n)'th root of unity
    :param n:
    :return:

    Reference: Soro and Lacan. FNT-based Reed-Solomon Erasure Codes
    Section II B (https://arxiv.org/pdf/0907.1788.pdf)
    """
    omega = omega2 ** 2
    k = len(zs)

    if n & (n - 1) != 0:
        raise ValueError(f"n must be a power of 2 (n = {n})")

    if k > n:
        raise ValueError(f"k must be lesser than or equal to n. (k={k})")

    if len(zs) != len(ys):
        raise ValueError(f"zs and ys arrays must be of same length.")

    if omega2 ** (2 * n) != 1:
        raise ValueError(f"omega2 must be 2n'th root of unity")

    if omega2 ** (n) == 1:
        raise ValueError(f"omega2 must be primitive 2n'th root of unity")

    xs = [omega ** z for z in zs]

    # Step 1: Generate polynomial A(x)
    def build_polynomial(roots):
        if len(roots) == 0:
            return Poly([1])
        elif len(roots) == 1:
            return Poly([-roots[0], 1])
        p = build_polynomial(roots[:len(roots) // 2])
        q = build_polynomial(roots[len(roots) // 2:])
        return p * q

    A = build_polynomial(xs)

    # Step 2: Differentiate polynomial
    Ad = A.derivative()

    # Step 3: Evaluate Ad at x0, x1, ..., x_(k-1)
    As_all = Ad.evaluate_fft(omega, n)
    As = [As_all[z] for z in zs]

    # Step 4: Process vi / Ai(xi)
    ns = [ys[i] / As[i] for i in range(k)]

    # Step 5: Calculate polynomial Q(x) = P(x) / A(x)
    # Step 5.1) Create polynomial N'(s) = \sum_j^{-n_i * x^{z_i}}
    # In other words, coefficient of x^{z_i} is n_i
    N_coeffs = [0 for _ in range(n)]
    for i in range(k):
        N_coeffs[zs[i]] = ns[i]
    N = Poly(N_coeffs)

    # Step 5.2) Generate Q(x)
    # Coefficient of x^i in Q(x) is -N'(omega^(-i - 1))
    Q = -Poly(N.evaluate_fft(omega, n)[::-1])
    # Q = Poly([-N(omega ** (-i - 1)) for i in range(n)])

    # Step 6) P(x) = Q(x) * A(x)
    # P = Q * A
    P = multiply_fft(Poly, Q, A, omega2, 2 * n)
    result = Poly(P.coeffs[:k])

    return result


def fnt_decode_step1(Poly, zs, omega2, n):
    """
    This needs to be run once for decoding a batch of secret shares
    It depends only on the x values (the points the polynomial is
    evaluated at, i.e. the IDs of the parties contributing shares) so
    it can be reused for multiple batches.
    Complexity: O(n^2)

    args:
        zs is a subset of [0,n)
        omega2 is a (2*n)th root of unity

    returns:
        A(X) evaluated at 1...omega2**(2n-1)
        Ai(xi) for each xi = omega**(zi)

    where:
        omega = omega2**2
        where A(X) = prod( X - xj ) for each xj
        Ai(xi) = prod( xi - xj ) for j != i
    """
    k = len(zs)
    omega = omega2 ** 2
    xs = [omega ** z for z in zs]

    # Compute A(X)
    A = Poly([1])
    for i in range(k):
        A *= Poly([-xs[i], 1])
    As = [A(omega2 ** i) for i in range(2 * n)]

    # Compute all Ai(Xi)
    Ais = []
    for i in range(k):
        Ai = A.field(1)
        for j in range(k):
            if i != j:
                Ai *= xs[i] - xs[j]
        Ais.append(Ai)
    return As, Ais


def fnt_decode_step2(Poly, zs, ys, As, Ais, omega2, n):
    """
    Returns a polynomial P such that P(omega**zi) = yi

    Complexity: O(n log n)

    args:
        zs is a subset of [0,n)
        As, Ais = fnt_decode_step1(zs, omega2, n)
        omega2 is a (2*n)th root of unity

    returns:
        P  Poly
    """
    k = len(ys)
    assert len(ys) == len(Ais)
    assert len(As) == 2 * n
    omega = omega2 ** 2

    # Compute N'(x)
    nis = [ys[i] / Ais[i] for i in range(k)]
    Ncoeffs = [0 for _ in range(n)]
    for i in range(k):
        Ncoeffs[zs[i]] = nis[i]

    N = Poly(Ncoeffs)

    # Compute P/A(X)
    # PoverA = -Poly([N(omega**(n-j-1)) for j in range(n)])
    Nevals = N.evaluate_fft(omega, n)
    PoverA = -Poly(Nevals[::-1])
    pas = PoverA.evaluate_fft(omega2, 2 * n)

    # Recover P(X)
    ps = [p * a for (p, a) in zip(pas, As)]
    Prec = Poly.interpolate_fft(ps, omega2)
    Prec.coeffs = Prec.coeffs[:k]
    return Prec


if __name__ == "__main__":
    field = GF.get(
        0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001)
    Poly = polynomialsOver(field)
    poly = Poly.random(degree=7)
    poly = Poly([1, 5, 3, 15, 0, 3])
    n = 2 ** 3
    omega = get_omega(field, n, seed=1)
    omega2 = get_omega(field, n, seed=4)
    # FFT
    # x = fft(poly, omega=omega, n=n, test=True, enable_profiling=True)
    x = poly.evaluate_fft(omega, n)
    # IFFT
    x2 = [b / n for b in fft_helper(x, 1 / omega, field)]
    poly2 = Poly.interpolate_fft(x2, omega)
    print(poly2)

    print('omega1:', omega ** (n // 2))
    print('omega2:', omega2 ** (n // 2))

    print('eval:')
    omega = get_omega(field, 2 * n)
    for i in range(len(x)):
        print(omega ** (2 * i), x[i])
    print('interp_extrap:')
    x3 = Poly.interp_extrap(x, omega)
    for i in range(len(x3)):
        print(omega ** i, x3[i])

    print("How many omegas are there?")
    for i in range(10):
        omega = get_omega(field, 2 ** 20)
        print(omega, omega ** (2 ** 17))
