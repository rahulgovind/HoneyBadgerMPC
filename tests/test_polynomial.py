from random import randint, shuffle
from honeybadgermpc.polynomial import get_omega, fnt_decode_step1, \
    fnt_decode_step2, fnt_decode


def test_poly_eval_at_k(GaloisField, Polynomial):
    poly1 = Polynomial([0, 1])  # y = x
    for i in range(10):
        assert poly1(i) == i

    poly2 = Polynomial([10, 0, 1])  # y = x^2 + 10
    for i in range(10):
        assert poly2(i) == pow(i, 2) + 10

    d = randint(1, 50)
    coeffs = [randint(0, GaloisField.modulus - 1) for i in range(d)]
    poly3 = Polynomial(coeffs)  # random polynomial of degree d
    x = GaloisField(randint(0, GaloisField.modulus - 1))
    y = sum([pow(x, i) * a for i, a in enumerate(coeffs)])
    assert y == poly3(x)


def test_evaluate_fft(GaloisField, Polynomial):
    d = randint(210, 300)
    coeffs = [randint(0, GaloisField.modulus - 1) for i in range(d)]
    poly = Polynomial(coeffs)  # random polynomial of degree d
    n = len(poly.coeffs)
    n = n if n & n - 1 == 0 else 2 ** n.bit_length()
    omega = get_omega(GaloisField, n)
    fftResult = poly.evaluate_fft(omega, n)
    assert len(fftResult) == n
    for i, a in zip(range(1, 201, 2), fftResult[1:201:2]):  # verify only 100 points
        assert poly(pow(omega, i)) == a


def test_interpolate_fft(GaloisField, Polynomial):
    d = randint(210, 300)
    y = [randint(0, GaloisField.modulus - 1) for i in range(d)]
    n = len(y)
    n = n if n & n - 1 == 0 else 2 ** n.bit_length()
    ys = y + [GaloisField(0)] * (n - len(y))
    omega = get_omega(GaloisField, n)
    poly = Polynomial.interpolate_fft(ys, omega)
    for i, a in zip(range(1, 201, 2), ys[1:201:2]):  # verify only 100 points
        assert poly(pow(omega, i)) == a


def test_interp_extrap(GaloisField, Polynomial):
    d = randint(210, 300)
    y = [randint(0, GaloisField.modulus - 1) for i in range(d)]
    n = len(y)
    n = n if n & n - 1 == 0 else 2 ** n.bit_length()
    ys = y + [GaloisField(0)] * (n - len(y))
    omega = get_omega(GaloisField, 2 * n)
    values = Polynomial.interp_extrap(ys, omega)
    for a, b in zip(ys, values[0:201:2]):  # verify only 100 points
        assert a == b


def test_fft_decode(GaloisField, Polynomial):
    d = randint(210, 300)
    d = 500
    coeffs = [randint(0, GaloisField.modulus - 1) for i in range(d)]
    P = Polynomial(coeffs)
    n = d
    n = n if n & n - 1 == 0 else 2 ** n.bit_length()
    omega2 = get_omega(GaloisField, 2 * n)
    omega = omega2 ** 2

    # Create shares and erasures
    zs = list(range(n))
    shuffle(zs)
    zs = zs[:d]
    ys = list(P.evaluate_fft(omega, n))
    ys = [ys[i] for i in zs]

    import time
    start = time.time()
    As_, Ais_ = fnt_decode_step1(Polynomial, zs, omega2, n)
    Prec_ = fnt_decode_step2(Polynomial, zs, ys, As_, Ais_, omega2, n)
    print("Decode: ", time.time() - start)

    assert Prec_(0) == P(0)
    assert Prec_.coeffs == P.coeffs


def test_fft_decode2(GaloisField, Polynomial):
    d = randint(210, 300)
    d = 500
    coeffs = [randint(0, GaloisField.modulus - 1) for _ in range(d)]
    P = Polynomial(coeffs)
    n = d
    n = n if n & n - 1 == 0 else 2 ** n.bit_length()
    omega2 = get_omega(GaloisField, 2 * n)
    omega = omega2 ** 2

    # Create shares and erasures
    zs = list(range(n))
    shuffle(zs)
    zs = zs[:d]
    ys = list(P.evaluate_fft(omega, n))
    ys = [ys[i] for i in zs]

    import time
    start = time.time()
    Prec_ = fnt_decode(Polynomial, zs, ys, omega2, n)
    print("Decode 2: ", time.time() - start)

    assert Prec_(0) == P(0)
    assert Prec_.coeffs == P.coeffs, (Prec_.coeffs, P.coeffs)


def time_decode():
    from honeybadgermpc.field import GF
    from honeybadgermpc.polynomial import polynomialsOver
    field = GF.get(0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001)
    Poly = polynomialsOver(field)
    test_fft_decode(field, Poly)
    test_fft_decode2(field, Poly)


if __name__ == "__main__":
    import timeit
    # timeit.timeit("time_decode()", setup="from __main__ import time_decode")
    time_decode()
