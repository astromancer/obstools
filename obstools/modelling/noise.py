import numpy as np

from scipy.stats import poisson  # norm
from scipy.special import erf



# __all__ = ['GaussUni', 'PoissonGauss', 'PoissonGaussAdaptive',
#            'PoissonGaussUni', 'PoissonGaussUniAdaptive']


# from decor.profiler import profiler

# @profiler.histogram()
def GaussUni(x, mu, sigma, l):
    """
    Joint pdf of combining Amplifier readout noise ~ Normal(σ, μ) and
    ADU digitization (a.k.a. quantization) noise ~Uniform(-l/2, l/2)

    Parameters     (in units of electrons)
    ----------
    b = bias level
    sigma = read noise
    delta = dark current rate
    l = least-significant bit of ADU
    """
    denom = np.sqrt(2) * sigma
    part1 = (x - mu) / denom
    ad = 0.5 * l / denom
    return (erf(part1 + ad) - erf(part1 - ad)) / (2 * l)


# def PoissonGauss(x, delta, mu, sigma, N=300):
#     """Poisson Gauss convolution"""
#     k = np.arange(N)
#     p = np.power(delta, k) * np.exp(-delta) / factorial(k)  # NOT ACCURATE!!
#     q = np.exp(-0.5 * np.square((x - k - mu) / sigma))
#     return np.sum(p * q) / (np.sqrt(2 * np.pi) * sigma)


def PoissonGauss(x, delta, mu, sigma, N=300):
    """
    Joint pdf of CCD pixel dark current Poiss(δ) and Amplifier readout
     noise N(σ, μ).
    i.e. Poisson Gauss convolution.

    Blind sum first N terms

    Parameters     (in units of electrons)
    ----------
    delta = dark current rate
    mu = bias level
    sigma = read noise
    """
    k = np.arange(N, dtype=int)
    p = poisson.pmf(k, delta)  # (delta)
    q = np.exp(-0.5 * np.square((x - k - mu) / sigma))
    return np.sum(p * q, -1) / (np.sqrt(2 * np.pi) * sigma)


# def poissonAdaptiveRange(delta):
#
#     poiss = poisson(delta)
#     kl, ku = poiss.interval(alpha)
#     return np.arange(kl, ku, dtype=int)


def PoissonGaussAdaptive(x, delta, mu, sigma, alpha=1 - 1e-6):
    """
    Joint pdf of CCD pixel dark current Poiss(δ) and amplifier readout
    noise N(σ, μ).
    i.e. Poisson Gauss convolution

    Adaptive sum based on confidence intervals on delta

    Parameters     (in units of electrons)
    ----------
    delta = dark current rate
    mu = bias level
    sigma = read noise
    """
    poiss = poisson(delta)
    kl, ku = poiss.interval(alpha)
    k = np.arange(kl, ku, dtype=int)[None]
    p = poiss.pmf(k)

    q = np.exp(-0.5 * np.square((x - k - mu) / sigma))
    return np.sum(p * q, -1) / (np.sqrt(2 * np.pi) * sigma)


def PoissonGaussUni(x, delta, mu, sigma, l, N=300):
    """
    Joint pdf of CCD pixel dark current Poiss(δ) and Amplifier readout
    noise N(σ, μ).
    i.e. Poisson Gauss convolution.

    Parameters     (in units of electrons)
    ----------
    delta = dark current rate
    mu = bias level
    sigma = read noise
    l = least-significant bit of ADU
    """
    k = np.arange(N, dtype=int)
    p = poisson(delta).pmf(k)
    q = GaussUni(x, k + mu, sigma, l)
    return np.sum(p * q, -1)  # / (np.sqrt(2 * np.pi) * sigma)


def PoissonGaussUniAdaptive(x, delta, mu, sigma, l, alpha=1 - 1e-6):
    """
    Joint pdf of CCD pixel dark current Poiss(δ) and Amplifier readout
    noise N(σ, μ).
    i.e. Poisson Gauss convolution.

    Parameters     (in units of electrons)
    ----------
    delta = dark current rate
    mu = bias level
    sigma = read noise
    l = least-significant bit of ADU
    """
    poiss = poisson(delta)
    kl, ku = poiss.interval(alpha)
    k = np.arange(kl, ku, dtype=int)
    p = poiss.pmf(k)
    q = GaussUni(x, k + mu, sigma, l)
    return np.sum(p * q, -1)  # / (np.sqrt(2 * np.pi) * sigma)


# def dark(z, b, delta, sigma, l):
#     """
#     Joint pdf of CCD pixel dark current (δ) and amplifier bias voltage (b).
#     * Gaussian current rate i.e. (delta >> 1) Gaussian approx to Poisson rate is appropriate
#     * Gaussian readout noise
#     * Uniform digitization (quantization noise)
#
#     Parameters
#     ----------
#     (in units of electrons)
#     b = bias level
#     sigma = read noise
#     delta = dark current rate
#     l = least-significant bit of ADU
#     """
#
#     # lsb = np.array([1, -1], ndmin=2).T * 0.5 *
#     a = b + delta
#     s = sigma + delta
#     denom = np.sqrt(2) * s
#     part1 = (z - a) / denom
#     ad = 0.5 * l / denom
#     return (erf(part1 + ad) - erf(part1 - ad)) / (2 * l)


if __name__ == '__main__':
    from scipy.integrate import quad


    def guess_range(delta, b, sigma, size=250):
        a = b + delta
        s = sigma + delta
        return np.linspace(max(0, a - 7 * s), a + 10 * s ** 0.1, size)


    def integrate(func, args, verbose=True):
        """check if proper distribution"""
        A, Ae = quad(func, -np.inf, np.inf, args)
        if verbose:
            print(func, A)
            # if round(A, 6) != 1:
        return A, Ae


    def test_integrate():
        mu = 10
        sigma = 1
        delta = 0.5
        l = 1
        alpha = 1 - 1e-6

        integrate(GaussUni, (mu, sigma, l))
        integrate(PoissonGauss, (delta, mu, sigma))
        integrate(PoissonGaussAdaptive, (delta, mu, sigma, alpha))
        integrate(PoissonGaussUni, (delta, mu, sigma, l))
        integrate(PoissonGaussUniAdaptive, (delta, mu, sigma, l, alpha))


    def profile(N=1e2):
        # from line_profiler import LineProfiler
        from decor.profiler import HLineProfiler

        # from recipes.io.tracewarn import warning_traceback_on
        # warning_traceback_on()

        profiler = HLineProfiler()
        for fname in __all__:
            profiler.add_function(eval(fname))
        profiler.enable_by_count()

        mu = 10
        sigma = 1
        delta = 0.5
        l = 1
        alpha = 1 - 1e-6
        x = guess_range(delta, mu, sigma, size=250)[None].T

        for _ in range(int(N)):
            GaussUni(x, mu, sigma, l)
            PoissonGauss(x, delta, mu, sigma)
            PoissonGaussAdaptive(x, delta, mu, sigma, alpha)
            PoissonGaussUni(x, delta, mu, sigma, l)
            PoissonGaussUniAdaptive(x, delta, mu, sigma, l, alpha)
            #

        profiler.print_stats()
        profiler.rank_functions()


    # test_integrate()
    profile()
