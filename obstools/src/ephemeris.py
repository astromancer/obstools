# coding: utf-8


# std libs
from functools import partial

# third-party libs
import numpy as np

# local libs
from recipes.lists import cosort

from matplotlib import ticker

from matplotlib.transforms import Affine2D


class Ephemeris(object):
    # TODO: def from_string(self):
    # TODO: fit method!!

    def __init__(self, *args):
        """
        to initialize use:
        >>> eph = Ephemeris(t0, P)
        or if you have uncertainties
        >>> eph = Ephemeris(t0, σ_t0, P, σ_t0)
        to initialize

        Parameters
        ----------
        t0:
            zero point for ephemeris
        P:
            period in days
        σ_t0:
            uncertainty on zero point
        σ_P:
            uncertainty on period

        """

        if len(args) == 2:
            self.t0, self.P = args
            self.e_t0 = self.e_P = None

        elif len(args) == 4:
            self.t0, self.e_t0, self.P, self.e_P = args
            # oom = 3 - len(str(self.P))
            # σp = self.e_P * 10 ** (oom - 1)
        else:
            raise TypeError('Invalid number of arguments')

        #

    def __str__(self):
        if self.e_t0:
            t = '{:f}({:02d})'.format(self.t0, self.e_t0)

        return 'HJD = %f + %f E' % (self.t0, self.P)

    def __call__(self, E):
        return self.t0 + E * self.P

    def phase(self, t):
        return np.atleast_1d(t - self.t0) / self.P

    def phase_mod1(self, t):
        return self.phase(t) % 1


def binner(t, data, bins, empty='mask'):
    """Calculate mean of data in bins.  Empty bins are masked"""
    if not np.all(np.diff(t) > 0):
        raise ValueError('t must be monotonic!')

    dz = np.digitize(t, bins)
    unq, ix = np.unique(dz, return_index=True)
    blc = np.vectorize(np.mean)(np.split(data, ix[1:]))

    binned = np.zeros_like(bins)
    binned[unq - 1] = blc

    return np.ma.masked_where(binned == 0, binned)


# def smart_bins(t):   # use bayesian blocks??
#     pass


_pr = False


def err(a, b, s):
    """
    Calculate the total "error" between two curves, (normalized by the number of
    overlapping points) given the time shift s.
    """
    if s < 0:
        a, b, s = b, a, -s  # exploit the symmetry of the problem
    o = len(a) - s  # number of overlapping points. N = len(a)

    if _pr:
        print('N, o, s')
        print(len(a), o, s)
        print(a[s:])
        print(b[:o])
    # n = b[:o].mask.sum()

    # np.sqrt(np.square(a[s:] - b[:o]).sum()) / o
    return abs(a[s:] - b[
                       :o]).sum() / o  # absolute sum counts small values with equal weight to large ones.


# fig, ax  = plt.subplots(figsize=(18,8))

def shifter(a, b, min_overlap=.25):
    """ """

    s0 = -int((~b.mask).sum() * (1 - min_overlap))
    se = int((~a.mask).sum() * (1 - min_overlap))

    S = range(s0, se)
    E = np.vectorize(partial(err, a, b))(S)

    return S, E


def brutus(a, b, min_overlap=.25):
    """brute force minimum search over integer shift"""
    # determine min and max shift

    S, E = shifter(a, b, min_overlap)
    # print(S)
    # print(E)
    # ax.plot(S, E)

    return S[np.nanargmin(E)]


def phaser(t, period):
    phase = t / period
    return phase - np.floor(phase[0])


def bin_centers(bins):
    return np.c_[bins, np.roll(bins, -1)].mean(1)[:-1]


def rephase(phase, offset, *data):
    phase = phase + offset  # local namespace
    phase -= np.floor(phase.min())
    phase %= 1
    if data is None:
        return phase
    else:
        data = np.array(cosort(phase, *data))

        phase = data[0]
        data = data[1:]
        return phase, data


def phase_splitter(ph, *data, **kws):
    """Split according to phase revolution"""
    revs, ix = np.unique(np.floor(ph), return_index=True)

    if kws.get('mod', True):
        ph = ph % 1

    phs, *datas = (np.split(item, ix[1:]) for item in (ph,) + data)
    return phs, datas
