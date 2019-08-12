# coding: utf-8

from functools import partial
import numpy as np

from recipes.list import sortmore


class Ephemeris():
    def __init__(self, t0, P):
        """
        t0 - zero point for ephemeris
        P - period in days
        """
        self.t0 = t0
        self.P = P

    def __str__(self):
        return 'HJD = %f + %f E' % (self.t0, self.P)

    def __call__(self, E):
        return self.t0 + E * self.P

    def phase(self, t):
        phase = np.atleast_1d(t - self.t0) / self.P
        return phase

    def phaseModulo1(self, t):
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
        data = np.array(sortmore(phase, *data))

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


# allocate bins based on phase range

# nbins = 20 #
# nphbw = 1. / nbins

# ph0 = max(egress) - min(ingress)           #start after eclipse egress
# phmx = fPh.max()
# ph1 = ((phmx // nphbw) + 1) * nphbw   #maximum phase bin upper edge

# phbins = np.linspace(ph0, 1, nbins, endpoint=True)
# phbins = np.r_[0, np.ravel(np.c_[:np.ceil(phmx)] + phbins)]


##collect segments in phase bins

# segdict = defaultdict(list)
##tdict = defaultdict(list)
# for i, ph in enumerate(Ph):
# dz = np.digitize(ph, phbins)
# unq, ix = np.unique(dz-1, return_index=True)  #NOTE: -1 ensures correct bin allocation!

# segs = np.split(D[i], ix[1:])
# tsegs = np.split(UT[i], ix[1:])
# phsegs = np.split(ph, ix[1:])

# for nq, phseg, tseg, seg in zip(unq, phsegs, tsegs, segs):
##Compute Power-spectrum of each segment
# f, (P,) = Spectral(tseg, seg, gaps=('mean', 25), detrend=('poly', 0))
##add data to bins (phased)
# segdict[nq % nbins].append((phseg, tseg, seg, f, P))


if __name__ == '__main__':
    # %%time

    # NOTE:  this doesn't work so well.  Find a way of fitting simultaneously

    binsize = 1  # in seconds
    tspans = [np.diff(t[[0, -1]]) for t in T]
    tmax = max(tspans)
    ir = 1  # np.argmax(tspans)  #this is the reference
    bins = np.arange(0, tmax, binsize)
    Nb = len(bins)

    tr, lcr = T[ir], D[ir]
    B = np.ma.zeros((len(T), Nb))
    B[ir] = br = binner(tr, lcr, bins)
    M = np.zeros(len(T))
    M[ir] = 0
    for i, (t, lc) in enumerate(zip(T, D)):
        if i == ir:
            continue

        B[i] = b = binner(t, lc, bins)
        # S, E = shifter(br, b, 0.4)

        # fig, ax  = plt.subplots(figsize=(18,8))
        # ax.plot(S, E)

        M[i] = brutus(br, b, 0.4)  # S[np.nanargmin(E)]#

    aB = np.ma.array(B)
