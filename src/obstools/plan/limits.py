from collections import defaultdict
import matplotlib.path as mpath
from matplotlib.patches import PathPatch
import itertools as itt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PPoly
from recipes.dicts import ManyToOneMap, TerseKws
import warnings

# Logic for resolving telescope name
TEMP = ManyToOneMap()
TEMP.add_trans({40: 1.,
                74: 1.9})
TEMP.add_mappings(TerseKws('40[ inch]', 1.),
                  TerseKws('1[.0 m]', 1.),
                  TerseKws('74[ inch]', 1.9),
                  TerseKws('1.9[ m]', 1.9),
                  float)
# some of these equivalence mappings will bork during the lookup, but we don't
# actually care, so suppress the warnings
TEMP.warn = False

#
HARD_LIMITS = TEMP.copy()
HARD_LIMITS.update(
    {1.0:
     {'east':  [[-90.0, -2.5],
                [-67.0, -4.0],
                [-67.0, -4.5],
                [-20.0, -3.6],
                [10.0, -2.5],
                [15.0, -2.0],
                [20.0, -0.75],
                [30.0, -0.25]],
      'west':   [[-90, 5.5],
                 [30, 5.5]]},

     1.9:
     {'east':  [[-90.0, -1.5],
                [-50.0, -4.5],
                [25.0, -1.5]],
      'west':   [[-90.0, 6.0],
                 [-78.0, 7.5],
                 [-44.0, 7.5],
                 [25, 4.5]]}
     })

SOFT_LIMITS = TEMP.copy()
SOFT_LIMITS.update(
    {1.0:
     {'east': [[-85.0, -2.5],
               [-60.0, -4.1],
               [-20.0, -3.3],
               [9.0, -2.2],
               [17.0, -0.5],
               [25.0, 0.0]],
      'west': [[-85.0, 5.0],
               [25, 5.0]]},

     1.9:
     {'east': [[-85.0, -1.5],
               [-50.0, -4.0],
               [20.0, -1.5]],
      'west': [[-85, 5],
               [-50, 5],
               [20, 4]]}
     })


LIMITS = {'soft': SOFT_LIMITS,
          'hard': HARD_LIMITS}

PLOT_LIMITS = {1.0: [(-5, 6), (-100, 40)],
               1.9: [(-5, 8), (-100, 40)]}

# clean up module namespace
del TEMP

_HS = ('hard', 'soft')
_EW = ('east', 'west')


def _check(s, ok):
    assert isinstance(s, str)
    s = s.lower()
    assert s in ok, f'{s!r} is not one of {ok}'
    return s


def _checks(where, which):
    return _check(where, _EW), _check(which, _HS)


def get_limits(tel, where, which):
    """
    [summary]

    Parameters
    ----------
    tel : [type]
        [description]
    which : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    where, which = _checks(where, which)
    return np.array(LIMITS[which][tel][where])


def get_poly(tel, where, which):
    """
    [summary]

    Parameters
    ----------
    tel : [type]
        [description]
    which : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # declination, HA
    x, y = np.transpose(get_limits(tel, where, which))
    dx = (x - np.roll(x, 1))[1:]
    dy = (y - np.roll(y, 1))[1:]
    use = (dx != 0)
    c = np.array([dy[use] / dx[use], y[:-1][use]])
    return PPoly(c, x[np.hstack([use, True])], False)


def arctan(xy):
    return np.arctan2(*xy)


def clockwise(x, y):
    """
    Sort a sequence of (x, y) points in azimuthal order progressing clockwise
    around the origin
    """
    return sorted(itt.product(x, y), key=arctan)


def get_polygon(inner, outer, **kws):
    """
    get the polygon (with a hole in the centre) described by the inner and
    outer sequence of points
    """
    verts = np.vstack([inner.T, clockwise(*outer)[::-1]])
    codes = np.ones(len(verts), dtype=mpath.Path.code_type) * mpath.Path.LINETO
    codes[[0, -4]] = mpath.Path.MOVETO
    path = mpath.Path(verts, codes)
    return PathPatch(path, **kws)


class TelescopeLimits:
    def __init__(self, tel):
        # get interpolators
        self.tel = HARD_LIMITS.resolve(tel)
        self.interpolators = defaultdict(dict)
        for where, which in itt.product(_EW, _HS):
            self.interpolators[where][which] = get_poly(tel, where, which)

    def get(self, where, which):
        return get_limits(self.tel, where, which)

    def get_visible_ha(self, dec, where='both', which='hard'):

        if where == 'both':
            return [self.get_visible_ha(dec, where_, which) for where_ in _EW]

        where, which = _checks(where, which)
        pp = self.interpolators[where][which]
        if pp.x.min() < dec < pp.x.max():
            return pp(dec)

    def plot(self, ax=None, which='both', hard_kws=None, soft_kws=None, **kws):
        if ax is None:
            fig, ax = plt.subplots()

        which = _HS if which == 'both' else _check(which, _HS)
        data = {}
        for key in which:
            east, west = LIMITS[key][self.tel].values()
            data[key] = np.vstack([east, west[::-1], east[0]]).T[::-1]

        art = []
        if 'hard' in which:
            xlim, ylim = corners = PLOT_LIMITS[self.tel]
            patch = get_polygon(data['hard'], corners,
                                **{**dict(facecolor='r', linewidth=0),
                                   **(hard_kws or kws)})
            ax.add_patch(patch)
            art.append(patch)

        if 'soft' in which:
            art.extend(
                ax.plot(*data['soft'], **{**dict(color='g', ls='--'),
                                          **(soft_kws or kws)})
            )

        # set title, labels, ax lims
        ax.set(title=f'{self.tel}m Limits',
               xlabel='HA', ylabel='DEC',
               xlim=xlim, ylim=ylim)
        ax.grid()

        # interpolators
        for which, (ha, dec) in data.items():
            d = np.linspace(dec.min(), dec.max())
            for where in _EW:
                pp = self.interpolators[where][which]
                ax.plot(pp(d), d, 'b:')

        return art
