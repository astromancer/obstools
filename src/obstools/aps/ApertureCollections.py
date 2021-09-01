# TODO: UPDATE ALL DOCSTRINGS (once stable)
# TODO: unit tests



# std
import re
from copy import copy

# third-party
import numpy as np
from scipy.spatial.distance import cdist
from matplotlib.colors import colorConverter
from matplotlib.collections import EllipseCollection, LineCollection
from matplotlib.transforms import (IdentityTransform,
                                   blended_transform_factory as btf)

# local
# from recipes.io import warn as Warn
from recipes.iter import as_sequence
from recipes.oo.meta import altflaggerFactory
from recipes.dicts import TransDict, ManyToOneMap


# from pprint import pprint



# from motley import banner

# from decor import expose, profile
# from decor.misc import unhookPyQt

# from PyQt4.QtCore import pyqtRemoveInputHook, pyqtRestoreInputHook

# alias
rgba_array = colorConverter.to_rgba_array


def rotation_matrix_2d(theta):
    """Rotation matrix"""
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([[cos, -sin],
                     [sin, cos]])


def rotate_2D(xy, theta):
    return rotation_matrix_2d(theta) @ xy


def pick_handler(artist, event):
    """
    The picker function
    """

    if not len(artist):
        return False, {}

    print('pick_handler', event.button)

    # if event.button != 0

    mouse_position = (event.xdata, event.ydata)
    if None in mouse_position:
        return False, {}

    pr = artist.get_pickradius()
    # NOTE: Can control individual pickability by making pickradius a sequence
    ep = artist.edge_proximity(mouse_position)
    # print(ep)
    hit = ep < pr
    anyhit = np.any(hit)

    props = {'index': np.where(hit)} if anyhit else {}
    # print(props)
    return anyhit, props


# def pick_handler(col, event):
#     """
#     The picker function
#     """
#     # TODO: ignore: scroll, button 2, 3???
#
#     return False, {}
#
#     if not col.size:
#         return False, {}
#
#     mouse_position = (event.xdata, event.ydata)
#     if None in mouse_position:
#         return False, {}
#
#     pr = col.get_pickradius()  # NOTE: Can control pickability by making pickradius a sequence
#
#     xyprox = col.coords[idx] - position
#     ang = np.arctan2(y, x)
#
#     hit = abs(artist.edge_proximity(mouse_position)) < pr
#     # print( 'HIT! '*10 )
#     # print( hit )
#     # except:
#     # print( 'PICKER FAIL!' , artist )
#     # pyqtRemoveInputHook()
#     # embed()
#     # pyqtRestoreInputHook()
#
#     # print( 'Picker', artist.edge_proximity( mouse_position )  )
#     # print( 'HITS', hit)
#     # hit &= artist.picable           #only hit if pickable
#     anyhit = np.any(hit)
#     # print( 'anyhit', anyhit)
#     # print()
#     props = {'index': np.where(hit)} if anyhit else {}
#     print(props)
#     return anyhit, props


class KeywordTranslator(TransDict):
    """Dictionary for translating keywords"""

    def __call__(self, dic=None, **kws):
        """translate keywords"""
        dic = dic or {}
        dic.update(kws)
        return {self._map.get(key, key): val for key, val in dic.items()}


class PropertyConverter(ManyToOneMap):
    """Keyword value conversion"""

    # @expose.args( pre='CONVERT!! '*10, post='DONE '*10 +'\n'*2 )
    def __call__(self, dic=None, **kws):
        dic = dic or {}
        dic.update(**kws)
        for key, val in dic.items():
            if key in self:
                dic[key] = self[key](val)
        return dic


class PropertyManager(dict):
    """State class for Collection properties."""

    # dictionary with property, default value. If any of these properties are
    # not specified in the initialisation keyword dict, the default values
    # here will be used
    _defaults = dict(widths=[],
                     heights=[],
                     angles=[0],
                     offsets=np.empty((0, 2)),
                     facecolor=['none'],
                     edgecolor=['g'],
                     linestyle='-',
                     linewidth=1.,
                     pickradius=2,
                     picker=pick_handler)

    # create object for keyword translation
    translator = KeywordTranslator()
    # The following relations map the ApertureCollection attributes to the
    # equivalent Collection attributes for user convenience.  All the values
    # in the translator dict map to the corresponding keyword upon
    # initialisation. This is purely for user convenience. eg. short-hand
    # notation when initialising
    translator.add_vocab(dict(coords='offsets',
                              ls='linestyle',
                              lw='linewidth',
                              w='width',  # TODO: a, b, theta, θ
                              h='height',
                              r='radii'))
    # many to one mappings
    translator.many2one({('ec', 'color', 'colour', 'colours'):
                             'edgecolor',
                         ('fc', 'facecolors', 'facecolour', 'facecolours'):
                             'facecolor'})

    # property convertion
    # Convert properties to arrays - easier this way to concatenate!
    converter = PropertyConverter(
            color=lambda val: rgba_array(as_sequence(val)),
            offsets=np.atleast_2d,
            radii=np.atleast_1d,
            widths=np.atleast_1d,
            heights=np.atleast_1d,
            angles=np.atleast_1d, )

    # Add equivalence mapping that maps *colour -> *color
    converter.add_mapping(
            lambda key: ''.join(re.search('(colo)u?(r)', key).groups()))

    # @expose.args( pre='%'*100, post = '\\'*100 )
    def __init__(self, defaults=None, **kws):

        # backup of the original properties
        self._original = kws.copy()

        # start from default values, then translate given keywords, then update
        new = (defaults or self._defaults).copy()

        # translate keywords
        translated = self.translator(**kws)
        for key, val in translated.items():
            # None => default
            if val is not None:
                new[key] = val

        # convert
        new = self.converter(new)

        # finally initialize dict
        dict.__init__(self, **new)

    # def set(self, key, value):

    # @classmethod
    # # @expose.args( pre='*'*100, post = '_'*100 )
    # def _prep_dict(cls, dic):  # TODO: combine with append method?
    #
    #     # radii, coords = dic.pop('radii'), dic.pop('offsets')
    #     coords = dic['offsets']
    #     # if radii.shape != coords.shape[:-1]:
    #     # try:
    #     # radii.shape = coords.shape[:-1]                 #broadcast the radii to match the coords
    #     # except:
    #     l = coords.size // 2
    #     for key in cls.__broadcast__:
    #         obj = dic[key]
    #         if np.size(obj) < l:
    #             if isinstance(obj, np.ndarray):
    #                 dic[key] = np.tile(obj, (l, 1))
    #             else:
    #                 dic[key] = list(cycleN(as_sequence(obj), l))
    #
    #                 # dic[ key ] = cls.broadcast( obj, key, dmax, True )
    #
    #     # dic['radii'] = radii
    #     # dic['offsets'] = coords
    #
    #     return dic

    # @classmethod
    # # @expose.args( pre='o'*100, post='~'*100 )
    # def concatenate(cls, current, adder):
    #
    #     if isinstance(current, np.ndarray):
    #         return stacker(current, adder)
    #         # try:            #attempt to stack by column (will only work if current and adder have the same length
    #         # return np.vstack( [current, adder] )
    #         # except ValueError:      #stack by appending
    #         # return  np.hstack( [current, adder] )
    #
    #     elif isinstance(current, list):
    #         return current + adder
    #
    #     else:
    #         banner('%s not list' % type(current), '!', bg='red')
    #         print(current, adder)
    #         print()
    #         return current

    # ===============================================================================================


def stacker(*arrays):
    """Stack array, filtering out empty ones"""
    args = ('0',) + tuple(filter(np.size, arrays))
    # NOTE: A string with three comma-separated integers allows specification of the
    # axis to concatenate along, the minimum number of dimensions to force the
    # entries to, and which axis should contain the start of the arrays which
    # are less than the specified number of dimensions.
    return np.r_[args]


#################################################################################################################################################################################################################
_AliasMixin, alias = altflaggerFactory(collection='aliases')


class AliasMixin(_AliasMixin):
    """class for creating method aliases by decoration"""

    def __init__(self):
        super().__init__()
        for method, aliases in self.aliases.items():
            for alias in aliases:
                setattr(self, alias, method)


# AliasMixin,

# TODO: Handle variable aperture shapes through generic ApertureCollection

class ApertureCollection(EllipseCollection):
    """`
    Class that handles methods common to collection of image apertures.
    """

    # TODO: __index__ !!!!!!!!!!!!!!!!!  THIS WILL BE REALLY REALLY REALLY REALLY AWESOME,
    # TODO: Annotation
    # TODO: def from_ellipses(ellipses)

    DEFAULT_RADIUS = 5.  # default radius
    PropertyManager = PropertyManager

    # @profile()
    def __init__(self, widths=None, heights=None, angles=None, **kws):
        """
        coords is the main parameter. all other parameters will be shaped
        accordingly
        """
        # NOTE: width, height, angles can be any size, as long as they are
        # all the same size, or broadcastable to such.  if their size is
        # less than the number of apertures (as implied by the coordinates)
        #  (self.coords.size // 2), their values will be cycled.

        # allow width / height to be set by radii keyword
        r = kws.pop('radii', kws.pop('r', None))
        if r is not None:
            r = np.asarray(r)
            widths = 2 * r
            heights = 2 * r

        kws['widths'] = widths
        kws['heights'] = heights
        kws['angles'] = angles
        # translate keywords, fill defaults, convert keyword values
        self._properties = _properties = self.PropertyManager(**kws)

        # remember shape
        off = _properties['offsets']
        self._coord_shape = off.shape

        # set colors to 'none' to avoid weird behaviour where empty Collection draws a
        # single ellipse in the axes
        self._restore_ec = None
        if off.size == 0:
            self._restore_ec = _properties['edgecolor'].copy()
            _properties['edgecolor'][:] = 0

        # NOTE:  for some bizarre reason the width / height passed to EllipseCollection
        # are divided by 2 when self._width / self._height are set. We therefore
        # pass in twice their value below
        # NOTE: transOffset=IdentityTransform() this is needed below so that Collection
        # uses self._offsets and not self._uniform_offsets for the coordinates which
        # does not  allow for dynamically extending the object.  This will be changed to
        # ax.transData when the collection is added to the axes via self.add_to_axes
        EllipseCollection.__init__(self,
                                   units='xy',
                                   # picker=pick_handler,
                                   transOffset=IdentityTransform(),
                                   # offset_position='data',
                                   **_properties)

        self._axes = None

    def __str__(self):
        # FIXME: better repr with widths, heights, angles
        return '%s of shape %s' % (self.__class__.__name__, self.shape)

    # def __repr__(self):
    #     return str(self)

    def __len__(self):
        return self.size

    def __add__(self, other):
        new = copy(self)
        new.append(other)
        return new

    # # NOTE: The following 2 methods allow dict-like unpacking to yield property key-value combinations
    # def __getitem__(self, key):
    #     return getattr(self, 'get_' + key)()
    #
    # def keys(self):
    #     return self._properties._defaults.keys()

    @property
    def shape(self):
        return self._coord_shape[:-1]

    @property
    def size(self):
        return self.coords.size // 2

    def get_offsets(self):
        """Returns a view of the coordinates that retains shape info"""
        return np.reshape(super().get_offsets(), self._coord_shape)

    def set_offsets(self, vals):
        converter = self._properties.converter['offsets']
        coords = converter(vals)

        # avoid zero size collection drawing erroneously
        if (self.size == 0) and np.size(coords):
            self.set_edgecolor(self._restore_ec)
        if self.size and np.size(coords) == 0:
            self._restore_ec = self.get_edgecolor()

        self._coord_shape = np.shape(coords)
        super().set_offsets(coords)  # will make Nx2

    # alias
    set_coords = set_offsets
    get_coords = get_offsets
    coords = property(get_coords, set_coords)

    def set_radii(self, radii):
        # NOTE: using set_radii will render circular apertures
        r = np.asarray(radii).ravel()
        self._heights = r
        self._widths = r
        self._angles = np.zeros_like(r)

    radii = property(None, set_radii)

    @property
    def semimajor(self):
        return self._widths
        # return np.max([self._widths, self._heights], 0)  # .reshape(self.shape)

    @semimajor.setter
    def semimajor(self, vals):
        converter = self._properties.converter['widths']
        self._widths = converter(vals)

    @property
    def semiminor(self):
        return self._heights
        # return np.min([self._widths, self._heights], 0)  # .reshape(self.shape)

    @semiminor.setter
    def semiminor(self, vals):
        converter = self._properties.converter['heights']
        self._heights = converter(vals)

    a = semimajor
    b = semiminor

    @property
    def angles(self):
        return self._angles

    @angles.setter
    def angles(self, vals):
        converter = self._properties.converter['angles']
        self._angles = converter(vals)

    # Various aliases
    def get_edgecolor(self):
        return super().get_edgecolor()

    get_ec = get_colours = get_edgecolor

    def set_edgecolor(self, colours):
        return super().set_edgecolor(colours)

    set_ec = set_colours = set_edgecolor

    def get_lw(self):
        return self.get_linewidth()

    def get_ls(self):
        return self.get_linestyle()

        # def prep_dict( self, dic, enforce_casting ):
        # return self._prep_dict( dic, self.size, enforce_casting )

        ##def prep_vals( self, vals, key='', warn=None ):
        # return self.broadcast( vals, key, self.size, 1, warn )

    # from misc import profile
    # @profile()
    def pre_append(self, aps=None, **props):
        """ """

        if aps:
            if not aps.coords.size:
                # coordinates are fundamental - cannot extend without these
                return
                # TODO: deal with aps.  Maybe implement .items ?

        elif props:
            # start from the original (initialised) collection properties as defaults, update this with the properties
            # passed as keywords.  This behaviour allows one to keep cycling the properties
            # with which the class was initialised.  Therefore one can pass only the changed
            # properties to the append method (along with the new coordinates)
            # NOTE:  This will only work if the shape of the adder is the same as that of
            # the original.  To make this work for variable shape appending, one will need to
            # specify all the properties to overwrite, or figure out a rule of how to slice the
            # original properties when appending.

            # print( '\n ORIGINAL PROPS' )
            # print( self._properties._original )
            # print( '\n INCOMING PROPS' )
            # print( props )
            # print()

            props = self._properties(self._properties._original,
                                     broadcast=False, **props)

            if not np.size(props.get('offsets',
                                     [])):  # 'coords'  #TODO: Manage as property to avoid this check
                return

        return props

    # @expose.args( pre='='*100, post = '~'*100 )
    # @unhookPyQt
    def append(self, aps=None, **props):
        """Dynamically extend the ApertureCollection.
        params: aps - instance of ApertureCollection / keywords with properties of ApertureCollection to append
        """

        props = self.pre_append(aps=None, **props)
        if props is None:
            print('#' * 300)
            return

        if not self.size:
            concatenate = lambda o, a: a
            # if the Collection was initialized as empty, set the new properties as current
        else:
            concatenate = props.concatenate

        # embed()
        oprops = self._properties._original
        # Find which properties differ and update those
        for key, val in props.items():
            if (not key in oprops) \
                    or (not np.array_equal(oprops[key], props[
                key])):  # `np.array_equal` here flags the empty properties as being unequal to the new ones, whereas `np.all` evaluates as True under the same conditions
                new = concatenate(self[key], val)

                # print( '8'*88 )
                # print('APPEND', key, self[key], val  )
                # print( 'NEW:', new )

                setter = getattr(self, 'set_%s' % key)
                setter(new)

                # print( )

                # pyqtRemoveInputHook()
                # embed()
                # pyqtRestoreInputHook()

                # OR:

    # def remove(self, idx=None):
    #     if idx:
    #         idx = idx[0]
    #         self.coords = np.delete(coords, idx, axis=0)
    #         # self.radii = np.delete(self.radii, idx, axis=0)
    #     else:
    #         self.coords = []
    # self.radii = []

    # def within_allowed_range(self, r):
    ##self.allowed_range
    # return True

    # def resize(self, relative_motion, idx=..., ):
    ##print( 'RESIZING!', relative_motion )
    # if not relative_motion is None:
    # rnew = self.radii
    # rnew[idx] += relative_motion
    # if not self.within_allowed_range( rnew ):
    # return

    # self.radii = rnew                   #remember this is a property

    def area(self, idx=...):
        """
        Return the area enclosed by the aperture(s)
        Specific aperture can be specified by idx.
        """
        return np.reshape(np.pi * self._widths * self._heights, self.shape)[idx]

    def area_between(self, idxs):
        """return the area enclosed between the two apertures given by idxs"""
        A = np.array(self.area(idxs), ndmin=2)
        area = np.abs(np.subtract(*A.T))
        return area

    def center_proximity(self, position, idx=...):
        """
        Calculate radial distances between aperture centres and position.
        If idx is given, do calculation for the aperture(s) indicated by the index.
        """
        if None in position:
            return

        # shape = self.radii.shape + (2,)
        # coords = self.coords.reshape( *shape )          # RESHAPING IN THE GETTER????
        # coords = self.coords[idx]                          #grabs offsets of aperture idx
        # xyprox = np.array( coords - position, ndmin=2 )
        xyprox = self.coords[idx] - position

        return np.linalg.norm(xyprox, axis=-1)

    def edge_proximity(self, position, idx=...):
        """
        Calculate the proximity of a given position to the aperture edge. Used
        for event control.

        Parameters
        ----------
        position : tuple with x, y coordinates

        Returns
        -------
        rp : array with radial distances in pixels from aperture radii.
        """
        if None in position:
            return

        n = len(self.coords)
        xyprox = np.subtract(position, self.coords)
        x, y = xyprox.T
        # angle between position and ellipses centre wrt x-axis
        theta = np.arctan2(y, x)
        alpha = np.radians(self._angles)
        a, b = self._widths, self._heights

        # tile a, b, alpha if different shape to coords
        # TODO: General method here - maybe can live in Properties class
        if n > len(a):
            a = np.tile(a, n // len(a))
        if n > len(b):
            b = np.tile(b, n // len(b))
        if n > len(alpha):
            alpha = np.tile(alpha, n // len(alpha))

        # angles between mouse position and semi-major axes
        phi = theta - alpha
        # radial equation for ellipse
        r = (a * b) / np.sqrt(np.square((b * np.cos(phi),
                                         a * np.sin(phi))).sum(0))
        # distance of (x, y)
        rp = np.sqrt(np.square(xyprox).sum())
        return np.abs((rp - r))

    # ==============================================================================================
    # TODO: maybe maxe this a propetry??
    def add_to_axes(self, ax=None):
        if not self in ax.collections:
            # print( 'Adding collection to axis' )

            # necessary for apertures to map correctly to data positions
            self._transOffset = ax.transData
            ax.add_collection(self)

            self._axes = ax

    # ==============================================================================================
    def annotate(self, names=None, offset=None, angle=45, **kws):
        """
        Annotate the apertures with text

        Parameters
        ----------
        names   :       sequence of name str's
        offset  :       float or sequence of 2 floats, or (N,2) array-like
            The offset from the center of the aperture
            if float, use as radial offset with givel angle
            if 2 floats, use as xy-offset and ignore angle
            if (N, 2) array, use
        angle   :       float
            angle (in degrees)

        Keywords
        --------
        all keyword arguments are passed directly to ax.annotate

        """
        if names is None:
            # provide numbered annotations by default
            names = np.arange(len(self)).astype(str)

        if len(names) != len(self):
            raise ValueError('Inadequate name specs', names,
                             'for %i Apertures' % len(self))

        if self._axes is None:
            raise Exception('First add the Collection to an axes')

        if offset is None:
            offset = np.ones_like(self.coords) * self.semimajor  # - 2
            # TODO: constant in display units?
            # NOTE, this will skew the offsets for differing radii

        offset = np.atleast_1d(offset)
        angle = np.radians(angle)
        if len(offset) == 1:
            offset = offset * (np.cos(angle), np.sin(angle))

        self.annotations = [self._axes.annotate(name, coo, off, **kws)
                            for coo, off, name in
                            zip(self.coords, self.coords + offset, names)]

        # ==============================================================================================
        # def numbering(self, TF): #convenience method for toggling number display?


# from recipes.list import UniformObjectContainer

# TODO: 2 options - UniformObjectContainer
# or slicing ApertureCollection to yield another ApertureCollection?
# OR subclass mpl.Container??

def duplicate(off):
    # duplicate the coordinates
    return np.swapaxes([off, off], 0, 1).reshape((-1, 2))


class SkyApsPropMan(PropertyManager):
    converter = PropertyManager.converter.copy()
    converter['offsets'] = duplicate

    # def __init__(self, defaults=None, **kws):
    #     PropertyManager.__init__(self, defaults, **kws)
    #     off = self['offsets']
    #     # duplicate the coordinates
    #     off = np.swapaxes([off, off], 0, 1).reshape((-1, 2))
    #     self['offsets'] = off

    # def duplicate(self, vals):
    #     # duplicate the coordinates
    #     off = np.swapaxes([off, off], 0, 1).reshape((-1, 2))
    #     self['offsets'] = off


class SkyApertures(ApertureCollection):
    """
    Class to handle sky apertures around individual stars within an image.
    """

    PropertyManager = SkyApsPropMan

    # def set_widths(self):

    #
    #
    #
    # def __init__(self, widths=None, heights=None, angles=None, **kws):
    #
    #
    #
    # def set_offsets(self):

    # def __init__(self, inner=None, outer=None, **kws):
    #     """Create sky apertures"""
    #     if inner is None:
    #         inner = ApertureCollection(**kws)
    #
    #     if outer is None:
    #         outer = ApertureCollection(**kws)
    #
    #     # TODO: move deeper
    #     assert isinstance(inner, EllipseCollection)
    #     assert isinstance(outer, EllipseCollection)
    #
    #     super().__init__([inner, outer])
    #
    #
    # @property
    # def inner(self):
    #     return self.data[0]
    #
    # @property
    # def outer(self):
    #     return self.data[1]
    #
    # def add_to_axes(self, ax):
    #     self.methodcaller('add_to_axes', ax)
    #
    # def get_visible(self):
    #     return self.inner.get_visible()
    #
    # def set_visible(self, v):
    #     self.methodcaller('set_visible', v)


#
#
#
#     RADII = R_IN, R_OUT = 10., 15.
#     R_OUT_UPLIM = int(np.ceil(R_OUT))
#     SKY_PIX_TOL = 20.
#
#     def __init__(self, **kwargs):
#         """
#         Have to specify properties as keywords if not using defaults.
#         """
#         # if not 'check' in kwargs:       kwargs['check'] = 'sky'
#         super().__init__(**kwargs)
#         # self.inner
#
#     def within_allowed_range(self, radii, lowlim=None, uplim=None):
#
#         lowlim = lowlim if lowlim else 0, self.R_IN
#         uplim = uplim if uplim else self.R_OUT, np.inf
#
#         ll = np.any(np.less(radii, lowlim))
#         if ll:
#             print('Outer Sky radius cannot be smaller than inner...')
#         return ~ll
#
#         lu = np.any(np.greater(radii, uplim))
#         if lu:
#             print('Inner Sky aperture cannot be larger than outer...')
#         return ~lu
#
#     def check_sky(self, idx=None, warn=1):
#
#         # print( 'counting sky pixels between ', self.radii )
#         # if idx is None:
#         # idx = slice(-2, None)       #assume the sky apertures are the last two in the sequence
#
#         sky_area = self.area_between(idx)
#         Nsky_pix = np.sqrt(sky_area)
#
#         skystate = Nsky_pix < self.SKY_PIX_TOL
#         skystate = np.array([skystate] * 2).ravel()
#
#         if np.any(skystate) and warn:
#             Warn('Sky annulus contains only {} pixels.'
#                  'Sky pixel tolerance is {:3.2f}.'.format(Nsky_pix, self.SKY_PIX_TOL))
#
#         return skystate
#
#     def check(self, **kw):
#         # super(SkyApertures, self).auto_colour( **kw )
#
#         check = kw.get('check')
#         sky = kw.get('sky')
#         if check == 'all':
#             check = True
#
#             # if not (check or edge or sky or cross):
#             # print( 'WARNING: Nothing being checked...' )
#
#         lc = super().check(**kw).ravel()
#
#         # print( 'SKYAPS AUTOCOLOUR', lc, self.check_sky( sky ) )
#         # print( 'check', check, 'sky', sky )
#         if check == 'sky' or check is True or sky:
#             lc |= self.check_sky(
#                 sky)  # sky here refers to an index. i.e. if all sky apertures are the same shape, you only have to check one of them!
#
#         # print( 'lc', lc )
#
#         return lc

# class Test:
# def connect(name):
# def _decorator(func):
# print( name, func )
# def trivial_wrapper( self, *args ):
# func( self, *args )
# return trivial_wrapper
# return _decorator

# @connect( 'something' )
# def bar( self, *args ) :
# print( "normal call:", args )


#################################################################################################################################################################################################################


class ApLineCollection(LineCollection):
    def __init__(self, apertures, ax, **kws):

        kws.setdefault('transform', btf(ax.transData, ax.transAxes))
        super().__init__([], **kws)  # initialise empty LineCollection
        # self.set_transform(  )

        # def add_to_axes(self, ax):

        # self.aplines = lines = LineCollection( [] )
        # self.update_aplines( )

        # return lines

    # @expose.args( pre='='*100, post='?'*100 )
    def make_segments(self, radii):

        if radii.size:
            return [list(zip((r, r), (0, 1))) for r in radii]
        else:
            return []

    def update_from(self, aps):

        self.set_segments(self.make_segments(aps.radii))

        # self._edgecolors_original = aps._edgecolors_original
        self._edgecolors = aps._edgecolors
        # self._facecolors_original = self._facecolors_original
        self._facecolors = aps._facecolors
        self._linewidths = aps._linewidths
        self._linestyles = aps._linestyles

        # self.update_from( self )
        # self.aplines.set_transform( self.line_transform )              #necessary becasue update_from kills the transform


######################################################################################################
class SameSizeMixin():
    SAME_SIZE_AXIS = 0  # axis of the aperture radii along which values are to be set equal

    # eg. 1 ==> each column holds constant value.

    # @expose.args()
    # @expose.returns()
    def samesize(self, index):
        """
        Takes an index (eg. from the picker) and returns indeces of all apertures with the
        same size as the picked one.
        """
        # index = list(index)
        # index[self.AXIS] = ...
        # return tuple(index)
        x = self.SAME_SIZE_AXIS
        ix = [...] * len(index)
        ix[x] = index[x]
        return tuple(
            ix)  # tuple(ix if i==aps.AXIS else ... for i,ix in enumerate(index))

    # @unhookPyQt
    def resize(self, relative_motion, idx=...):
        print('RESIZING SAMESIZEMIXIN!', relative_motion, self.radii, idx)

        # embed()
        # Resize all apertures along the AXIS with linked radii
        super().resize(relative_motion,
                       self.samesize(
                           idx))  # NOTE: does not enforce the same size, only same relative increment
        # self.samesize()

    # @expose.args()
    # @unhookPyQt
    def set_radii(self, radii):
        # NOTE:  This constructs the index array that will be used to set all elements along
        # the required axis to the same value as the 0th element on that axis.  A column vector
        # is used as index because numpy handles indeces of the form (np.array([0]),) differently
        # to indeces of the form (0,).  An array index yields a sliced array that already has the
        # correct shape, and we can use it to fill the array along the required dimension via a
        # simple one-liner using the Ellipses. BOOM!

        # embed()

        x = self.SAME_SIZE_AXIS
        # radii = np.array(radii, ndmin=x+1)
        ix = list(
            np.zeros((radii.ndim, 1), int))  # use zeroth radius for all radii
        ix[x] = ...
        radii[...] = radii[ix]
        super().set_radii(radii)


######################################################################################################
class InteractionProperties(PropertyManager):  # TODO: class factory
    translator = KeywordTranslator()
    translator._map.update(PropertyManager.translator._map)
    translator.many2one({
        ('gc', 'goodcolor', 'goodcolors', 'goodcolour', 'goodcolours'):
            'edgecolor',
        ('bc', 'badcolor', 'badcolors', 'badcolours'):
            'badcolour'})

    converter = PropertyConverter()
    converter['radii'] = np.atleast_2d
    converter['offsets'] = lambda x: np.r_['0,3,-1', x]

    # _defaults = dict( PropertyManager._defaults,
    # **{#'goodcolour'    :       'g',
    # 'badcolour'     :       'y'     } )


class InteractionMixin():
    # """Enable user interaction with mouse/keyboard"""
    _properties = InteractionProperties

    # TODO:  deal with interactions between edge and other apertures separately  -->  different colour changes

    # @expose.args( pre='*'*100, post = '_'*100 )
    def __init__(self, **kws):
        print('InteractionMixin')
        print(kws)
        print()

        # self.__propdict__.many2one( ('gc', 'goodcolour', 'goodcolours'), 'edgecolor' )

        # pyqtRemoveInputHook()
        # embed()
        # pyqtRestoreInputHook()

        # kws = self._properties( **kws )
        # NOTE:  This means the property translation / conversion will be attempted twice,
        #       which might be slightly inefficient.  Give PropertyManager.__init__ a
        #       `translate` switch to overcome this.
        # self.goodcolour = kws['edgecolor']   #TODO:  should InteractionProperties inherit TransDict??
        # self.badcolour = kws.pop('badcolour', rgba_array('y'))

        # self.translator = InteractionMixin.translator

        # pyqtRemoveInputHook()
        # embed()
        # pyqtRestoreInputHook()

        print('super().__init__', kws)

        self.badcolour = kws.pop('badcolour', rgba_array(
            'y'))  # FIXME:  This will cause errors eventually.

        super().__init__(broadcast=True, **kws)

        self.goodcolour = self._edgecolors_original  # HACK

        # if self.size and check:
        # self.auto_colour( check=check )

        # 'badcolour' :       BAD_COLOUR,

        # self.badcolour = kws.pop('badcolour', self.BAD_COLOUR )
        # self.goodcolour = kws.pop('edgecolor', self.GOOD_COLOUR )

        # def set_badcolour(self, colour):
        # self.badcolour = colour

        # def update(self, props):
        # banner( '\nRUNNING UPDATE\n', props, bg='red' )

        # super().update( props )

        # def set_radii(self, radii):
        # print( 'INTERCEPT '*10 )
        # super().set_radii(radii)

    # def get_pickradius(self):
    # pr = np.array( artist.get_pickradius() ).reshape(-1,2) #TODO:

    # from misc import profile
    # @profile()
    # FIXME: SMELLY CODE!!!!!!!!!!!!
    def pre_append(self, aps=None, **props):
        """ """

        if aps:
            if not aps.coords.size:
                # coordinates are fundamental - cannot extend without these
                return
                # TODO: deal with aps.  Maybe implement .items ?

        elif props:
            # start from the original (initialised) collection properties as defaults, update this with the properties
            # passed as keywords.  This behaviour allows one to keep cycling the properties
            # with which the class was initialised.  Therefore one can pass only the changed
            # properties to the append method (along with the new coordinates)
            # NOTE:  This will only work if the shape of the adder is the same as that of
            # the original.  To make this work for variable shape appending, one will need to
            # specify all the properties to overwrite, or figure out a rule of how to slice the
            # original properties when appending.

            print('\n ORIGINAL PROPS')
            print(self._properties._original)
            print('\n INCOMING PROPS')
            print(props)
            print()

            # banner( 'embed', '~', bg='cyan', text='bold' )

            # pyqtRemoveInputHook()
            # embed()
            # pyqtRestoreInputHook()

            # TODO:  self._properties?????????
            # props = self._properties( self._properties._original, broadcast=True, **props )
            # TODO: 'coords'?????  need PropertyManager(TransDict)
            #      Manage as property to avoid this check

            props = self._properties.translator(props)
            props = self._properties.converter(props)

            if not np.size(props.get('offsets', [])):
                return

        return props

    # @expose.args( pre='*'*100, post = '_'*100 )
    # @unhookPyQt
    def append(self, aps=None, **props):
        """Dynamically extend the ApertureCollection.
        params: aps - instance of ApertureCollection / keywords with properties of ApertureCollection to append
        """

        # embed()

        props = self.pre_append(aps=None, **props)

        if props is None:
            return

        # casting properties to the size of the instance in preparation for appending the
        # properties of the two instances (which might have unequal shapes)
        # NOTE: This is only really necessary when one will interact with the Apertures via
        # the figure canvas, and that interaction may change the properties of a individual
        # aperture.  in that case, one wants all properties to have the same shape, so they
        # can all be changed at a specific index.

        # FIXME: SMELLY CODE!!!!!!!!!!!!
        if not self.size:
            concatenate = lambda o, a: a
            # HACK! if the Collection was initialized as empty, set the new properties as current
        else:
            concatenate = PropertyManager.concatenate

        # for key in self._properties.__broadcast__:
        for key, val in props.items():

            # if not key in self._properties.__broadcast__:
            # banner( key, bg='red' )

            try:
                print('APPEND', key, self[key], val)
                new = concatenate(self[key], val)
                print('NEW:', new)
            except Exception as e:
                print('stacker fails')
                print(e)
                embed()

            setter = getattr(self, 'set_%s' % key)

            print()
            # try:
            setter(new)

    def within_allowed_range(self, r):
        # self.allowed_range
        return True

    # @unhookPyQt
    def resize(self, relative_motion, idx=..., ):
        print('RESIZING!', relative_motion, self.radii, idx)

        if not relative_motion is None:
            rnew = self.radii
            rnew[idx] += relative_motion
            if not self.within_allowed_range(rnew):
                return

            self.radii = rnew  # remember this is a property

    # @unhookPyQt
    def intersection(self, aps=None, unique=1):
        """
        Checks whether one set of apertures intersects (crosses) another set.
        If unique, returns only the crossings of apertures with unique coordinates.
        """
        if aps is None:
            aps = self  # determine self intersections by default

        # pyqtRemoveInputHook()
        # embed()
        # pyqtRestoreInputHook()

        # embed()

        # distance matrix for aperture centres (in pixels)
        dist_mat = cdist(self._offsets, aps._offsets)

        if unique:
            # distances between apertures with non-unique (duplicate) coordinates are not considered  i.e. self-intersection ignored
            dist_mat[dist_mat == 0] = np.inf

        # print( '=====>   ', self.coords, aps.coords )
        # print( self.radii.ravel()[None].T )
        # print( aps.radii.ravel() )

        # print( dist_mat )
        # print( '_'*88 )

        l = dist_mat - self.radii.ravel()[
            None].T - aps.radii.ravel() < 0  # which apertures intersect (cross) one another (excluding self intersection)

        # if aps is self:
        l = np.any(l, 1)  # self intersections

        return l

    # @unhookPyQt
    # @expose.args()
    def check_edge(self, edges):
        """
        Checks which apertures cross the image edge.

        Returns
        -------
        l : boolean array
        """
        # if edges is None:
        # edges = (10,10)     #selector.image_shape

        cshape = self.radii.shape + (2,)
        rshape = self.radii.shape + (1,)
        r = self.radii.reshape(*rshape)  # reshapes radii for arithmetic
        # try:
        coords = self.coords.reshape(
            *cshape)  # reshapes coords for arithmetic       # RESHAPING IN THE GETTER????
        # except:
        # print('\n'*2, 'check_edge fail', '\n'*2 )
        # embed()

        # coords = self.coords
        # r = self.radii.T

        l1 = coords + r >= edges
        l2 = coords - r <= (0, 0)
        return np.any(l1 | l2, axis=-1)

    def check(self, **kws):

        check = kws.get('check')
        edge = kws.get('edge')
        cross = kws.get('cross')
        if check == 'all':
            check = 1

        lc = np.zeros(len(self), bool)
        if not (check or edge or cross):
            Warn('Nothing being checked...')
            return lc

        # print( 'APERTURECOLLECTION AUTOCOLOUR' )
        # print( 'check', check, 'edge', edge, 'cross', cross )
        # print( 'self.check_edge( edge )',       self.check_edge( edge ) )
        # print( 'self.intersection( cross )', self.intersection( cross ) )

        if check == 'edge' or check == 1 or edge:
            lc |= self.check_edge(edge).ravel()
        if check == 'cross' or check == 1 or cross:
            lc |= self.intersection(cross)

        # print( 'lc', lc )

        return lc

    def auto_colour(self, **kws):
        lc = self.check(**kws)
        self.colourise(lc)

    # @unhookPyQt
    def colourise(self, lc):
        lc = lc.ravel()[None].T
        colours = np.where(lc, self.badcolour, self.goodcolour)
        try:
            self.set_ec(colours)
        except:
            print('colourise FAIL')
            embed()
