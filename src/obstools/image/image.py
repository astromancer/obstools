"""
Image and image container classes
"""


# std
import pickle
import warnings
from collections import abc

# third-party
import numpy as np
import more_itertools as mit
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

# local
from scrawl.image import CanvasBlitHelper, ImageDisplay
from pyxides import ListOf
from pyxides.getitem import IndexingMixin
from pyxides.vectorize import AttrVector, Vectorized
from recipes.oo import SelfAware
from recipes.oo.slots import SlotHelper
from recipes.oo.repr_helpers import qualname
from recipes.oo.property import cached_property
from recipes.utils import duplicate_if_scalar, not_null
from recipes.dicts import isdict, AttrDict as ArtistContainer

# relative
from .detect import SourceDetectionMixin
from .calibration import ImageCalibratorMixin


# ---------------------------------------------------------------------------- #
SCALE_DFLT = 1
UNIT_CORNERS = np.array([[0., 0.],
                         [1., 0.],
                         [1., 1.],
                         [0., 1.]])

IMAGE_STYLE = dict(cmap=None,  # 'cmr.voltage_r',
                   hist=False,
                   sliders=False,
                   cbar=False)

CONTOUR_STYLE = dict(cmap='hot',
                     lw=1.5)

FRAME_STYLE = dict(fc='none',
                   lw=1,
                   ec='0.5')

MARKER_STYLE = dict(marker='x',
                    color='w',
                    ls='none')

TEXT_STYLE = dict(size='xx-small',
                  color='w',
                  weight='heavy',
                  offset=5)

# ---------------------------------------------------------------------------- #


def isdict(obj):
    return isinstance(obj, dict)


class Image(SelfAware, SlotHelper):  # AliasManager
    """
    A simple image class.
    """

    # ------------------------------------------------------------------------ #
    __slots__ = ('data', 'meta', 'art')

    _repr_style = dict(SlotHelper._repr_style,
                       attrs=['shape'],
                       maybe=['meta'],)
    #    brackets='<>',
    #    hang=True)

    # ------------------------------------------------------------------------ #
    def __init__(self, data, **kws):
        """
        Create a Image

        Parameters
        ----------
        data : array-like
            The image data as a 2d
        """

        # data array
        self.data = np.asanyarray(data)
        # meta data
        self.meta = kws
        # artists
        self.art = self._init_art()

    def _init_art(self):
        return ArtistContainer(display=None, image=None, frame=None)

    def __getstate__(self):
        # remove artists that can't be pickled
        return {**super().__getstate__(), 'art': self._init_art()}

    def __array__(self):
        return self.data

    def __getitem__(self, key):
        return self.data[key]

    def copy(self,):
        return pickle.loads(pickle.dumps(self))

    # ------------------------------------------------------------------------ #

    @property
    def shape(self):
        return self.data.shape

    @property
    def transform(self):
        return Affine2D()

    @property
    def corners(self):
        """
        xy coords of image corners anti-clockwise from lower left.
        """
        return self.transform.transform(UNIT_CORNERS * self.shape - 0.5)

    # ------------------------------------------------------------------------ #
    # @alias('plot')
    def show(self, image=True, frame=True, set_lims=True, **kws):
        #  ,
        """
        Display the image in the axes.
        """

        # logger.debug(f'{corners=}')
        ax = None
        if image:
            display = ImageDisplay(self.data, **{**IMAGE_STYLE, **kws})
            self.art.image = display.image
            ax = display.ax

        # Add frame around image
        if frame:
            frame_kws = dict(**FRAME_STYLE, alpha=kws.get('alpha'))
            if isdict(frame):
                frame_kws.update(frame)

            self.art.frame = frame = Rectangle((-0.5, -0.5), *self.shape[::-1],
                                               **frame_kws)
            if (ax := kws.get('ax', ax)) is None:
                raise TypeError('`ax` keyword required')

            ax.add_patch(frame)

        return display, self.art

    # alias
    plot = show

    @cached_property
    def grid(self):
        return np.indices(self.data.shape)


class TransformedImage(Image):
    # TODO: array_to_pixel, pixel_to_world, array_to_axes etc.

    # ------------------------------------------------------------------------ #
    __slots__ = ('_origin', '_angle', '_scale')

    _repr_style = dict(Image._repr_style,
                       attrs=('shape', 'scale', 'origin', 'angle'))

    # ------------------------------------------------------------------------ #
    # @doc.inherit('Parameters')
    def __init__(self, data, origin=(0, 0), angle=0, scale=SCALE_DFLT, **kws):
        """
        A translated, scaled, rotated image.

        Parameters
        ----------
        origin : tuple, optional
            Position of the lower left corner of the image array with respect to
            some user-defined coordinate system, by default (0, 0).
        angle : int, optional
            Rotation angle in radians counter-clockwise from horizontal axis, by
            default 0.
        scale : float or array_like of float, optional, by default 1
            Pixel scale in units/pixel. 

        Raises
        ------
        ValueError
            If both `fov` and `scale` are None

        """
        super().__init__(data, **kws)
        self.origin = origin
        self.angle = angle
        self.scale = scale
        """scale in data units per pixel for required coordinate system"""

    # ------------------------------------------------------------------------ #
    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, angle):
        self._angle = float(angle)

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, origin):
        origin = np.asarray(origin, float)
        assert origin.size == 2
        self._origin = origin.squeeze()

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
        self._scale = np.array(duplicate_if_scalar(scale), float)

    # alias
    pixel_scale = scale

    @property
    def params(self):
        return np.array([*self._origin, self.angle])

    @params.setter
    def params(self, params):
        *self.origin, self.angle = params

    @property
    def transform(self):
        return Affine2D().scale(*self.scale).rotate(self.angle).translate(*self.origin)

    @cached_property
    def grid(self):
        """transformed pixel location grid cartesian xy coordinates"""
        grid = Image.grid.fget(self)
        return self.transform(grid.reshape(2, -1).T[:, ::-1])
    # ------------------------------------------------------------------------ #
    # def contains(self, xy):
    #     # tr = Affine2D().scale(*( / reg.scale)).rotate(self.angle).translate(*self.origin)
    #     tr = Affine2D().translate(*-self.origin).rotate(-self.angle).scale(*reg.scale)
    #     xyi = tr.transform(xy)
    #     return (0 > xyi & xyi < 1).any()

    # ------------------------------------------------------------------------ #
    def show(self, image=True, frame=True, set_lims=None, coords='world', **kws):

        display, art = super().show(image, frame, set_lims, **kws)
        ax = next(filter(None, mit.collapse(art.values()))).axes

        # Rotate + offset the image by setting the transform
        if coords == 'world':
            art_trans = self.transform + ax.transData
            for artist in filter(None, mit.collapse(art.values())):
                artist.set_transform(art_trans)

        # update axes limits
        if set_lims := (set_lims or (coords == 'world')):
            corners = self.corners
            xlim, ylim = np.array([corners.min(0), corners.max(0)]).T
            # self.logger.debug('Updating axes limits {}', dict(xlim=xlim, ylim=ylim))
            ax.set(xlim=xlim, ylim=ylim)

        # add artists for blitting
        display.add_art(art.values())

        return display, art

    # alias
    plot = show


class CCDImage(ImageCalibratorMixin):

    ndim = 2

    def __getitem__(self, key):
        return self.calibrated[key]


class SkyImage(CCDImage, TransformedImage, SourceDetectionMixin):
    """
    Helper class for image registration. Represents an image with some
    associated meta data like pixel scale and pixel map of detected sources,
    their coordinates and integrated counts.
    """

    __slots__ = ('seg', 'xy', 'counts')

    _repr_style = dict(TransformedImage._repr_style,
                       maybe=['angle'])

    # ------------------------------------------------------------------------ #
    # @doc.inherit('Parameters')
    # filename = None

    @classmethod
    # @caches.to_file(cachePaths.skyimage, typed={'hdu': _hdu_hasher})
    def from_hdu(cls, hdu, sample_stat='median', depth=5, interval=...,
                 report=False, **kws):
        """
        Construct a SkyImage from an HDU by first drawing a sample image, then
        running the source detection algorithm on it.

        Parameters
        ----------
        hdu : [type]
            [description]
        sample_stat : str, optional
            [description], by default 'median'
        depth : int, optional
            [description], by default 10

        Returns
        -------
        SkyImage
            [description]
        """

        from .sample import ImageSamplerMixin

        if not isinstance(hdu, ImageSamplerMixin):
            raise TypeError(
                f'Received object {hdu} of type: {type(hdu)}. '
                'Can only initialize from HDUs that inherit from '
                f'`{qualname(ImageSamplerMixin)}`. Alternatively use the '
                '`from_image` constructor, which runs source detection '
                'on an image array.'
            )

        # logger.info(str(kws))

        # use `hdu.detection` which caches the detections on the hdu filename
        seg = hdu.detect(sample_stat, depth, interval, report, **kws)
        del seg.slices # FIXME: since this may be incorrect in the cache!!? 

        # pull the sample image (computed in the line above) from the cache
        image = hdu.get_sample_image(sample_stat, depth, interval)

        # TODO: if wcs is defined, use that as default
        return cls(image, hdu.fov, angle=hdu.pa, segments=seg, **dict(hdu.header))

    @classmethod
    def from_image(cls, image, fov=None, scale=SCALE_DFLT, **kws):
        return cls(image, fov, scale=scale,
                   segments=super().from_image(image, **kws))

    # ------------------------------------------------------------------------ #
    def __init__(self, data, fov=None, origin=(0, 0), angle=0, scale=SCALE_DFLT,
                 segments=None, **kws):
        """
        Create and SkyImage object with a know size on sky.

        Parameters
        ----------
        fov : float or array_like of float, optional
            Field-of-view of the image in arcminutes. The default None, however
            either `fov` or `scale` must be given.
        scale : float or array_like of float, optional
            Pixel scale in arcminutes/pixel. The default None, however
            either `fov` or `scale` must be given.

        Raises
        ------
        ValueError
            If both `fov` and `scale` are None
        """

        if fov is scale is None:
            raise ValueError('Either field-of-view, or pixel scale must be '
                             'given and not be `None`.')

        if scale is None:
            fov = np.array(duplicate_if_scalar(fov))
            scale = fov / np.shape(data)

        # init
        TransformedImage.__init__(self, data, origin, angle, scale, **kws)

        # segmentation data
        self.seg = segments     # : SegmentedArray or None
        self.xy = None      # : np.ndarray: center-of-mass coordinates pixels
        self.counts = None  # : np.ndarray: pixel sums for segmentation

    def __getstate__(self):
        state = super().__getstate__()
        for name in ('oriented', 'calibrated'):
            state[name] = getattr(self, name, None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        # handle copying descriptor states
        # (not handled by superclass since they are not slots)
        for name in ('oriented', 'calibrated'):
            if val := state.get(name, None):
                setattr(self, name, val)

    # @lazyproperty
    # def xy(self):
    #     self.detect()

    @property
    def fov(self):
        """Field of view"""
        return self.shape * self.scale

    # ------------------------------------------------------------------------ #
    def detect(self, snr=3, **kws):  # max_iter=1,

        if self.seg is None:
            self.seg = super().detect(self.data, snr=snr, **kws)

        # centre of mass, counts
        yx = self.seg.com(self.data)
        counts, noise = self.seg.flux(self.data)

        # sometimes, we get nans from the center-of-mass calculation
        ok = np.isfinite(yx).all(1)
        if not ok.any():
            warnings.warn('No detections for image.')

        self.xy = yx[ok, ::-1]
        self.counts = counts[ok]
        # return xy coordinates in pixels
        return self.seg, self.xy

    def remove_segment(self, label):
        self.seg.remove_label(label)
        self.xy = np.delete(self.xy, label - 1, 0)
        self.counts = np.delete(self.counts, label - 1)

    def _resolve_points(self, points):
        if (points is True):
            points = {}
            xy = self.xy
        if isinstance(points, str):
            colour, marker = points
            points = dict(marker=marker, color=colour)
            xy = self.xy
        elif isinstance(points, abc.Container):
            xy = np.asanyarray(points)
            assert xy.ndim == 2
            assert xy.shape[-1] == 2
        else:
            raise TypeError(f'Invalid points object {points} of type '
                            f'{type(points)}.')

        points_style = {**MARKER_STYLE,
                        **(points if isdict(points) else {})}
        
        return xy, points_style

    def show(self, image=True, frame=True, points=False, regions=False,
             labels=False, set_lims=None, coords='world', **kws):
        """
        Display the image in the axes, applying the transformation to 
        coordinate system given in `coords`.
        """
        assert coords in {'pixel', 'world'}  # TODO: array

        display, art = super().show(image, frame, set_lims, coords, **kws)
        ax = next(filter(None, mit.collapse(art.values()))).axes

        # add xy position markers (centre of mass)
        if not_null(points):
            xy, points_style = self._resolve_points(points)
            art.points, = ax.plot(*xy.T, **points_style)

        # add segmentation contours
        tr = self.transform
        transform = ax.transData if coords == 'pixel' else tr + ax.transData
        if regions:
            regions = regions if isdict(regions) else {}
            regions.setdefault('alpha', kws.get('alpha'))
            art.contours = self.seg.show.contours(ax, transform=transform,
                                                  **{**CONTOUR_STYLE, **regions})

        # add label artists
        if labels:
            art.texts = self.seg.show.labels(
                ax, **{**TEXT_STYLE, 'transform': transform,
                       **(labels if isdict(labels) else {})}
            )

        # add artists for blitting
        display.add_art(art.values())

        return display, art

    # alias
    plot = show

    def add_label(self, ax, name=None, **kws):
        return ax.text(*self.corners[-1],
                       (self.meta.get('name', '') if name is None else name),
                       rotation=np.degrees(self.angle), rotation_mode='anchor',
                       va='top', **kws)


class ImageContainer(IndexingMixin, ListOf(SkyImage), Vectorized):
    def __init__(self, images=(), fovs=()):
        """
        A container of `SkyImages`'s

        Parameters
        ----------
        images : sequence, optional
            A sequence of `SkyImages` or 2d `np.ndarrays`, by default ()
        fovs : sequence, optional
            A sequence of field-of-views, each being of size 1 or 2. If each
            item in the sequence is of size 2, it is the field-of-view along the
            image array dimensions (rows, columns). It an fov is size 1, it is
            as being the field-of-view along each dimension of a square image.
            If `images` is a sequence of `np.ndarrays`, `fovs` is a required
            parameter

        Raises
        ------
        ValueError
            If `images` is a sequence of `np.ndarrays` and `fovs` is not given.
        """
        # check init parameters.  If `images` are arrays, also need `fovs`
        n = len(images)
        if n != len(fovs):
            # items = self.check_all_types(images, silent=True)
            types = set(map(type, images))

            if len(types) == 1 and issubclass(types.pop(), SkyImage):
                fovs = [im.fov for im in images]
            else:
                raise ValueError(
                    'When initializing this class from a stack of images, '
                    'please also proved the field-of-views `fovs`.')
                # as well as the set transformation parameters `params`.')

            # create instances of `SkyImage`
            images = map(SkyImage, images, fovs)

        # initialize container
        super().__init__(images)

        # ensure we get lists back from getitem lookup since the initializer
        # works differently to standard containers
        self.set_returned_type(list)

    def __repr__(self):
        n = len(self)
        return f'{self.__class__.__name__}: {n} image{"s" * bool(n)}'

    # properties: vectorized attribute getters on `SkyImage`
    images = AttrVector('data')
    shapes = AttrVector('data.shape', output=np.array)
    detections = AttrVector('seg')
    coms = centroids = AttrVector('xy')
    fovs = AttrVector('fov', output=np.array)
    scales = AttrVector('scale', output=np.array)
    params = AttrVector('params', output=np.array)
    origins = AttrVector('origin', output=np.array)
    angles = AttrVector('angles', output=np.array)
    corners = AttrVector('corners', output=np.array)

    # @property
    # def params(self):
    #     return np.array(self._params)
