"""
Image and image container classes
"""


# std
import warnings
from copy import copy

# third-party
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from pyxides import ListOf
from pyxides.getitem import IndexerMixin
from pyxides.vectorize import Vectorized, AttrVectorizer

# local
from scrawl.imagine import ImageDisplay
from recipes.oo import SelfAware
from recipes.pprint import qualname
from recipes.misc import duplicate_if_scalar
from recipes.dicts import pformat, AttrDict as ArtistContainer

# relative
from .detect import SourceDetectionMixin


UNIT_CORNERS = np.array([[0., 0.],
                         [1., 0.],
                         [1., 1.],
                         [0., 1.]])


def attr_dict(obj, keys):
    return {key: getattr(obj, key) for key in keys}


class Image(SelfAware):
    """
    A simple image class
    """

    _repr_keys = ('shape', )

    def __init__(self, data):
        """
        Create a Image

        Parameters
        ----------
        data : array-like
            The image data as a 2d

        """

        # data array
        self.data = np.asanyarray(data)
        # artists
        self.art = ArtistContainer(image=None, frame=None)

    def __repr__(self):
        return pformat(attr_dict(self, self._repr_keys),
                       self.__class__.__name__,
                       brackets='<>',
                       hang=True)

    def copy(self):
        return copy(self)

    @property
    def shape(self):
        return self.data.shape

    @property
    def transform(self):
        return Affine2D()

    @property
    def corners(self):
        """
        xy coords of image corners anti-clockwise from lower left
        """
        return self.transform.transform(UNIT_CORNERS * self.shape - 0.5)

    def __array__(self):
        return self.data

    def plot(self, ax=None, frame=True, set_lims=True, **kws):
        #  ,
        """
        Display the image in the axes
        """

        # logger.debug(f'{corners=}')
        im = ImageDisplay(self.data, ax=ax,
                          **{**dict(hist=False,
                                    sliders=False,
                                    cbar=False,
                                    interpolation='none'),
                             **kws})
        self.art.image = im.imagePlot
        ax = im.ax

        # Add frame around image
        if frame:
            frame_kws = dict(fc='none',
                             lw=1,
                             ec='0.5',
                             alpha=kws.get('alpha'))
            if isinstance(frame, dict):
                frame_kws.update(frame)

            self.art.frame = frame = Rectangle((-0.5, -0.5), *self.shape,
                                               **frame_kws)
            ax.add_patch(frame)

        return self.art

    # @lazyproperty
    # def grid(self):
    #     return np.indices(self.data.shape)

    # def grid(self, p, scale):
    #     """transformed pixel location grid cartesian xy coordinates"""
    #     g = np.indices(self.data.shape).reshape(2, -1).T[:, ::-1]
    #     return transforms.affine(g, p, scale)


class TransformedImage(Image):

    _repr_keys = ('shape', 'scale', 'offset', 'angle')

    # @doc.inherit('Parameters')
    def __init__(self, data, offset=(0, 0), angle=0, scale=1):
        """
        A translated, scaled, rotated image.

        Parameters
        ----------
        offset : tuple, optional
            [description], by default (0, 0)
        angle : int, optional
            [description], by default 0
        scale : float or array_like of float, optional, by default 1
            Pixel scale in units/pixel. 

        Raises
        ------
        ValueError
            If both `fov` and `scale` are None

        """
        super().__init__(data)
        self.offset = offset
        self.angle = angle
        self.scale = scale
        """scale in data units per pixel for required coordinate system"""

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, angle):
        self._angle = float(angle)

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, offset):
        offset = np.asarray(offset, float)
        assert offset.size == 2
        self._offset = offset.squeeze()

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
        self._scale = np.array(duplicate_if_scalar(scale), float)

    pixel_scale = scale

    @property
    def params(self):
        return np.array([*self._offset, self.angle])

    @params.setter
    def params(self, params):
        *self.offset, self.angle = params

    @property
    def transform(self):
        return Affine2D().scale(*self.scale).rotate(self.angle).translate(*self.offset)

    def plot(self, ax=None, frame=True, set_lims=True, **kws):

        art = super().plot(ax, frame, set_lims, **kws)
        ax = art.image.axes

        # Rotate + offset the image by setting the transform
        art_trans = self.transform + ax.transData
        for artist in art.values():
            artist.set_transform(art_trans)

        # update axes limits
        if set_lims:
            corners = self.corners
            xlim, ylim = np.sort([corners.min(0), corners.max(0)]).T
            ax.set(xlim=xlim, ylim=ylim)

        return art


class SkyImage(TransformedImage, SourceDetectionMixin):
    """
    Helper class for image registration. Represents an image with some
    associated meta data like pixel scale as well as detected sources and their
    counts.
    """
    # @doc.inherit('Parameters')

    # _repr_keys = 'shape', 'scale' #, 'offset', 'angle'
    # filename = None

    @classmethod
    # @caches.to_file(cachePaths.skyimage, typed={'hdu': _hdu_hasher})
    def from_hdu(cls, hdu, sample_stat='median', depth=10, **kws):
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
                '`from_image` constructor (which also runs source detection), '
                'or initialize the class directly with an image array .'
            )

        # self.filename

        # use `hdu.detect` so we cache the detections on the hdu filename
        seg = hdu.detect(sample_stat, depth, **kws)
        # pull the sample image (computed in the line above) from the cache
        image = hdu.get_sample_image(sample_stat, depth)
        
        # TODO: if wcs is defined, use that as default
        
        return cls(image, hdu.fov, angle=hdu.pa, segmentation=seg)

    @classmethod
    def from_image(cls, image, fov=None, scale=None, **kws):
        image = cls(image, fov, scale=scale)
        image.detect(**kws)
        return image

    def __init__(self, data, fov=None, offset=(0, 0), angle=0, scale=None,
                 segmentation=None):
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
                             'given and not be `None`')

        if scale is None:
            fov = np.array(duplicate_if_scalar(fov))
            scale = fov / data.shape

        # init
        TransformedImage.__init__(self, data, offset, angle, scale)

        # segmentation data
        self.seg = segmentation     # : SegmentedArray or None
        self.xy = None      # : np.ndarray: center-of-mass coordinates pixels
        self.counts = None  # : np.ndarray: pixel sums for segmentation

    @property
    def fov(self):
        """Field of view"""
        return self.shape * self.scale

    def detect(self, snr=3, **kws):  # max_iter=1,

        if self.seg is None:
            self.seg = super().detect(self.data, snr=snr, **kws)

        # centre of mass, counts
        yx = self.seg.com_bg(self.data)
        counts, noise = self.seg.flux(self.data)

        # sometimes, we get nans from the center-of-mass calculation
        ok = np.isfinite(yx).all(1)
        if not ok.any():
            warnings.warn('No detections for image.')

        self.xy = yx[ok, ::-1]
        self.counts = counts[ok]
        # return xy coordinates in pixels
        return self.seg, self.xy

    def plot(self, ax=None, frame=True, positions=False, regions=False,
             labels=False, set_lims=True, **kws):
        #  ,
        """
        Display the image in the axes, applying the affine transformation for
        parameters `p`
        """
        art = super().plot(ax, frame, set_lims, **kws)

        # add xy position markers (centre of mass)
        if positions:
            art.points, = ax.plot(
                *self.transform.transform(self.xy).T,
                **{**dict(marker='x', color='w', ls='none'),
                   **(positions if isinstance(positions, dict) else{})}
            )

        # add segmentation contours
        transform = self.transform + ax.transData
        if regions:
            regions = regions if isinstance(regions, dict) else {}
            regions.setdefault('alpha', kws.get('alpha'))
            art.seg = self.seg.draw_contours(ax, transform=transform, **kws)

        if labels:
            art.texts = self.seg.draw_labels(
                ax, **{**dict(size=8,
                              color='w',
                              weight='heavy',
                              transform=transform),
                       **(labels if isinstance(labels, dict) else {})}
            )
        return art

    def label_image(self, ax, name='', **kws):
        return ax.text(*self.corners[-1], name,
                       rotation=np.degrees(self.angle),
                       rotation_mode='anchor',
                       va='top',
                       **kws)


class ImageContainer(IndexerMixin, ListOf(SkyImage), Vectorized):
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
            # items = self.checks_type(images, silent=True)
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
    images = AttrVectorizer('data')
    shapes = AttrVectorizer('data.shape', convert=np.array)
    detections = AttrVectorizer('seg')
    coms = AttrVectorizer('xy')
    fovs = AttrVectorizer('fov', convert=np.array)
    scales = AttrVectorizer('scale', convert=np.array)
    params = AttrVectorizer('params', convert=np.array)
    xy_offsets = AttrVectorizer('offset', convert=np.array)
    angles = AttrVectorizer('angles', convert=np.array)
    corners = AttrVectorizer('corners', convert=np.array)

    # @property
    # def params(self):
    #     return np.array(self._params)
