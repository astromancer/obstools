

# std libs
import warnings

# third-party libs
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

# local libs
from scrawl.imagine import ImageDisplay
from recipes.oo import SelfAware
from recipes.misc import duplicate_if_scalar
from recipes.dicts import AttrDict as ArtistContainer

# relative libs
from .segmentation import SegmentedImage


UNIT_CORNERS = np.array([[0., 0.],
                         [1., 0.],
                         [1., 1.],
                         [0., 1.]])


# from obstools.phot.image import SourceDetectionMixin


def detect_measure(image, mask=False, background=None, snr=3., npixels=5,
                   edge_cutoff=None, deblend=False, dilate=0):
    #
    seg = SegmentedImage.detect(image, mask, background, snr, npixels,
                                edge_cutoff, deblend, dilate)

    counts = seg.sum(image) - seg.median(image, [0]) * seg.areas
    coords = seg.com_bg(image)
    return seg, coords, counts


class Image(SelfAware):
    """
    A simple image class
    """

    def __init__(self, data):
        """
        Create a Image

        Parameters
        ----------
        data : array-like
            The image data as a 2d

        """

        # data array
        self.data = np.asarray(data)
        # artists
        self.art = ArtistContainer(image=None, frame=None)

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
        return self.transform.transform(UNIT_CORNERS * self.shape)

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
        offset = np.asarray(offset)
        assert offset.size == 2
        self._offset = offset.squeeze()

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
        self._scale = np.array(duplicate_if_scalar(scale))

    @property
    def transform(self):
        return Affine2D().scale(*self.scale).rotate(self.angle).translate(*self.offset)

    def plot(self, ax=None, frame=True, set_lims=True, **kws):

        art = super().plot(ax, frame, set_lims, **kws)
        ax = art.image

        # Rotate + offset the image by setting the transform
        art_trans = self.transform + ax.transData
        art.image.set_transform(art_trans)
        art.frame.set_transform(art_trans)

        # update axes limits
        if set_lims:
            corners = self.corners
            xlim, ylim = np.sort([corners.min(0), corners.max(0)]).T
            ax.set(xlim=xlim, ylim=ylim)

        return art


class SkyImage(TransformedImage):
    """
    Helper class for image registration. Represents an image with some
    associated meta data like pixel scale as well as detected sources and their
    counts.
    """
    # @doc.inherit('Parameters')

    def __init__(self, data, fov=None, scale=None):
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

        if (fov is scale is None):
            raise ValueError('Either field-of-view, or pixel scale must be '
                             'given and not be `None`')

        if scale is None:
            fov = np.array(duplicate_if_scalar(fov))
            scale = fov / data.shape

        # init
        super().__init__(data, scale=scale)

        # segmentation data
        self.seg = None     # : SegmentedArray:
        self.xy = None      # : np.ndarray: center-of-mass coordinates pixels
        self.counts = None  # : np.ndarray: pixel sums for segmentation

    @property
    def fov(self):
        """Field of view"""
        return self.shape * self.scale

    def detect(self, **kws):
        self.seg, yx, counts = detect_measure(self.data, **kws)

        # sometimes, we get nans from the center-of-mass calculation
        ok = np.isfinite(yx).all(1)
        if not ok.any():
            warnings.warn('No detections for image.')

        self.xy = yx[ok, ::-1]
        # self.xy = self.xyp * self.pixel_scale
        self.counts = counts[ok]
        return self.seg, self.xy, self.counts

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
        if regions:
            regions = regions if isinstance(regions, dict) else {}
            regions.setdefault('alpha', kws.get('alpha'))
            self.draw_segments(ax, labels, **regions)

        return art

    def draw_segments(self, ax, labels=True, **kws):
        transform = self.transform + ax.transData
        self.art.seg = self.seg.draw_contours(ax, transform=transform, **kws)

        if labels:
            self.art.texts = self.seg.draw_labels(
                ax, **{**dict(size=8,
                              color='w',
                              weight='heavy',
                              transform=transform),
                       **(labels if isinstance(labels, dict) else {})}
            )
        return self.art.seg, self.art.texts

    def label_image(self, ax, name='', **kws):
        #
        return ax.text(*self.corners[-1], name,
                       rotation=np.degrees(self.angle),
                       rotation_mode='anchor',
                       va='top',
                       **kws)
