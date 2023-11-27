

# third-party
from astropy.utils import lazyproperty
from astropy.io.fits.hdu import PrimaryHDU

# local
from recipes.logging import LoggingMixin
from recipes.decorators import update_defaults

# relative
from ..io import _FilePicklable
from . import CONFIG
from .noise import CCDNoiseModel
from .sample import ImageSamplerMixin
from .detect import SourceDetectionMixin
from .calibrate import ImageCalibratorMixin


# ---------------------------------------------------------------------------- #

class ImageHDU(PrimaryHDU,
               ImageSamplerMixin,
               ImageCalibratorMixin,
               SourceDetectionMixin,
               LoggingMixin):
    """
    Some extra methods and properties to support PhotCampaign features.
    """

    @classmethod
    def readfrom(cls, fileobj, checksum=False, ignore_missing_end=False, **kws):

        if not isinstance(fileobj, _FilePicklable):
            fileobj = _FilePicklable(fileobj)

        return PrimaryHDU.readfrom(fileobj, checksum, ignore_missing_end, **kws)

    @update_defaults(CONFIG.sample)
    def detect(self, stat, min_depth, interval=..., report=True, **kws):
        """
        Cached source detection for HDUs.

        Parameters
        ----------
        stat : str, optional
            Statistic to use, by default 'median'.
        min_depth : int, optional
            [description], by default 5
        snr : int, optional
            [description], by default 3

        Returns
        -------
        seg
            SegmentedImage
        """
        # NOTE: `get_sample_image` and `detection` are both cached for performance
        image = self.get_sample_image(stat, min_depth, interval)

        if report is True:
            report = CONFIG.detect.report
        if report:
            report = {**report, 'title': self.file.name}

        return super().detect(image, **kws, report=report)

    @property
    def file(self):
        return self._FilenameHelperClass(self)

    @property
    def ishape(self):
        """Image frame shape"""
        return self.shape[-2:]

    @property
    def ndim(self):
        return len(self.shape)

    @lazyproperty
    def fov(self):
        # field of view
        return self.get_fov()

    def get_fov(self):
        raise NotImplementedError

    @property
    def pixel_scale(self):
        return self.fov / self.ishape

    @lazyproperty
    def pa(self):
        return self.get_rotation()

    def get_rotation(self):
        """
        Get the instrument rotation (position angle) wrt the sky in radians
        """
        raise NotImplementedError

    @lazyproperty
    def noise_model(self):
        return CCDNoiseModel(self.readout.noise, self.readout.preAmpGain)

    # plotting
    def show(self, **kws):
        """Display the data. """

        if (nd := self.ndim) == 2:
            from scrawl.image import ImageDisplay

            im = ImageDisplay(self.data, **kws)
            # Note: `self.section` fails with 2d data

        elif nd == 3:
            from .image.display import FitsVideo

            im = FitsVideo(self, **kws)

        else:
            raise TypeError(f'Can only display 2D or 3D data. Your data is {nd}D.')

        im.figure.canvas.manager.set_window_title(self.file.name)
        return im

