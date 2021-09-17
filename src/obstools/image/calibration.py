"""
Helpers for performing image calibration.
"""


# std
from collections import abc

# third-party
import numpy as np
from astropy.utils import lazyproperty
from astropy.io.fits.hdu import PrimaryHDU

# local
from recipes.dicts import pformat


class keep:
    pass


class IndexHelper:
    """
    Get `slice`s or tuples of slices by indexing this object.

    Examples
    --------
    >>> IndexHelper()[..., :, 1::2]
    (Ellipsis, slice(None, None, None), slice(1, None, 2))
    """

    def __getitem__(self, key):
        return key


_s = IndexHelper()


class ImageOrienter:
    """
    Simple base class that stores the orientation state. Images are
    re-oriented upon item access.
    """

    def __init__(self, hdu, flip='', x=False, y=False):
        """
        Image orientation helper

        Parameters
        ----------
        hdu : obstools.
            [description]
        flip : str or tuple, optional
            String(s) listing the xy-axes to flip (eg. 'xy'), by default ''.
            This parameter is mutually exclusive with parameters `x` and `y`
            which are also provided for convenience.
        x : bool, optional
            Flip x (columns) left right, by default False.
        y : bool, optional
            Flip y (rows) up down, by default False.
        """

        # assert isinstance(hdu, PrimaryHDU)
        assert hdu.ndim >= 2
        self.hdu = hdu

        # set item getter for dimensionality
        self.getitem = (self._getitem3d, self._getitem2d)[hdu.ndim == 2]

        # set some array-like attributes
        self.ndim = self.hdu.ndim
        self.shape = self.hdu.shape

        # setup tuple of slices for array
        orient = list(_s[..., :, :])
        for i, (s, t) in enumerate(zip('xy', (x, y)), 1):
            if (s in flip) or t:
                orient[-i] = _s[::-1]

        self.orient = tuple(orient)

    def __len__(self):
        return self.shape[0]
    
    def __call__(self, data):
        return data[self.orient]

    def __getitem__(self, key):
        return self.getitem(key)

    def _getitem2d(self, key):
        # `section` fails with 2d data
        return self.hdu.data[self.orient][key]

    def _getitem3d(self, key):
        # reading section for performance
        return self.hdu.section[key][self.orient]

    def __array__(self):
        return self.hdu.data[self.orient]


class CalibrationImage:
    """Descriptor class for calibration images"""

    # Orientation = ImageOrientBase

    def __init__(self, name):
        self.name = f'_{name}'

    def __get__(self, instance, kls):
        if instance is None:
            return self  # lookup from class
        return getattr(instance, self.name)

    def __set__(self, instance, value):
        if value is keep:
            return

        # Sub-framing
        sub = getattr(instance.hdu, 'subrect', ...)
        # set array as ImageCalibrator instance attribute '_dark' or '_flat'
        setattr(instance, self.name, get_array(value)[sub])

    def __delete__(self, instance):
        setattr(instance, self.name, None)


def get_array(hdu):
    if hdu is None:
        return

    if isinstance(hdu, ImageCalibratorMixin):
        # The image is an HDU object
        # ensure consistent orientation between image - and calibration data
        # NOTE: The flat fields will get debiased here. An array is returned
        img = hdu.calibrated  # [instance.hdu.oriented.orient]
    elif isinstance(img, PrimaryHDU):
        img = img.data

    img = np.asanyarray(img)
    if img.ndim != 2:
        raise ValueError(f'Calibration image must be 2d, not '
                         f'{img.ndim}d.')

    return img


class ImageCalibrator(ImageOrienter):
    """
    Do calibration arithmetic for CCD images on the fly!
    """
    # init the descriptors
    dark = CalibrationImage('dark')
    flat = CalibrationImage('flat')

    def __init__(self, hdu, dark=keep, flat=keep):

        xy = []
        if hasattr(hdu, 'oriented'):
            xy = [(o == _s[::-1]) for o in hdu.oriented.orient[:0:-1]]

        super().__init__(hdu, '', *xy)

        self._dark = self._flat = None
        self.dark = dark
        self.flat = flat

    def __str__(self):
        return pformat(dict(dark=self.dark,
                            flat=self.flat),
                       self.__class__.__name__)

    def __repr__(self):
        return str(self)

    def __call__(self, data):
        """
        Do calibration arithmetic on `data` ignoring orientation

        Parameters
        ----------
        data

        Returns
        -------

        """
        # dedark
        if self.dark is not None:
            data = data - self.dark

        # flat field
        if self.flat is not None:
            data = data / self.flat

        return data

    #
    def __getitem__(self, item):
        return self(super().__getitem__(item))


class ImageCalibratorMixin:
    """
    A calibration mixin for HDUs.
    """
    @lazyproperty
    def oriented(self):
        """Manage on-the-fly image orientation."""
        return ImageOrienter(self)

    @lazyproperty
    def calibrated(self):
        """Manage on-the-fly calibration for large files."""
        return ImageCalibrator(self)

    def set_calibrators(self, dark=keep, flat=keep):
        """
        Set calibration images for this observation. Default it to keep
        previously set image if none are provided here.  To remove a
        previously set calibration image pass a value of `None` to this
        function, or simply delete the attribute `self.calibrated.dark` or
        `self.calibrated.flat`

        Parameters
        ----------
        dark
        flat

        Returns
        -------

        """

        for name, hdu in dict(dark=dark, flat=flat).items():
            if isinstance(hdu, abc.Container) and (len(hdu) == 1):
                #              FIXME Campaign
                hdu = hdu[0]

            setattr(self.calibrated, name, hdu)
