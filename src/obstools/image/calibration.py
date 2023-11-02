"""
Helpers for efficient image calibration.
"""


# std
from recipes import op
from collections import abc

# third-party
import numpy as np
from astropy.utils import lazyproperty
from astropy.io.fits.hdu import PrimaryHDU

# local
from recipes.dicts import pformat
from recipes.oo.property import ForwardProperty


# ---------------------------------------------------------------------------- #

# API flag
keep = object()


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

# ---------------------------------------------------------------------------- #


def _get_array(hdu):
    # used when setting a calibration data (hdu) on the science hdu to retrieve
    # the data array

    if hdu is None:
        return

    if isinstance(hdu, ImageCalibratorMixin):
        # The image is an HDU object
        # ensure consistent orientation between image - and calibration data
        # NOTE: The flat fields will get debiased here. An array is returned
        return hdu.calibrated

    if isinstance(hdu, PrimaryHDU):
        return hdu.data

    if isinstance(hdu, np.ndarray):
        return hdu

    raise TypeError(f'Received invalid object of type {type(hdu)}. '
                    'Calibration frames should be one of the following '
                    'types: ImageCalibratorMixin, PrimaryHDU, np.ndarray.')


def get_array(hdu):
    # used when setting a calibration data (hdu) on the science hdu to retrieve
    # the data array

    img = _get_array(hdu)

    if img is None:
        return

    if img.ndim != 2:
        raise ValueError(f'Calibration image must be 2D, not {img.ndim}D.')

    return img


class CalibrationImageDescriptor:
    """
    Descriptor class for calibration images (flat/dark).
    """

    def __init__(self, name):
        self.name = f'_{name}'

    def __get__(self, instance, kls):
        # lookup from class                  lookup from instance
        return self if instance is None else getattr(instance, self.name)

    def __set__(self, instance, value):
        if value is keep:
            return

        # set array as ImageCalibrator instance attribute '_dark' or '_flat'
        img = get_array(value)

        if img is not None:
            # Sub-framing the calibration frames to match target
            img = img[getattr(instance.hdu, 'subrect', ...)]

        # attach to instance
        setattr(instance, self.name, img)

        # update the noise model to account for dark / flat noise
        if isinstance(value, ImageCalibratorMixin):
            setattr(instance.calibrated.noise_model, self.name[1:], value.noise_model.var(img))

    def __delete__(self, instance):
        setattr(instance, self.name, None)


class ImageOrienter:
    """
    Simple base class that stores the orientation state. Images are re-oriented
    upon item access. To conserve working memory during large array operations,
    `astropy.io.fits.hdu.image.Section` is used for 3d data slices.
    """

    # forward array-like properties to hdu
    ndim = ForwardProperty('hdu.ndim')
    shape = ForwardProperty('hdu.shape')

    def __init__(self, hdu, flip='', x=False, y=False):
        """
        Image orientation helper.

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
        self.getitem = self._getitem2d if hdu.ndim == 2 else self._getitem3d

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

    def __array__(self):
        return self.hdu.data[self.orient]

    def __getitem__(self, key):
        return self.getitem(key)

    def _getitem2d(self, key):
        # `section` fails with 2d data
        return self.hdu.data[self.orient][key]

    def _getitem3d(self, key):
        # reading section for performance
        return self.hdu.section[key][self.orient]


class ImageCalibrator(ImageOrienter):
    """
    Do calibration arithmetic for CCD images on the fly!
    """

    # init the descriptors
    dark = CalibrationImageDescriptor('dark')
    flat = CalibrationImageDescriptor('flat')

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, gain):
        gain = float(gain)
        assert gain > 0
        self._gain = gain

    @gain.deleter
    def gain(self):
        self._gain = 1

    def __init__(self, hdu, dark=keep, flat=keep, gain=None):

        xy = []
        if hasattr(hdu, 'oriented'):
            xy = [(o == _s[::-1]) for o in hdu.oriented.orient[:0:-1]]

        super().__init__(hdu, '', *xy)

        self._dark = self._flat = None
        self.dark = dark
        self.flat = flat

        if gain is None:
            gain = op.attrgetter('readout.preAmpGain', default=1)(hdu)
        self.gain = gain

        self.noise_model = hdu.noise_model.copy()

    def __str__(self):
        return pformat(dict(gain=self.gain,
                            dark=self.dark,
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
        # Since darks and flats are already gain calibrated, we have to do that
        # gain correction: image now in units of electrons
        data = data * self.gain

        # dark subtract
        if self.dark is not None:
            # subtract dark
            data = data - self.dark

        # flat field
        if self.flat is not None:
            data = data / self.flat

        return data

    def __getitem__(self, item):
        # calibration done on getitem!
        return self(super().__getitem__(item))


class ImageCalibratorMixin:
    """
    A calibration mixin for HDU objects.
    """

    # dark = ForwardProperty('calibrted.dark')
    # flat = ForwardProperty('calibrted.flat')
    # gain = ForwardProperty('calibrted.gain')

    @lazyproperty
    def oriented(self):
        """Manage on-the-fly image orientation."""
        return ImageOrienter(self)

    @lazyproperty
    def calibrated(self):
        """Manage on-the-fly image calibration."""
        return ImageCalibrator(self)

    def set_calibrators(self, dark=keep, flat=keep, gain=keep):
        """
        Set calibration images for this observation. Default it to keep
        previously set image if none are provided here. To remove a previously
        set calibration image pass a value of `None` to this function, or simply
        delete the attribute `self.calibrated.dark`,  `self.calibrated.flat` or
        `self.calibrated.gain`.

        Parameters
        ----------
        dark
        flat

        Returns
        -------

        """

        for name, val in dict(dark=dark, flat=flat, gain=gain).items():
            if isinstance(val, abc.Container) and (len(val) == 1):  # FIXME Campaign
                val = val[0]

            setattr(self.calibrated, name, val)
