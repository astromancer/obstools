from .orient import ImageOrienter
from recipes.dicts import pformat


class keep:
    pass


class CalibrationImage:
    """Descriptor class for calibration images"""

    # Orientation = ImageOrientBase

    def __init__(self, name):
        self.name = f'_{name}'

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.name)

    def __set__(self, instance, value):
        if value is keep:
            return

        if value is not None:
            # ensure consistent orientation
            # note getting array here!!
            assert len(value.shape) == 2, 'Calibration image must be 2d'
            value = value.oriented[:]

        setattr(instance, self.name, value)

    def __delete__(self, instance):
        setattr(instance, self.name, None)


class ImageCalibration(ImageOrienter):
    """
    Do calibration arithmetic for CCD images on the fly
    """
    # init the descriptors
    bias = CalibrationImage('bias')
    flat = CalibrationImage('flat')

    def __init__(self, hdu, bias=keep, flat=keep):
        super().__init__(hdu)
        self._bias = self._flat = None
        self.bias = bias
        self.flat = flat

    def __str__(self):
        return pformat(dict(bias=self.bias,
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
        # debias
        if self.bias is not None:
            data = data - self.bias

        # flat field
        if self.flat is not None:
            data = data / self.flat

        return data

    #
    def __getitem__(self, item):
        return self(super().__getitem__(item))
