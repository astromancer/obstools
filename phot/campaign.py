import operator
from pathlib import Path
import itertools as itt

import numpy as np

from recipes.logging import LoggingMixin
from astropy.io.fits.hdu import HDUList
from astropy.io.fits.hdu.base import _BaseHDU
from recipes.set import OrderedSet


class SelfAwareContainer(object):
    _skip_init = False

    def __new__(cls, *args):
        # this is here to handle initializing the object from an already
        # existing instance of the class
        if len(args) and isinstance(args[0], cls):
            instance = args[0]
            instance._skip_init = True
            return instance
        else:
            return super().__new__(cls)


# ****************************************************************************************************
# NOTE: there can be a generalized construction layer here that should be
#  able to easily make containers of their constituent class.
#  automatically get attributes as list if subclass has those
#  attributes. ie vectorized operations across instances

# Create an abstraction layer that can split and merge multiple time
# series data sets

class FilenameHelper(list):
    def __init__(self, data=()):
        # type enforcement
        list.__init__(self, map(Path, data))

    def attr_getter(self, *attrs):
        """Fetch attributes from objects inside the container"""
        return list(map(operator.attrgetter(*attrs), self))

    @property
    def names(self):
        return self.attr_getter('name')


#

# class PhotCampaign:

# TODO: multiprocess generic methods
# TODO: make picklable !!!!!!!!!


class ObservationList(SelfAwareContainer, LoggingMixin):
    """
    Class for collecting / sorting / grouping / heterogeneous set of CCD
    observations
    """

    @classmethod
    def load(cls, filenames, **kws):
        """
        Load data from file(s).

        Parameters
        ----------
        filenames

        Returns
        -------

        """

        cls.logger.info('Loading data')

        # sanitize filenames:  input filenames may contain None - remove these
        filenames = filter(None, filenames)
        hdus = []
        for i, name in enumerate(filenames):
            hdu = _BaseHDU.readfrom(name, **kws)  # note can pass header!!!!
            hdus.append(hdu)

            # set the pretty printer as same object for all shocObs in the Run
            # obs.pprinter = cls.pprinter

        return cls(hdus)

    @classmethod
    def load_dir(cls, path, extensions=('fits',)):
        """
        Load all files with given extension(s) from a directory
        """
        path = Path(path)
        if not path.is_dir():
            raise IOError('%r is not a directory' % str(path))

        if isinstance(extensions, str):
            extensions = extensions,

        #
        iterators = (path.glob(f'*.{ext.lstrip(".")}') for ext in extensions)
        obj = cls.load(itt.chain(*iterators))

        if len(obj) == 0:
            # although we could load an empty run here, least surprise
            # dictates throwing an error
            raise IOError("Directory %s contains no valid '*.fits' files"
                          % str(path))

        return obj

    def __init__(self, hdus=None, groupId=None):

        if hdus is None:
            hdus = []

        self.files = FilenameHelper()
        for i, hdu in enumerate(hdus):
            if not isinstance(hdu, _BaseHDU):
                raise TypeError('%s items must derive from `_BaseHDU`. Item %i '
                                'is of type %r'
                                % (self.__class__.__name__, i, type(hdu)))
            self.files.append(Path(hdu._file.name))

        # put hdu objects in array to ease item getting.
        self.data = np.empty(len(hdus), dtype='O')
        self.data[:] = hdus
        self.groupId = OrderedSet(groupId)
        # self.label = label

    def __str__(self):
        return '%s of %i file%s: %s' \
               % (self.__class__.__name__,
                  len(self),
                  's' * bool(len(self) - 1),
                  '| '.join(self.files.names))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """
        Can be indexed numerically, or by corresponding filename / basename.
        """
        if isinstance(key, str):
            if key.endswith('.fits'):
                key = key.replace('.fits', '')
            key = self.files.names.index(key)
            return self.data[key]

        elif isinstance(key, slice):
            return self.__class__(self.data[key])
        else:
            return self.data[key]

    def attr_getter(self, *attrs):
        """fetch attributes from the inner class"""
        return list(map(operator.attrgetter(*attrs), self))

    # TODO: attrsetter
