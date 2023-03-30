"""
Utilities for working with observing campaigns that consist of multiple
observation files.
"""


# std
import glob
import inspect
import fnmatch as fnm
import warnings as wrn
import itertools as itt
from pathlib import Path
from collections import UserList, abc

# third-party
import numpy as np
from loguru import logger
from astropy.utils import lazyproperty
from astropy.io.fits.hdu import PrimaryHDU
from astropy.io.fits.hdu.base import _BaseHDU

# local
import docsplice as doc
from motley.table.attrs import AttrTable
from pyxides.typing import ListOf
from pyxides.getitem import IndexingMixin
from pyxides.grouping import AttrGrouper, Groups
from pyxides.vectorize import AttrVector, Vectorized
from pyxides.pprint import PPrintContainer, PrettyPrinter
from recipes import bash, io, op
from recipes.dicts import groupby
from recipes.oo import Null, SelfAware
from recipes.logging import LoggingMixin
from recipes.string.brackets import braces
from recipes.string import pluralize, strings

# relative
from .io import _FilePicklable
from .image.noise import CCDNoiseModel
from .image.sample import ImageSamplerMixin
from .image.detect import SourceDetectionMixin
from .image.calibration import ImageCalibratorMixin


# TODO: multiprocess generic methods
# TODO: Create an abstraction layer that can split and merge multiple time
#        series data sets
# TODO: # each item in this container should have  MultivariateTimeSeries
#  containing changes of observed brightness (etc ...) over time

# ---------------------------------------------------------------------------- #
# defaults
SAMPLE_STAT = 'median'
DEPTH = 5

DETECT_REPORT_STYLE = {'title_style': ('B', '_'),
                       'extend': 2}


# ---------------------------------------------------------------------------- #
get_msg = op.attrgetter('message')


def is_property(v):
    return isinstance(v, property)

# ---------------------------------------------------------------------------- #


class NoFile(Null):
    pass


NOFILE = NoFile()


class FilenameHelper:
    """
    Helper class for working with filenames
    """

    def __init__(self, hdu):
        self._path = Path(hdu._file.name) if hdu._file else NOFILE

    def __str__(self):
        return str(self.path)

    __repr__ = __str__

    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return str(self._path.name)

    @property
    def stem(self):
        return str(self._path.stem)

    @property
    def basename(self):
        return str(self._path.stem).split('.', 1)[0]


class FileList(UserList, Vectorized):  # ListOf(FilenameHelper)
    """
    Helper class for working with lists of filenames
    """
    def __new__(cls, campaign):
        obj = super().__new__(cls)
        # vectorize all properties of `FilenameHelper`
        kls = campaign._allowed_types[0]._FilenameHelperClass
        for name, _ in inspect.getmembers(kls, is_property):
            # print('creating property', name)
            setattr(cls, f'{name}s', AttrVector(name))
        return obj

    def __init__(self, campaign):
        super().__init__(campaign.attrs('file'))

    def common_root(self):
        parents = {p.parent for p in self.paths}
        if (root := parents.pop()):  # single root
            return root

        # multiple roots: return None


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

    def detect(self, stat=SAMPLE_STAT, depth=DEPTH, interval=..., report=True, **kws):
        """
        Cached source detection for HDUs.

        Parameters
        ----------
        stat : str, optional
            Statistic to use, by default 'median'.
        depth : int, optional
            [description], by default 5
        snr : int, optional
            [description], by default 3

        Examples
        --------
        >>> 

        Returns
        -------
        seg
            SegmentedImage
        """
        # NOTE: `get_sample_image` and `detection` are both cached for performance
        image = self.get_sample_image(stat, depth, interval)

        if report is True:
            report = DETECT_REPORT_STYLE
        if report:
            report = {**report, 'title': self.file.name}

        return self.detection(image, **kws, report=report)

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

        im.figure.canvas.set_window_title(self.file.name)
        return im


# class PPrintHelper(AttrTable):
#     pass


class GlobIndexing(IndexingMixin):
    """
    Mixin that allows retrieving items from the campaign by indexing with
    filename(s) or glob expressions.

    Examples
    --------
    >>> run['SHA_20200729.0010']
    >>> run['*10.fits']
    >>> run['*1[0-5].fits']
    >>> run['*0[^1-7].*']    # same as     run['*0[!1-7].*']
    >>> run['*1?.fits']
    >>> run['*0{01,22}.*']
    >>> run['*0{0?,1?}.*']
    >>> run['*{0[1-9],1[0-9]}.*']
    >>> run['*1{12..21}.*']

    For the full set of globbing expression syntax see:
        https://linuxhint.com/bash_globbing_tutorial/
    """

    def _get_index(self, filename):
        files = self.files.names
        for trial in (filename, f'{filename}.fits'):
            if trial in files:
                return files.index(trial)

    def __getitem__(self, key):
        getitem = super().__getitem__
        original = key

        if isinstance(key, (str, Path)):
            # If key is a pattern, always return a sequence of items - this
            # means items will be wrapped in the container at the superclass
            key = str(key)

            is_glob = glob.has_magic(key)
            special = bool(braces.match(key, must_close=True))

            # handle filename
            if not (is_glob | special):
                # trial key and key with fits extension added to support
                # both 'SHA_20200729.0010' and 'SHA_20200729.0010.fits'
                # patterns for convenience
                return getitem(self._get_index(key))

            files = self.files.names
            # handle special numeric range specification here

            if special:
                key = list(itt.chain.from_iterable(
                    (fnm.filter(files, key)
                        for key in bash.brace_expand(key))))

            elif is_glob:
                key = list(fnm.filter(files, key))

        # all the cases handeled above should resolve to list of filenames
        if isinstance(key, (list, tuple, np.ndarray)):
            # if not all are strings, TypeError will happend at super
            if not (set(map(type, key)) - {str, np.str_}):
                # handle list / tuple of filenames
                key = list(map(self._get_index, key))
                # this will raise IndexError for invalid filenames

            if len(key) == 0:
                raise IndexError(f'Could not resolve {original!r} '
                                 'as filename(s) in the campaign')

        return super().__getitem__(key)

#


class CampaignType(SelfAware, ListOf):
    """metaclass to avoid conflicts"""


class PhotCampaign(PPrintContainer,
                   GlobIndexing,     # needs to be before `UserList` in mro
                   CampaignType(_BaseHDU),  # pylint: disable=abstract-method
                   AttrGrouper,
                   Vectorized,
                   LoggingMixin):
    """
    A class containing multiple CCD observations potentially from different
    instruments and telescopes. Provides an interface for basic operations on
    sets of image stacks obtained during photometric observing campaigns. Each
    item in this container is a `astropy.io.fits.hdu.HDU` object encapsulating
    the FITS data and header information of the observation.

    Built in capabilities include:
        * sort, select, filter, group, split and merge operations based on
          attributes of the contained HDUs or arbitrary functions via
          :meth:`sort_by`, :meth:`select_by`, :meth:`filter_by`,
          :meth:`group_by` and :meth:`join` methods
        * Removing of duplicates :meth:`filter_duplicates`
        * Vectorize attribute lookup and method calling on the contained
          objects courtesy of :class:`Vectorized` via :meth:`attrs` and
          :meth:`calls`
        * Pretty printing in table format via :meth:`pprint` method
        * Image registration. ie. Aligning sample images from
          the stacks with respect to each other via :meth:`coalign`
        * Basic Astrometry. Aligning sample images from the stacks with
          respect to some survey image (eg DSS, SkyMapper) and infering WCS via
          the :meth:`coalign_survey` method

    """
    # init helpers

    # Pretty representations for __str__ and __repr__
    pretty = PrettyPrinter(max_lines=25,
                           max_items=100,
                           sep=' | ',
                           item_str=op.attrgetter('file.name'))

    # Initialize pprint helper
    tabulate = AttrTable(
        ['name', 'target', 'obstype', 'nframes', 'ishape', 'binning']
    )

    #
    @classmethod
    def load(cls, files_or_dir, recurse=False, extensions=('fits', 'FITS'),
             **kws):
        """
        Load files into the campaign.

        Parameters
        ----------
        files : str or Path or Container or Iterable
            The filename(s) or directory to load. Can also be a glob pattern of
            filenames to load eg: '/path/SHA_20200715.000[1-5].fits' or any
            iterable that yields successive filenames
        extensions : tuple, optional
            The file extensions to consider if `files` is a directory, by
            default ('fits',).  All files ending on any of the extensions in
            this list will be included
        recurse : bool
            Whether to step down into sub-directories to find files

        Returns
        -------
        PhotCampaign
        """

        files = files_or_dir
        # original = files_or_dir

        # resolve input from text file with list of file names
        if isinstance(files, str) and files.startswith('@'):
            logger.debug('Loading from plain text list: {!r}.', files[1:])
            files = io.read_lines(files.lstrip('@'))

        if isinstance(files, (str, Path)):
            logger.debug("Loading: '{!s}'", files)  # files.relative_to(files.parent.parent))
            if Path(files).is_file():
                files = [files]
            else:
                # files either special pattern or directory
                files = str(files)

                if braces.match(files):
                    files = bash.brace_expand(files)
                else:
                    try:
                        files = io.iter_files(files, extensions, recurse)
                    except ValueError:
                        raise ValueError(
                            f'{files!r} could not be resolved as either a '
                            'single filename, a glob pattern, or a directory.'
                        ) from None

        if not isinstance(files, (abc.Container, abc.Iterable)):
            raise TypeError(f'Invalid input type {type(files)} for `files`.')

        loader = kws.pop('loader', None)
        allow_empty = kws.pop('allow_empty')
        obj = cls.load_files(files, loader, allow_empty=True, **kws)

        if len(obj) == 0 and not allow_empty:
            # although we could load an empty run here, least surprise
            # dictates throwing an error
            raise ValueError(
                f'Could not resolve any valid files with extensions: '
                f'{extensions} from input {type(files_or_dir).__name__}: '
                f'{str(files_or_dir)!r}.')

        return obj

    @classmethod
    def load_files(cls, filenames, loader=None, allow_empty=False, **kws):
        """
        Load data from (list of) filename(s).

        Parameters
        ----------
        filenames

        Returns
        -------

        """

        # sanitize filenames:  input filenames may contain None - remove these
        # note sort filenames here by alphanumeric order
        files = groupby(sorted(filenames), bool)
        if False in files:
            cls.logger.info('Filtering `None` from filename list.')

        filenames = files.get(True, ())
        if not (filenames or allow_empty):
            raise ValueError('No data found.')

        if loader is None:
            loaders = op.AttrVector('readfrom', default=None)(cls._allowed_types)
            loader = next(filter(None, loaders))

        i = 0
        hdus = []
        for i, name in enumerate(sorted(filenames), 1):
            # load the HDU
            cls.logger.debug('Loading {!r}.', str(name))

            # catch all warnings
            with wrn.catch_warnings(record=True) as warnings:
                wrn.simplefilter('always')

                # load file
                hdu = loader(name, **kws)
                hdus.append(hdu)

                # handle warnings
                if warnings:
                    cls.logger.warning(
                        'Loading file: {!r} triggered the following {}:\n{}',
                        name, pluralize('warning', warnings),
                        '\n'.join(strings(map(get_msg, warnings)))
                    )
        if i:
            cls.logger.success('Loaded {:d} {:s}.', i, pluralize('file', hdus))

        return cls(hdus)

    def __init__(self, hdus=None):
        """
        Initialize a PhotCampaign from a list of hdus

        Parameters
        ----------
        hdus : list of astropy.io.fits.hdu.PrimaryHDU, optional
            The list of HDU objects that make up the observational campaign.
            The default is None, which creates an empty campaign
        """
        if hdus is None:
            hdus = []

        # init container
        super().__init__(hdus)

    @property
    def files(self):
        return FileList(self)

    def pformat(self, attrs=None, **kws):
        return self.tabulate(attrs, **kws)

    def pprint(self, attrs=None, **kws):
        print(self.pformat(attrs, **kws))

    def join(self, other):
        if not other:
            return self

        if isinstance(other, self.new_groups().__class__):
            other = other.to_list()

        if not isinstance(other, self.__class__):
            raise TypeError(
                f'Cannot join {type(other)!r} with {self.__class__!r}.')

        return self.__class__(np.hstack((self.data, other.data)))

    def coalign(self, sample_stat=SAMPLE_STAT, depth=DEPTH, plot=False, **kws):
        """
        Perform image alignment internally for sample images from all stacks in
        this campaign by the method of point set registration.  This is
        essentially a search heuristic that finds the positional and rotational
        offset between partially or fully overlapping images.  The
        implementation of the image registration algorithm is handled inside the
        :class:`ImageRegister` class.

        See: https://en.wikipedia.org/wiki/Image_registration for the basics

        Parameters
        ----------
        depth : float
            Simulated exposure depth (in seconds) of sample images drawn from
            each of the image stacks in the run. This determined how many images
            from the stack will be used to create the sample image.
        sample_stat : str or callable, default='median'
            The statistic that will be used to compute the sample image from the
            stack of sample images drawn from the original stack.
        find_kws : dict
            Keywords for object detection algorithm.
        plot: bool
            Whether to plot diagnostic figures


        Returns
        -------

        """
        # group observations by telescope / instrument
        groups, indices = self.group_by('telescope', return_index=True)

        # start with the group having the most observations.  This will help
        # later when we need to align the different groups with each other
        keys, indices = zip(*indices.items())
        order = np.argsort(list(map(len, indices)))[::-1]

        # create data containers
        registers = np.empty(len(groups), 'O')

        # For each telescope, align images wrt each other
        for i in order:
            registers[i] = groups[keys[i]]._coalign(
                sample_stat, depth, plot=plot, **kws)

        # match coordinates of registers against each other
        reg = registers[order[0]]
        for i in order[1:]:
            reg.fit(registers[i])
            reg.register()

        # refine alignment
        # refine = 5
        for _ in range(5):
            # likelihood ratio for gmm model before and and after refine
            _, lhr = reg.refine()
            if lhr < 1.01:
                break

            reg.recentre()

        reg.order = np.hstack([indices[o] for o in order])
        # reg.data, _ = cosort(reg.order, reg.data)
        return reg

    def _coalign(self, sample_stat=SAMPLE_STAT, depth=DEPTH, primary=None,
                 plot=False, **kws):
        # check
        assert not self.varies_by('telescope')  # , 'camera')

        from .image.registration import ImageRegister

        # If no reference image indicated by user-specified `primary`, choose
        # image with highest resolution if any, otherwise, just take the first.
        # primary, *_ = np.argmin(self.attrs.pixel_scale, 0)

        # self.logger.debug('PRIMARY = {}', primary)
        reg = ImageRegister.from_hdus(self, sample_stat, depth, primary, **kws)
        reg.fit()

        # first = self[primary or 0]
        # First fit detects sources and measures their CoM to establish a point
        # cloud for the coherent point drift model
        # kws = dict(sample_stat=sample_stat, depth=depth, refine=False)
        # reg(first, **kws)
        # Other images are fit concurrently
        # order = [primary, *np.delete(np.arange(len(self)), primary)]
        # rest = self[order]
        # reg(rest, **kws)
        # reg.order = order

        #
        if plot:
            reg.mosaic()

        # return reg, idx

        # make sure we have the best possible alignment amongst sample images.
        # register constellation of stars by fitting clusters to center-of-mass
        # measurements. Refine the fit, by ...
        reg.register(plot=plot)

        # refine alignment
        # refine = 5
        for _ in range(5):
            # likelihood ratio for gmm model before and and after refine
            _, lhr = reg.refine()
            if lhr < 1.01:
                break

            reg.recentre()

        # reg.refine(plot=plot)
        # reg.recentre(plot=plot)
        return reg

    @doc.splice(coalign, 'Parameters')
    def coalign_survey(self, survey=None, fov=None, fov_stretch=1.2,
                       sample_stat=SAMPLE_STAT, depth=DEPTH, primary=None,
                       plot=False, **kws):
        """
        Align all the image stacks in this campaign with a survey image centred
        on the same field. In astronomical parlance, this is a first order wcs /
        astrometry estimation fitting only for 3 parameters per frame: 
            xy-offset : The offset in pixels of the source position wrt to the
                coordinates given in the header
            theta : The rotation (in radians) of the image wrt equatorial
                coordinates.

        Parameters
        ----------
        primary : int, default None
            The index of the image that will be used as the reference image for
            the alignment. If `None`, the highest resolution image amongst the
            observations will be used.
        fov : float or array-like of size 2 or None
            Field of view of survey image in arcminutes. If not given, the
            field size will be taken as the maximal extent of the aligned
            images multiplied by the scaling factor `fov_stretch`.
        fov_stretch : float
            Scaling factor for automatically choosing the field of view size of
            the survey image. This factor is multiplied by the maximal extent of
            the (partially overlapping) aligned images to get the field of view
            size of the survey image.


        Returns
        -------
        `ImageRegisterDSS` object

        """

        from .image.registration import ImageRegisterDSS

        survey = survey.lower()
        if survey != 'dss':
            raise NotImplementedError('Only support for DSS image lookup atm.')

        # coalign images with each other
        reg = self.coalign(sample_stat, depth, plot,
                           primary=primary, **kws)

        # pick the DSS FoV to be slightly larger than the largest image
        if fov is None:
            fov = np.ceil(np.max(reg.fovs, 0)) * fov_stretch

        #
        dss = ImageRegisterDSS(self[reg.primary].coords, fov, **kws)
        dss.fit_register(reg, refine=False)
        dss.register()
        dss.order = reg.order

        # dss.recentre(plot=plot)
        # _, better = imr.refine(plot=plot)
        dss.build_wcs(self)
        return dss

    def coalign_dss(self, fov=None, fov_stretch=1.2,
                    sample_stat=SAMPLE_STAT, depth=DEPTH, primary=None,
                    plot=False, **kws):
        return self.coalign_survey('dss', fov, fov_stretch, sample_stat, depth,
                                   primary, plot, **kws)

    def close(self):
        # close all files
        self.calls('_file.close')

    @property
    def phot(self):
        # The photometry interface
        from .phot.core import PhotInterface

        return PhotInterface(self)


class ObsGroups(Groups, LoggingMixin):
    """
    Emulates dict to hold multiple `Campaign` instances keyed by their common
    attribute values. The attribute names given in `group_id` are the ones by
    which the original Campaign is separated into unique segments (which are
    also `Campaign` instances).

    This class attempts to eliminate the tedium of doing computations on
    multiple files with identical observational setups by enabling flexible
    looping over many such groupings.
    """

    def __init__(self, factory=PhotCampaign, *args, **kw):
        super().__init__(factory, *args, **kw)
