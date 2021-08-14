"""
Utilities for working with observing campaigns that consist of multiple
observation files.
"""


# std libs
import re
import glob
import inspect
import fnmatch as fnm
import operator as op
import warnings as wrn
import itertools as itt
from pathlib import Path
from collections import UserList, abc

# third-party libs
import numpy as np
from astropy.utils import lazyproperty
from astropy.io.fits.hdu import PrimaryHDU
from astropy.io.fits.hdu.base import _BaseHDU
from pyxides.typing import ListOf
from pyxides.getitem import IndexerMixin
from pyxides.grouping import Groups, AttrGrouper
from pyxides.vectorize import Vectorized, AttrVectorizer
from pyxides.pprint import PrettyPrinter, PPrintContainer

# local libs
import docsplice as doc
from motley.table import AttrTable
from recipes.oo import SelfAware
from recipes.oo.null import NULL
from recipes import caches, io, bash
from recipes.logging import LoggingMixin
from recipes.string.brackets import braces

# relative libs
from .. import cachePaths, _hdu_hasher
from ..image.detect import SourceDetectionMixin
from ..image.sample import ImageSamplerMixin
from ..image.calibration import ImageCalibratorMixin
# from ..image.registration import ImageRegister, ImageRegisterDSS


# translation for special "[22:34]" type file globbing
REGEX_SPECIAL = re.compile(r'(.*?)\[(\d+)\.{2}(\d+)\](.*)')


# TODO: multiprocess generic methods
# TODO: Create an abstraction layer that can split and merge multiple time
#  series data sets

# TODO: # each item in this container should have  MultivariateTimeSeries
#  containing changes of observed brightness (etc ...) over time


# class ImageRegisterMixin:
#     'todo maybe'


# def _get_file_name(hdu):
#     # gets the item string for pprinting Campaign
#     return hdu.file.name
#

def is_property(v):
    return isinstance(v, property)


class FilenameHelper:
    """
    Helper class for working with filenames
    """

    def __init__(self, hdu):
        self._path = Path(hdu._file.name) if hdu._file else NULL

    def __str__(self):
        return str(self.path)

    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return str(self._path.name)

    @property
    def stem(self):
        return str(self._path.stem)


class FileList(UserList, Vectorized):  # ListOf(FilenameHelper)
    """
    Helper class for working with lists of filenames
    """
    def __new__(cls, campaign):
        obj = super().__new__(cls)
        # use all
        kls = campaign._allowed_types[0]._FilenameHelperClass
        for name, _ in inspect.getmembers(kls, is_property):
            # print('creating property', name)
            setattr(cls, f'{name}s', AttrVectorizer(name))

        return obj

    def __init__(self, campaign):
        super().__init__(campaign.attrs('file'))

    def common_root(self):
        parents = {p.parent for p in self.paths}
        root = parents.pop()
        if parents:  # more than one root
            return
        if root:
            return root


class HDUExtra(PrimaryHDU, ImageSamplerMixin, ImageCalibratorMixin,
               LoggingMixin, SourceDetectionMixin):
    """
    Some extra methods and properties that help PhotCampaign
    """

    #detect = SourceDetection('sigma_threshold')

    @caches.to_file(cachePaths.detection, typed={'self': _hdu_hasher})
    def detect(self,  stat='median', depth=5, snr=3, **kws):
        """
        Cached source detection for HDUs

        Parameters
        ----------
        stat : str, optional
            [description], by default 'median'
        depth : int, optional
            [description], by default 5
        snr : int, optional
            [description], by default 3

        Examples
        --------
        >>> 

        Returns
        -------
        [type]
            [description]
        """
        image = self.get_sample_image(stat, depth)
        return super().detect(image, snr=snr, **kws)

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

    # plotting
    def display(self, **kws):
        """Display the data"""

        if self.ndim == 2:
            from scrawl.imagine import ImageDisplay
            im = ImageDisplay(self.data, **kws)
            # `section` fails with 2d data

        elif self.ndim == 3:
            from scrawl.imagine import VideoDisplay
            # FIXME: this does not work since VideoDisplay tries to interpret
            #  `self.section` as an array
            im = VideoDisplay(self.section, **kws)

        else:
            raise TypeError('Data is not image or video.')

        im.figure.canvas.set_window_title(self.file.name)
        return im


# class PPrintHelper(AttrTable):
#     pass


class GlobIndexing(IndexerMixin):
    """
    Mixin that allows retrieving items from the campaign by indexing with
    filename(s) or glob expressions

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
            special = bool(braces.match(key, False, must_close=True))

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
                   CampaignType(_BaseHDU),
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
          the :meth:`coalign_sky` method

    """
    # init helpers

    # Pretty representations for __str__ and __repr__
    pretty = PrettyPrinter(max_lines=25,
                           max_items=100,
                           sep=' | ',
                           item_str=op.attrgetter('file.name'))

    # Initialize pprint helper
    tabulate = AttrTable(
        ['name', 'target', 'obstype', 'nframes', 'ishape', 'binning'])

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
        kws.setdefault('loader', _BaseHDU.readfrom)
        files = files_or_dir

        # resolve input from text file with list of file names
        if isinstance(files, str) and files.startswith('@'):
            files = io.read_lines(files.lstrip('@'))

        if isinstance(files, (str, Path)):
            if Path(files).is_file():
                files = [files]
            else:
                # files either special pattern or directory
                files = str(files)

                if all(map(files.__contains__, '{}')):
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

        obj = cls.load_files(files, **kws)
        if len(obj) == 0:
            # although we could load an empty run here, least surprise
            # dictates throwing an error
            raise ValueError(f'Could not resolve any valid files with '
                             f'extensions: {extensions} from input '
                             f'{files_or_dir!r}.')

        return obj

    @classmethod
    def load_files(cls, filenames, loader=_BaseHDU.readfrom, **kws):
        """
        Load data from (list of) filename(s).

        Parameters
        ----------
        filenames

        Returns
        -------

        """

        # cls.logger.info('Loading..')
        # kws.setdefault('memmap', True)

        from time import time
        from recipes import pprint as ppr

        # sanitize filenames:  input filenames may contain None - remove these
        filenames = filter(None, filenames)
        hdus = []
        said = False
        i = 0
        # note sort filenames here by alphanumeric order
        for i, name in enumerate(sorted(filenames), 1):
            if name is None:
                if not said:
                    cls.logger.info('Filtering filenames that are `None`.')
                    said = True
                continue

            # load the HDU
            cls.logger.debug('Loading %s: %s.', name, ppr.hms(time() % 86400))

            # catch all warnings
            with wrn.catch_warnings(record=True) as w:
                wrn.simplefilter('always')

                # load file
                hdu = loader(name, **kws)

                # handle warnings
                if w:
                    cls.logger.warning(
                        f'Loading file: {name!r} triggered the following '
                        f'warning{"s"*(len(w) > 1)}:\n' +
                        '\n'.join((str(warning.message) for warning in w)))
            hdus.append(hdu)

        cls.logger.info('Loaded %i file%s.', i, 's' * bool(i))
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
        return self.tabulate(self, attrs, **kws)

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

    def coalign(self, sample_stat='median', depth=10, plot=False, **find_kws):
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
        groups, indices = self.group_by('telescope',  # 'date', # 'camera',
                                        return_index=True)

        # start with the group having the most observations.  This will help
        # later when we need to align the different groups with each other
        keys, indices = zip(*indices.items())
        order = np.argsort(list(map(len, indices)))[::-1]

        # create data containers
        registers = np.empty(len(groups), 'O')

        # For each telescope, align images wrt each other
        for i in order:
            run = groups[keys[i]]
            registers[i] = run._coalign(
                sample_stat, depth, plot=plot, **find_kws)

        # match coordinates of registers against each other
        reg = registers[order[0]]
        for i in order[1:]:
            reg.fit(registers[i])
            reg.register()

        # refine alignment
        refine = 5
        for _ in range(refine):
            # likelihood ratio for gmm model before and and after refine
            _, lhr = reg.refine()
            if lhr < 1.01:
                break
            
            reg.recentre()

        return reg

    def _coalign(self, sample_stat='median', depth=10, primary=None,
                 plot=False, **find_kws):
        
        # check
        assert not self.varies_by('telescope')  # , 'camera')

        from ..image.registration import ImageRegister
        
        
        reg = ImageRegister(**find_kws)
        # If no reference image indicated by user-specified `primary`, choose
        # image with highest resolution if any, otherwise, just take the first.
        primary, *_ = np.argmin(self.attrs.pixel_scale, 0)
        first = self[primary or 0]
        # First fit detects sources and measures their CoM to establish a point
        # cloud for the coherent point drift model
        kws = dict(sample_stat=sample_stat, depth=depth, refine=False)
        reg(first, **kws)
        # Other images are fit concurrently
        rest = self[np.delete(np.arange(len(self)), primary)]
        reg(rest, **kws)
        
        # 
        if plot:
            reg.mosaic()

        # return reg, idx

        # make sure we have the best possible alignment amongst sample images.
        # register constellation of stars by fitting clusters to center-of-mass 
        # measurements. Refine the fit, by ...
        reg.register(plot=plot)
        reg.refine(plot=plot)
        reg.recentre(plot=plot)

        return reg

    @doc.splice(coalign, 'Parameters')
    def coalign_survey(self, survey=None, fov=None, fov_stretch=1.2,
                       sample_stat='median', depth=10, primary=0,
                       plot=False, **find_kws):
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
        primary : int, default 0
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

        from ..image.registration import ImageRegisterDSS
        
        survey = survey.lower()
        if survey != 'dss':
            raise NotImplementedError('Only support for DSS image lookup atm.')

        # coalign images with each other
        reg = self.coalign(sample_stat, depth, plot, **find_kws)

        # pick the DSS FoV to be slightly larger than the largest image
        if fov is None:
            fov = np.ceil(np.max(reg.fovs, 0)) * fov_stretch

        #
        dss = ImageRegisterDSS(self[primary].coords, fov, **find_kws)
        dss.fit_register(reg, refine=False)
        dss.register()
        
        # dss.recentre(plot=plot)
        # _, better = imr.refine(plot=plot)
        dss.build_wcs(self)
        return dss

    def coalign_dss(self, fov=None, fov_stretch=1.2,
                    sample_stat='median', depth=10, primary=0,
                    plot=False, **find_kws):
        return self.coalign_survey('dss', fov, fov_stretch, sample_stat, depth,
                                   primary, plot, **find_kws)

    def close(self):
        # close all files
        self.calls('_file.close')
    
    @property
    def phot(self):
        # The photometry interface
        from .core import PhotInterface
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
