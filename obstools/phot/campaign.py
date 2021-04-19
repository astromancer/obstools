# std libs
# from recipes.oo.meta import classmaker
import re
# from recipes.string import replace
import fnmatch as fnm
import inspect
from recipes.oo.null import Null
import functools as ftl
import itertools as itt
from pathlib import Path
from collections import UserList, abc
import glob
import operator as op

# third-party libs
import numpy as np
from astropy.io.fits.hdu import PrimaryHDU
from astropy.io.fits.hdu.base import _BaseHDU
from astropy.utils import lazyproperty

# local libs
from recipes.logging import LoggingMixin
from recipes.containers import (AttrGrouper, AttrMapper, Grouped,
                                PrettyPrinter, PPrintContainer, OfType,
                                ItemGetter, AttrProp, is_property)
from motley.table import AttrTable
from obstools.image.sample import BootstrapResample
from obstools.image.calibration import ImageCalibration, keep
from recipes import io
from recipes.oo import SelfAware
# from recipes.regex import glob_to_regex
from recipes.string import braces

# translation for special "[22:34]" type file globbing
REGEX_SPECIAL = re.compile(r'(.*?)\[(\d+)\.{2}(\d+)\](.*)')

# TODO: multiprocess generic methods
# TODO: Create an abstraction layer that can split and merge multiple time
#  series data sets

# TODO: # each item in this container should have  MultivariateTimeSeries
#  containing changes of observed brightness (etc ...) over time


# class ImageRegisterMixin(object):
#     'todo maybe'


# def _get_file_name(hdu):
#     # gets the item string for pprinting Campaign
#     return hdu.file.name
#


class FnHelp:
    # def __init_subclass(cls):

    def __init__(self, hdu):
        if hdu._file is None:
            self._path = Null()
        else:
            self._path = Path(hdu._file.name)  #

    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return str(self._path.name)

    @property
    def stem(self):
        return str(self._path.stem)


class FileHelper(UserList,  AttrMapper):  # OfType(FnHelp)
    def __new__(cls, campaign):
        obj = super().__new__(cls)
        # use all
        kls = campaign._allowed_types[0]._FnHelper
        for name, p in inspect.getmembers(kls, is_property):
            # print('creating property', name)
            setattr(cls, f'{name}s', AttrProp(name))

        return obj

    def __init__(self, campaign):
        super().__init__(campaign.attrs('file'))


class _HDUExtra(PrimaryHDU, LoggingMixin):
    """
    Some extra methods and properties that help PhotCampaign
    """

    _FnHelper = FnHelp

    @property
    def file(self):
        return self._FnHelper(self)

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
    def oriented(self):
        # manage on-the-fly image orientation
        from obstools.image.orient import ImageOrienter
        return ImageOrienter(self)

    @lazyproperty  # ImageCalibrationMixin ?
    def calibrated(self):
        # manage on-the-fly calibration for large files
        return ImageCalibration(self)

    def set_calibrators(self, bias=keep, flat=keep):
        """
        Set calibration images for this observation. Default it to keep
        previously set image if none are provided here.  To remove a
        previously set calibration image pass a value of `None` to this
        function, or simply delete the attribute `self.calibrated.bias` or
        `self.calibrated.flat`

        Parameters
        ----------
        bias
        flat

        Returns
        -------

        """
        self.calibrated.bias = bias
        self.calibrated.flat = flat

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


class ImageSamplerHDU(_HDUExtra):
    # _sampler = None

    @lazyproperty  # lazyproperty ??
    def sampler(self):
        """
        Use this property to get sample images from the stack

        >>> stack.sampler.median(10, 100)

        """
        #  allow higher dimensional data (multi channel etc), but not lower
        #  than 2d
        if self.ndim < 2:
            raise ValueError('Cannot create image sampler for data with '
                             f'{self.ndim} dimensions.')

        # ensure NE orientation
        data = self.oriented

        # make sure we pass 3d data to sampler. This is a hack so we can use
        # the sampler to get thumbnails from data that is a 2d image,
        # eg. master flats.  The 'sample' will just be the image itself.

        if self.ndim == 2:
            # insert axis in front
            data = self.data[None]

        return BootstrapResample(data)

    @ftl.lru_cache()
    def get_sample_image(self, stat='median', min_depth=5):
        """
        Get sample image to a certain minimum simulated exposure depth by 
        averaging data


        Parameters
        ----------
        stat : str, optional
            The statistic to use when computing the sample image, by default 
            'median'
        min_depth : int, optional
            Minimum simulated exposure depth, by default 5 seconds

        Returns
        -------
        [type]
            [description]
        """
        # FIXME: get this to work for SALTICAM

        n = int(np.ceil(min_depth // self.timing.exp)) or 1

        self.logger.info(f'Computing {stat} of {n} images (exposure depth of '
                         f'{float(min_depth):.1f} seconds) for sample image '
                         f'from {self.file.name!r}')

        sampler = getattr(self.sampler, stat)
        return self.calibrated(sampler(n, n))


class HDUExtra(ImageSamplerHDU):
    pass


# class PPrintHelper(AttrTable):
#     pass


class ItemGlobber(ItemGetter):
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
                        for key in io.bash_expansion(key))))

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


# metaclass to avoid conflicts
class CampaignType(SelfAware, OfType):
    pass


class PhotCampaign(PPrintContainer,
                   ItemGlobber,
                   UserList, CampaignType(_BaseHDU),
                   AttrGrouper,
                   LoggingMixin):
    """
    A class containing multiple CCD observations potentially from different
    instruments and telescopes. Provides an interface for basic operations on
    sets of image stacks obtained during photometric observing campaigns. Each
    item in this container is a `astropy.io.fits.hdu.HDU` object encapsulating
    the FITS data and header information.

    Built in capabilities include:
        * sort, select, filter, group, split and merge operations based on
          attributes of the contained HDUs or arbitrary functions via
          :meth:`sort_by`, :meth:`select_by`, :meth:`filter_by`,
          :meth:`group_by` and :meth:`join` methods
        * Removing of duplicates :meth:`filter_duplicates` 
        * Vectorized attribute lookup and method calling on the contained
          objects courtesy of :class:`AttrMapper` via :meth:`attrs` and
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
    table = AttrTable(
        ['name', 'target', 'obstype', 'nframes', 'ishape', 'binning'])

    #
    @classmethod
    def load(cls, files_or_dir, recurse=False, extensions=('fits', 'FITS'),
             **kws):
        """
        Load files into the campaign

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
                # is_special =
                files = str(files)

                if all(map(files.__contains__, '{}')):
                    files = io.bash.brace_expand(files)
                else:
                    try:
                        files = io.iter_files(files, extensions, recurse)
                    except ValueError:
                        raise ValueError(
                            f'{files!r} could not be resolved as either a '
                            'single filename, a glob pattern, or a directory'
                        ) from None

        if not isinstance(files, (abc.Container, abc.Iterable)):
            raise TypeError(f'Invalid input type {type(files)} for `files`')

        obj = cls.load_files(files, **kws)
        if len(obj) == 0:
            # although we could load an empty run here, least surprise
            # dictates throwing an error
            raise ValueError(f'Could not resolve any valid files with '
                             f'extensions: {extensions} from input '
                             f'{files_or_dir!r}')

        return obj

    @classmethod
    def load_files(cls, filenames, loader=_BaseHDU.readfrom, **kws):
        """
        Load data from file(s).

        Parameters
        ----------
        filenames

        Returns
        -------

        """

        # cls.logger.info('Loading..')
        # kws.setdefault('memmap', True)

        from time import time
        from recipes import pprint

        # sanitize filenames:  input filenames may contain None - remove these
        filenames = filter(None, filenames)
        hdus = []
        said = False
        i = 0
        # note sort filenames here by alphanumeric order
        for i, name in enumerate(sorted(filenames), 1):
            if name is None:
                if not said:
                    cls.logger.info('Filtering filenames that are `None`')
                    said = True
                continue

            # load the HDU
            cls.logger.debug('Loading %s: %s', name,
                             pprint.hms(time() % 86400))
            hdu = loader(name, **kws)
            hdus.append(hdu)

        cls.logger.info('Loaded %i files', i)
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
        return FileHelper(self)

    def pprint(self, attrs=None, **kws):
        print(self.table(self, attrs, **kws))

    def join(self, other):
        if isinstance(other, self.new_groups().__class__):
            other = other.to_list()

        if not isinstance(other, self.__class__):
            raise TypeError(
                f'Cannot join {type(other)!r} with {self.__class__!r}')

        return self.__class__(np.hstack((self.data, other.data)))

    def _coalign(self, depth=10, sample_stat='median', reference_index=None,
                 plot=False, **find_kws):

        # check
        assert not self.varies_by('telescope', 'instrument')

        from obstools.image.registration import ImageRegister

        # get sample images etc
        images = self.calls('get_sample_image', sample_stat, depth)
        fovs, angles = zip(*self.attrs('fov', 'pa'))
        #
        matcher = ImageRegister.from_images(images, fovs, **find_kws)

        if plot:
            matcher.mosaic(coords=matcher.xyt)

        # return matcher, idx

        # make sure we have the best possible alignment amongst sample images.
        # register constellation of stars
        matcher.register_constellation(plot=plot)
        # for i in range(3):
        matcher.refine(plot=plot)
        matcher.recentre(plot=plot)
        return matcher

    def coalign(self, depth=10, sample_stat='median', plot=False, **find_kws):
        """
        Perform image alignment of all stacks in this PhotCampaign by
        the method of point set registration.  This is essentially a search
        heuristic that finds the positional and rotational offset between
        partially or fully overlapping images.  The implementation of the
        image registration algorithm is handled inside the
        :class:`ImageRegister` class.

        See: https://en.wikipedia.org/wiki/Image_registration for the basics

        Parameters
        ----------
        depth: float
            Exposure depth (in seconds) for the sample image
        sample_stat: str
            statistic to use when retrieving sample images
        reference_index: int
            index of observation to use as reference for aligning others.
            If `None`, the highest resolution image amongst the
            observations will be used.
        plot: bool
            Whether to plot diagnostic figures

        find_kws

        Returns
        -------

        """
        # group observations by telescope / instrument
        groups, indices = self.group_by('telescope', 'instrument',
                                        return_index=True)

        # start with the group having the most observations.  This will help
        # later when we need to align the different groups with each other
        keys, indices = zip(*indices.items())
        seq = np.argsort(list(map(len, indices)))[::-1]

        # create data containers
        ng = len(groups)
        registers = np.empty(ng, 'O')

        # For each telescope, align images wrt each other
        for i in seq:
            run = groups[keys[i]]
            registers[i] = run._coalign(depth, sample_stat, plot=plot,
                                        **find_kws)

        # return registers, seq

        # match coordinates of registers against each other
        imr = registers[seq[0]]
        for i in seq[1:]:
            reg = registers[i]

            imr.match_reg(reg)
            imr.register_constellation()

        count = 0
        lhr = 10
        while (lhr > 1.01) and (count < 5):
            _, lhr = imr.refine()
            imr.recentre()
            count += 1

        #
        # imr.recentre(plot=plot)
        # lhr = 2
        # while lhr > 1.0005:
        # _, lhr = imr.refine(plot=plot)

        return imr

    # TODO: coalign_survey
    def coalign_dss(self, depth=10, sample_stat='median', reference_index=0,
                    fov=None, plot=False, **find_kws):
        """
        Perform image alignment of all images in this campaign with
        Digital Sky Survey image centred on the same field.  In astro-speak,
        this is a first order wcs / astrometry estimation fitting only for 3
        parfameters per frame: xy-offsets and rotation.

        Parameters
        ----------
        depth
        sample_stat
        reference_index: int
        fov: float or 2-tuple
            Field of view for DSS image
        find_kws

        Returns
        -------
        `ImageRegisterDSS` object

        """
        from obstools.image.registration import ImageRegisterDSS

        reg = self.coalign(depth, sample_stat, plot, **find_kws)

        # pick the DSS FoV to be slightly larger than the largest image
        if fov is None:
            fov = np.ceil(np.max(reg.fovs, 0)) * 1.1

        dss = ImageRegisterDSS(self[reference_index].coords, fov, **find_kws)
        dss.match_reg(reg)
        dss.register_constellation()
        # dss.recentre(plot=plot)
        # _, better = imr.refine(plot=plot)
        return dss

        # group observations by telescope / instrument
        # groups, indices = self.group_by('telescope', 'instrument',
        #                                 return_index=True)

        # start with the group having the most observations

        # create data containers
        # n = len(self)
        # images = np.empty(n, 'O')
        # params = np.empty((n, 3))
        # fovs = np.empty((n, 2))
        # coords = np.empty(n, 'O')
        # ng = len(groups)
        # aligned_on = np.empty(ng, int)
        # matchers = np.empty(ng, 'O')

        # # For each image group, align images wrt each other
        # # ensure that `params`, `fovs` etc maintains the same order as `self`
        # for i, (gid, run) in enumerate(groups.items()):
        #     idx = indices[gid]
        #     m = matchers[i] = run.coalign(depth, sample_stat, plot=plot,
        #                                **find_kws)

        #     aligned_on[i] = idx[m.idx]

        # try:

        #     #
        #     dss = ImageRegisterDSS(self[reference_index].coords, fov_dss,
        #                             **find_kws)

        #     for i, gid in enumerate(groups.keys()):
        #         mo = matchers[i]
        #         theta = self[aligned_on[i]].get_rotation()
        #         p = dss.match_points(mo.yx, mo.fov, theta)
        #         params[indices[gid]] += p
        # except:
        #     from IPython import embed
        #     embed()

        return dss


class ObsGroups(Grouped, LoggingMixin):
    """
    Emulates dict to hold multiple ObservationList instances keyed by their
    shared common attributes. The attribute names given in groupId are the
    ones by  which the run is separated into unique segments (which are also
    ObservationList instances). This class attempts to eliminate the tedium
    of computing calibration frames for different observational setups by
    enabling flexible looping over various such groupings.
    """

    def __init__(self, factory=PhotCampaign, *a, **kw):
        super().__init__(factory, *a, **kw)

    def to_list(self):
        out = self.factory()
        for item in self.values():
            if item is None:
                continue
            if isinstance(item, PrimaryHDU):
                out.append(item)
            elif isinstance(item, out.__class__):
                out.extend(item)
            else:
                raise TypeError(f'{item.__class__}')
        return out
