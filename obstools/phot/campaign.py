# std libs
import functools as ftl
import itertools as itt
from pathlib import Path

# third-party libs
import numpy as np
from astropy.io.fits.hdu import PrimaryHDU
from astropy.io.fits.hdu.base import _BaseHDU
from astropy.utils import lazyproperty

# local libs
from recipes.logging import LoggingMixin
from recipes.containers import (SelfAwareContainer, AttrGrouper,
                                AttrMapper, AttrTable, Grouped,
                                ReprContainer, ReprContainerMixin, OfType,
                                ObjectArray1D
                                )
from obstools.image.sample import BootstrapResample
from obstools.image.calibration import ImageCalibration, keep


# from sklearn.cluster import MeanShift
# from obstools.image.registration import register_constellation


# TODO: multiprocess generic methods
# TODO: Create an abstraction layer that can split and merge multiple time
#  series data sets

# TODO: # each item in this container should have  MultivariateTimeSeries
#  containing changes of observed brightness (etc ...) over time


class FilenameHelper(AttrMapper):
    def __init__(self, parent):
        self.parent = parent

    @property
    def names(self):
        return [Path(hdu._file.name).name for hdu in self.parent]

    @property
    def paths(self):
        return [Path(hdu._file.name) for hdu in self.parent]


# class ImageRegisterMixin(object):
#     'todo maybe'


class ReprCampaign(ReprContainer):
    def item_str(self, hdu):
        return hdu.filename


class _HDUExtra(PrimaryHDU, LoggingMixin):
    """
    Some extra methods and properties that help PhotCampaign
    """

    @property
    def filepath(self):  # fixme. won't work if self._file is None!!!
        return Path(self._file.name)

    @property
    def filename(self):
        if self._file is not None:
            return self.filepath.stem
        return 'None'

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
            from graphing.imagine import ImageDisplay
            im = ImageDisplay(self.data, **kws)
            # `section` fails with 2d data

        elif self.ndim == 3:
            from graphing.imagine import VideoDisplay
            im = VideoDisplay(self.section, **kws)

        else:
            raise TypeError('Data is not image or video.')

        im.figure.canvas.set_window_title(self.filepath.name)
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
                             f'{self.ndim} dimensiofcons.')

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
    def get_sample_image(self, stat='median', depth=5):
        # get sample image to a certain simulated exposure depth by averaging
        # data

        # FIXME: get this to work for SALTICAM

        n = int(np.ceil(depth // self.timing.t_exp))

        self.logger.info(f'Computing {stat} of {n} images (exposure depth of '
                         f'{float(depth):.1f} seconds) for sample image from '
                         f'{self.filepath.name!r}')

        sampler = getattr(self.sampler, stat)
        return self.calibrated(sampler(n, n))


class HDUExtra(ImageSamplerHDU):
    """"""


class PPrintHelper(AttrTable):
    pass


class PhotCampaign(ObjectArray1D, OfType(_BaseHDU),
                   SelfAwareContainer, AttrGrouper,
                   ReprContainerMixin, LoggingMixin):
    """
    A class containing multiple CCD observations potentially from different
    instruments and telescopes. Provides an interface for basic operations on
    sets of image stacks such as obtained during photometric observing
    campaigns. Each item in this container is a `astropy.io.fits.hdu.HDU` object
    encapsulating the FITS data.

    Built in capabilities include: 
        * sort, select, filter, group, split and merge operations based on attributes of
          the contained HDUs or arbitrary functions via :meth:`sort_by`,
          :meth:`select_by`, :meth:`filter_by`, :meth:`group_by` and :meth:`join` methods
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

    # Initialize pprint helper
    pprinter = PPrintHelper(
        ['name', 'target', 'obstype', 'nframes', 'ishape', 'binning'])

    @classmethod
    # @profile(report='bars')
    def load(cls, filenames, loader=_BaseHDU.readfrom, **kws):
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

        # sanitize filenames:  input filenames may contain None - remove these
        filenames = filter(None, filenames)
        hdus = []
        said = False
        i = 0
        for i, name in enumerate(filenames, 1):
            if name is None:
                if not said:
                    cls.logger.info('Filtering filenames that are `None`')
                    said = True
                continue

            # load the HDU
            # cls.logger.info('Loading %s', name)
            hdu = loader(name, **kws)
            hdus.append(hdu)

        cls.logger.info('Loaded %i files', i)
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

    def __init__(self, hdus=None):

        if hdus is None:
            hdus = []

        # make sure objects derive from _BaseHDU
        # TypeEnforcer.__init__(self, _BaseHDU)

        # init container
        ObjectArray1D.__init__(self, hdus)

        # init helpers
        self.files = FilenameHelper(self)
        self._repr = ReprCampaign(self, sep=' | ')  # TODO: merge with PPrint??

    def pprint(self, attrs=None, **kws):
        return self.pprinter(self, attrs, **kws)

    def join(self, other):
        assert other.__class__.__name__ == self.__class__.__name__
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
                    fov_dss=None, plot=False, **find_kws):
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
        fov_dss: float or 2-tuple
            Field of view for DSS image
        find_kws

        Returns
        -------
        `ImageRegisterDSS` object

        """
        from obstools.image.registration import ImageRegisterDSS

        reg = self.coalign(depth, sample_stat, plot, **find_kws)

        # pick the DSS FoV to be slightly larger than the largest image
        if fov_dss is None:
            fov_dss = np.ceil(np.max(reg.fovs, 0)) * 1.1

        dss = ImageRegisterDSS(self[reference_index].coords, fov_dss,
                                   **find_kws)

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


# class ObservationList(SelfAwareContainer, AttrGrouper, ReprContainerMixin,
#                       LoggingMixin):
#     """
#     Class for collecting / sorting / grouping / heterogeneous set of CCD
#     observations
#     """
#
#     @classmethod
#     def load(cls, filenames, **kws):
#         """
#         Load data from file(s).
#
#         Parameters
#         ----------
#         filenames
#
#         Returns
#         -------
#
#         """
#
#         cls.logger.info('Loading data')
#
#         # sanitize filenames:  input filenames may contain None - remove these
#         filenames = filter(None, filenames)
#         hdus = []
#         for i, name in enumerate(filenames):
#             hdu = _BaseHDU.readfrom(name, **kws)  # note can pass header!!!!
#             hdus.append(hdu)
#
#             # set the pretty printer as same object for all shocObs in the Run
#             # obs.pprinter = cls.pprinter
#
#         return cls(hdus)
#
#     @classmethod
#     def load_dir(cls, path, extensions=('fits',)):
#         """
#         Load all files with given extension(s) from a directory
#         """
#         path = Path(path)
#         if not path.is_dir():
#             raise IOError('%r is not a directory' % str(path))
#
#         if isinstance(extensions, str):
#             extensions = extensions,
#
#         #
#         iterators = (path.glob(f'*.{ext.lstrip(".")}') for ext in extensions)
#         obj = cls.load(itt.chain(*iterators))
#
#         if len(obj) == 0:
#             # although we could load an empty run here, least surprise
#             # dictates throwing an error
#             raise IOError("Directory %s contains no valid '*.fits' files"
#                           % str(path))
#
#         return obj
#
#     def __init__(self, hdus=None):
#
#         if hdus is None:
#             hdus = []
#
#         # init helpers
#         self.files = FilenameHelper()
#         self._repr = ReprContainer(self, sep='|', brackets=None)
#         #
#         for i, hdu in enumerate(hdus):
#             if not isinstance(hdu, _BaseHDU):
#                 raise TypeError('%s items must derive from `_BaseHDU`. Item %i '
#                                 'is of type %r'
#                                 % (self.__class__.__name__, i, type(hdu)))
#             self.files.append(Path(hdu._file.name))
#
#         # put hdu objects in array to ease item getting.
#         self.data = np.empty(len(hdus), dtype='O')
#         self.data[:] = hdus
#
#         # self.label = label
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, key):
#         """
#         Can be indexed numerically, or by corresponding filename / basename.
#         """
#         if isinstance(key, str):
#             if not key.endswith('.fits'):
#                 key += '.fits'
#             key = self.files.names.index(key)
#             return self.data[key]
#
#         elif isinstance(key, slice):
#             return self.__class__(self.data[key])
#         else:
#             return self.data[key]


class ObsGroups(Grouped, LoggingMixin):
    """
    Emulates dict to hold multiple ObservationList instances keyed by their
    shared common attributes. The attribute names given in groupId are the
    ones by  which the run is separated into unique segments (which are also
    ObservationList instances). This class attempts to eliminate the tedium
    of computing calibration frames for different observational setups by
    enabling flexible looping over various such groupings.
    """

    def __init__(self, default_factory=PhotCampaign, *a, **kw):
        super().__init__(default_factory, *a, **kw)
