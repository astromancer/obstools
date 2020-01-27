# std libs
import itertools as itt
from pathlib import Path

# third-party libs
import numpy as np
from astropy.io.fits.hdu import PrimaryHDU
from astropy.io.fits.hdu.base import _BaseHDU

# local libs
from recipes.logging import LoggingMixin
from recipes.containers import (SelfAwareContainer, AttrGrouper,
                                AttrMapper, AttrTable, Grouped,
                                ReprContainer, ReprContainerMixin, TypeEnforcer,
                                ObjectArray1D
                                )
from obstools.image.sample import BootstrapResample

from sklearn.cluster import MeanShift
from obstools.image.registration import register_constellation


# TODO: multiprocess generic methods
# TODO: Create an abstraction layer that can split and merge multiple time
#  series data sets


class FilenameHelper(AttrMapper):
    def __init__(self, parent):
        self.parent = parent

    @property
    def names(self):
        return [Path(hdu._file.name).name for hdu in self.parent]

    @property
    def paths(self):
        return [Path(hdu._file.name) for hdu in self.parent]


# class ImageRegistrationMixin(object):
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

    # def get_instrumental_setup(self):
    #     raise NotImplementedError

    # plotting
    def display(self, **kws):
        """Display the data"""
        n_dim = len(self.shape)
        if n_dim == 2:
            from graphing.imagine import ImageDisplay
            im = ImageDisplay(self.data, **kws)

        elif n_dim == 3:
            from graphing.imagine import VideoDisplay
            im = VideoDisplay(self.section, **kws)

        else:
            raise TypeError('Data is not image or video.')

        im.figure.canvas.set_window_title(self.filepath.name)
        return im


class ImageSamplerHDU(_HDUExtra):
    _sampler = None

    @property  # todo: lazyproperty
    def sampler(self):
        # reading subset of data for performance
        self._sampler = BootstrapResample(self.section)
        return self._sampler

    def get_sample_image(self, stat='median', depth=5):
        # get sample image
        n = int(np.ceil(depth // self.timing.t_exp))

        self.logger.info(f'Computing {stat} of {n} images (exposure depth of '
                         f'{float(depth):.1f} seconds) for sample image from '
                         f'{self.filepath.name!r}')

        sampler = getattr(self.sampler, stat)
        return self.calibrated(sampler(n, n))


class HDUExtra(ImageSamplerHDU):
    pass


# import os
# from recipes import pprint

# TODO: # each item in this container is a MultivariateTimeSeries containing
#  changes of observed brightness (etc ...) over time


class PPrintHelper(AttrTable):
    pass


class PhotCampaign(ObjectArray1D, TypeEnforcer, SelfAwareContainer, AttrGrouper,
                   ReprContainerMixin, LoggingMixin):
    """
    A class containing multiple CCD observations.  Useful for photometric
    campaigns.

    Each item in this container is a `HDU` object containing contained FITS data

    Built in capabilities allow basic group, split and merge operations,
    pretty printing in table format.


    Class for collecting / sorting / grouping / heterogeneous set of CCD
    observations

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
        TypeEnforcer.__init__(self, _BaseHDU)

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

    def coalign(self, depth=10, sample_stat='median', reference_index=None,
                plot=False, **find_kws):
        """
        Perform image alignment of all images in this PhotCampaign by
        the method of point set registration.  This is essentially a search
        heuristic that finds the positional and rotational offset between
        partially or fully overlapping images.  The implementation of the
        image registration algorithm is handled inside the `ImageRegistration`
        class.

        See: https://en.wikipedia.org/wiki/Image_registration for the basics

        Parameters
        ----------
        reference_index: int
            index of observation to use as reference for aligning others.
            If `None`, the highest resolution image amongst the
            observations will be used.
        depth:
            Exposure depth (in seconds) for the sample image
        sample_stat

        find_kws

        Returns
        -------

        """

        from obstools.image.registration import ImageRegistration, \
            roto_translate_yx

        n_par = 3  # x, y, θ
        n = len(self)
        fovs = np.empty((n, 2))
        scales = np.empty((n, 2))
        angles = np.empty(n)
        images = []

        # get sample images etc
        for i, hdu in enumerate(self):
            image = hdu.get_sample_image(sample_stat, depth)
            images.append(image)
            fovs[i] = fov = hdu.get_fov()
            angles[i] = hdu.get_rotation()
            scales[i] = fov / image.shape

        # align on highest res image if not specified
        idx = reference_index
        if reference_index is None:
            idx = scales.argmin(0)[0]
        others = set(range(n)) - {idx}

        self.logger.info('Aligning %i images on image %i: %r', len(self), idx,
                         self[idx].filename)
        matcher = ImageRegistration(images[idx], fovs[idx], **find_kws)
        for i in others:
            # match image
            p, yx = matcher(images[i], fovs[i],
                            angles[i] - angles[idx],  # relative angles!
                            plot=plot)
        if plot:
            matcher.mosaic(coords=matcher.yxt)

        # return matcher, idx

        # make sure we have the best possible alignment amongst sample images.
        # register constellation of stars
        matcher.register_constellation(plot=True)
        # make the cluster centres the target constellation
        matcher.recentre(plot=True)

        return matcher, idx

    def coalign_dss(self, depth=10, sample_stat='median', reference_index=0,
                    fov_dss=None, plot=False, **find_kws):
        """
        Perform image alignment of all images in this ObservationList with
        Digital Sky Survey image centred on the same field.  In astro-speak,
        this is a first order wcs / astrometry estimation.

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

        dss
        params
        yx:
            coordinates of sources in each image
        indices:
        aligned_on:
              index of base image on which others are aligned
        """
        from obstools.image.registration import ImageRegistrationDSS, \
            roto_translate_yx

        # group observations by telescope / instrument
        groups, indices = self.group_by('telescope', 'instrument',
                                        return_index=True)

        # create data containers
        n = len(self)
        images = np.empty(n, 'O')
        params = np.empty((n, 3))
        fovs = np.empty((n, 2))
        coords = np.empty(n, 'O')
        ng = len(groups)
        aligned_on = np.empty(ng, int)
        matchers = np.empty(ng, 'O')

        # For each image group, align images wrt each other
        # ensure that `params`, `fovs` etc maintains the same order as `self`
        for i, (gid, run) in enumerate(groups.items()):
            idx = indices[gid]
            matchers[i], images[idx], fovs[idx], params[idx], coords[idx], ali \
                = run.coalign(depth, sample_stat, plot=plot, **find_kws)

            aligned_on[i] = idx[ali]

        # pick the DSS FoV to be slightly larger than the largest image
        if fov_dss is None:
            fov_dss = np.ceil(fovs.max(0)) * 1.2

        #
        dss = ImageRegistrationDSS(self[reference_index].coords, fov_dss,
                                   **find_kws)

        for i, gid in enumerate(groups.keys()):
            mo = matchers[i]
            theta = self[aligned_on[i]].get_rotation()
            p = dss.match_points(mo.yx, mo.fov, theta)
            params[indices[gid]] += p

        # # transform coords
        # for i, (yx, p) in enumerate(zip(coords, params)):
        #     coords[i] = roto_translate_yx(yx, p)

        # return dss, matchers, images, params, coords, indices, aligned_on

        # make sure we have the best possible alignment amongst sample images.
        # register constellation of stars
        pixel_size = matchers[0].pixel_size[0]
        clustering = MeanShift(bandwidth=4 * pixel_size, cluster_all=False)
        centres, σ_xy, xy_offsets, outliers, xy = register_constellation(
                clustering, coords, pixel_size, plot=False,
                pixel_size=pixel_size)

        # some of the offsets may be masked. ignore those
        good = ~xy_offsets.mask.any(1)
        params[good, 1::-1] -= xy_offsets[good]
        # xy -= xy_offsets[:, None]

        # xy[:, ::-1].swapaxes(1, 2)
        return dss, images, params, xy[:, ::-1], indices, aligned_on


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
