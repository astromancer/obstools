

# std libs
import operator
import itertools as itt
import collections as col
from pathlib import Path

# third-party libs
import numpy as np
from astropy.io.fits.hdu.base import _BaseHDU

# local libs
from recipes.dict import pformat
from recipes.set import OrderedSet
from recipes.logging import LoggingMixin





# from astropy.io.fits.hdu import HDUList


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

class ObsGroups(col.OrderedDict, LoggingMixin):
    """
    Emulates dict to hold multiple shocRun instances keyed by their shared
    common attributes. The attribute names given in groupId are the ones by
    which the run is separated into unique segments (which are also shocRun
    instances). This class attempts to eliminate the tedium of computing
    calibration frames for different  observational setups by enabling loops
    over various such groupings.
    """

    def __repr__(self):
        return pformat(self)

    def to_list(self):
        """Concatenate values to list"""
        hdus = []
        for hdus_ in filter(None, self.values()):
            hdus.extend(hdus_)

        # construct child
        return ObservationList(hdus)

    def group_by(self, *keys):
        # if self.groupId == keys:
        #     return self
        return self.to_list().group_by(*keys)

    def varies_by(self, key):
        """
        Check whether the attribute value mapped to by `key` varies across
        the set of observing runs

        Parameters
        ----------
        key

        Returns
        -------
        bool
        """
        attrValSet = {getattr(o, key) for o in filter(None, self.values())}
        return len(attrValSet) > 1


class ObservationList(SelfAwareContainer, LoggingMixin):
    """
    Class for collecting / sorting / grouping / heterogeneous set of CCD
    observations
    """

    _repr_filecount = True
    _repr_sep = '|'

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
        sep = self._repr_sep + ' '
        s = '%s: %s%s' % (self.__class__.__name__,
                          sep.join(self.files.names),
                          sep)
        if self._repr_filecount:
            s += '(%i file%s)' % (len(self), 's' * bool(len(self) - 1))
        return s

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

    # TODO: attr_setter

    def group_by(self, *keys, **kws):
        """
        Separate a run according to the attribute given in keys.
        keys can be a tuple of attributes (str), in which case it will
        separate into runs with a unique combination of these attributes.

        Parameters
        ----------
        keys
        kws

        optional keyword: return_index

        Returns
        -------
        att_dic: dict
            (val, run) pairs where val is a tuple of attribute values mapped
            to by `keys` and run is the shocRun containing observations which
            all share the same attribute values
        flag:
            1 if attrs different for any cube in the run, 0 all have the same
            attrs

        """
        attrs = self.attr_getter(*keys)
        keys = OrderedSet(keys)
        return_index = kws.get('return_index', False)
        if self.groupId == keys:  # is already separated by this key
            gr = ObsGroups(zip([attrs[0]], [self]))
            gr.groupId = keys
            # gr.name = self.name
            if return_index:
                return gr, {attrs[0]: list(range(len(self)))}
            return gr

        att_set = set(attrs)  # unique set of key attribute values
        att_dic = col.OrderedDict()
        idx_dic = col.OrderedDict()
        if len(att_set) == 1:
            # all input files have the same attribute (key) value(s)
            self.groupId = keys
            att_dic[attrs[0]] = self
            idx_dic[attrs[0]] = np.arange(len(self))
        else:  # key attributes are not equal across all shocObs
            for ats in sorted(att_set):
                # map unique attribute values to shocObs (indices) with those
                # attributes. list comp for-loop needed for tuple attrs
                l = np.array([attr == ats for attr in attrs])
                # this class object of images with equal attr value this key
                eq_run = self.__class__(self[l], keys)
                att_dic[ats] = eq_run  # put into dictionary
                idx_dic[ats], = np.where(l)

        gr = ObsGroups(att_dic)
        gr.groupId = keys
        # gr.name = self.name
        if return_index:
            return gr, idx_dic
        return gr

    def coalign(self, reference_index=None, sample_size=10,
                sample_interval=None, combine='median',
                return_index=False, return_coords=False, plot=False,
                **find_kws):
        """
        Perform image alignment of all images in this ObservationList by point
        set registration.  This is essentially a search heuristic that finds
        the positional and rotational offset between partially or
        fully overlapping images.  The implementation of the image registration
        algorithm is handled inside the `ImageRegistration` class.

        See: https://en.wikipedia.org/wiki/Image_registration

        Parameters
        ----------
        reference_index: int
            index of observation to use as reference for aligning others.
            If `None`, the highest resolution image amongst the
            observations
            will be used.
        sample_size: int
            Size of the sample to use in creating sample image
        sample_interval:
        combine
        return_index:

        find_kws

        Returns
        -------

        """

        # TODO: bork if no overlap ?
        from pySHOC.wcs import ImageRegistration

        n_par = 3  # x, y, θ
        n = len(self)
        params = np.zeros((n, n_par))
        fovs = np.empty((n, 2))
        scales = np.empty((n, 2))
        angles = np.empty(n)
        images = []

        inv = sample_interval
        self.logger.info('Extracting median images (%d%s) frames',
                         sample_size, '' if inv is None else ' ' + str(inv))
        for i, hdu in enumerate(self):
            image = hdu.get_sample_image(sample_size, inv, combine)
            images.append(image)
            fovs[i] = fov = hdu.get_fov()
            scales[i] = fov / image.shape
            angles[i] = hdu.get_rotation()

        # align on highest res image if not specified
        a = reference_index
        if reference_index is None:
            a = scales.argmin(0)[0]
        others = set(range(n)) - {a}

        self.logger.info('Aligning run of %i images on %r', len(self),
                         self[a]._file.name)

        matcher = ImageRegistration(images[a], fovs[a], **find_kws)
        for i in others:
            # print(i, angles[i], '!'*23)
            p, *yx = matcher.match_image(images[i], fovs[i], angles[i],
                                         return_coords, plot)
            params[i] = p

        out = [images, fovs, params]
        if return_coords:
            out.append(yx)
        if return_index:
            out.append(a)
        return out

    def coalign_dss(self, reference_index=0, sample_size=10,
                    sample_interval=None, combine='median', **find_kws):
        """
        Perform image alignment of all images in this ObservationList with
        Digital Sky Survey image centred on the same field.  In astro-speak,
        this is a first order wcs / astrometry estimation.

        Parameters
        ----------
        reference_index
        sample_size
        find_kws

        Returns
        -------

        """
        from pySHOC.wcs import ImageRegistrationDSS

        # TODO: bork if no overlap !!

        # group observations by telescope / instrument
        sr, idx_dic = self.group_by('telescope', return_index=True)

        # create data containers
        n = len(self)
        images = np.empty(n, 'O')
        params = np.empty((n, 3))
        fovs = np.empty((n, 2))
        coords = np.empty(n, 'O')
        aligned_on = np.empty(len(sr), int)

        # For each image group, align images wrt each other
        # ensure that `params`, `fovs` etc maintains the same order as `self`
        for i, (tel, run) in enumerate(sr.items()):
            idx = idx_dic[tel]
            images[idx], fovs[idx], params[idx], coords[idx], ali = \
                run.coalign(None, sample_size, sample_interval, combine,
                            return_coords=True, return_index=True, **find_kws)
            aligned_on[i] = idx[ali]

        # pick the DSS FoV to be slightly larger than the largest image
        fov_dss = np.ceil(fovs.max(0))
        dss = ImageRegistrationDSS(self[reference_index].coords, fov_dss,
                                   **find_kws)

        for i, tel in enumerate(sr.keys()):
            a = aligned_on[i]
            p = dss.match_image(images[a], fovs[a])
            params[idx_dic[tel]] += p

        return dss, images, fovs, params, coords, idx_dic
