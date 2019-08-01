# std libs
import time
import operator
import itertools as itt
from pathlib import Path
from collections import defaultdict, MutableMapping, OrderedDict

# third-party libs
import numpy as np
from photutils.aperture import EllipticalAperture, EllipticalAnnulus

# local libs
from recipes.list import tally
from recipes.logging import LoggingMixin
from recipes.dict import Record, AttrDict
from graphical.imagine import VideoDisplay

# relative libs
from ..parameters import Parameters
from ...phot.utils import LabelGroupsMixin
from ..utils import load_memmap, int2tup
from ..core import Model, CompoundModel, UnconvergedOptimization
from ...phot.segmentation import SegmentationGridHelper


# TODO: class that fits all stars simultaneously with MCMC. compare for speed
# / accuracy etc...
# TODO: option to fit centres or use Centroids
# idea: detect stars that share windows and fit simultaneously


#


class SegmentedImageModel(CompoundModel, LabelGroupsMixin, LoggingMixin):
    """
    Model fitting and comparison on segmented image frame
    """

    # TODO: refactor to use list of Segments  ListOfSegments from
    #  self.segm.segments / self.segmentation.segments
    #  to keep up with recent photutils dev...

    # TODO: mask_policy.  filter / set error to inf

    # FIXME: implement group stuff in different class HierarchicalImageModel ?

    # use record arrays for fit results (structured parameters)
    use_record = False
    use_params = True

    def __init__(self, seg, models=()):  # , label_groups=None
        """

        Parameters
        ----------
        seg: np.ndarray, SegmentationHelper
            The segmentation
        models: {sequence, dict}
            The sequence of models.
            If a sequence, assume the mapping from models to image sections is
            one to one. warn if model index has no corresponding label in
            segmentation image. `models` can also be an empty sequence,
            in which case the object will initially not contain any models.
            The `add_model` method can be used to add models retro-actively
            before evaluation.
            If dict, keys correspond to labels in the segmentation image
        """

        # init container
        CompoundModel.__init__(self)

        # init segments
        self.set_segments(seg)

        # add models
        self.set_models(models)

        # optional unique names for parameter construction
        # self._names = None

        # probably check the labels in the groups
        # # optional named groups
        LabelGroupsMixin.__init__(self)
        # self.groups.info = Record()

    # def evaluate(self, model, labels, mask=False, extract=False):
    #     # segmented = self.segm.coslice(self.segm.grid,
    #     #                               labels=labels, mask=mask, extract=extract)
    #     # for i, grid in enumerate(segmented):
    #     for lbl in self.segm.resolve_labels(labels):
    #         grid = self.segm.coord_grids[lbl]
    # g = model.adapt_grid(grid)

    @property
    def nlabels(self):
        return self.segm.nlabels

    def set_models(self, models):
        #
        if isinstance(models, Model):
            models = [models]

        n_models = len(models)
        if not isinstance(models, MutableMapping):
            if n_models not in (0, self.segm.nlabels):
                raise ValueError("Mapping from segments to models is not "
                                 "1-to-1")
            models = dict(zip(self.segm.labels, models))
        else:
            self.segm.check_labels(list(models.keys()))

        # init container
        CompoundModel.set_models(self, models)

    def set_segments(self, seg):
        if isinstance(seg, SegmentationGridHelper):
            # detecting the class of the SegmentationImage allows some
            # optimization by avoiding unnecessary recompute of lazyproperties.
            # Also allows custom subclasses of SegmentationGridHelper to be
            # used
            self.segm = seg
            # TODO: figure out if you can do this inside init of
            #  SegmentationHelper
        else:
            self.segm = SegmentationGridHelper(seg)

    def get_dtype(self, labels=all):
        # build the structured np.dtype object for a particular set of labels.
        # default is to use the full set of models

        dtypes = []
        for mdl, lbls in self.models.invert(labels).items():
            dtype = self._adapt_dtype(mdl, len(lbls))
            dtypes.append(dtype)

        return dtypes

    def fit(self, data, stddev=None, **kws):
        """
        Fit frame data by looping over segments and models

        Parameters
        ----------
        data
        std
        mask
        models
        labels

        Returns
        -------

        """
        return self.fit_sequential(data, stddev, **kws)

    def fit_sequential(self, data, stddev=None, labels=None,
                       reduce=False, **kws):
        """
        Fit data in the segments with labels.
        """

        full_output = kws.pop('full_output', False)
        # full results container is returned with nans where not fit / not
        # converged
        p0 = kws.pop('p0', None)

        # if full_output:
        #     labels = None

        if labels is None:
            labels = list(self.models.keys())

        # output
        results = self._results_container(None if full_output else labels)
        residuals = np.ma.getdata(data).copy() if reduce else None

        self.fit_worker(data, stddev, labels, p0, results, residuals, **kws)

        if reduce:
            return results, residuals

        return results

    def fit_worker(self, data, stddev, labels, p0, result, residuals, **kws):
        # iterator for data segments
        # subs = self.segm.coslice(data, stddev, labels=labels,
        #                          masked_bg=mask_bg, flatten=flatten)

        # get slices
        slices = self.segm.get_slices(labels)
        if data.ndim > 2:
            slices = list(map((...,).__add__, slices))

        #
        reduce = residuals is not None
        for label, slice_ in zip(labels, slices):
            model = self.models[label]

            # skip models with 0 free parameters
            if model.dof == 0:
                continue

            if p0 is not None:
                kws['p0'] = p0[model.name]

            # select data
            sub = np.ma.array(data[slice_])
            sub[..., self.segm.masks[label]] = np.ma.masked
            std = None if stddev is None else stddev[..., slice_]

            # get coordinate grid
            grid = self.segm.coord_grids[label]

            # if flatten:
            #     grid = grid.reshape(2, -1)

            # print(sub.shape, grid.shape)

            # minimize
            # kws['jac'] = model.jacobian_wrss
            # kws['hess'] = model.hessian_wrss

            #
            r = model.fit(sub, grid, std, **kws)

            if r is None:
                # TODO: optionally raise here based on cls.raise_on_failure
                #  can do this by catching above and raising from.
                #  raise_on_failure can also be function / Exception ??
                msg = (f'{self.models[label]!r} fit to segment {label} '
                       f'failed to converge.')
                med = np.ma.median(sub)
                if np.abs(med - 1) > 0.3:
                    msg += '\nMaybe try median rescale? data median is %f' % med
                raise UnconvergedOptimization(msg)
            else:
                # print(label, model.name, i)
                result[model.name] = r.squeeze()

                if reduce:
                    # resi = model.residuals(r, np.ma.getdata(sub), grid)
                    # print('reduce', residuals.shape, slice_, resi.shape)
                    residuals[slice_] = model.residuals(r, np.ma.getdata(sub),
                                                        grid)

        return r

    def minimized_residual(self, data, stddev=None, **kws):
        # loop over all models and all segments
        results = self.fit(data, stddev, **kws)

        # minimized_residual
        resi = self.residuals(results, data)
        return results, resi

    def _fit_model_sequential(self, model, data, std, labels, results, **kws):
        """
        Fit this model with data for the segments with labels
        """

        # skip models with 0 free parameters
        if model.dof == 0:
            return results

        # this method is for production
        mask = kws.pop('mask', True)
        flatten = kws.pop('flatten', False)

        # iterator for data segments
        subs = self.segm.coslice(data, std, labels=labels, masked_bg=mask,
                                 flatten=flatten)

        # indexer for results container
        if len(labels) == 1:
            # hack for models with single label to keep results array 1d
            rix = itt.repeat(slice(None))
        else:
            rix = itt.count()

        # loop over image segments for this model
        for i, lbl, (sub, substd) in zip(rix, labels, subs):
            grid = self.segm.coord_grids[lbl]
            r = model.fit(sub, grid, substd, **kws)

            if r is not None:
                # print(i, lbl, repr(model), 'results', r, r.shape, results[
                #     i].shape)
                results[i] = r

        return results  # squeeze for models that only have one label

    def animate(self, shape):
        """
        Animate the image model by stepping through the parameter values in
        plist

        Parameters
        ----------
        plist
        image

        Returns
        -------

        """

        return ImageModelAnimation(self, shape)

    # def model_selection(self, gof):
    #     """
    #     Select best model based on goodness of fit metric(s)
    #     """
    #     msg = None  # info message
    #
    #     # no convergence
    #     if np.isnan(gof).all():
    #         return -99, None, 'No model converged.'
    #
    #     # only one model
    #     if len(gof) == 1:
    #         return 0, self.models[0], msg
    #
    #     # the model indices at the minimal gof metric value
    #     best = np.nanargmin(gof, 0)
    #     ub, ib, cb = np.unique(best, return_index=True, return_counts=True)
    #
    #     # check cross metric consistency
    #     if len(ub) > 1:
    #         msg = 'GoF metrics inconsistent. Using %r.' % self.metrics[0]
    #         # NOTE: This probably means none of the models is an especially good fit
    #
    #     # choose model that most metrics agree is best
    #     ix_bf = ub[cb.argmax()]
    #     best_model = self.models[ix_bf]
    #
    #     return ix_bf, best_model, msg


class HierarchicalImageModel(LabelGroupsMixin):  # CompoundModel ??

    def __init__(self, segm, groups=None):

        """

        Parameters
        ----------
        segm
        groups
                    if label_groups given, match labels
        label_groups: dict, optional
            Create one to many mapping between model and segments. i.e. The
            same model will be used for multiple segments.
        """
        LabelGroupsMixin.__init__(groups)

        self.groups.info = Record()

        for grp, labels in groups.items():
            SegmentedImageModel(segm, models)

    @property
    def ngroups(self):
        return len(self.groups)

    def add_model(self, model, group, labels):
        """

        Parameters
        ----------
        model:

        labels: array-like
            image segments for which the model will be used
        group: str
            the group to which this model segment belongs

        Returns
        -------

        """

        group_labels = self.groups.get(group, ())
        group_labels = np.append(group_labels, labels)
        self.groups[group] = group_labels

        self._names = None

    def groups_to_labels(self, labels):
        _, groups = self.get_dtype(labels, all)
        return groups

    def get_dtype(self, groups=all, labels=None):
        # if return_labels:
        #     out = dtypes, out_lbl
        # else:
        #     out = dtypes
        #
        #     aggregate =
        # else:
        #     aggregate = dtypes.append

        # if groups is None and labels is None:
        #     groups = all

        if labels is None:
            if groups is all:
                groups = self.groups.keys()
        else:
            groups = None

        if groups is not None:
            # use groups to make nested dtype
            if isinstance(groups, str):
                groups = groups,

            for g in groups:
                # recurse
                dtype, lbls = self._get_dtype(self.groups[g])
                if len(dtype):
                    dtypes.append((g, dtype, ()))
                    out_lbl[g] = lbls

            return dtypes, out_lbl

        # group structure not included in hierarchy:    i.e dtype not nested
        for mdl, lbls in self.models_to_labels(labels).items():
            dtype = self._adapt_dtype(mdl, len(lbls))
            dtypes.append(dtype)
            out_lbl[mdl.name] = lbls

        return dtypes, out_lbl

    # def fit(self):
    # # resolve groups / labels
    # resolve groups / labels
    # group = kws.pop('group', None)
    # if group:
    #     groups = group,
    # else:
    #     groups = kws.pop('groups', None)

    # if groups is None:
    #     valid_labels = list(self.models.keys())
    #     if labels is None:
    #         labels = valid_labels
    #         # not using `self.segm.resolve_labels` here since there may
    #         # be labels that do not have a corresponding model
    #     else:
    #         labels = np.atleast_1d(labels).astype(int)
    #         bad = np.setdiff1d(labels, valid_labels)
    #         if len(bad):
    #             raise ValueError('Invalid labels: %s' % bad)
    #
    #     groups = {...: labels}
    # else:
    #     groups = {g: self.groups[g] for g in groups}

    # if groups is not None and len(groups) == 1 and labels is None:
    #     labels = self.groups.get(groups[0])
    #     # groups = {...: labels}
    # else:

    # dtype, groups = self.get_dtype(labels, groups)

    # # model_counter = defaultdict(int)
    # for grp, labels in groups.items():
    #     # print(grp, labels)
    #     if isinstance(labels, dict):  # nested
    #         # print('fit_groups')
    #         self.fit_groups(data, stddev, labels, p0,
    #                         results[grp], residuals, **kws)
    #     else:
    #         # print('fit_inner')
    #         self.fit_inner(data, stddev, labels, p0,
    #                        results, residuals, **kws)
    #
    #     if reduce:
    #         data = residuals

    #
    # if len(groups) == 1:
    #     # de-nest
    #     results = results[list(groups.keys())[0]]

    # def fit_groups(self, data, stddev, groups, p0, out, residuals, **kws):
    #
    #     for grp, labels in groups.items():
    #         # print(grp, labels)
    #         self.fit_inner(data, stddev, labels, p0, out, residuals, **kws)


class PSFModeller(SegmentedImageModel):

    # TODO: option to fit centres or use Centroids

    def emperical_psf(self, image, centres, labels=None):

        from scipy.stats import binned_statistic_2d

        if labels is None:
            labels = self.use_labels

        # NOTE: this is pretty crude

        # bgstd = np.empty(len(labels))
        # for i, m in enumerate(self.segm.to_annuli(2, 5, labels)):
        #     bgstd[i] = image[m].std()

        # imbg = tracker.background(image)
        # resi, p_bg = mdlr.background_subtract(image, imbg.mask)

        x, y, z = [], [], []
        v = []
        for i, (thumb, (sly, slx)) in enumerate(
                self.segm.iter_segments(image, labels, True, True)):
            g = self.grid[:, sly, slx] - centres[i, None, None].T
            x.extend(g[0].ravel())
            y.extend(g[1].ravel())
            v.extend((thumb / thumb.max()).ravel())

        rng = np.floor(min(np.min(x), np.min(y))), np.ceil(
                max(np.max(x), np.max(y)))
        stat, xe, ye, bn = binned_statistic_2d(x, y, v,
                                               bins=np.ptp(rng),
                                               range=(rng, rng))
        return stat


# class BackgroundModel(SegmentedImageModel):

# def

# def background_subtract(self, image, mask=None, p0=None, **kws):
#
#     # background subtraction
#     imbg = self.segm.mask_segments(image)
#
#     if mask is not None:
#         imbg.mask |= mask
#
#     # return imbg
#
#     # fit background
#     mdl = self.bg
#     if mdl:
#         # try:
#         results = mdl.fit(imbg, p0, **kws)
#         # background subtracted image
#         im_bg = mdl.residuals(results, image)
#         # except Exception as err:
#         #     self.logger.exception('BG')
#
#         return im_bg, results
#     else:
#         raise NotImplementedError


# class ModelData(LoggingMixin):
#     def __init__(self, model, n, nfit, folder, clobber=False):
#         """Shared data containers for model"""
#
#         # folder = Path(folder)
#         # if not folder.exists():
#         #     self.logger.info('Creating folder: %s', str(folder))
#         #     folder.mkdir(parents=True)
#         #
#         # self.loc = str(folder)
#
#         model._init_mem(self.loc, (n, nfit), clobber)
#
#         # npars = model.npar
#
#         # fitting parameters
#         # shape = (n, nfit, npars)
#         # locPar = '%s.par' % model.name
#         # locStd = '%s.std' % model.name
#         # self.params = load_memmap(locPar, shape, 'f', np.nan, clobber)
#         # # standard deviation on parameters
#         # self.params_std = load_memmap(locStd, shape, 'f', np.nan, clobber)
#
#         # self.
#         #
#         # if hasattr(model, 'integrate'):
#         #     locFlx = folder / 'flx'
#         #     locFlxStd = folder / 'flxStd'
#         #     shape = (n, nfit)
#         #     self.flux = load_memmap(locFlx, shape, np.nan)
#         #     self.flux_std = load_memmap(locFlxStd, shape, np.nan)
#         #     # NOTE: can also be computed post-facto
#
#     def save_params(self, i, j, p, pu):
#         self.params[i, j] = p
#         self.params_std[i, j] = pu


class ModellingResultsMixin(object):
    def __init__(self, save_residual=False):

        # Data containers
        # self.coords = None    # reference star coordinates across frames
        # self.data = {}
        # self.metricData = AttrDict()
        self.best = AttrDict()
        self.resData = defaultdict(list)
        self.saveRes = save_residual

        self.loc = None
        self.data = None

    def init_mem(self, shape, loc=None, clobber=False):

        if loc is None:
            import tempfile
            loc = Path(tempfile.mkdtemp())

        self.loc = str(loc)
        self.data = load_memmap(loc, shape, self.get_dtype(), np.nan,
                                clobber)
        return self.data

        # for k, model in enumerate(self.models):
        #     self.data[k] = model._init_mem(loc, (n, nfit), clobber=clobber)
        #     # if self.saveRes
        #         sizes = self.segm.box_sizes(self.use_labels)
        #         for j, (sy, sx) in enumerate(sizes.astype(int)):
        #             r = SyncedArray(shape=(sy, sx), fill_value=0)
        #             self.resData[model].append(r)

        # global bg model
        # if self.bg:
        #     self.data[-1] = self.bg._init_mem(loc, n, clobber=clobber)
        #
        # if self.nmodels:
        #     locMetric = loc / 'GoF.dat'
        #     shape = (n, nfit, self.nmodels, len(self.metrics))
        #     self.metricData = load_memmap(
        #             locMetric, shape, 'f', np.nan, clobber)
        #
        #     # shared Data containers for best fit flux
        #     # columns:
        #     locBest = loc / 'bestIx.dat'
        #     self.best.ix = load_memmap(
        #             locBest, (n, nfit), ctypes.c_int, -99, clobber)
        #
        #     # Data containers for best fit flux
        #     locFlx = loc / 'bestFlx.dat'
        #     # locFlxStd = loc / 'bestFlxStd.dat'
        #     self.best.flux = load_memmap(
        #             locFlx, (n, nfit, 2), 'f', np.nan, clobber)
        #     # self.best.flux_std = load_memmap(locFlxStd, (n, nfit), np.nan)
        #
        # return loc

    # def get_best_models(self, bestIx):
    #     # get the best fit models
    #     bestIx = np.asarray(bestIx)
    #     hasBest = (bestIx != -99)
    #     # init with None
    #     best_models = np.empty(len(bestIx), 'O')
    #     # fill best models
    #     best_models[hasBest] = self.models[list(bestIx[hasBest])]
    #     return best_models

    def save_best_flux(self, i, best):
        bestIx, bestModels, params, pstd = best

        self.best.ix[i] = bestIx  # per detected object
        for j, (m, p, pu) in enumerate(zip(bestModels, params, pstd)):
            if m is not None:
                self.best.flux[i, j] = m.flux(p), m.fluxStd(p, pu)
                # self.best.flux_std[i] =  m.fluxStd(p, pu)

    def save_params(self, i, results, best):

        p, pu, gof = results
        # loop over models # FIXME: eliminate this loop
        for k, m in enumerate(self.models):
            self.data[k].params[i] = p[k]  # (nstars, npars)
            self.data[k].params_std[i] = pu[k]

            # loop over stars
            # for j, r in enumerate(zip(p[k], pu[k], gof[k])):
            #     self._save_params(m, i, j, r)

        self.save_best_flux(i, best)

        # bestIx, params, pstd = best
        # best_models = mdlr.models[bestIx]  #
        # self.best.ix[i] = bestIx # per detected object
        # self.best.flux[i] =
        # self.best.flux_std[i] = pstd

    # def _save_params(self, model, i, j, results):  # sub=None, grid=None
    #     """save fitted / derived paramers for this model"""
    #     if i is None:
    #         return
    #
    #     p, pu, gof = results
    #
    #     # set shared memory
    #     psfData = self.data[model]
    #     psfData.params[i, j] = p
    #     psfData.params_std[i, j] = pu
    #
    #     print('saved', psfData.params)
    #
    #     Goodness of fit statistics
    #     if gof is not None:
    #         k = self.models.index(model)
    #         self.metricData[i, j] = gof.swapaxes(0, 1) # (nstars, nmodels, nmetrics)
    #         for m, metric in enumerate(self.metrics):
    #             self.metricData[i, j, k,] = gof[m]


class AperturesFromModel(LoggingMixin):
    """
    class that creates apertures based on modelling results
    """

    def __init__(self, r, rsky):
        self.r = np.asarray(r)
        self.rsky = np.asarray(rsky)

        self._pars = None

    # @property
    # def coo(self):
    #     return self._pars[:, :2]
    #
    # @property
    # def sigmaXY(self):
    #     return self._pars[:, 2:4]
    #
    # @property
    # def theta(self):
    #     return self._pars[:, -1]

    def init_mem(self, n, loc, clobber=False):
        self.coords = load_memmap(loc, (n, 2), 'f', np.nan, clobber)
        self.sigmaXY = load_memmap(loc, (n, 2), 'f', np.nan, clobber)
        self.theta = load_memmap(loc, (n,), 'f', np.nan, clobber)

    def combine_results(self, models, p, pstd, cfunc=np.nanmedian, axis=(), ):
        # nan_policy=None):
        """
        Get aperture parameters from best model for each star

        Parameters
        ----------
        p, pstd
        r_sigma
        cfunc
        axis

        Returns
        -------

        """
        coo = np.full((len(p), 2), np.nan)
        sxy = np.full((len(p), 2), np.nan)
        th = np.full((len(p), 1), np.nan)
        for i, (mdl, q) in enumerate(
                zip(models, p)):  # TODO: eliminate this loop
            if mdl is not None:
                coo[i] = q[1::-1]
                sxy[i] = mdl.get_sigma_xy(q)  # [None]
                th[i] = mdl.get_theta(q)  # [None]

        r = cfunc(sxy, axis)
        th = cfunc(th, axis)

        return coo, r, th

    def save(self, i, appars):

        coo, sxy, th = appars
        self.coords[i] = coo
        self.sigmaXY[i] = sxy
        self.theta[i] = th

    # def forced_apertures(self, coo, r, th):

    # def _fallback(self, a, fallback):
    #     """
    #     Used to substite aperture parameters for unconvergent models
    #     (i.e force photometry)
    #     """
    #
    #     nans = np.isnan(a).any(1)
    #     if nans.any() and (fallback is not None):
    #         sz = np.size(fallback)
    #         if isinstance(fallback, Callable):
    #             a[nans] = fallback(a)
    #         elif sz == 1:  # a number
    #             a[nans] = fallback
    #         elif sz > 1:  # presumably an array
    #             a[nans] = fallback[nans]
    #
    # for i, (a, f) in enumerate(zip(appars, fallbacks)):
    #     nans = np.isnan(a).any(1)
    #     a[nans] = f(a, nans)

    def create_apertures(self, appars, coords=None, sky=False, fallbacks=None):
        """
        Initialize apertures at com coordinates with radii determined from
        model dispersion (standard deviation (sigma) or fwhm) and the scaling
        attributes *r*, *rsky* (in units of sigma)

        Note: even though photutils supports multiple positions for apertures
        with the same radii, this is not preferred if we want to mask background
        stars for each star aperture. So we return a list of apertures
        """
        fcoords, sigma_xy, theta = appars
        rx, ry = sigma_xy * self.r

        if coords is None:
            coords = fcoords  # use fit coordinates for aperture positions

        if sky:
            sx, sy = sigma_xy
            rxsky = sx * self.rsky
            rysky = sy * self.rsky[1]
            return [EllipticalAnnulus(coo, *rxsky, rysky, theta)
                    for coo in coords[:, 1::-1]]

        else:
            return [EllipticalAperture(coo, rx, ry, theta)
                    for coo in coords[:, 1::-1]]

        # TODO: use_fit_coords
        # TODO: handle bright and faint seperately here
        # ixm = mdlr.segm.indices(mdlr.ignore_labels) # TODO: don't recompute every time

        # coo, r, th = appars

        # coords = np.empty_like(com)
        # rnew = np.ones_like(r)

        # self.modeller.use_labels
        # self.tracker.use_labels

        # coo, r, th = (self._fallback(a, f) for a, f in zip(appars, fallbacks))

    def check_aps_sky(self, i, rsky, rmax):
        rskyin, rskyout = rsky
        info = 'Frame {:d}, rin={:.1f}, rout={:.1f}'
        if np.isnan(rsky).any():
            self.logger.warning('Nans in sky apertures: ' + info, i, *rsky)
        if rskyin > rskyout:
            self.logger.warning('rskyin > rskyout: ' + info, i, *rsky)
        if rskyin > rmax:
            self.logger.warning('Large sky apertures: ' + info, i, *rsky)


class ImageModeller(SegmentedImageModel, ModellingResultsMixin):
    def __init__(self, segm, psf, bg, metrics=None, use_labels=None,
                 fit_positions=True, save_residual=True):
        SegmentedImageModel.__init__(self, segm, psf, bg, self._metrics,
                                     use_labels)
        ModellingResultsMixin.__init__(self, save_residual)


class ImageModelAnimation(VideoDisplay):
    """
    Visualise an image model by stepping through random parameter choices and
    display the resulting images as video.


    """

    def __init__(self, model, ishape, pscale=1, **kws):
        shape = (1,) + ishape
        super().__init__(np.zeros(shape), **kws)
        self.model = model
        self.pscale = float(pscale)
        self.rng = np.random.randn

    def _scroll(self, event):
        pass  # scroll disabled

    def get_image_data(self, i):
        p = self.rng(self.model.dof)  # * scale
        return self.model(p)

    def set_frame(self, i):
        self._frame = i  # store current frame

    def run(self, n=np.inf, pause=50):
        """
        Show a video of images in the model space

        Parameters
        ----------
        n: int
            number of frames in the animation
        pause: int
            interval between frames in miliseconds

        Returns
        -------

        """
        # n: number of iterations
        # pause: inter-frame pause (milisecond)
        seconds = pause / 1000
        i = 0
        while i <= n:
            self.update(i)
            i += 1
            time.sleep(seconds)
