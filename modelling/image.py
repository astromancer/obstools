import ctypes
# from pprint import pprint
import warnings
from collections import defaultdict
from pathlib import Path
import itertools as itt

import numpy as np
from photutils.aperture import EllipticalAperture, EllipticalAnnulus
from recipes.dict import AttrDict
# from recipes.array import neighbours
from recipes.logging import LoggingMixin
from recipes.string import seq_repr_trunc

from .core import Model, CompoundModel
from .utils import make_shared_mem, int2tup
from ..phot.trackers import LabelUser, LabelGroupsMixin, GriddedSegments


# from IPython import embed

# TODO: class that fits all stars simultaneously with MCMC. compare for speed
# / accuracy etc...
# TODO: option to fit centres or use Centroids
# idea: detect stars that share windows and fit simultaneously


def analyze_image_fit(mdl, image, p0=None):
    # original
    fig, axes = plt.subplots(3, 1, )

    im = ImageDisplay(image, ax= axes[0])


    # fit
    t0 = time.time()
    r = mdl.fit(image, p0=p0, method='nelder-mead')
    δt = time.time() - t0
    if r is None:
        raise Exception('Fail')

    # model
    im = ImageDisplay(mdl(r))
    display(im.figure)

    # residual
    im = ImageDisplay(mdl.residuals(r, image))
    display(im.figure)

    print(mdl)
    print('Parameters (%i)' % len(r))
    print('Optimization took: %3.2f s' % δt)
    print(
        r)  # '\n'.join(map(numeric_repr, r))  # mdl.format_params(r, precision=3))
    print()
    print('chi2', mdl.redchi(r, mim))

    return r  # , figs


class ImageSegmentsModeller(CompoundModel, LabelGroupsMixin, LoggingMixin):
    """
    Model fitting and comparison on segmented image frame
    """

    # TODO: this is a heirarchical model.  make it so!

    # TODO: mask_policy.  filter / set error to inf
    # TODO: option to fit centres or use Centroids
    # self.bg = bg  # TODO: many models - combinations of psf + bg models

    # use record arrays for fit results (structured parameters)
    use_record = True

    # @classmethod
    # def from_image(cls, image, mask=None, snr=(10, 7, 5, 3),
    #                npixels=(7, 5, 3), edge_cutoff=None, deblend=(True, False),
    #                dilate=(4, 1), models=(), **kws):
    #     """Multi-threshold blob detection"""
    #
    #
    #     from obstools.phot.trackers import SegmentationHelper, LabelGroups, \
    #         Record, iter_repeat_last
    #
    #     cls.logger.info('Running detection loop')
    #
    #     # make iterables
    #     variters = tuple(map(iter_repeat_last, (snr, npixels, dilate, deblend)))
    #     vargen = zip(*variters)
    #
    #     # segmentation data
    #     data = np.zeros(image.shape, int)
    #     original_mask = mask
    #     if mask is None:
    #         mask = np.zeros(image.shape, bool)
    #
    #     # first round detection without background model
    #     residual = image
    #     results = None
    #     # keep track of group info + detection meta data
    #     groups = LabelGroups(bg=[0])
    #     groups.info = Record()
    #     groups._auto_name_fmt = groups.info._auto_name_fmt = 'stars%i'
    #     counter = itt.count(0)
    #     go = True
    #     j = 0
    #     while go:
    #         # keep track of iteration number
    #         count = next(counter)
    #         group_name = 'stars%i' % count
    #
    #         # detect
    #         _snr, _npix, _dil, _debl = next(vargen)
    #         sh = SegmentationHelper.detect(residual, mask, None, _snr, _npix,
    #                                        edge_cutoff, _debl, _dil)
    #
    #         # update mask, get new labels
    #         new_mask = sh.to_bool()
    #         new_data = new_mask & np.logical_not(mask)
    #
    #         # since we dilated the detection masks, we may now be overlapping
    #         # with previous detections. Remove overlapping pixels here
    #         if dilate:
    #             overlap = data.astype(bool) & new_mask
    #             # print('overlap', overlap.sum())
    #             sh.data[overlap] = 0
    #             new_data[overlap] = False
    #
    #         if not new_data.any():
    #             break
    #
    #         # aggregate
    #         new_labelled = sh.data[new_data]
    #         new_labels = np.unique(new_labelled)
    #         data[new_data] += new_labelled + j
    #         mask = mask | new_mask
    #         # group
    #         group = new_labels + j
    #         groups[group_name] = group
    #         groups.info[group_name] = \
    #             dict(snr=_snr, npixels=_npix, dilate=_dil, deblend=_debl)
    #         # update
    #         j += new_labels.max()
    #
    #         #
    #         # logger.info('detect_loop: round nr %i: %i new detections: %s',
    #         #       count, len(group), tuple(group))
    #         cls.logger.info(
    #                 'detect_loop: round nr %i: %i new detections: %s',
    #                 count, len(group), seq_repr_trunc(tuple(group)))
    #
    #         if count == 0:
    #             # initialise the background model
    #             model = cls(sh, kws, groups)
    #         else:
    #             # add segments to ignore
    #             cls.logger.info('adding segments')
    #             _, labels = model.segm.add_segments(sh, replace=True)
    #
    #             model.groups.append(labels)
    #
    #         # return model, results, data, mask, groups
    #
    #         # fit background
    #         mimage = np.ma.MaskedArray(image, original_mask)
    #         results, residual = model._fit_reduce(mimage)
    #
    #     # TODO: log what you found
    #     return model, results, data, mask, groups

    def __init__(self, segm, models, label_groups=None):
        """

        Parameters
        ----------
        segm: SegmentationHelper
        models: {array-like, dict}
            The sequence of models. If an array-like, assume the mapping from
            models to image sections is one to one. warn if model index has
            no corresponding label.
            If dict, and label_groups is None:
                ?
            if label_groups given, match labels
        label_groups:
            Create one to many mapping between model and segments. i.e. The
            same model will be used for multiple segments.
        """

        self.segm = GriddedSegments(segm)
        # fixme: no need to recompute slices. happening implicitly in line above
        # allow zero => we allow label 0 (bg) slice region
        self.segm.allow_zero = True

        if isinstance(models, Model):
            models = [models]

        # if isinstance(models, dict) and isinstance(label_groups, dict):

        # default group contains all segment labels
        # print('label_groups', label_groups)
        LabelGroupsMixin.__init__(self, label_groups)

        # at least one model, and equal number of models and segment groups
        # assert len(models) == len(self.groups)
        CompoundModel.__init__(self, models)

        self.dtype = self.get_dtype()
        # self.group_sizes = self.groups.sizes  # not dynamic

    def __reduce__(self):
        # helper method for unpickling.
        return self.__class__, \
               (self.segm, list(self.models), self.groups)

    def model_to_labels(self):
        """
        Show which model refers to which label(s)
        """
        return list(zip(self.models, self.groups.values()))
        # note: case with groups that are not sequential may not work

    def add_model(self, model, labels):
        """

        Parameters
        ----------
        model:

        labels: array-like
            image segments for which the model will be used

        Returns
        -------

        """
        self[model.name] = model
        self.groups[model.name] = labels
        # overwrite the full-model-hierarchy dtype to include additional model
        self.dtype = self.get_dtype()

    @property
    def ngroups(self):
        return len(self.groups)

    @property
    def nlabels(self):
        return self.segm.nlabels

    # @property
    # def dtype(self):
    #     if self._dtype in None:
    #         self._dtype = self.get_dtype()
    #     else:
    #         return self._dtype


    def get_dtype(self, models=None, groups=None, ):
        # build the structured np.dtype object for a particular model or
        # group of models. default is to use the full set of models an
        if models is None:
            models = self.models

        if groups is None:
            groups = self.groups

        dtype = []
        for i, mdl in enumerate(models):
            nlabels = len(groups[i])
            dt = self._adapt_dtype(mdl, nlabels)
            dtype.append(dt)
        return dtype

    def _adapt_dtype(self, model, shape):
        # adapt the dtype of a component model so that it can be used with
        # other dtypes in a structured dtype

        # make sure size in a tuple
        shape = int2tup(shape)
        dt = model.get_dtype()
        if len(dt) == 1:                            # simple model
            name, base, dof = dt[0]
            dof = int2tup(dof)
            # extend shape of dtype
            return model.name, base, shape + dof
        else:                                       # compound model
            # structured dtype - nest!
            return model.name, dt, shape

    def _get_output(self, shape=(), models=None, groups=None, fill=np.nan):
        # create the container `np.recarray` for the result of an
        # optimization run on `models` in `groups`
        if (models, groups) == (None, None):
            dtype = self.dtype
        else:
            dtype = self.get_dtype(models, groups)

        try:
            out = np.full(shape, fill, dtype)
        except Exception as err:
            print('Hi' * 100, dtype)
            # from IPython import embed
            # embed()
            raise

        if self.use_record:
            return np.rec.array(out)
        return out

    # def evaluate(self, model, labels, mask=False, extract=False):
    #     # segmented = self.segm.coslice(self.segm.grid,
    #     #                               labels=labels, mask=mask, extract=extract)
    #     # for i, grid in enumerate(segmented):
    #     for lbl in self.segm.resolve_labels(labels):
    #         grid = self.segm.subgrids[lbl]
    # g = model.adapt_grid(grid)

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
        # if std is None:
        #     std = np.ones_like(data)  # slow?
        #
        # if mask is not None:
        #     data = np.ma.MaskedArray(data, mask)

        # loop over models and fit
        results = self._get_output()
        for i, model in enumerate(self.models):
            labels = self.groups[i]
            self._fit_model_sequential(model, data, stddev, labels,
                                       results[model.name], **kws)
        return results

    def _fit_model_sequential(self, model, data, std, labels, results, **kws):
        """
        Fit this model for all segments
        """
        # this method is for production
        mask = kws.pop('mask', True)
        extract = kws.pop('extract', False)
        # labels = self.segm.groups[group]
        segmented = self.segm.coslice(data, std, labels=labels, mask_bg=mask,
                                      flatten=extract)
        for i, (sub, substd) in enumerate(segmented):
            grid = model.adapt_grid(self.segm.subgrids[labels[i]])
            # print(sub)
            # p0 = model.p0guess(sub, grid, substd)
            # print('p0', p0)
            r = model.fit(None, sub, grid, substd, **kws)

            if r is not None:
                print('results', r.shape, results[i].shape)
                results[i] = r

        return results

    # def fit_model_simul(self, model, data, std, labels, results, **kws):

    # def fit_model(self, *args, **kws):
    #     """
    #     call signature:
    #         fit_model(model, data, std=None, labels=None)
    #     or
    #         fit_model(data, std=None, labels=None) if only one model
    #
    #     Fit this model for all segments
    #     """
    #     # For convenience, allow(data, std) signature if there is only 1 model
    #     if isinstance(args[0], np.ndarray):
    #         if self.nmodels == 1:
    #             args = self.models[0], *args
    #         else:
    #             raise TypeError('Need model')
    #
    #     model, data, *rest = args
    #     std, *labels = rest if len(rest) else None, *()
    #     labels = labels[0] if len(labels) else None
    #     labels = self.resolve_labels(labels)
    #
    #     shape = 1 if len(labels) == 1 else ()
    #     dtype = self.get_dtype((model,), labels, squeeze=True)
    #     results = np.full(shape, np.nan, dtype)
    #     self._fit_model_sequential(model, data, std, labels, results, **kws)
    #
    #     return results[model.name].squeeze()

    def _fit_reduce(self, data, stddev=None, **kws):
        # loop over models and fit
        results = self._get_output()
        resi = data.copy()
        for i, model in enumerate(self.models):
            labels = self.groups[i]
            r = results[model.name]
            r, resi = self._fit_model_reduce(model, resi, stddev, labels, r,
                                             **kws)
        return results, resi

    def _fit_model_reduce(self, model, data, std, labels, results, **kws):
        # maximum_likelihood_residual # ml_res
        """
        Fit this model for all segments
        """
        resi = data
        extra = kws.pop('extrapolate', 0)  # todo better name?# grow_segments
        mask_bg = kws.pop('mask_bg', True)
        flatten = kws.pop('flatten', False)
        # labels = self.segm.groups[group]
        segmented = self.segm.coslice(resi, std, labels=labels, mask_bg=mask_bg,
                                      flatten=flatten, enum=True)
        i = 0
        if extra:
            slices = self.segm.slices.enlarge(labels, extra)
        else:
            slices = self.segm.get_slices(labels)

        for lbl, (sub, substd) in segmented:
            grid = model.adapt_grid(self.segm.subgrids[lbl])
            r = model.fit(sub, grid, substd, **kws)

            # resi[self.segm.slices[labels[i]]] -= model.residuals(r, sub, grid)

            # resi.append( model.residuals(r, sub, grid) )

            # print('III', resi[self.segm.slices[labels[i]]].shape)
            # print('JJ', rr.shape)

            if r is not None:
                try:
                    # print('results', model.name, r)
                    results[i] = r
                    slice = slices[i]
                    resi[slice] = model.residuals(r, data[slice], grid)
                    # foo.append(rr)
                    # resi[slice] = rr
                except Exception as err:
                    from IPython import embed
                    import traceback, textwrap
                    header = textwrap.dedent(
                        """\
                        Caught the following %s:
                        ------ Traceback ------
                        %s
                        -----------------------
                        Exception will be re-raised upon exiting this embedded interpreter.
                        """) % (err.__class__.__name__, traceback.format_exc())
                    embed(header=header)
                    raise

            else:
                'what to do with parameters here? Keep as nans? mask?'
            #     warnings.warn('Model fit for %r failed.' % model.name)
            i += 1

        return results, resi

    #
    # def _fit_segment(self, sub, sub_std, grid, models=None):
    #     """
    #     Fit various models for single segment
    #     """
    #     # this method is for convenience
    #     models = models or self.models
    #     gof = np.full((len(models), len(self.metrics)), np.nan)
    #     par, paru = [], []  # parameter values and associated uncertainty
    #     for m, r in enumerate(self._gen_fits(sub, sub_std, grid, models)):
    #         if r is None:
    #             continue
    #
    #         # aggregate results
    #         p, pu, gof[m] = r
    #         par.append(p)
    #         paru.append(pu)
    #
    #         # convert here??
    #         # if i:
    #         #     p, _, gof[m] = r
    #         #     model = models[m]
    #         # self.save_params(i, j, model, r, sub)
    #
    #     # choose best fitting model and save those fluxes
    #     # ix_bm, best_model = self.model_selection(i, j, gof, models)
    #
    #     # if i:
    #     #     # save best fit flux
    #     #     self.best.ix[i, j] = ix_bm
    #     #     self.best.flux[i, j] = self.data[best_model].flux[i, j]
    #     #     self.best.flux_std[i, j] = self.data[best_model].flux_std[i, j]
    #
    #     return par, paru, gof
    #
    # def _gen_fits(self, sub, sub_stddev, grid, models=None):
    #     # TODO: use this task producer more efficiently
    #     models = models or self.models
    #     for model in models:
    #         p0 = model.p0guess(sub, grid, sub_stddev)
    #         yield model.fit(p0, sub, grid, sub_stddev)
    #
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


class PSFModeller(ImageSegmentsModeller):

    # def

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


class BackgroundModeller(ImageSegmentsModeller):

    def __init__(self, segmentation, models, use_labels=(0,)):
        segm = segmentation.copy()
        segm.allow_zero = True
        ImageSegmentsModeller.__init__(self, segm, models, use_labels)

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
#         # self.params = make_shared_mem(locPar, shape, 'f', np.nan, clobber)
#         # # standard deviation on parameters
#         # self.params_std = make_shared_mem(locStd, shape, 'f', np.nan, clobber)
#
#         # self.
#         #
#         # if hasattr(model, 'integrate'):
#         #     locFlx = folder / 'flx'
#         #     locFlxStd = folder / 'flxStd'
#         #     shape = (n, nfit)
#         #     self.flux = make_shared_mem(locFlx, shape, np.nan)
#         #     self.flux_std = make_shared_mem(locFlxStd, shape, np.nan)
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
        self.data = make_shared_mem(loc, shape, self.dtype, np.nan, clobber)
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
        #     self.metricData = make_shared_mem(
        #             locMetric, shape, 'f', np.nan, clobber)
        #
        #     # shared Data containers for best fit flux
        #     # columns:
        #     locBest = loc / 'bestIx.dat'
        #     self.best.ix = make_shared_mem(
        #             locBest, (n, nfit), ctypes.c_int, -99, clobber)
        #
        #     # Data containers for best fit flux
        #     locFlx = loc / 'bestFlx.dat'
        #     # locFlxStd = loc / 'bestFlxStd.dat'
        #     self.best.flux = make_shared_mem(
        #             locFlx, (n, nfit, 2), 'f', np.nan, clobber)
        #     # self.best.flux_std = make_shared_mem(locFlxStd, (n, nfit), np.nan)
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
        self.coords = make_shared_mem(loc, (n, 2), 'f', np.nan, clobber)
        self.sigmaXY = make_shared_mem(loc, (n, 2), 'f', np.nan, clobber)
        self.theta = make_shared_mem(loc, (n,), 'f', np.nan, clobber)

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


class ImageModeller(ImageSegmentsModeller, ModellingResultsMixin):
    def __init__(self, segm, psf, bg, metrics=None, use_labels=None,
                 fit_positions=True, save_residual=True):
        ImageSegmentsModeller.__init__(self, segm, psf, bg, self._metrics,
                                       use_labels)
        ModellingResultsMixin.__init__(self, save_residual)


import time
from graphical.imagine import VideoDisplay


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

    def _scroll(self, event):
        pass  # scroll disabled

    def get_image_data(self, i):
        p = np.random.randn(self.model.dof)  # * scale
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
