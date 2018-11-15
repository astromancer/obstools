import ctypes
# from pprint import pprint
import operator
import warnings
from collections import defaultdict, MutableMapping, OrderedDict
from pathlib import Path
import itertools as itt

import numpy as np
from IPython import embed
from photutils.aperture import EllipticalAperture, EllipticalAnnulus
from recipes.dict import AttrDict
from recipes.list import tally
# from recipes.array import neighbours
from recipes.logging import LoggingMixin
from recipes.string import seq_repr_trunc
from scipy.optimize import leastsq

from ..core import Model, UnconvergedOptimization #, CompoundModel
from ..parameters import Parameters
from ..utils import make_shared_mem, int2tup
from ...phot.tracking import LabelGroupsMixin  # LabelUser
from ...phot.segmentation import SegmentationGridHelper


# from IPython import embed

# TODO: class that fits all stars simultaneously with MCMC. compare for speed
# / accuracy etc...
# TODO: option to fit centres or use Centroids
# idea: detect stars that share windows and fit simultaneously


class ModelContainer(OrderedDict, LoggingMixin):

    def __init__(self, *args, **kws):
        super().__init__(*args, **kws)
        self._names = None

    @property
    def dofs(self):
        return self.attrgetter('dof')

    @property
    def dof(self):
        """Total number of free parameters considering all constituent models"""
        return sum(self.dofs)

    @property
    def names(self):
        """unique model names"""
        if self._names is None:
            self._names = self.unique_names()
            self.rename_models(self._names)
        return self._names

    def attrgetter(self, *attrs):
        getter = operator.attrgetter(*attrs)
        return list(map(getter, self.values()))

    def unique_names(self, default_name='model'):
        """
        Get mapping from labels to a set of unique model names.  Names are
        taken from component models where possible, substituting
        `default_name` for unnamed models. Numbers are underscore appended to
        ensure uniqueness of names. Useful for nested parameter construction.

        Returns
        -------
        names: dict
            model names keyed on segment labels
        """
        assert isinstance(default_name, str)
        #
        names = [getattr(m, 'name', None) or default_name
                 for m in self.values()]

        # check for duplicate names
        unames = set(names)
        if len(unames) != len(names):
            # models have duplicate names
            self.logger.info('Renaming %i models', len(unames))
            new_names = []
            for name, indices in tally(names).items():
                fmt = '%s_{:i}' % name
                new_names.extend(
                        map(fmt.format, range(len(indices))))
            names = new_names
        return names

    def rename_models(self, names):
        for model, name in zip(self.values(), names):
            model.name = name





from recipes.dict import Record


class SegmentedImageModel(Model, LabelGroupsMixin, LoggingMixin):
    """
    Model fitting and comparison on segmented image frame
    """

    # TODO: refactor to use list of Segments  ListOfSegments from
    # self.segm.segments / self.segmentation.segments

    # TODO: mask_policy.  filter / set error to inf

    # self.bg = bg  # TODO: many models - combinations of psf + bg models

    # use record arrays for fit results (structured parameters)
    use_record = False
    use_params = True

    def __init__(self, segm, models=(), label_groups=None):
        """

        Parameters
        ----------
        segm: SegmentationHelper
        models: {sequence, dict}
            The sequence of models.
            If a sequence, assume the mapping from models to image sections is
            one to one. warn if model index has no corresponding label in
            segmentation image. `models` can also be an empty sequence,
            in which case the object will initially not contain any models.
            The `add_model` method can be used to add models retro-actively
            before compute.
            If dict, keys correspond to labels in the segmentation image

            if label_groups given, match labels
        label_groups: dict, optional
            Create one to many mapping between model and segments. i.e. The
            same model will be used for multiple segments.
        """

        if isinstance(segm, SegmentationGridHelper):
            # detecting the class of the SegmentationImage allows some
            # optimization by avoiding unnecessary recompute of lazyproperties.
            # Also allows custom subclasses of SegmentationGridHelper to be
            # used
            self.segm = segm
        else:
            self.segm = SegmentationGridHelper(segm)

        #
        if isinstance(models, Model):
            models = [models]

        n_models = len(models)
        if not isinstance(models, MutableMapping):
            if n_models not in (0, self.segm.nlabels):
                raise ValueError("Mapping from segments to models is not "
                                 "1-to-1 ")
            models = dict(zip(self.segm.labels, models))
        else:
            self.segm.check_labels(models.keys())

        # probably check the labels in the groups

        # init container
        self.models = ModelContainer(models)

        # optional unique names for parameter construction
        self._names = None

        # optional named groups
        LabelGroupsMixin.__init__(self, label_groups)
        self.groups.info = Record()

    # def __reduce__(self):
    #     # helper method for unpickling.
    #     return self.__class__, (self.segm, list(self.models))

    # def __getattr__(self, key):
    #     names = self.models.names
    #     # FIXME RecursionError: maximum recursion depth exceeded
    #     if key in names:
    #         return self.names_to_models()[key]
    #     return super().__getattribute__(key)

    @property
    def dof(self):
        return self.models.dof

    @property
    def nlabels(self):
        return self.segm.nlabels

    @property
    def ngroups(self):
        return len(self.groups)

    @property
    def nmodels(self):
        """number of segments with models"""
        return len(self.models)

    def add_model(self, model, labels, group=None):
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
        labels = self.segm.resolve_labels(labels)
        # check if any models will be clobbered ?

        for lbl in labels:
            self.models[lbl] = model
            # one model may be used for many labels

        if group is not None:
            group_labels = self.groups.get(group, ())
            group_labels = np.hstack([group_labels, labels])
            self.groups[group] = group_labels

        self._names = None

    # @property
    # def dtype(self):
    #     if self._dtype in None:
    #         self._dtype = self.get_dtype()
    #     else:
    #         return self._dtype

    def models_to_labels(self):
        """Mapping from models to labels"""
        #  Inverse of `self.models`
        m2l = defaultdict(list)
        for lbl, mdl in self.models.items():
            m2l[mdl].append(lbl)
        return m2l

    def names_to_models(self):
        return dict(zip(self.models.names, self.models.values()))

    def get_dtype(self, labels=None, groups=None):
        # build the structured np.dtype object for a particular model or
        # group of models. default is to use the full set of models and groups

        dtypes = []

        # TODO: use groups to make nested dtype
        if groups is not None:
            if isinstance(groups, str):
                groups = groups,
            for g in groups:
                dtype = self.get_dtype(self.groups[g])
                self.append((g, dtype, ()))
        elif labels is None:
            labels = list(self.models.keys())

        labels = self.segm.resolve_labels(labels)
        models = [self.models[_] for _ in labels]

        result_keys = {}
        # check if there are models that map to many segments
        if len(set(models)) < len(models):
            raise NotImplementedError
            # these can be reshaped to a block by adapting dtype shape
            for mdl, lbls in self.models_to_labels():
                dtype = self._adapt_dtype(mdl, len(lbls))
                dtypes.append(dtype)
                result_keys.update({lbl: (self.names[lbl], i)
                                    for i, lbl in enumerate(lbls)})
                # WARNING: may require different iteration ???
        else:
            for lbl in labels:
                model = self.models[lbl]
                dt = self._adapt_dtype(model, ())
                dtypes.append(dt)
            return dtypes

    # def get_dtype(self, models=None, groups=None):  # todo get from segments
    #     # build the structured np.dtype object for a particular model or
    #     # group of models. default is to use the full set of models an
    #     if models is None:
    #         models = self.models
    #
    #     if groups is None:
    #         groups = self.groups
    #
    #     dtype = []
    #     for i, mdl in enumerate(models):
    #         g = groups.get(i, None)
    #         nlabels = len(g) if g else ()
    #         dt = self._adapt_dtype(mdl, nlabels)
    #         dtype.append(dt)
    #     return dtype
    #
    def _adapt_dtype(self, model, out_shape):
        # adapt the dtype of a component model so that it can be used with
        # other dtypes in a (possibly nested) structured dtype. `out_shape`
        # allows for results (optimized parameters) of models that are used in
        # more than one segment to be represented by a 2D array.

        # make sure size in a tuple
        out_shape = int2tup(out_shape)
        dt = model.get_dtype()
        if len(dt) == 1:  # simple 1 component model
            name, base, dof = dt[0]
            dof = int2tup(dof)
            # extend shape of dtype
            return model.name, base, out_shape + dof
        else:  # compound model
            # structured dtype - nest!
            return model.name, dt, out_shape

    def _results_container(self, labels=None, groups=None, fill=np.nan,
                           shape=()):
        # create the container `np.recarray` for the result of an
        # optimization run on `models` in `groups`

        # if (models, groups) == (None, None):
        #     dtype = self.dtype
        #     # FIXME: if the dof of the component models change after init,
        #     # this will be wrong!!!
        #
        # else:
        #     dtype = self.get_dtype(models, groups)

        # build model dtype
        dtype = self.get_dtype(labels, groups)

        # create array
        out = np.full(shape, fill, dtype)

        if self.use_record:
            return np.rec.array(out)

        if self.use_params:
            return Parameters(out)

        return out

    # --------------------------------------------------------------------------
    # detect stars over the background and update segmentation
    # def detect(self, image, mask=False, background=None, snr=3., npixels=7,
    #            edge_cutoff=None, deblend=False, dilate=0):
    #     """
    #     Image object detection that returns a SegmentationHelper instance
    #
    #     Parameters
    #     ----------
    #     image
    #     mask
    #     background
    #     snr
    #     npixels
    #     edge_cutoff
    #     deblend
    #     dilate
    #
    #     Returns
    #     -------
    #
    #     """
    #
    #     # segmentation based on sigma-clipping
    #     obj = detect(image, mask, background, snr, npixels, edge_cutoff,
    #                  deblend)
    #
    #     # Initialize
    #     new = cls(obj.data)
    #
    #     #
    #     if dilate:
    #         new.dilate(iterations=dilate)
    #     #
    #     return new

    # def detection_loop(self, image, mask=None, snr=(10, 7, 5, 3),
    #                    npixels=(7, 5, 3), edge_cutoff=None,
    #                    deblend=(True, False),
    #                    dilate=(4, 1)):
    #     """
    #     Multi-threshold blob detection on minimized residual background for
    #     modelled image
    #     """
    #
    #     from recipes.string import seq_repr_trunc  # todo move to pprint
    #     from obstools.phot.utils import iter_repeat_last
    #     from obstools.phot.segmentation import SegmentationHelper
    #
    #     #
    #     self.logger.info('Running detection loop')
    #
    #     # make iterables
    #     variters = tuple(map(iter_repeat_last, (snr, npixels, dilate, deblend)))
    #     vargen = zip(*variters)
    #
    #     # segmentation data
    #     data = np.zeros(image.shape, int)  # for new segments only!
    #     if mask is None:
    #         mask = np.zeros(image.shape, bool)
    #
    #     # first round detection on raw data
    #     residual = image
    #     results = None
    #     counter = itt.count(0)
    #     while True:
    #         # keep track of iteration number
    #         count = next(counter)
    #         group_name = 'stars%i' % count
    #
    #         # detect
    #         # -----------------------------------------------------------------
    #         _snr, _npix, _dil, _debl = next(vargen)
    #         # or self.segm.__class__.detect
    #         sh = SegmentationHelper.detect(residual, mask, None, _snr, _npix,
    #                                        edge_cutoff, _debl, _dil)
    #
    #         # update mask, get new labels
    #         new_mask = sh.to_bool()
    #         new_data = new_mask & np.logical_not(mask)
    #         mask = mask | new_mask
    #
    #         # since we dilated the detection masks, we may now be overlapping
    #         # with previous detections. Remove overlapping pixels here
    #         if dilate:
    #             overlap = data.astype(bool) & new_mask
    #             sh.data[overlap] = 0
    #             new_data[overlap] = False
    #
    #         # aggregate
    #         _, new_labels = self.segm.add_segments(sh)
    #         # add new segments
    #         data[new_data] = sh.data[new_data]
    #         # todo: method here ??
    #         # -----------------------------------------------------------------
    #
    #         # log what has been found
    #         self.logger.info(
    #                 'detect_loop: round %i: %i new detections: %s',
    #                 count, len(new_labels), seq_repr_trunc(tuple(new_labels)))
    #
    #         # break the loop if there are no new detections
    #         if not len(new_labels):
    #             break
    #
    #         # add group info, detection meta data
    #         self.groups[group_name] = new_labels
    #         self.groups.info[group_name] = \
    #             dict(snr=_snr, npixels=_npix, dilate=_dil, deblend=_debl)
    #
    #         if count == 1:
    #             mimage = np.ma.MaskedArray(image, mask)
    #             self.optimize_knots(mimage)
    #
    #         # fit background
    #         results = self.fit(image)
    #         residual = self.residuals(results, image)
    #         #
    #
    #         # TODO: start at previous optimum
    #
    #     # TODO: log summary of what you found
    #     return results, residual, data, mask

    # def evaluate(self, model, labels, mask=False, extract=False):
    #     # segmented = self.segm.coslice(self.segm.grid,
    #     #                               labels=labels, mask=mask, extract=extract)
    #     # for i, grid in enumerate(segmented):
    #     for lbl in self.segm.resolve_labels(labels):
    #         grid = self.segm.coord_grids[lbl]
    # g = model.adapt_grid(grid)

    # def fit(self, data, stddev=None, **kws):
    #     """
    #     Fit frame data by looping over segments and models
    #
    #     Parameters
    #     ----------
    #     data
    #     std
    #     mask
    #     models
    #     labels
    #
    #     Returns
    #     -------
    #
    #     """
    #
    #     # loop over models and fit
    #     results = self._results_container()
    #
    #     for i, model in enumerate(self.models):
    #         labels = self.groups[i]
    #         self._fit_model_sequential(model, data, stddev, labels,
    #                                    results[model.name], **kws)
    #
    #     return results

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

    def fit_sequential(self, data, stddev=None, labels=None, **kws):
        """
        Fit data in the segments with labels.
        """

        # full results container is returned with nans where not fit / not
        # converged

        mask = kws.pop('mask', True)
        flatten = kws.pop('flatten', False)
        full_output = kws.pop('full_output', False)
        p0 = kws.pop('p0', None)

        # resolve labels
        all_labels = list(self.models.keys())
        if labels is None:
            labels = all_labels
        else:
            labels = np.atleast_1d(labels).astype(int)
            # TODO: segm.resolve_labels()
            bad = np.setdiff1d(labels, all_labels)
            if len(bad):
                raise ValueError('Invalid labels: %s' % bad)

        # output
        results = self._results_container(None if full_output else labels)
        # labels

        # iterator for data segments
        subs = self.segm.coslice(data, stddev, labels=labels, masked_bg=mask,
                                 flatten=flatten)
        for label, (sub, std) in zip(labels, subs):
            model = self.models[label]

            # skip models with 0 free parameters
            if model.dof == 0:
                continue

            if p0 is not None:
                kws['p0'] = p0[model.name]

            # get coordinate grid
            grid = self.segm.coord_grids[label]
            # minimize
            # kws['jac'] = model.jacobian_wrss
            # kws['hess'] = model.hessian_wrss
            r = model.fit(sub, grid, std, **kws)

            if r is None:
                raise UnconvergedOptimization(
                        'Fit for segment %i (%r) did not converge'
                        % (label, self.models[label]))
            else:
                results[model.name] = r


        return results
        # if len(labels) > 1:
        #     return results
        # else:
        #     return results.flattened

    # def _fit_segment(self, data, label):

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

    # def _fit_model_simul(self, model, data, std, labels, results, **kws):

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

    # def _fit_model_reduce(self, model, data, std, labels, results, **kws):
    #     """
    #     Get the minimized residual for model on labels, given data (and
    #     uncertainties.
    #
    #     Fit this model for all segments
    #     """
    #     resi = data
    #     extra = kws.pop('grow_segments', 0)
    #     mask_bg = kws.pop('mask_bg', True)
    #     flatten = kws.pop('flatten', False)
    #     # labels = self.segm.groups[group]
    #     segmented = self.segm.coslice(resi, std, labels=labels, mask_bg=mask_bg,
    #                                   flatten=flatten, enum=True)
    #     i = 0
    #     if extra:
    #         slices = self.segm.slices.grow(labels, extra)
    #     else:
    #         slices = self.segm.get_slices(labels)
    #
    #     for lbl, (sub, substd) in segmented:
    #         grid = model.adapt_grid(self.segm.coord_grids[lbl])
    #         r = model.fit(sub, grid, substd, **kws)
    #
    #         # resi[self.segm.slices[labels[i]]] -= model.residuals(r, sub, grid)
    #
    #         # resi.append( model.residuals(r, sub, grid) )
    #
    #         # print('III', resi[self.segm.slices[labels[i]]].shape)
    #         # print('JJ', rr.shape)
    #
    #         if r is not None:
    #             try:
    #                 # print('results', model.name, r)
    #                 results[i] = r
    #                 slice = slices[i]
    #                 resi[slice] = model.residuals(r, data[slice], grid)
    #                 # foo.append(rr)
    #                 # resi[slice] = rr
    #             except Exception as err:
    #                 from IPython import embed
    #                 import traceback, textwrap
    #                 header = textwrap.dedent(
    #                         """\
    #                         Caught the following %s:
    #                         ------ Traceback ------
    #                         %s
    #                         -----------------------
    #                         Exception will be re-raised upon exiting this embedded interpreter.
    #                         """) % (
    #                              err.__class__.__name__, traceback.format_exc())
    #                 embed(header=header)
    #                 raise
    #
    #         else:
    #             'what to do with parameters here? Keep as nans? mask? bork!!?'
    #         #     warnings.warn('Model fit for %r failed.' % model.name)
    #         i += 1
    #
    #     return results, resi

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
        self.data = make_shared_mem(loc, shape, self.get_dtype(), np.nan,
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


class ImageModeller(SegmentedImageModel, ModellingResultsMixin):
    def __init__(self, segm, psf, bg, metrics=None, use_labels=None,
                 fit_positions=True, save_residual=True):
        SegmentedImageModel.__init__(self, segm, psf, bg, self._metrics,
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
