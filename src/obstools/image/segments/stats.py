
# std
import types
import contextlib as ctx

# third-party
import numpy as np
from scipy import ndimage
from joblib import Parallel, delayed

# local
from recipes.io import load_memmap
from recipes.logging import LoggingMixin


class MaskedStatistic(LoggingMixin):
    # Descriptor class that enables statistical computation on masked
    # input data with segmented images
    _doc_template = \
        """
        %s pixel values in each segment ignoring any masked pixels.

        Parameters
        ----------
        image:  array-like, or masked array
            Image for which to calculate statistic.
        labels: array-like
            Labels to use in calculation.

        Returns
        -------
        float or 1d array or masked array.
        """

    def __init__(self, func):
        self.func = func
        self.__name__ = name = func.__name__
        self.__doc__ = self._doc_template % name.title().replace('_', ' ')

    def __get__(self, seg, kls=None):
        # sourcery skip: assign-if-exp, reintroduce-else

        if seg is None:  # called from class
            return self

        # Dynamically bind this class to the seg instance from whence the lookup
        # came. Essentially this binds the first argument `seg` in `__call__`
        # below.
        return types.MethodType(self, seg)

    def __call__(self, seg, image, labels=None, njobs=-1, *results):
        # handling of masked pixels for all statistical methods done here
        seg._check_input_data(image)
        labels = seg.resolve_labels(labels, allow_zero=True)

        # ensure return array or masked array and not list.
        return self.run(image, seg, labels, njobs, *results)

    def run(self, data, seg, labels, njobs=-1, *results, **kws):

        if (nd := data.ndim) < 2:
            raise ValueError(f'Cannot compute image statistic for {nd}D data. '
                             f'Data should be at least 2D.')

        result, mask = self._run(data, seg, labels, njobs, *results, **kws)

        if mask is None:
            return np.array(result)

        return np.ma.MaskedArray(result, mask)

    def _run(self, data, seg, labels, njobs=-1, *results, **kws):

        # result shape and dtype
        nlabels = len(labels)
        shape = (nlabels, *seg._result_dims.get(self.__name__, ()))
        dtype = 'i' if 'position' in self.__name__ else 'f'

        is2d = (data.ndim == 2)
        if not is2d:
            shape = ((n := len(data)), *shape)
            njobs = int((n == 1) or njobs)

        isma = np.ma.is_masked(data)

        # short circuit
        if nlabels == 0:
            return np.empty(shape), (np.empty(shape, bool) if isma else None)

        #
        self.logger.trace(
            'Computing {} with {} jobs on input data with shape {} using '
            'labels: {}. Output array{} shape: {}.',
            self.__name__, njobs, data.shape, labels, ' is masked of' * isma, shape
        )

        # get worker
        worker = self.worker_ma if isma else self.worker
        masked = None
        if results:
            results, *masked = results
        elif is2d or njobs == 1:
            results = np.empty(shape, dtype)
            masked = np.empty(shape, bool)
        else:
            # create memmap(s)
            results = load_memmap(shape=shape, dtype=dtype)
            if isma:
                masked = load_memmap(shape=shape, fill=False)

        self._runner(data, seg, labels, worker, njobs, results, masked)
        return results, (masked if isma else None)

    def _runner(self, data, seg, labels, worker, njobs, results, *masked):
        is2d = (data.ndim == 2)
        if is2d:
            # run single task
            worker(data, seg, labels, results, *masked)
        else:
            # run many tasks concurrently
            self._run_parallel(data, seg, labels, worker, njobs, results, *masked)

        return results, masked

    def _run_parallel(self, data, seg, labels, worker, njobs, results, *masked, **kws):

        #  3D data
        if njobs == 1:
            # sequential
            context = ctx.nullcontext(tuple)
            # to_close = ()
        else:
            # concurrent
            # faster serialization with `backend='multiprocessing'`
            context = Parallel(n_jobs=njobs,
                               **{'backend': 'multiprocessing', **kws})
            worker = delayed(worker)

        with context as compute:
            # compute = stack.enter_context(context)
            compute(worker(im, seg, labels, results, *masked, i)
                    for i, im in enumerate(data))

        if njobs > 1:
            self.logger.debug('{} Processes successfully shut down.', n_jobs)

        return results, masked

    def worker(self, image, seg, labels, output, _ignored_=None, index=...):
        output[index] = self.func(image, seg.data, labels)

    def worker_ma(self, image, seg, labels, output, output_mask, index=...):
        # ignore masked pixels
        seg_data = seg.data.copy()
        # original = seg_data[image.mask]
        seg_data[image.mask] = seg.max_label + 1
        # this label will not be used for statistic computation.
        # NOTE: intentionally not using 0 here since that may be one of the
        # labels for which we are computing the statistic.

        # compute
        output[index] = self.func(image, seg_data, labels)

        # get output mask
        # now we have to check which labels may be completely masked in
        # image data, so we can mask those in output.
        n_masked = ndimage.sum(~image.mask, seg_data, labels)
        # for functions that return array-like results per segment (eg.
        # center_of_mass), we have to up-cast the mask
        if seg._result_dims.get(self.__name__, ()):
            n_masked = n_masked[:, None]
        output_mask[index] = (n_masked == 0)


class MaskedStatsMixin:
    """
    This class gives inheritors access to methods for doing statistics on
    segmented images (from `scipy.ndimage.measurements`).
    """

    # Each supported method is wrapped in the `MaskedStatistic` class upon
    # construction.  `MaskedStatistic` is a descriptor and will dynamically
    # attach to inheritors of this class that invoke the method via attribute
    # lookup eg:
    # >>> obj.sum(image)
    #

    # define supported statistics
    _supported = ['sum',
                  'mean', 'median',
                  'minimum', 'minimum_position',
                  'maximum', 'maximum_position',
                  # 'extrema',
                  # return signature is different, not currently supported
                  'variance', 'standard_deviation',
                  'center_of_mass']

    _result_dims = {'center_of_mass':   (2,),
                    'minimum_position': (2,),
                    'maximum_position': (2,)}

    # define some convenient aliases for the ndimage functions
    _aliases = {'minimum': 'min',
                'maximum': 'max',
                'minimum_position': 'argmin',
                'maximum_position': 'argmax',
                'maximum_position': 'peak',
                'standard_deviation': 'std',
                'center_of_mass': 'com'}

    def __init_subclass__(cls, **kws):
        super().__init_subclass__(**kws)
        # add methods for statistics on masked image to inheritors
        for stat in cls._supported:
            method = MaskedStatistic(getattr(ndimage, stat))
            setattr(cls, stat, method)
            # also add aliases for convenience
            if alias := cls._aliases.get(stat):
                setattr(cls, alias, method)
