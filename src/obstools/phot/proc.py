
# std
import contextlib as ctx
import multiprocessing as mp

# third-party
import numpy as np
import more_itertools as mit
from tqdm import tqdm
from loguru import logger
from joblib import Parallel, delayed

# local
import motley
from recipes import api
from recipes.string import pluralize
from recipes.config import ConfigNode
from recipes.logging import LoggingMixin
from recipes.flow.contexts import ContextStack
from recipes.concurrent.joblib import initialized

# relative
from .logging import TqdmLogAdapter, TqdmStreamAdapter


# TODO: filter across frames for better shift determination ???
# TODO: wavelet sharpen / lucky imaging for better relative positions
# TODO
#  simulate how different centre measures performs for sources with decreasing snr
#  super resolution images
#  lucky imaging ?


# ---------------------------------------------------------------------------- #
CONFIG = ConfigNode.load_module(__file__)

# stylize progressbar
prg = CONFIG.progress
prg['bar_format'] = motley.stylize(prg.bar_format)
del prg


# ---------------------------------------------------------------------------- #
# Multiprocessing
sync_manager = mp.Manager()

# default lock - does nothing
memory_lock = ctx.nullcontext()


def set_lock(mem_lock, tqdm_lock):
    """
    Initialize each process with a global variable lock.
    """
    global memory_lock
    memory_lock = mem_lock
    tqdm.set_lock(tqdm_lock)


# ---------------------------------------------------------------------------- #

class FrameProcessor(LoggingMixin):

    def __init__(self):
        self.measurements = self.frame_info = None

    def init_memory(self, n, loc=None, overwrite=False):
        """
        Initialize shared memory synchronised access wrappers. Should only be
        run in the main process.

        Parameters
        ----------
        n:
            number of frames
        loc
        overwrite

        Returns
        -------

        """
        raise NotImplementedError

    def __call__(self, data, indices=None, mask=None):
        """
        Track the shift of the image frame from initial coordinates

        Parameters
        ----------
        image
        mask

        Returns
        -------

        """

        if self.measurements is None:
            raise FileNotFoundError('Initialize memory first by calling the '
                                    '`init_memory` method.')

        if data.ndim == 2:
            data = [data]

        return self.loop(data, indices, mask)

    def __str__(self):
        name = type(self).__name__
        if (m := self.measurements) is None:
            return f'{name}(0/0)'
        return f'{name}({self.measured.sum()} / {len(m)})'

    __repr__ = __str__

    @api.synonyms({'n_jobs': 'njobs'})
    def run(self, data, indices=None, njobs=-1, backend='multiprocessing',
            batch_size=None, progress_bar=True):
        """
        Start a worker pool of source trackers. The workload will be split into
        chunks of size ``

        Parameters
        ----------
        data : array-like
            Image stack.
        indices : Iterable, optional
            Indices of frames to compute, the default None, runs through all the
            data.
        njobs : int, optional
            Number of concurrent woorker processes to launch, by default -1

        progress_bar : bool, optional
            _description_, by default True

        Raises
        ------
        FileNotFoundError
            If memory has not been initialized prior to calling this method.
        """
        # preliminary checks
        if self.measurements is None:
            raise FileNotFoundError('Initialize memory first by calling the '
                                    '`init_memory` method.')

        if indices is None:
            indices, = np.where(~self.measured)

        if len(indices) == 0:
            self.logger.info('All frames have been measured. To force a rerun, '
                             'you may do >>> tracker.measurements[:] = np.nan')
            return

        if njobs in (-1, None):
            njobs = mp.cpu_count()
        

        # main compute
        self.main(data, indices, njobs, batch_size, progress_bar, backend)

    def main(self, data, indices, njobs, batch_size, progress_bar, backend):

        # setup compute context
        context = ContextStack()
        if njobs == 1:
            worker = self.loop
            context.add(ctx.nullcontext(list))
        else:
            worker = self._setup_compute(njobs, backend, context, progress_bar)

        logger.remove()
        logger.add(TqdmStreamAdapter(), colorize=True, enqueue=True)

        # execute
        with context as compute:
            compute(worker(data, *args) for args in
                    self.get_workload(indices, njobs, batch_size, progress_bar))

        # self.logger.debug('With {} backend, pickle serialization took: {:.3f}s.',
        #              backend, time.time() - t_start)

    def _setup_compute(self, njobs, backend, context, progress_bar):
        # locks for managing output contention
        tqdm.set_lock(mp.RLock())
        memory_lock = mp.Lock()

        # NOTE: object serialization is about x100-150 times faster with
        # "multiprocessing" backend. ~0.1s vs 10s for "loky".
        worker = delayed(self.loop)
        executor = Parallel(njobs, backend)
        context.add(initialized(executor, set_lock,
                                (memory_lock, tqdm.get_lock())))

        # Adapt logging for progressbar
        if progress_bar:
            # These catch the print statements
            context.add(TqdmLogAdapter())

        return worker

    def get_workload(self, indices, njobs, batch_size=None, progress_bar=True):
        
        n  = len(indices)
        njobs = int(njobs)
        
        if batch_size is None:
            batch_size == (n // njobs) + (n % njobs)
        
        # divide work
        batches = mit.chunked(indices, batch_size)
        n_batches = round(n / batch_size)

        #
        self.logger.opt(lazy=True).info(
            'Work split into {0[0]} batches of {0[1]} frames each, using {0[2]} {0[3]}.',
            lambda: (n_batches, batch_size, njobs,
                     pluralize('worker', plural='concurrent workers', n=njobs))
        )

        # workload iterable with progressbar if required
        return tqdm(batches,
                    initial=(self.measured.sum() // batch_size),
                    total=(n // batch_size), unit_scale=batch_size,
                    disable=not progress_bar, **CONFIG.progress)

    def loop(self, data, indices, *args, **kws):

        indices = np.atleast_1d(indices)

        # measure
        for i in indices:
            self.frame_info[i] = self.measure(data, i, *args, **kws)

    def measure(self, data, index, mask=None):
        self.logger.trace('Measuring frame {}.', index)
        image = np.ma.MaskedArray(data[index], mask)
        return self._measure(image, index)

    def _measure(self, image, index):
        raise NotImplementedError()

    @property
    def measured(self):
        # boolean array flags True if frame has any measurement(s)
        return ~np.isnan(self.measurements).all(
            tuple({*range(self.measurements.ndim)} - {0})
        )
