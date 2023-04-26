
# std
import sys
import math
import tempfile
import functools as ftl
import itertools as itt
import contextlib as ctx
import multiprocessing as mp
from pathlib import Path

# third-party
import numpy as np
import more_itertools as mit
from tqdm import tqdm
from loguru import logger
from joblib import Parallel, delayed
from bottleneck import nanmean, nanstd
from astropy.utils import lazyproperty
from scipy.spatial.distance import cdist

# local
from recipes.io import load_memmap
from recipes.pprint import describe
from recipes.dicts import AttrReadItem
from recipes.logging import LoggingMixin
from recipes.parallel.joblib import initialized

# relative
from ...image.noise import CCDNoiseModel
from ...image.detect import make_border_mask
from ...image.segmentation.user import LabelUser
from ...image.segmentation.utils import merge_segmentations
from ...image.segmentation import (LabelGroupsMixin, SegmentedImage,                                  SegmentsModelHelper)
from ...image.registration import (ImageRegister, compute_centres_offsets,                                  report_measurements)
from ..config import CONFIG
from ..proc import ContextStack
from .display import SourceTrackerPlots


# ---------------------------------------------------------------------------- #
def check_image_drift(cube, nframes, mask=None, snr=5, npixels=10):
    """Estimate the maximal positional drift for sources"""

    #
    logger.info('Estimating maximal image drift for {:d} frames.', nframes)

    # take `nframes` frames evenly spaced across data set
    step = len(cube) // nframes
    maxImage = cube[::step].max(0)  #

    segImx = SegmentedImage.detect(maxImage, mask, snr=snr, npixels=npixels,
                                   dilate=3)

    mxshift = np.max([(xs.stop - xs.start, ys.stop - ys.start)
                      for (xs, ys) in segImx.slices], 0)
    return mxshift, maxImage, segImx

#