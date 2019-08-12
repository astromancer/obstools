
# third-party libs
import numpy as np

# local libs
from obstools.phot.segmentation import SegmentationHelper

def test_self_awareness():
    seg = SegmentationHelper(np.zeros(10, 10))
    seg2 = SegmentationHelper(seg)

