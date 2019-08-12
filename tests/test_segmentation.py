from obstools.phot.segmentation import SegmentationHelper
import numpy as np

def test_self_awareness():
    seg = SegmentationHelper(np.zeros(10, 10))
    seg2 = SegmentationHelper(seg)

