"""
A module for image segmentation and source detection
"""

from .user import LabelUser
from .neighbours import get_neighbour_index, get_neighbours
from .core import (GlobalSegmentation, SegmentedImage, SegmentsMasksHelper,
                   SegmentsModelHelper, get_masking_flags, image_sub, resolve_bg)
# from .masks import SegmentsMasksHelper
