import pickle

import numpy as np
from IPython import embed

from obstools.phot.trackers import SegmentationHelper
from obstools.modelling.core import Model
from obstools.modelling.image import ImageSegmentsModeller


#test = ImageSegmentsModeller(mdlr.segm, list(mdlr.models), mdlr.groups)
#test_clone = pickle.loads(pickle.dumps(test))

sy, sx = ishape = (48, 342)
z = np.zeros(ishape)
z[20:25, 10:20] = 1
sh = SegmentationHelper(z)

orders = orders_x, orders_y = (5, 1), (3, 1, 3)
breaks = (0, 10, sx), (0, 10, 38, sy)  # 3x3
smoothness = (False, False)

from slotmode.vignette import Vignette2DCross

vignette = Vignette2DCross(orders, breaks, smoothness)
vignette.set_grid(z)

ism = ImageSegmentsModeller(sh, (vignette, None), )


test = pickle.loads(pickle.dumps(ism, 3))

#
# m = Model()
# m.npar = 1


# class ImSegMdl(ImageSegmentsModeller):
#
#     # def __reduce__
#
#     def __getnewargs_ex__(self):
#         print('hello')
#         return
#
#     def __getnewargs__(self):
#         print('hello')
#         return
#
# test = ImSegMdl(sh, m)
#
# embed()
