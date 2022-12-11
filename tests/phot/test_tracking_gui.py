# std
import sys

# third-party
import numpy as np
from loguru import logger
from mpl_multitab import QtWidgets

# local
from pyxides.vectorize import repeat
from obstools.phot import SourceTracker
from shoc import shocCampaign
from shoc.pipeline.calibrate import calibrate


# import matplotlib.pyplot as plt
# from matplotlib import use

# use('qt5agg')

if __name__ != '__main__':
    exit()


logger.enable('obstools')
logger.enable('scrawl')

#
run = shocCampaign.load(
    '/media/Oceanus/work/Observing/data/sources/CVs/polars/1RXS_J035410.4-165244'
)

run.attrs.set(repeat(target='1RXS J035410.4-165244', obstype='object'))
missing_telescope_info = run[~np.array(run.attrs.telescope, bool)]
missing_telescope_info.attrs.set(repeat(telescope='1.9m'))

run.pprint()


gobj, mdark, mflat = calibrate(run, overwrite=False)


np.random.seed(1111)
hdu = run[1]
image = hdu.get_sample_image()


tracker = SourceTracker.from_image(image, detect=dict(dilate=4))
tracker.init_memory(hdu.nframes)


app = QtWidgets.QApplication(sys.argv)
# ui = MplMultiTab()
# fig = ui.add_tab()
# vid = TrackerVideo(tracker, hdu.calibrated, fig=fig)


# for i in range(20):
#     print(tracker(hdu.calibrated[i], i))

gui = tracker.gui(hdu)
gui.show()
sys.exit(app.exec_())
# plt.show()
