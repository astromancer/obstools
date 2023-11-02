
# std
from pathlib import Path

# third-party
import numpy as np
from loguru import logger
from pyshoc import calDB, shocCampaign
from pyshoc.pipeline.calibrate import calibrate

# local
from pyxides.vectorize import repeat
from obstools.phot import SourceTracker


# logger.add(sys.stderr, level='INFO')
logger.enable('obstools')
logger.enable('recipes')
# logger.enable('scrawl')

# class MyFilter:

#     def __init__(self, level):
#         self.level = level

#     def __call__(self, record):
#         levelno = logger.level(self.level).no
#         return record["level"].no >= levelno

# my_filter = MyFilter("WARNING")


if __name__ != '__main__':
    exit()

# 1RXS_J035410.4-165244
loc = Path('/media/Oceanus/work/Observing/data/sources/CVs/polars/V834_Cen')
run = shocCampaign.load(loc)
run.attrs.set(repeat(target='V834 Cen', obstype='object'))
missing_telescope_info = run[~np.array(run.attrs.telescope, bool)]
missing_telescope_info.attrs.set(repeat(telescope='1.9m'))

run.pprint()


f = calDB.get(run, 'flat', False)

embed(header="Embedded interpreter at 'tests/phot/test_tracking.py':51")

raise SystemExit

gobj, mdark, mflat = calibrate(run, overwrite=False)


np.random.seed(1111)
hdu = run[1]

# Initialize object tracker
tracker, image = SourceTracker.from_hdu(hdu, dilate=4)
tracker.init_memory(hdu.nframes, loc / 'phot/tracking')
tracker.run(hdu.calibrated, n_jobs=1)


# # class TaskQueue(QueueLoader):
# #     def trigger(self):
# #         self.queue.put()
# #         super().trigger()

# # def iter_tasks(self):
# #     _compute_snr = self.snr_weighting or self.snr_cut
# #     for batch in mit.ichunked(range(len(hdu.calibrated)), self._update_centres_every):
# #         for indices in mit.chunked(batch, self._update_weights_every):
# #             yield Delayed(track_loop)(hdu.calibrated, indices, _compute_snr)
# #         yield Delayed(tracker.compute_centres_offsets)()


# queue = mp.Queue()
# trigger = mp.Event()
# loading_done = mp.Event()

# _compute_snr = self.snr_weighting or self.snr_cut
# indices = range(len(hdu.calibrated))
# track_loop_task = Delayed(track_loop)
# tasks = (track_loop_task(hdu.calibrated, indices, _compute_snr)

#          )

# # batch_size = self._update_centres_every // self._update_weights_every
# loader = TaskQueue(queue, trigger, loading_done, tasks, self._update_centres_every)
# loader.start()


# def consumer(queue):
#     print('CONSUMER', queue.get())

# loader.start()
# for _ in range(11):
#     print(f'TRIGGER {_}')
#     loader.trigger()
#     print(f'SLEEPING {_}')
#     time.sleep(1.5)


# loader.join()

# from IPython import embed
# embed(header="Embedded interpreter at 'tests/phot/test_tracking.py':79")

# logger.remove()
# logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
# for i in tqdm(range(hdu.nframes)):
#     logger.info('Frame #{}', i)

# tracker.run(hdu.calibrated)
# tracker.run(hdu.calibrated, range(100))

# clone = pickle.loads(pickle.dumps(tracker.seg))

# from IPython import embed
# embed(header="Embedded interpreter at 'tests/phot/test_tracking.py':68")

# def test(hdu):
#     #memloc = Path(tracker.xy_offsets.filename).parent
#     n = 10
#     tracker, reg = SourceTracker.from_hdu(hdu)

#     # xy lower left corner first frame wrt global seg
#     llc = reg.xy_offsets.min(0).round().astype(int)
#     self = tracker = SourceTracker(reg.xy, reg.global_seg(), llc=llc)
#     tracker.init_mem(n)

#     im = tracker.plot(hdu.calibrated[0])
#     im.ax.plot(*tracker.coords.T, 'rx')

#     #
#     seg = self.seg.select_region(-llc[::-1], image.shape)
#     lines = seg.draw_contours(im.ax, color='g')


#     image = hdu.oriented[0]
#     xy = self.measure_source_locations(image, None, self.llc)

#     im.ax.plot(*xy.T, 'gx')
