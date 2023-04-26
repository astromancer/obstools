"""
Photometry.
"""

# std
import functools as ftl
import itertools as itt

# third-party
import numpy as np
from loguru import logger

# local
from motley.textbox import textbox
from recipes import op
from recipes.lists import split_like
from recipes.iter import split_slices

# relative
from .. import io
from ..lc.ascii import write
from ..campaign import PhotCampaign
from ..image.segments import display


def ragged(hdu, seg, top=5, dilate=0, filename=None):
    # ragged aperture photometry (no tracking!!)
    seg = seg.dilate(dilate, copy=True)

    logger.opt(lazy=True).info(
        'Source images and ragged aperture regions for photometry:\n{}',
        lambda: textbox(display.source_thumbnails_terminal(
            hdu.get_sample_image(), seg, top, title=hdu.file.name
        ))
    )
    #
    return seg.flux(hdu.calibrated, seg.labels[:top])


# @caches.to_file(cachePaths.phot, typed={'self': _hdu_hasher},
# filename = cachePaths.phot / '{hdu.file.name}-{fun.__name__}.phot')
def phot(hdu, fun, **kws):
    return fun(hdu, **kws)


class PhotInterface:
    """Interface for photometry algorithms"""

    filename_template = '{hdu.file.stem}.{fun.__name__}.txt'

    # def __get__(self, run, kls):
    #     if run is None:
    #         return self

    def __init__(self, run, reg=None, path='.'):
        assert isinstance(run, PhotCampaign)
        self.run = run
        self.reg = reg
        self.path = path

    def _runner(self, fun, day, segs, top=5, save=True, **kws):
        # run photometry for multiple hdus of the same target and aggregate a
        # light curve

        for hdu, seg in zip(day, segs):
            logger.info('Running {:s} aperture photometry on hdu: {:s}',
                        fun.__name__, hdu.file.name)

            flx, err = fun(hdu, seg, top, **kws)

            if save:
                # save to plain text
                folder = self.path / hdu.file.name.split('.', 1)[0]
                if not folder.exists():
                    folder.mkdir()
                filename = self.filename_template.format(hdu=hdu, fun=fun)
                write(folder / filename,
                      hdu.timing.bjd, flx.T, err.T,
                      obj_name=hdu.target)

            yield hdu.timing.bjd, flx, err

    def diff(self, fun, day, segs, top=5, save=True, **kws):
        date = day[0].date
        target = day[0].target
        basepath = self.path / f'{date:d}-{fun.__name__}'

        sizes = day.attrs.nframes
        n = sum(sizes)
        labels = self.reg.attrs('seg.labels')[1:]
        nstars = max(map(max, labels)) - 1
        top = min(top, nstars)

        logger.info('Light curves for {:d} sources will be extracted.', top)

        target_label = 1
        cmp = ftl.reduce(set.intersection, map(set, labels)) - {target_label}
        cmp = np.array(list(cmp)) - 1
        cmp = cmp[cmp < top]

        # logger.info('Light curves for {:d} sources will be extracted.', top)
        logger.info('Light curves for {:d} / {:d} detected sources with labels '
                    '{} will be extracted.', top, nstars, tuple(cmp))

        times = np.empty(n)
        # path = self.path / f'{basename}.dat'
        results = io.load_memmap(basepath.with_suffix('.npy'), (2, n, top))
        sections = split_slices(itt.accumulate(sizes))
        datagen = self._runner(fun, day, segs, top, save, **kws)
        for section, seg, (t, flx, std) in zip(sections, segs, datagen):
            times[section] = t

            # rescale
            idx = np.argsort(seg.labels[:top])

            results[:, section, idx] = \
                (flx, std) / np.ma.median(flx[:, cmp], 1, keepdims=True)

        if save:
            # save to plain text
            write(basepath.with_suffix('.txt'),
                  times, *np.swapaxes(results, 1, 2),
                  obj_name=target)

        return times, *results

    def _phot_daily(self, fun, **kws):

        assert not self.run.varies_by('target')

        ts = []
        daily, indices = self.run.group_by('date', return_index=True)
        order = op.itemgetter(*sum(indices.values(), []))(self.reg.order)
        segs = op.itemgetter(*order)(self.reg.detections[1:])
        segs = split_like(segs, daily.values())
        for (date, sub), segs in zip(daily.items(), segs):
            logger.info('Starting photometry for {!r:} on {!r:}',
                        sub[0].target, date)

            ts.append(self.diff(fun, sub, segs, **kws))
            break
        return ts

    def ragged(self, top=5, dilate=0):
        # ragged aperture photometry (no tracking!!)
        return self._phot_daily(ragged, top=top, dilate=dilate)

    # def opt(self):
    #     pass

    # def aperture(self):
    #     pass

    # def cog(self):
    #     pass

    # def psf(self):
    #     pass
