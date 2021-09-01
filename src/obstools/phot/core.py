

# std
import functools as ftl
import itertools as itt

# third-party
import numpy as np

# local
from recipes.lists import split_like
from recipes.iter import split_slices
from recipes.logging import logging, get_module_logger

# relative
from ..lc.ascii import write
from ..campaign import PhotCampaign
from .. import io

# module level logger
logger = get_module_logger()
logging.basicConfig()
logger.setLevel(logging.INFO)


def ragged(hdu, seg, top=5, dilate=0, filename=None):
    # ragged aperture photometry (no tracking!!)
    logger.info('Running ragged aperture photometry on hdu: %s',
                hdu.file.name)
    seg = seg.dilate(dilate, copy=True)
    return seg.flux(hdu.calibrated, seg.labels[:top])


# @caches.to_file(cachePaths.phot, typed={'self': _hdu_hasher},
# filename = cachePaths.phot / '{hdu.file.name}-{fun.__name__}.phot')
def phot(hdu, fun, **kws):
    return fun(hdu, **kws)


class PhotInterface:
    """Interface for photometry algorithms"""

    filename_template = '{hdu.file.stem}-{fun.__name__}.phot'

    # def __get__(self, run, kls):
    #     if run is None:
    #         return self

    def __init__(self, run, reg=None, path='.'):
        assert isinstance(run, PhotCampaign)
        self.run = run
        self.reg = reg
        self.path = path

    def _runner(self, fun, day, segs, top=5, save=True, **kws):
        # run photometry for multiple hdus of the same target and aggregate a light
        # curve

        for hdu, seg in zip(day, segs):
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

        sizes = day.attrs.nframes
        n = sum(sizes)
        labels = self.reg.attrs('seg.labels')[1:]
        nstars = max(map(max, labels)) - 1
        top = min(top, nstars)

        target = 1
        cmp = list(ftl.reduce(set.intersection, map(set, labels)) - {target})
        cmp = np.array(cmp) - 1

        io.load_memmap(, (2, n, top))
        
        times = np.empty(n)
        results = np.ma.zeros((2, n, top)) # FIXME: memmap!!
        sections = split_slices(itt.accumulate(sizes))
        datagen = self._runner(fun, day, segs, top, save, **kws)
        for section, seg, (t, flx, std) in zip(sections, segs, datagen):
            times[section] = t

            # rescale
            idx = seg.labels[:top] - 1
            results[:, section, idx] = \
                (flx, std) / np.ma.median(flx[:, cmp], 1, keepdims=True)

                
        if save:
            # save to plain text
            hdu = day[0]
            filename = self.path / f'{hdu.date:d}-{fun.__name__}.txt'
            write(self.path / filename,
                  times, *np.swapaxes(results, 1, 2),
                  obj_name=hdu.target)

        return times, *results

    def _phot_daily(self, fun, **kws):

        assert not self.run.varies_by('target')

        ts = []
        daily = self.run.group_by('date')
        segs = split_like(self.reg.detections[1:], daily.values())
        for (date, sub), segs in zip(daily.items(), segs):
            if f'{date:d}' == '20131002':
                ts.append(self.diff(fun, sub, segs, **kws))
            
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
