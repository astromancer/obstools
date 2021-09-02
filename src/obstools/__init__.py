
from pathlib import Path
from recipes.dicts import AttrReadItem

from recipes.caches import Reject


def _hdu_hasher(hdu):
    # Cache on the hdu filename. hdu.file is NULL --> unsaved file, ignore
    return str(hdu.file) if hdu.file else Reject(silent=False)

# set hashing algorithm for HDU types
# config(typed={HDUExtra: _hdu_hasher})


# persistent caches for faster coordinate and image retrieval
cachePath = _ = Path.home() / '.cache/obstools'  # NOTE only for linux!
cachePaths = AttrReadItem(
    base=_,
    coo=_ / 'coords.pkl',
    site=_ / 'sites.pkl',
    dss=_ / 'dss.pkl',
    sky=_ / 'skymapper.pkl',
    samples=_ / 'samples.pkl',
    detection=_ / 'detection.pkl',
    skyimage=_ / 'skyimage.pkl'
)
