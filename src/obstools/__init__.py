
from pathlib import Path
from recipes.dicts import AttrReadItem

from recipes.caches import Reject


def _hdu_hasher(hdu):
    # Cache on the hdu filename. hdu.file is NULL --> unsaved file, ignore
    return str(hdu.file) if hdu.file else Reject(silent=False)


# persistent caches for faster coordinate and image retrieval
cachePath = Path.home() / '.cache/obstools'  # NOTE only for linux!
cachePaths = AttrReadItem(
    base=cachePath,
    coo=cachePath / 'coords.pkl',
    site=cachePath / 'sites.pkl',
    dss=cachePath / 'dss.pkl',
    sky=cachePath / 'skymapper.pkl',
    samples=cachePath / 'samples.pkl',
    skyimage=cachePath / 'skyimage.pkl'
)
