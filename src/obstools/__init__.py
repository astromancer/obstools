
# std
from pathlib import Path

# third-party
from loguru import logger

# local
from recipes.config import ConfigNode
from recipes.dicts import AttrReadItem
from recipes.caching import Reject, hashers


# ---------------------------------------------------------------------------- #
# config
CONFIG = ConfigNode.load_module(__file__)

#
logger.disable('obstools')

# ---------------------------------------------------------------------------- #


def _cal_image_hasher(image):
    if image is None:
        return
    return hashers.array(image)


def _hdu_hasher(hdu):
    if hdu.file:
        # Cache on the hdu filename.
        return (str(hdu.file),
                _cal_image_hasher(hdu.calibrated.dark),
                _cal_image_hasher(hdu.calibrated.flat))

    # hdu.file is NULL --> unsaved file, don't cache
    return Reject(silent=False)


# set hashing algorithm for HDU types
# config(typed={ImageHDU: _hdu_hasher})

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
