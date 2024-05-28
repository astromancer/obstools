
# third-party
from loguru import logger
from platformdirs import user_cache_path

# local
from recipes.config import ConfigNode
from recipes.caching import Reject, hashers
from recipes.containers.dicts import AttrReadItem


# Config
# ---------------------------------------------------------------------------- #
CONFIG = ConfigNode.load_module(__file__)
CONFIG.image.detect['roundness'] = tuple(CONFIG.image.detect.roundness)

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
cachePath = _ = user_cache_path('obstools')
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
