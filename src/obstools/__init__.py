
from pathlib import Path
from recipes.dicts import AttrReadItem


# persistent caches for faster coordinate and image retrieval
cachePath = Path.home() / '.cache/obstools'  # NOTE only for linux!

cachePaths = AttrReadItem(
    base=cachePath,
    coo=cachePath / 'coords.pkl',
    site=cachePath / 'sites.pkl',
    dss=cachePath / 'dss.pkl',
    sky=cachePath / 'skymapper.pkl',
    samples=cachePath / 'samples.pkl',
)
