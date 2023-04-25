
# std
from pathlib import Path

# third-party
import yaml

# local
import motley
from recipes.dicts import AttrReadItem, DictNode


# ---------------------------------------------------------------------------- #

class ConfigNode(DictNode, AttrReadItem):
    pass


CONFIG = ConfigNode(
    **yaml.load((Path(__file__).parent / 'config.yaml').read_text(),
                Loader=yaml.FullLoader)
)

# stylize progressbar
CONFIG.progress['bar_format'] = motley.stylize(CONFIG.progress.bar_format)
