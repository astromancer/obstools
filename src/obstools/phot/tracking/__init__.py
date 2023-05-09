# local
import motley
from recipes.config import ConfigNode

# ---------------------------------------------------------------------------- #
CONFIG = ConfigNode.load_module(__file__, 'yaml')


# stylize progressbar
prg = CONFIG.progress
prg['bar_format'] = motley.stylize(prg.bar_format)
del prg


# relative
from .gui import *
from .display import *
from .tracking import *
