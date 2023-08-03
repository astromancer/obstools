
# std
from pathlib import Path

# local
from recipes.config import ConfigNode, find_config

config_file = find_config((path := Path(__file__)), 'yaml', True)
CONFIG = ConfigNode.load(config_file)

