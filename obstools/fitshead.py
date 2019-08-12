

# std libs
from sys import argv

# third-party libs
from fastfits import quickheader


print(repr(quickheader( argv[1] )))
