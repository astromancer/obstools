from sys import argv
from fastfits import quickheader

print(repr(quickheader( argv[1] )))