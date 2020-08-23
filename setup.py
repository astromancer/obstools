import site
import inspect
import subprocess
from pathlib import Path
import os

def get_site():
    dest = site.getsitepackages()[0]
    if os.access(dest, os.W_OK):
        return dest

    dest = site.getusersitepackages()
    if os.access(dest, os.W_OK):
        return dest
    
    raise Exception(f'{dest} not writable')

def link_install_cheat():
    """
    Cheat install that links the repo's main directory into system python
    path.  Good for development libraries - changes to the code base are
    immediately available to the interpreter :)

    Can be used as is for any simple python library
    """
    here = inspect.getfile(inspect.currentframe())
    here = Path(here)
    src = here.parent.resolve()

    pkg_name = src.name
    src = str(src / src.name)
    dest = str(Path(get_site()) / pkg_name)

    print('linking', src, dest)
    ok = subprocess.call(['ln', '-sfn', src, dest])
    # ln -s path/to/repo/eeg `python3 -m site --user-site`/eeg
    if ok == 0:
    	print('Installed', pkg_name, 'at', dest, 'via cheat')
    return ok

happiness = link_install_cheat()
