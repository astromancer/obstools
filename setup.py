import site
import inspect
import subprocess
from pathlib import Path


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
    dest = site.getusersitepackages()
    dest = str(Path(dest) / pkg_name)

    print('linking', src, dest)
    ok = subprocess.call(['ln', '-sf', src, dest])
    # ln -s path/to/repo/eeg `python3 -m site --user-site`/eeg
    if ok == 0:
    	print('Installed', pkg_name, 'at', dest, 'via cheat')
    return ok

happiness = link_install_cheat()
