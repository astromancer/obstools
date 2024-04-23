"""
Input / output helpers
"""

# std
import os
import mmap

# third-party
from astropy.io import fits

# local
from recipes.io import FileIOPicklable


class FilePicklable(fits.file._File):

    def _open_filename(self, filename, mode, overwrite):
        """Open a FITS file from a filename string."""
        if mode == "ostream":
            self._overwrite_existing(overwrite, None, True)

        if os.path.exists(self.name):
            with open(self.name, "rb") as f:
                magic = f.read(4)
        else:
            magic = b""

        ext = os.path.splitext(self.name)[1]

        if not self._try_read_compressed(self.name, magic, mode, ext=ext):
            mode = fits.file.IO_FITS_MODES[mode]
            if 'r' in mode:
                self._file = FileIOPicklable(filename, mode)
            else:
                self._file = open(self.name, mode)
            self.close_on_error = True

        # Make certain we're back at the beginning of the file
        # BZ2File does not support seek when the file is open for writing, but
        # when opening a file for write, bz2.BZ2File always truncates anyway.
        if not (fits.file._is_bz2file(self._file) and mode == "ostream"):
            self._file.seek(0)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_mmap']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._mmap = mmap.mmap(self._file.fileno(), 0,
                               access=fits.file.MEMMAP_MODES[self.mode],
                               offset=0)
