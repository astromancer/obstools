"""
Input / output helpers
"""

# std
import io
import mmap,os

# third-party
from loguru import logger
from astropy.io import fits

# local
from recipes.oo.temp import temporarily


class _FilePicklable(fits.file._File):

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
    
    # def _open_filename(self, filename, mode, overwrite):
    #     if 'r' in mode:
    #         logger.debug('Injecting process-inheritable file wrapper.')
    #         self._file = FileIOPicklable(filename, fits.file.IO_FITS_MODES[mode])

    #     super()._open_filename(filename, mode, overwrite)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_mmap']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._mmap = mmap.mmap(self._file.fileno(), 0,
                               access=fits.file.MEMMAP_MODES[self.mode],
                               offset=0)


class FileIOPicklable(io.FileIO):
    """
    File object (read-only) that can be pickled.

    This class provides a file-like object (as returned by :func:`open`,
    namely :class:`io.FileIO`) that, unlike standard Python file objects,
    can be pickled. Only read mode is supported.
    When the file is pickled, filename and position of the open file handle in
    the file are saved. On unpickling, the file is opened by filename,
    and the file is seeked to the saved position.
    This means that for a successful unpickle, the original file still has to
    be accessible with its filename.

    Note
    ----
    This class only supports reading files in binary mode. If you need to open
    a file in text mode, use the :func:`pickle_open`.

    Parameters
    ----------
    name : str
        either a text or byte string giving the name (and the path
        if the file isn't in the current working directory) of the file to
        be opened.
    mode : str
        only reading ('r') mode works. It exists to be consistent
        with a wider API.

    Example
    -------
    ::
        >>> file = FileIOPicklable(PDB)
        >>> file.readline()
        >>> file_pickled = pickle.loads(pickle.dumps(file))
        >>> print(file.tell(), file_pickled.tell())
            55 55

    See Also
    ---------
    TextIOPicklable
    BufferIOPicklable
    .. versionadded:: 2.0.0
    """

    def __init__(self, name, mode='r'):
        self._mode = mode
        super().__init__(name, mode)

    def __getstate__(self):
        if 'r' not in self._mode:
            raise RuntimeError(f'Can only pickle files that were opened in'
                               f' read mode, not {self._mode}')
        return self.name, self.tell()

    def __setstate__(self, args):
        name = args[0]
        super().__init__(name, mode='r')
        self.seek(args[1])
