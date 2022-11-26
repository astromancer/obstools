"""
Input / output helpers
"""

# std
import io
import mmap

# third-party
from loguru import logger
from astropy.io import fits

# local
from recipes.oo.temp import temporarily


def fileobj_open_picklable(filename, mode):

    if 'r' in mode:
        logger.debug('Injecting process-inheritable file wrapper.')
        return FileIOPicklable(filename, mode)

    return fits.util.fileobj_open(filename, mode)


class _FilePicklable(fits.file._File):

    def _open_filename(self, filename, mode, overwrite):
        with temporarily(fits.file, fileobj_open=fileobj_open_picklable):
            super()._open_filename(filename, mode, overwrite)

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
