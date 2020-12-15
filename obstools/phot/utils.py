"""
Miscellaneous utility functions
"""

# std libs
import logging
import itertools as itt

# third-party libs
import numpy as np
import more_itertools as mit

# local libs
from motley.progress import ProgressBar
from recipes.logging import LoggingMixin
from recipes.dicts import AttrReadItem, ListLike
from recipes.logging import get_module_logger

# module level logger
logger = get_module_logger()


def null_func(*_):
    pass


def iter_repeat_last(it):
    """
    Yield items from the input iterable and repeat the last item indefinitely
    """
    it, it1 = itt.tee(mit.always_iterable(it))
    return mit.padded(it, next(mit.tail(1, it1)))


# class ContainerAttrGetterMixin(object):
#     """"""
#     # def __init__(self, data):
#     #     self.data = data
#     #     # types = set(map(type, data.flat))
#     #     # assert len(types) == 1
#
#     def __getattr__(self, key):
#         if hasattr(self, key):
#             return super().__getattr__(key)
#
#         # if hasattr(self.data[0, 0], key)
#         getter = operator.attrgetter(key)
#         return np.vectorize(getter, 'O')(self)


class ProgressLogger(ProgressBar, LoggingMixin):
    # def __init__(self, **kws):
    #     ProgressBar.__init__(self, **kws)
    #     if not log_progress:
    #         self.progress = null_func

    def create(self, end):
        self.end = end
        self.every = np.ceil((10 ** -(self.sigfig + 2)) * self.end)
        # only have to update text every so often

    def progress(self, state, info=None):
        if self.needs_update(state):
            bar = self.get_bar(state)
            self.logger.info('Progress: %s' % bar)


# class ProgressPrinter(ProgressBar):
#     def __init__(self, **kws):
#         ProgressBar.__init__(self, **kws)
#         if not print_progress:
#             self.progress = self.create = null_func

def progressFactory(log=True, print_=True):
    if not log:
        global ProgressLogger  # not sure why this is needed

        class ProgressLogger(ProgressLogger):
            progress = null_func

    if not print_:
        class ProgressPrinter(ProgressBar):
            progress = create = null_func

    return ProgressLogger, ProgressBar


# noinspection NonAsciiCharacters


class Record(AttrReadItem, ListLike):
    """
    Ordered dict with key access via attribute lookup. Also has some
    list-like functionality: indexing by int and appending new data.
    Best of both worlds.
    """
    pass


class LabelGroups(Record):
    """
    Makes sure values (labels) are always arrays.
    """
    _auto_name_fmt = 'group%i'

    def _allow_item(self, item):
        return bool(len(item))

    def _convert_item(self, item):
        return np.atleast_1d(item).astype(int)

    @property
    def sizes(self):
        return list(map(len, self.values()))
        # return [len(item) for item in self.values()]

    def inverse(self):
        # fixme: if a label belongs to more than one group
        return {lbl: gid for gid, labels in self.items() for lbl in labels}


class LabelGroupsMixin(object):
    """Mixin class for grouping and labelling image segments"""

    def __init__(self, groups=None):
        self._groups = None
        self.set_groups(groups)

    @property
    def groups(self):
        return self._groups

    @groups.setter
    def groups(self, groups):
        self.set_groups(groups)

    def set_groups(self, groups):
        self._groups = LabelGroups(groups)

    # todo
    # def remove_group()
