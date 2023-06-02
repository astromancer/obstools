
# third-party
import numpy as np

# local
from recipes.dicts import AttrReadItem, ListLike


class auto_id:
    """Enable automatic group labeling"""


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

    auto_key_template = 'group%i'

    def check_item(self, item):
        return bool(len(item))

    def convert_item(self, item):
        return np.atleast_1d(item).astype(int)

    @property
    def sizes(self):
        return list(map(len, self.values()))
        # return [len(item) for item in self.values()]

    def inverse(self):
        # fixme: if a label belongs to more than one group
        return {lbl: gid for gid, labels in self.items() for lbl in labels}


class LabelGroupsMixin:
    """
    Mixin class for grouping and labelling image segments.
    """

    _groups = ()

    def __init__(self, groups=()):
        self.groups = groups

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
