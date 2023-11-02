
import numpy as np


class LabelUser:
    """
    Mixin class for objects that use `SegmentedImage`.

    Adds the `use_label` and `ignore_label` properties.  Whenever a function
    that takes the `label` parameter is called with the default value `None`,
    the labels in `use_labels` will be used instead. This helps with dynamically
    including / excluding labels and with grouping labelled segments together.

    If initialized without any arguments, all labels in the segmentation image
    will be in use.

    """

    def __init__(self, use_labels=None, ignore_labels=None):
        if use_labels is None:
            use_labels = self.seg.labels
        if ignore_labels is None:
            ignore_labels = []

        self._use_labels = np.setdiff1d(use_labels, ignore_labels)
        self._ignore_labels = np.asarray(ignore_labels)

    @property
    def ignore_labels(self):
        return self._ignore_labels

    @ignore_labels.setter
    def ignore_labels(self, labels):
        self._ignore_labels = np.asarray(labels, int)
        self._use_labels = np.setdiff1d(self.seg.labels, labels)

    @property
    def use_labels(self):
        return self._use_labels

    @use_labels.setter
    def use_labels(self, labels):
        self._use_labels = np.asarray(labels, int)
        self._ignore_labels = np.setdiff1d(self.seg.labels, labels)

    def resolve_labels(self, labels=None):
        return self.use_labels if labels is None else self.seg.has_labels(labels)

    @property
    def nlabels(self):
        return len(self.use_labels)

    # @property
    # def sizes(self):
    #     return [len(labels) for labels in self.values()]
