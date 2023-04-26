import numpy as np


class SegmentedArray(np.ndarray):
    """
    WORK IN PROGRESS

    Array subclass for keeping image segmentation data. Keeps a reference to
    the `SegmentedImage` object that created it so that changing the
    segmentation array data triggers the lazyproperties to recompute the next
    time they are accessed.

    # note inplace operations on array will not trigger reset of parent
    # lazyproperties, but setting data explicitly should

    """

    def __new__(cls, input_array, parent):
        # Input data is array-like structure
        obj = np.array(input_array)

        # initialize with data
        super_ = super(SegmentedArray, cls)
        obj = super_.__new__(cls, obj.shape, int)
        super_.__setitem__(obj, ..., input_array)  # populate entire array

        # add SegmentedImage instance as attribute to be updated upon
        # changes to segmentation data
        obj.parent = parent  # FIXME: this will be missed for new-from-template
        return obj

    def __setitem__(self, key, value):
        np.ndarray.__setitem__(self, key, value)
        # set the data in the SegmentedImage
        # print('Hitting up set data')
        self.parent.data = self

    def __reduce__(self):
        return SegmentedArray, (np.array(self), self.parent)

    #     constructor, init_args, *rest = np.ndarray.__reduce__(self)

    # def __getnewargs__(self):
    #     # These will be passed to the __new__() method upon unpickling.
    #     print('HI!!!!! ' * 10)
    #     return self, self.parent

    def __array_finalize__(self, obj):
        #
        if obj is None:
            # explicit constructor:  `SegmentedArray(data)`
            return

        # view casting or new-from-template constructor
        if hasattr(obj, 'parent'):
            self.parent = obj.parent

    def __array_wrap__(self, out_arr, context=None):
        # return a plain old array so we don't accidentally reset the parent
        # lazyproperties by edits on data of derived array
        return np.array(out_arr)

        # def __array_ufunc__(self, ufunc, method, *inputs, **kws):
        #     # return a plain old array so we don't accidentally reset the parent
        #     # lazyproperties by edits on data of derived array
        #     result = super(SegmentedArray, self).__array_ufunc__(ufunc, method,
        #                                                          *inputs, **kws)
        #     return np.array(result)

        # class Slices(np.recarray):
        #     # maps semantic corner positions to slice attributes
        #     _corner_slice_mapping = {'l': 'start', 'u': 'stop', 'r': 'stop'}
        #
        #     def __new__(cls, parent):
        #         # parent in SegmentedImage instance
        #         # get list of slices from super class SegmentationImage
        #         slices = SegmentationImage.slices.fget(parent)
        #         if parent.use_zero:
        #             slices = [(slice(None), slice(None))] + slices
        #
        #         # initialize np.ndarray with data
        #         super_ = super(np.recarray, cls)
        # dtype = np.dtype(list(zip('yx', 'OO')))


#         obj = super_.__new__(cls, len(slices), dtype)
#         super_.__setitem__(obj, ..., slices)
#
#         # add SegmentedImage instance as attribute
#         obj.parent = parent
#         return obj
#
#     def __array_ufunc__(self, ufunc, method, *inputs, **kws):
#         return NotImplemented
