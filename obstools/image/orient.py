# from astropy.io.fits import PrimaryHDU


class ImageOrienter(object):
    def __init__(self, hdu, flip=(), x=False, y=False):
        """
        Simple base class that stores the orientation state. Images are
        re-oriented upon item access.
        """

        # assert isinstance(hdu, PrimaryHDU)
        assert hdu.ndim >= 2
        self.hdu = hdu

        # setup tuple of slices for array
        orient = [slice(None)] * self.hdu.ndim

        # orient = [..., slice(None), slice(None)]
        for i, (s, t) in enumerate(zip('xy', (x, y)), 1):
            if (s in flip) or t:
                orient[-i] = slice(None, None, -1)

        self.orient = tuple(orient)

    def __call__(self, data):
        return data[self.orient]

    def __getitem__(self, item):
        if self.hdu.ndim == 2:
            # `section` fails with 2d data
            return self.hdu.data[self.orient][item]

        else:
            # reading section for performance
            return self.hdu.section[item][self.orient]
