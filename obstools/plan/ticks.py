
# std libs
import warnings

# third-party libs
import numpy as np
from matplotlib import ticker
from matplotlib.transforms import Transform, IdentityTransform




class TransFormatter(ticker.ScalarFormatter):
    _transform =  IdentityTransform()

    def __init__(self, transform=None, infinite=1e15, useOffset=None, useMathText=True, useLocale=None):
        super(TransFormatter, self).__init__(useOffset, useMathText, useLocale)
        self.inf = infinite


        if transform is not None:
            if isinstance(transform, Transform):
                self._transform = transform
            else:
                raise ValueError('bork!')

    def __call__(self, x, pos=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xt = self._transform.transform(x)

        return super(TransFormatter, self).__call__(xt, pos)

    def pprint_val(self, x):
        #make infinite if beyond threshold
        #print('PPP')
        if abs(x) > self.inf:
            x = np.sign(x) * np.inf

        if abs(x) == np.inf:
            if self.useMathText:
                sign = '-' * int(x<0)
                return r'{}$\infty$'.format(sign)

        return decimal_repr(x,2)
#         #return super().pprint_val(x)   #FIXME: does not produce correct range of ticks


#****************************************************************************************************
class InfiniteAwareness():
    def __call__(self, x, pos=None):
        xs = super(InfiniteAwareness, self).__call__(x, pos)

        if xs == 'inf':
            return r'$\infty$'
        else:
            return xs #


#****************************************************************************************************
class DegreeFormatter(ticker.Formatter):
    def __init__(self, precision=0):
        self.precision = precision

    def __call__(self, x, pos=None):
        # \u00b0 : degree symbol
        return '{:.{}f}\u00b0'.format(x, self.precision)
