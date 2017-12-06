
from ..lm_compat import lmModelFactory
from .models import ConstantBG, CircularGaussianPSF, EllipticalGaussianPSF

# Now we do the actual conversion

# ConstantBG
convert = ('__call__', 'validate')
pnames = ['bg']
ConstantBG = lmModelFactory(ConstantBG, convert, pnames)

# CircularGaussianPSF
pnames = 'x0, y0, z0, a, d'.split(', ')
convert = ('__call__', 'validate')
CircularGaussianPSF = lmModelFactory(
    CircularGaussianPSF, convert, pnames)

# EllipticalGaussianPSF
pnames = 'x0, y0, z0, a, b, c, d'.split(', ')
convert = ('__call__', 'reparameterize', 'integrate', 'integration_uncertainty',
           'int_err', 'get_fwhm', 'get_description',
           'correlation', 'covariance_matrix', 'precision_matrix',
           'get_sigma_xy', 'get_theta', 'validate')
EllipticalGaussianPSF = lmModelFactory(
    EllipticalGaussianPSF, convert, pnames)

if __name__ == "__main__":
    # test
    from pickle import dumps, loads


    def pr(*x):
        for xx in x:
            print(type(xx), xx, xx.params)


    a, b, c = ConstantBG(), CircularGaussianPSF(), EllipticalGaussianPSF()

    pr(a, b, c)
    a_p = dumps(a)
    b_p = dumps(b)
    c_p = dumps(c)

    del a, b
    a = loads(a_p)
    b = loads(b_p)
    c = loads(c_p)
    pr(a, b, c)

    from IPython import embed

    embed()
