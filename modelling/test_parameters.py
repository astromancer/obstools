from IPython import embed
from scipy import stats

from .parameters import Parameters, Priors

if __name__ == '__main__':
    # some tests for `Parameters`

    # construction
    p0 = Parameters([1, 2, 3])
    p00 = Parameters(p0)

    p1 = Parameters(x=[1, 2, 3], y=1)
    p2 = Parameters(x=[1, 2, 3], y=p1)

    p3 = Parameters(v=p1, w=p2)

    for p in [p0, p1, p2, p3]:
        print(p)

    # some tests for priors
    pr = Priors(u=stats.uniform(),
                z=dict(q=stats.uniform(0, 100),
                       x=stats.uniform(1e-12, 2e12)))
