

# third-party
import pytest
import numpy as np

# local
from recipes.transforms import rigid, rotate
from obstools.image.gmm import CoherentPointDrift, MultiGauss


# from pytes


def make_id(name, n):
    for i in range(n):
        yield f'{name}{i}'


def _generic(request):
    return request.param

# dynamically generate some simple fixtures for combinatorial tests
# sourcery skip: remove-dict-items
for name, params in dict(
        xy=[((5, 10), (5, 6), (8, 2))],
        sigmas=[1,
                (1, 0.5),
                (1, 0.5, 0.6),
                ((0.5, 0.1),
                 (0.5, 0.6),
                 (0.8, 0.2))
                ],
        amplitudes=[10,
                    (1, 2, 3)
                    ]
).items():
    fix = pytest.fixture(params=params, ids=make_id(name, len(params)))(_generic)
    # setattr(__)
    exec(f'{name} = fix')
    
    # exec(textwrap.dedent(
    #     f"""
    #     @pytest.fixture(params=params, ids=make_id(name, len(params)))
    #     def {name}(request):
    #         return request.param
    #     """))


@pytest.fixture
def model(xy, sigmas, amplitudes):
    # this actually already tests the initialization, but it's basically
    # impossible to re-use the model created in a test :(
    return MultiGauss(xy, sigmas, amplitudes)

@pytest.fixture
def cpd(xy, sigmas, amplitudes):
    # this actually already tests the initialization, but it's basically
    # impossible to re-use the model created in a test :(
    weights = amplitudes / np.sum(amplitudes)
    return CoherentPointDrift(xy, sigmas, weights)

# @pytest.mark.skip


# class TestTransforms:
#     def test_rotate(self):
#         ''

#     def test_rigid(self):
#         ''

#     def test_affine(self):
#         ''


# @pytest.mark.skip       # TODO autouse
class TestMultiGauss:
    """test MultiGauss"""

    xy_test = (5, 5)
    xy_test2 = [(3, 3), (6, 6)]
    expected = iter([6.066578,
                     1.213435,
                     1.353353,
                     0.270671,
                     1.353390,
                     0.270674,
                     2.493522,
                     0.498704,
                     ])
    expected2 = iter([[0.015057, 6.06779528],
                      [0.00301366, 1.21340099],
                      [5.06408816e-06, 6.06530660e+00],
                      [1.51716529e-06, 1.21306132e+00],
                      [8.20811573e-11, 1.35538752e+00],
                      [1.33176199e-11, 2.70874035e-01],
                      [1.25016514e-08, 1.35335283e+00],
                      [2.50034256e-09, 2.70670566e-01]]
                     )

    def test_init(self, model):
        assert model.xy.shape == model.sigmas.shape

    def test_set_attr(self, model):
        # test setting hyper parameters
        amplitudes = (3, 4, 5)
        model.amplitudes = amplitudes

        # test bork for negative amplitudes
        with pytest.raises(ValueError):
            model.amplitudes = np.negative(amplitudes)

        # test bork for bad shape
        with pytest.raises(ValueError):
            model.amplitudes = [1, 2]

        # test bork bad sigma shape
        with pytest.raises(ValueError):
            model.sigmas = [1, 2, 3, 4]

        # TODO: test borks for non-finite / masked amp / sigma

    # TODO: def test_flux(self, model): check integral

    # @pytest.mark.parametrize('expected', (1,2))

    def test_call(self, model):
        # model call
        # use the next 2 lines to generate expected answers
        # with open('foo.txt', 'a') as fp:
        #     fp.write('%f\n' % model((), self.xy_test))

        pytest.approx(model((), self.xy_test), next(self.expected))

    def test_call2(self, model):
        # model call
        pytest.approx(model((), self.xy_test2), next(self.expected2))

    @pytest.mark.skip
    @pytest.mark.mpl_image_compare(baseline_dir='images',
                                   remove_text=True)
    def test_plot(self, model):
        # print('running test_plot')
        im = model.plot(sliders=False, hist=False)
        # im.ax.plot(*self.xy_test, 'mx')
        # im.figure.savefig(f'{next(counter)}.png')
        return im.figure

# TODO: testGMM - check always integrates to 1


class TestCoherentPointDrift:
    @classmethod
    def setup_class(cls):
        cls.model = CoherentPointDrift(((5, 10), (5, 6), (8, 2)),
                                       2.5)

    # @pytest.mark.skip
    def test_transform(self):
        xy = np.random.rand(5, 2)
        assert xy == pytest.approx(self.model.transform(xy, (0, 0, 0)))

        self.model.fit_rotation = False
        assert xy + 1 == pytest.approx(self.model.transform(xy, (1, 1)))

    # @pytest.mark.skip
    def test_fit_translate(self, cpd):
        cpd.fit_rotation = False
        off = (0.5, 0.5)
        r = cpd.fit(cpd.gmm.xy + off)
        assert pytest.approx(off, 0.01) == -r

    def test_fit(self, cpd):
        cpd.fit_rotation = True
        θ = np.pi / 12
        off = (0.5, 0.5)

        xy = rigid(cpd.gmm.xy, np.hstack([off, θ]))
        r = cpd.fit(xy)

        assert pytest.approx(r, 0.01) == np.hstack([-rotate(off, -θ), -θ])
