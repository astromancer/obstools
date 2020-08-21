import itertools as itt
import matplotlib.pyplot as plt
from obstools.plan.visibilities import Visibilities
from obstools.plan.limits import HARD_LIMITS, SOFT_LIMITS, TelescopeLimits
import pytest
import re
from pathlib import Path
import more_itertools as mit


# TODO:
# test_add_target
# test_remove_target
# - check legend, tracks, blitting
# test_limits
# test_highlight
# test_reset
# test_resize
# - check blitting still works
# - labels get redone
# test_watch_folder
# test_scroll_dates
# test_twilight_text

# TestMoon
# - test get markers
# TestObjTrack
#   - test_chaching
#

def get_readme_code():
    path = (Path(__file__).parent / '../README.md').resolve()
    regex = re.compile("```python(.*?Vis.*?)```", re.S)
    for code in regex.finditer(path.read_text()):
        yield code.group(1)


# def idfn(val):
#     return str(next(count))


# count = itt.count()
# ---------------------------------------------------------------------------- #


@pytest.mark.mpl_image_compare(baseline_dir='images',
                               # filename='example0.png',
                               style='default')
@pytest.mark.parametrize('code', get_readme_code(), ids=itt.count())
def test_readme_example(code):
    locals_ = {}
    exec(code, None, locals_)
    vis = locals_['vis']
    vis.canvas.draw()
    # vis.close()
    return vis.figure

# ---------------------------------------------------------------------------- #


@pytest.mark.mpl_image_compare(baseline_dir='images',
                               # filename='example1.png',
                               style='default')
def test_example3():

    MCVs = ['AR Sco',
            'V895 Cen',
            # 'IGR J14536-5522',
            '1RXS J174320.1-042953',
            'RX J1610.1+0352',
            # 'RX J1745.5-2905',
            'SDSS J151415.65+074446.4',
            # '2XMM J154305.5-522709',
            '2MASS J19283247-5001344',
            'QS Tel',
            'V1432 Aql',
            # 'CRTS SSS100805 J194428-420209',
            # 'IGR J19552+0044',
            'HU Aqr',
            # 'CD Ind',
            # 'CE Gru',
            'BL Hyi',
            # 'RX J0154.0-5947',
            'FL Cet',
            'FO Aqr']

    vis = Visibilities(MCVs, date='2020-08-14', colors='jet')  #
    vis.canvas.draw()
    return vis.figure


# class TestCelestialTrack:
#     def test_cache(self, name, date):

class TestLimits:
    @pytest.mark.parametrize(
        'ok, expected',
        mit.flatten([((f'{tel}inch', nr),
                      (f'{tel} inch', nr),
                      (f'{tel}in', nr),
                      (f'{tel} in', nr),
                      (int(tel), nr),
                      (nr, nr),
                      (f'{nr}m', nr),
                      (f'{nr} m', nr))
                     for tel, nr in {'40': 1., '74': 1.9}.items()])
    )
    def test_resolve(self, ok, expected):
        assert HARD_LIMITS[ok] == HARD_LIMITS[expected]
        assert SOFT_LIMITS[ok] == SOFT_LIMITS[expected]

    @pytest.mark.parametrize('tel', (1, 1.9))
    @pytest.mark.mpl_image_compare(baseline_dir='images', remove_text=True)
    def test_plot(self, tel):
        limits = TelescopeLimits(tel)
        art = limits.plot()
        return art[0].figure

    # @pytest.mark.skip
    # @pytest.mark.parametrize('tel', (1, 1.9))
    # def test_get_ha(self, dec):
    #     lim = TelescopeLimits(tel)
    #     lim.plot(ax)

    #     for which in _HS:  # itt.product(_HS
    #         lims = lim.get('east', which)
    #         dec = lims[:, 0]
    #         for d in np.linspace(dec.min(), dec.max()):
    #             ha = lim.get_visible_ha(d, which=which)
    #             if ha is None:
    #                 continue
    #             ax.plot(ha, [d] * 2, 'm*')


# def test_limits():
#     vis = Visibilities(MCVs, date='2020-08-14', colors='jet', tel='1m')  #
#     vis.canvas.draw()
#     return vis.figure
