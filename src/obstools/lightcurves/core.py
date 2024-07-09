"""
Multivariate Light Curves as extension of Time Series
"""

# third-party
from astropy.time import Time

# local
from scrawl.ticks import DateTick
from recipes.config import ConfigNode
from tsa.ts.ts import MultiVariate, TimeSeries
from tsa.ts.plotting import TimeSeriesPlot, make_twin_relative


# ---------------------------------------------------------------------------- #
# Config
CONFIG = ConfigNode.load_module(__file__)


# ---------------------------------------------------------------------------- #
# Module constants

# seconds per day
SPD = 86400


# ---------------------------------------------------------------------------- #

class LightCurvePlot(TimeSeriesPlot):

    def setup_figure(self, ax, figure=None, twinx=True,
                     figsize=CONFIG.plots.figure.size, **kws):

        fig, ax, hax = super().setup_figure(ax, figure, figsize, twinx, **kws)

        ts = self.parent
        jd0 = int(ts.t[0]) - 0.5
        utc0 = Time(jd0, format='jd').utc.iso.split()[0]

        # plot
        cfg = CONFIG.plots
        #  twinx='period' / '
        xlabel = cfg.axes.labels.x
        axp = make_twin_relative(ax, -(ts.t[0] - jd0) * SPD,
                                 tick_label_angle=xlabel.ticks.rotation)

        axp.xaxis.set_minor_formatter(DateTick(utc0))
        # _rotate_tick_labels(axp, 45, True)

        ax.set(xlabel=xlabel.bottom, ylabel=cfg.axes.labels.y.left)
        axp.set_xlabel(xlabel.top, labelpad=xlabel.pad)

        # fig.tight_layout()
        fig.subplots_adjust(**cfg.figure.margins)

        return fig, ax, hax

    def __call__(self, *data, **kws):
        kws = {**dict(t0=[0], tscale=SPD, show_masked=True), **kws}
        return super().__call__(*data, **kws)


class LightCurve(TimeSeries):

    plot = LightCurvePlot(CONFIG.plots.axes.plims)

    def write(self, filename, **kws):
        return super().write(filename,
                             **{**CONFIG.io.txt.rename('columns', 'col_info'),
                                **kws})


class MultiVariateLightCurve(MultiVariate, LightCurve):
    pass
