

# third-party
from astropy.time import Time

# local
from scrawl.ticks import DateTick
from recipes.config import ConfigNode
from tsa.ts.ts import MultiVariate, TimeSeries
from tsa.ts.plotting import TimeSeriesPlot, make_twin_relative

# relative
from .. import lightcurves as lc


# ---------------------------------------------------------------------------- #
# Config
CONFIG = ConfigNode.load_module(__file__)


# ---------------------------------------------------------------------------- #
# Module constants

SPD = 86400

# ---------------------------------------------------------------------------- #


class LightCurvePlot(TimeSeriesPlot):

    def setup_figure(self, ax, figsize=(14, 8), twinx='period', **kws):

        fig, ax, hax = super().setup_figure(ax, **kws)

        ts = self.parent
        jd0 = int(ts.t[0]) - 0.5
        utc0 = Time(jd0, format='jd').utc.iso.split()[0]

        # plot
        axp = make_twin_relative(ax, -(ts.t[0] - jd0) * SPD, 1, 45)
        axp.xaxis.set_minor_formatter(DateTick(utc0))
        # _rotate_tick_labels(axp, 45, True)

        cfg = CONFIG.plots
        ax.set(xlabel=cfg.xlabel.bottom, ylabel=cfg.ylabel)
        axp.set_xlabel(cfg.xlabel.top, labelpad=cfg.xlabel.pad)

        # fig.tight_layout()
        fig.subplots_adjust(**cfg.subplotspec)

        return fig, ax, hax

    def __call__(self, *data, **kws):
        kws = {**dict(t0=[0], tscale=SPD, show_masked=True), **kws}
        super().__call__(*data, **kws)


class LightCurve(TimeSeries):

    plot = LightCurvePlot(plims=(-0.1, 99.99))

    @classmethod
    def load(cls, filename, hdu=None):
        return cls(*lc.io.read(filename, hdu))

    def save(self, filename, **kws):
        return lc.io.write(filename, self.t, self.x.T, self.u.T, **kws)


class MultiVariateLightCurve(MultiVariate, LightCurve):
    pass
