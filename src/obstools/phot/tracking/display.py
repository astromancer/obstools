# std
from collections import defaultdict

# third-party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import AffineDeltaTransform

# local
from motley.utils import vbrace

# relative
from ...image import SkyImage
from ..config import CONFIG


# ---------------------------------------------------------------------------- #
CONFIG = CONFIG.tracking

# ---------------------------------------------------------------------------- #
IntFormatter = ticker.FuncFormatter(lambda x, pos: f'{round(x):d}')

# ---------------------------------------------------------------------------- #


class SourceTrackerPlots:

    def __init__(self, tracker):
        self.tracker = tracker

    def image(self, image, ax=None, points='rx', contours=True,
              labels=CONFIG.labels, **kws):

        tracker = self.tracker
        sim = SkyImage(image, segments=tracker.seg)
        sim.xy = tracker.coords

        display, art = sim.show(True, False, points, False, labels,
                                coords='pixel', ax=ax, **kws)
        ax = display.ax
        
        if contours:
            if contours is True:
                contours = {}
            art.contours = self.contours(ax, **contours)

        return art

    def video(self, *args, **kws):
        TrackerVideo()

    def contours(self, ax, **kws):
        return self.tracker.seg.show.contours(
            ax,
            **{'offsets':     self.tracker.origin,
               'transOffset': AffineDeltaTransform(ax.transData),
               **kws}
        )

    def positions(self, labels=None, section=..., figsize=(6.5, 7),
                  legend=True, show_weights=True, show_null_weights=False,
                  show_avg=None, **kws):
        """
        For sets of measurements (m, n, 2), plot each (m, 2) feature set as on
        its own axes as scatter / density plot.  Additionally plot the shifted
        points in the neighbouring axes to the right. Mark the pixel size and
        diffraction limit for reference.
        """
        trk = self.tracker

        if labels is None:
            labels = trk.use_labels

        coords = trk.coords[labels - 1]
        residue = trk.measurements[section] - coords[None, None] - trk._origins[section, None, None]
        delta_xy = trk.measure_avg[section] - coords - trk.delta_xy[section, None]
        # delta_xy = trk.delta_xy[section]  # - trk._origins[section]
        fig, axes = plt.subplots(len(labels), 2, sharex=True, sharey=True,
                                 figsize=figsize)
        axes = np.atleast_2d(axes)
        style = dict(lw=1, ls='--', fc='none', ec='c', zorder=100)
        for i, ax in enumerate(axes.ravel()):
            self._setup_scatter_axes(ax, i)
            self._show_pixel(ax, style)

        style = dict(ls='', mfc='none', zorder=1, alpha=0.35, **kws)
        features = self.tracker.centrality

        if show_avg is None:
            show_avg = sum(np.fromiter(features.values(), float) > 0) > 1

        # loop features
        art = defaultdict(list)
        for feature, (marker, color, _) in CONFIG.centroids.items():
            style = {**style,
                     'marker': marker,
                     'color': color,
                     'label': feature}

            if (weight := features.get(feature)) is not None:
                if not (weight or show_null_weights):
                    continue
                data = residue[:, list(features).index(feature)]
            elif (feature == 'avg'):
                if not show_avg:
                    continue
                data = delta_avg
            else:
                raise ValueError(f'Unknown feature {feature!r}')

            # loop sources
            for j, xy in enumerate(data.T):
                ax1, ax2 = axes[j]
                art[feature].extend(ax1.plot(*xy, **style))
                ax2.plot(*delta_xy.T, **style)

        fig.tight_layout()
        if legend:
            self._legend(axes[0, 0], art, show_weights)

    def _setup_scatter_axes(self, ax, i):

        ax.set_aspect('equal')
        ax.tick_params(bottom=True, top=True, left=True, right=True,
                       labelright=(lr := (i % 2)), labelleft=(not lr),
                       labeltop=(lt := (i // 2) == 0), labelbottom=(not lt))

        for axx in (ax.xaxis, ax.yaxis):
            axx.set_major_locator(ticker.IndexLocator(1, -1))
            axx.set_major_formatter(IntFormatter)

        # ax.set(xlabel='$\delta x$', ylabel=)

        ax.grid()

    def _show_pixel(self, ax, style):
        # add pixel size rect
        r = Rectangle((-0.5, -0.5), 1, 1, **style)
        c = Circle((0, 0), self.tracker.precision, **style)
        for p in (r, c):
            ax.add_patch(p)
        return r, c

    def _legend(self, ax, art, show_weights, **kws):

        captions = dict(self._get_legend_labels(show_weights))

        handles, labels = [], []
        for key, lines in art.items():
            handles.append(lines[0])
            labels.append(captions[key])

        if show_weights:
            spacer = Line2D(*np.empty((2, 1, 1)), marker='', ls='')
            handles.insert(-1, spacer)
            labels.insert(-1, '\n ')

        ax.figure.subplots_adjust(top=0.8)
        ax.legend(
            handles, labels,
            **{**dict(ncol=2,
                      loc='lower left',
                      bbox_to_anchor=(-0.1, 1.2),
                      handletextpad=0.25,
                      labelspacing=-0.175,
                      columnspacing=0.25),
               **kws}
        )

    def _get_legend_labels(self, show_weights=True):
        centroids = self.tracker.centrality
        weights = iter(centroids.values())
        ends = iter(vbrace(len(centroids)).splitlines())

        *_, labels = zip(*map(CONFIG.centroids.get, centroids))
        if not show_weights:
            yield from zip(centroids, labels)
            return

        # add weight to label
        w = max(map(len, labels)) + 1
        for stat, (*_, label) in CONFIG.centroids.items():
            if stat in centroids:
                yield stat, f'{label: <{w}}$(w={next(weights)}) ${next(ends)}'
            else:
                yield stat, label


# ---------------------------------------------------------------------------- #
