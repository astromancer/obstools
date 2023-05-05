# std
import itertools as itt
from collections import defaultdict
from IPython import embed

# third-party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import AffineDeltaTransform

# local
from motley.utils import vbrace
from scrawl import scatter
from scrawl.video import VideoFeatureDisplay

# relative
from ...image import SkyImage
from ..config import CONFIG
from recipes import pprint
from mpl_multitab import MplTabGUI

# ---------------------------------------------------------------------------- #
CONFIG = CONFIG.tracking
CENTROIDS = CONFIG.plots.centroids


# ---------------------------------------------------------------------------- #


# def _part_map(mapping, keys):
#     for key in keys:
#         yield key, mapping[key]


# def part_map(mapping, keys):
#     return dict(_part_map(mapping, keys))


# ---------------------------------------------------------------------------- #


def _format_int(x, pos):
    # print(f'{x=}')
    return f'{round(x):d}'
    # return f'{(-1, 1)[x > 1] * round(abs(x)):d}'
    # return f'{x:.1f}'


# IntFormatter = ticker.FuncFormatter(_format_int)


class IntLocator(ticker.IndexLocator):
    def __call__(self):
        """Return the locations of the ticks"""
        dmin, dmax = self.axis.get_data_interval()
        return self.tick_values(np.floor(dmin), np.ceil(dmax))


def format_coord(x, y):
    return f'{x = :4.3f}; {y = :4.3f}'

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #


class SourceTrackerPlots:

    def __init__(self, tracker):
        self.tracker = tracker

    def image(self, image, ax=None, points='rx', contours=True,
              labels=CONFIG.plots.labels, **kws):

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
        return TrackerVideo(self.tracker, *args, **kws)

    # alias
    vid = video

    def contours(self, ax, **kws):
        return self.tracker.seg.show.contours(
            ax,
            **{'offsets':     self.tracker.origin,
               'transOffset': AffineDeltaTransform(ax.transData),
               **kws}
        )

    def positions(self, labels=None, section=...,
                  show=(cfg := CONFIG.plots.position).show,  # 'nulls',
                  legend=cfg.legend, figsize=cfg.figsize, **kws):
        """
        For sets of measurements (m, n, 2), plot each (n, 2) feature set as on
        its own axes as scatter / density plot. Additionally plot the shifted
        points in the neighbouring axes to the right. Mark the pixel size and
        diffraction limit for reference.
        """
        # trk = self.tracker

        if labels is None:
            labels = self.tracker.use_labels

        ui = None
        if _ui := len(labels) > 1:
            ui = MplTabGUI(title='Position Measurements')

        for label in labels:
            if _ui:
                tab = ui.add_tab(f'Source {label}', fig={'figsize': figsize})
                fig = tab.figure
            else:
                fig = plt.figure(figsize=figsize)

            art = self._positions_source(fig, label - 1, section, show, legend,
                                         **kws)
            # fig.tight_layout()

        return ui

    def _positions_source(self, fig, source, section, show, legend, **kws):

        tracker = self.tracker
        features = tracker.centrality

        if ('nulls' in show):
            n_feats = len(features)
        else:
            n_feats = sum(np.fromiter(features.values(), float) > 0)

        n_rows = n_feats + (('avg' in show) and (n_feats > 1))

        axes = fig.subplots(n_rows, 2,
                            sharex=True, sharey=True)
        fig.subplots_adjust(bottom=0.05,
                            left=0.05,
                            right=1,
                            hspace=0.05,
                            wspace=0.05)

        axes = np.atleast_2d(axes)
        for i, ax in enumerate(axes.ravel()):
            self._setup_scatter_axes(ax, i, i >= (n_rows - 1) * 2)
            self._show_pixel(ax, **CONFIG.plots.position.pixel)

        # loop features
        count = itt.count()
        art = defaultdict(list)
        scatter = dict(**CONFIG.plots.position.scatter, **kws)

        captions = {}
        if legend:
            captions = dict(self._get_legend_labels('weights' in show))

        for feature, (color, marker, label) in CENTROIDS.items():
            #
            if feature == 'avg':
                if 'avg' not in show:
                    continue

            # check requested feature was part of compute
            elif (weight := features.get(feature)) is not None:
                # have weight
                if not (weight or ('nulls' in show)):
                    # zero weight
                    continue
            #
            else:
                raise ValueError(f'Unknown feature {feature!r}')

            # get data
            residue = tracker.get_coords_residual(section, feature, source)
            shifted = tracker.get_coords_residual(section, feature, source, False)
            # shifted = residue + tracker.delta_xy[section]

            # plot residuals vs shifted residuals
            art[feature] = self._compare_scatter_density(
                axes[next(count)], residue, shifted,
                {**dict(color=color,
                        marker=marker,
                        label=captions.get(feature, label)),
                 **scatter},
                legend
            )

        x, y = pprint.uarray(tracker.coords[source], tracker.sigma_xy[source], 2)
        fig.suptitle(f'Source {tracker.use_labels[source]}:    {x = :s}; {y = :s}')

        return art

        # if legend:
        #     self._legend(axes[0, 0], art, show_weights)

    def _compare_scatter_density(self, axes, xy, shifted,
                                 scatter_kws=CONFIG.plots.position.scatter,
                                 legend=True, **kws):
        ax1, ax2 = axes
        common = dict(CONFIG.plots.position.density, **kws,
                      scatter_kws=scatter_kws)
        # print(common)
        _, *art = scatter.density(ax1, xy, **common)

        if legend:
            ax1.legend()  # loc='lower left', bbox_to_anchor=(0, 1.05)

        _, polys, points = scatter.density(ax2, shifted, **common)
        ax1.figure.colorbar(polys, ax=axes, shrink=0.7, pad=0.025)

        return art, (polys, points)

    def _setup_scatter_axes(self, ax, i, last):

        ax.set_aspect('equal')
        ax.tick_params(bottom=True, top=True, left=True, right=True,
                       labelright=False,  # (lr := (i % 2))
                       labelleft=(i % 2) == 0,
                       labeltop=(lt := (i // 2) == 0), labelbottom=last)

        # for axx in (ax.xaxis, ax.yaxis):
        #     axx.set_major_locator(IntLocator(1, 0))
        #     axx.set_major_formatter(ticker.FuncFormatter(_format_int))

        # ax.set(xlabel='$\delta x$', ylabel=)
        # ax.format_coord=format_coord
        ax.grid()

    def _show_pixel(self, ax, **style):
        # add pixel size rect
        r = Rectangle((-0.5, -0.5), 1, 1, **style)
        c = Circle((0, 0), self.tracker.precision, **style)
        for p in (r, c):
            ax.add_patch(p)
        return r, c

    def _legend(self, ax, art, show_weights, **kws):

        captions = dict(self._get_legend_labels(show_weights, braced=True))

        handles, labels = [], []
        for key, lines in art.items():
            handles.append(lines[0])
            labels.append(captions[key])

        if show_weights:
            spacer = Line2D(*np.empty((2, 1, 1)), marker='', ls='')
            handles.insert(-1, spacer)
            labels.insert(-1, '\n ')

            for _ in range(((len(labels) - 1) // 2) - 1):
                handles.append(spacer)
                labels.append('\n ')

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

    def _get_legend_labels(self, show_weights=True, braced=False, align=None):
        centroids = self.tracker.centrality
        * _, labels = zip(*map(CENTROIDS.get, centroids))
        if not show_weights:
            yield from zip(centroids, labels)
            return

        # add weight to label
        weights = iter(centroids.values())
        align = braced if align is None else align
        w = max(map(len, labels)) + 1 if align else 0
        ends = iter(vbrace(len(centroids)).splitlines() if braced else ())
        for stat, (*_, label) in CENTROIDS.items():
            if stat in centroids:
                yield (stat, f'{label: <{w or len(label) + 1}}'
                             f'$(w={next(weights)}) ${next(ends, "")}')
            else:
                yield stat, label


# ---------------------------------------------------------------------------- #

class TrackerVideo(VideoFeatureDisplay):

    # style = AttrDict()

    default_marker_style = {
        **VideoFeatureDisplay.default_marker_style,
        'cmap': 'rainbow',
        'emboss': 1.5
    }
    # default_marker_style.pop('edgecolor')
    # default_marker_style.pop('facecolor')

    label_style = {
        'offset': 5,
        'size': 'x-small',
        'alpha': 1
    }

    def __init__(self, tracker, data, marker_cycle=(), marker_style=(),
                 update=True, legend=False, **kws):

        self.tracker = tracker

        # setup scatter property cycle
        marker_cycle = defaultdict(list)
        # weights = iter(tracker.feature_weights.squeeze())
        for stat in (*tracker.centrality, 'avg'):
            for key, val in zip(('color', 'marker', 'label'), CENTROIDS[stat]):
                marker_cycle[key].append(val)

        # kws passed to ImageDisplay
        kws.setdefault('clim_every', 0)

        # init video + feature marks
        _n, *shape = tracker.measurements.shape
        shape[0] += 1
        VideoFeatureDisplay.__init__(self, data, np.full(shape, np.nan),
                                     marker_cycle, marker_style, **kws)

        # Source segments
        seg = self.tracker.seg
        contour_style = dict(marker_style)
        contour_style.pop('s', None)

        self.regions = tracker.show.contours(self.ax, **contour_style)
        self.label_texts = seg.show.labels(self.ax, **self.label_style)

        if self.cbar:
            # NOTE, we don't want the cmap to switch for these when scrolling,
            # so add directly to artists
            self.cbar.scroll.artists.add(self.regions)

        # update for frame 0
        if update:
            self.update(0)

        # Link segment contours and labels to redraw upon slider move (since
        # image redraws)
        # self.sliders.add_art(self.regions, self.label_texts, self.marks)
        self.sliders.link(self.regions, self.label_texts, self.marks)

        if legend:
            self.legend()

    def get_coords(self, i):
        # logger.debug('GETCOO', i)
        tracker = self.tracker
        if np.isnan(tracker.delta_xy[i]).any():
            self.logger.debug('No measurements yet for frame {}.', i)
            tracker(self.data[i], i)

        # tracker.get_coords(i)
        if 'avg' in CENTROIDS:
            return np.vstack([tracker.measurements[i],
                              #   np.full((1, 2, 2), np.nan),
                              tracker.measure_avg[i, None]])
        return tracker.measurements[i]

    def update(self, i, draw=False):
        # logger.debug('UPDATE', i)
        # logger.debug('GRUMBLE' * np.isnan(tracker.delta_xy[i]).any())
        i = int(i)
        tracker = self.tracker
        if np.isnan(tracker.measurements[i]).any():
            self.logger.debug('No measurements yet for frame {}.', i)
            tracker(self.data[i], i)

        # update region offsets
        # print(tracker._origins[i])
        self.regions.set_offsets(-tracker._origins[i, ::-1])

        return [*super().update(i, draw), self.regions, self.label_texts]

    # def mark(self, xy, marker_cycle, emboss=1.5, alpha=1, **style):
    #     art = super().mark(xy, marker_cycle, emboss, alpha, **style)
    #     self.ax.plot()
    #     return art

    def legend(self, show_weights=True, **kws):
        spacer = self.divider.append_axes('top', 1.25, pad=0.05)
        spacer.set_axis_off()

        art = dict(zip((*self.tracker.centrality, 'avg'), zip(self.marks)))
        self.tracker.show._legend(spacer, art, show_weights,
                                  bbox_to_anchor=(-0.02, 0),)
        # labels = dict(self.tracker.show._get_legend_labels(show_weights))

        # spacer.legend(     art,
        #                     bbox_to_anchor=(-0.02, 1), **kws)
