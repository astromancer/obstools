
# std
import sys
import itertools as itt
from collections import defaultdict, deque

# third-party
import numpy as np
from mpl_multitab import MplMultiTab
from mpl_toolkits.axes_grid1.mpl_axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import AffineDeltaTransform

# local
from motley.utils import vbrace
from scrawl import scatter
from scrawl.video import VideoFeatureDisplay

# relative
from ...image import SkyImage
from . import CONFIG


# ---------------------------------------------------------------------------- #
CENTROIDS = CONFIG.plots.centroids

SUBPLOTSPEC = dict(
    bottom=0.05,
    top=0.8,
    left=0.075,
    right=0.85,
    hspace=0.05,
    wspace=0.05
)

# ---------------------------------------------------------------------------- #


# def _part_map(mapping, keys):
#     for key in keys:
#         yield key, mapping[key]


# def part_map(mapping, keys):
#     return dict(_part_map(mapping, keys))


# ---------------------------------------------------------------------------- #

class HackAxesToNotShare(Axes):
    def __init__(self, fig, rect, **kws):
        super().__init__(fig, rect, **{**kws, **dict(sharex=None, sharey=None)})


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


def add_colorbar(mappable, aspect=20, pad_fraction=0.5, **kws):
    """Add a vertical colour bar to a plot."""

    # Adapted from https://stackoverflow.com/a/33505522/1098683
    ax = mappable.axes
    divider = make_axes_locatable(ax)
    # width = axes_size.AxesY(ax, aspect=1. / aspect)
    # pad = axes_size.Fraction(pad_fraction, width)
    cax = divider.append_axes('right', size=0.1, pad=1)
    return ax.figure.colorbar(mappable, cax, **kws)

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

    def _get_figure(self, ui, label):
        if ui:
            tab = ui.add_tab(f'Source {label}')
            return tab.figure
        
        if plt := sys.modules.get('matplotlib.pyplot'):
            return plt.figure()
        
        return Figure()

    def positions(self, labels=None, section=slice(None),
                  show=(cfg := CONFIG.plots.position).show,  # 'nulls',
                  legend=cfg.legend, figsize=cfg.figsize, **kws):
        """
        For sets of measurements (m, n, 2), plot each (n, 2) feature set as on
        its own axes as scatter / density plot. Additionally plot the shifted
        points in the neighbouring axes to the right. Mark the pixel size and
        diffraction limit for reference.
        """
        # trk = self.tracker

        # print(figsize, 'BITCHES')

        if labels is None:
            labels = self.tracker.use_labels

        ui = None
        if len(labels) > 1:
            ui = MplMultiTab(title='Position Measurements', pos='WN')

        art = {}
        for label in labels:
            fig = self._get_figure(ui, label)

            art[label] = self._positions_source(
                fig, label - 1, section, show, legend, **kws
            )
            # fig.tight_layout()
        # fig.set_size_inches(figsize)
        return ui, art

    def _positions_source(self, fig, source, section, show, legend, **kws):

        tracker = self.tracker
        weights = dict(zip(tracker.features, tracker.feature_weights.squeeze()))
        features = list(self._get_features(show, weights))
        n_cols = len(features)

        axes = fig.subplots(2, n_cols, sharex='row', sharey='row',
                            gridspec_kw=SUBPLOTSPEC)

        for idx, ax in np.ndenumerate(axes):
            self._setup_scatter_axes(ax, idx, n_cols, features[idx[1]],
                                     'weights' in show)
            self._show_pixel(ax, **CONFIG.plots.position.pixel)
            self._show_precision(ax, **CONFIG.plots.position.precision)

        # loop features
        count = itt.count()
        art = defaultdict(list)

        scatter = dict({**CONFIG.plots.position.scatter, **kws})
        captions = dict(self._get_legend_labels('weights' in show)) if legend else {}

        for feature in features:
            # plot residuals vs shifted residuals
            color, marker, label = CENTROIDS[feature]
            art[feature] = tuple(self._compare_scatter_density(
                axes[:, next(count)],
                (section, feature, source),
                {**dict(color=color,
                        marker=marker,
                        label=captions.get(feature, label)),
                 **scatter},
                legend
            ))

        polys = next(zip(*deque(art.values())[-1]))
        # add_colorbar(polys[0])

        for i, poly in enumerate(polys):
            l, b, w, h = axes[i, -1].get_position().bounds
            cax = fig.add_axes([l + w + 0.075, b, 0.01, h])
            fig.colorbar(poly, cax=cax, label='Density')

        #     # cax = grids[i].cbar_axes[0]
        #     cax = grid.cbar_axes[i]
        #     cax.colorbar(poly, label='Density')
        #     cax.toggle_label(True)
        #     cax.axis[cax.orientation].set_label()

        # share x axes between rows
        # for row in axes:
        #     first, *rest = row
        #     for ax in rest:
        #         ax.sharex(first)
        #         ax.sharey(first)

        #
        # view = axes[1, 0].viewLim.get_points()
        # view = view.mean(0) + view.ptp(0).max() * np.array([[-0.5], [0.5]])
        # axes[1, 0].viewLim.set_points(view)


#        **CONFIG.plots.position.cbar
        # cbar.ax.set_ylabel('Density')

        # x, y = pprint.uarray(tracker.coords[source], tracker.sigma_xy[source], 2)
        # Source {tracker.use_labels[source]}:
        # cap = dict(CONFIG.plots.position.caption)
        # fig.text(*cap.pop('pos'), cap.pop('text').format(x=x, y=y), **cap)

        return art
        # if legend:
        #     self._legend(axes[0, 0], art, show_weights)

    def _setup_scatter_axes(self, ax, index, n_cols, feature, show_weights):

        row, col = index
        ax.tick_params(
            bottom=True, top=True, left=True, right=True,
            labelright=(right := (col == n_cols - 1)),
            labelleft=(left := (col == 0)),
            labeltop=(top := (row == 0)),
            labelbottom=(bot := (row == 1)),
            direction='inout',
            length=7
        )

        # for axx in (ax.xaxis, ax.yaxis):
        #     axx.set_major_locator(IntLocator(1, 0))
        #     axx.set_major_formatter(ticker.FuncFormatter(_format_int))

        # ax.format_coord=format_coord

        delta = '' if top else ' - \delta'
        ax.set(xlabel=f'$x{delta}$')  # , aspect='equal'

        if top:
            ax.xaxis.set_label_position('top')
            
            title = CONFIG.plots.centroids[feature][-1]
            if show_weights:
                title += '\n'
                if (i := index[1]) < len(weights := self.tracker.feature_weights):
                    title += f'(w = {weights[i].item():3.2f})'
            ax.set_title(title, size='small', fontweight='bold')
            

        if left or right:
            ax.set_ylabel(f'$y{delta}$')
        if right:
            ax.yaxis.set_label_position('right')

        ax.grid()

    def _get_features(self, show, weights):

        for feature in CENTROIDS:
            #
            if feature == 'avg':
                if 'avg' not in show:
                    continue

            # check requested feature was part of compute
            elif (weight := weights.get(feature)) is not None:
                # have weight
                if not (weight or ('nulls' in show)):
                    # zero weight
                    continue
            #
            else:
                raise ValueError(f'Unknown feature {feature!r}')

            yield feature

    def _compare_scatter_density(self, axes, index,
                                 scatter_kws=CONFIG.plots.position.scatter,
                                 legend=False, **kws):

        #
        scatter_kws = dict(CONFIG.plots.position.density,
                           scatter_kws=scatter_kws,
                           **kws)
        for i, ax in enumerate(axes):
            data = self.tracker.get_coords_residual(*index, bool(i))
            _, *art = scatter.density(ax, data, **scatter_kws)
            yield art

        if legend:
            ax2.legend(loc='upper left')  # , bbox_to_anchor=(0, 1.05)

    def _show_pixel(self, ax, pos=(0, 0), **style):
        # add pixel size rect
        r = Rectangle(np.array(pos) - 0.5, 1, 1, **style)
        ax.add_patch(r)
        return r

    def _show_precision(self, ax, pos=(0, 0), **style):
        # add pixel size rect
        c = Circle(pos, self.tracker.precision, **style)
        ax.add_patch(c)
        return c

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
        centroids = self.tracker.features
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
        for stat in (*tracker.features, 'avg'):
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
            tracker(self.data, i)

        # update region offsets
        # print(tracker._origins[i])
        self.regions.set_offsets(tracker._origins[i, ::-1])

        return [*super().update(i, draw), self.regions, self.label_texts]

    # def mark(self, xy, marker_cycle, emboss=1.5, alpha=1, **style):
    #     art = super().mark(xy, marker_cycle, emboss, alpha, **style)
    #     self.ax.plot()
    #     return art

    def legend(self, show_weights=True, **kws):
        spacer = self.divider.append_axes('top', 1.25, pad=0.05)
        spacer.set_axis_off()

        art = dict(zip((*self.tracker.features, 'avg'), zip(self.marks)))
        self.tracker.show._legend(spacer, art, show_weights,
                                  bbox_to_anchor=(-0.02, 0),)
        # labels = dict(self.tracker.show._get_legend_labels(show_weights))

        # spacer.legend(     art,
        #                     bbox_to_anchor=(-0.02, 1), **kws)
