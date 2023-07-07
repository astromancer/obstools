
# std
import sys
import itertools as itt
import functools as ftl
import contextlib as ctx
from collections import defaultdict, deque

# third-party
import numpy as np
from mpl_multitab import MplMultiTab
from bottleneck import nanmax, nanmin
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import AffineDeltaTransform

# local
from motley.utils import vbrace
from recipes import pprint
from recipes.logging import LoggingMixin
from scrawl import density
from scrawl.video import VideoFeatureDisplay

# relative
from ...image import SkyImage
from . import CONFIG


# ---------------------------------------------------------------------------- #
__all__ = ['SourceTrackerPlots', 'TrackerVideo']


# ---------------------------------------------------------------------------- #
with ctx.suppress(AttributeError):  # autoreload hack
    CONFIG = CONFIG.plots

CENTROIDS = CONFIG.centroids
LABEL_CONFIG = CONFIG.labels
CONFIG = CONFIG.position

SUBPLOTSPEC = dict(
    bottom=0.075,
    top=0.9,
    left=0.075,
    right=0.85,
    hspace=0.025,
    wspace=0.05
)

# ---------------------------------------------------------------------------- #


# def _part_map(mapping, keys):
#     for key in keys:
#         yield key, mapping[key]


# def part_map(mapping, keys):
#     return dict(_part_map(mapping, keys))


# ---------------------------------------------------------------------------- #

# class HackAxesToNotShare(Axes):
#     def __init__(self, fig, rect, **kws):
#         super().__init__(fig, rect, **{**kws, **dict(sharex=None, sharey=None)})


def _format_int(x, pos):
    # print(f'{x=}')
    return f'{round(x):d}'
    # return f'{(-1, 1)[x > 1] * round(abs(x)):d}'
    # return f'{x:.1f}'


# IntFormatter = ticker.FuncFormatter(_format_int)


class IntLocator(ticker.IndexLocator):
    def __call__(self):
        """Return the locations of the ticks"""
        dmin, dmax = self.axis.get_view_interval()
        return self.tick_values(np.ceil(dmin), np.floor(dmax))


def format_coord(x, y):
    return f'{x = :4.3f}; {y = :4.3f}'


def nanptp(a, axis):
    return nanmax(a, axis) - nanmin(a, axis)


# ---------------------------------------------------------------------------- #


class SourceTrackerPlots(LoggingMixin):

    def __init__(self, tracker):
        self.tracker = tracker

    def image(self, image, ax=None, points='rx', contours=True,
              labels=LABEL_CONFIG, **kws):

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

    def _get_figure(self, ui=None, label='', **kws):
        if ui:
            tab = ui.add_tab(f'Source {label}', fig=kws)
            return tab.figure

        if plt := sys.modules.get('matplotlib.pyplot'):
            return plt.figure(**kws)

        return Figure(**kws)

    def positions(self, labels=None, section=slice(None),
                  show=CONFIG.show, legend=CONFIG.legend, figsize=CONFIG.figsize,
                  ui=None, **kws):
        """
        For sets of measurements (m, n, 2), plot each (n, 2) feature set as on
        its own axes as scatter / density plot. Additionally plot the shifted
        points in the neighbouring axes to the right. Mark the pixel size and
        diffraction limit for reference.
        """
        # trk = self.tracker

        if labels is None:
            labels = self.tracker.use_labels

        if len(labels) > 1 and ui or (ui is None):
            ui = MplMultiTab(title='Position Measurements', pos='WN')

        art = {}
        figures = []
        for label in labels:
            fig = self._get_figure(ui, label)
            fig.set_size_inches(figsize)
            figures.append(fig)

            art[label] = self.positions_source(
                fig, label - 1, (), section, show, legend, **kws
            )

        return (ui or figures), art

    def positions_source(self, fig, source, features=(), section=slice(None),
                         show=('weights', 'caption'), legend=False, **kws):

        self.logger.debug('Plotting position measurements for source {}.', source)

        if fig is None:
            fig = self._get_figure()

        if features:
            if isinstance(features, str):
                features = [features]
        else:
            tracker = self.tracker
            weights = dict(zip(tracker.features, tracker.feature_weights.squeeze()))
            features = list(self._get_features(show, weights))

        #
        assert (n_cols := len(features))

        if n_cols == 1:
            gridspec_kw = dict(right=0.8, top=0.82)
        else:
            gridspec_kw = SUBPLOTSPEC
            if 'caption' in show:
                SUBPLOTSPEC['bottom'] += 0.125

        axes = fig.subplots(2, n_cols, sharex='row', sharey='row',
                            gridspec_kw=gridspec_kw)
        if n_cols == 1:
            axes = axes[:, None]

        for idx, ax in np.ndenumerate(axes):
            self._setup_scatter_axes(ax, idx, n_cols, features[idx[1]],
                                     'weights' in show)
            # if idx[0]:  # bottom row
            # self._show_pixel(ax, **CONFIG.pixel)
            # self._show_precision(ax, **CONFIG.precision)

        # loop features
        count = itt.count()
        art = defaultdict(list)

        scatter = {**CONFIG.scatter, **kws}
        legends = dict(self._get_legend_labels('weights' in show)) if legend else {}

        for feature in features:
            # plot residuals vs shifted residuals
            color, marker, label = CENTROIDS[feature]
            art[feature] = tuple(self._compare_density_maps(
                axes[:, next(count)],
                (section, feature, source),
                {**dict(color=color,
                        marker=marker,
                        label=legends.get(feature, label)),
                 **scatter},
                legend,

            ))

        polys = next(zip(*deque(art.values())[-1]))
        # add_colorbar(polys[0])

        self._cid = fig.canvas.mpl_connect(
            'draw_event', ftl.partial(self._on_first_draw, polys=polys)
        )

        # Add caption
        if 'caption' in show:
            coords = self.tracker.coords
            x, y = pprint.uarray(coords['xy'][source],
                                 coords['sigma'][source], 2)
            # Source {tracker.use_labels[source]}:
            cap = dict(CONFIG.caption)
            fig.text(*cap.pop('pos'), cap.pop('text').format(x=x, y=y), **cap)

        return art
        # if legend:
        #     self._legend(axes[0, 0], art, show_weights)

    def _setup_scatter_axes(self, ax, index, n_cols, feature, show_weights):

        row, col = index
        multicol = (n_cols != 1)
        ax.tick_params(
            bottom=True, top=True, left=True, right=True,
            labelright=(right := (col == n_cols - 1)) and multicol,
            labelleft=(left := (col == 0)),
            labeltop=False,
            labelbottom=True,  # (bot := (row == 1)) or (n_cols == 1),
            direction='inout',
            length=7
        )

        # ax.format_coord=format_coord
        top = (row == 0)
        delta = '' if top else ' - \delta'
        ax.set(xlabel=f'$x - x_0{delta}$', aspect='equal')

        if top:
            ax.xaxis.set_label_position('top')

            title = CENTROIDS[feature][-1]
            if show_weights:
                title += '\n' + '\n' * (1 - title.count('\n')) * multicol

                if feature != 'avg':
                    title += f'(w = {self.tracker.get_weight(feature):3.2f})'
            ax.set_title(title, **CONFIG.title)
        else:
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(IntLocator(1, 0))
                axis.set_major_formatter(ticker.FuncFormatter(_format_int))

        if left or right:
            ax.set_ylabel(f'$y - y_0{delta}$')
        if right and n_cols != 1:
            ax.yaxis.set_label_position('right')

        ax.grid()

    def _on_first_draw(self, event, polys):

        # match axes ranges ottom axes
        ax = polys[1].axes
        view = ax.viewLim._points
        xlim, ylim = view.mean(0, keepdims=1).T + view.ptp(0).max(keepdims=1) * [-0.5, 0.5]
        ax.set(xlim=xlim, ylim=ylim)

        # add colorbar
        self._add_colorbars(polys)

        event.canvas.mpl_disconnect(self._cid)
        # event.canvas.draw()

    def _add_colorbars(self, polys):
        self.logger.debug('Adding colorbars')
        for i, poly in enumerate(polys):
            ax = poly.axes
            fig = ax.figure
            l, b, w, h = ax.get_position().bounds
            cax = fig.add_axes([l + w + 0.075, b, 0.01, h])
            fig.colorbar(poly, cax=cax, label=CONFIG.cbar[f'label{i}'])

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

    def _compare_density_maps(self, axes, index,
                              scatter_kws=CONFIG.scatter,
                              legend=False, **kws):

        #
        density_kws = {**CONFIG.density, **kws}

        for i, ax in enumerate(axes):
            data = self.tracker.get_coords_residual(*index, shifted=bool(i))

            # special case
            special_case = {}
            feature = index[1]
            if (i == 0) and (feature == 'peak'):
                special_case = dict(tessellation='rect',
                                    bins=nanptp(data, 0).astype(int))

            # plot
            _, *art = density.scatter_map(ax, data, scatter_kws=scatter_kws,
                                          **{**density_kws, **special_case})

            yield art

        if legend:
            ax.legend(loc='upper left')  # , bbox_to_anchor=(0, 1.05)

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
            return np.vstack([tracker.measurements['xy'][i],
                              #   np.full((1, 2, 2), np.nan),
                              tracker.soure_info['xy'][i, None]])
        return tracker.measurements[i]

    def update(self, i, draw=False):
        # logger.debug('UPDATE', i)
        # logger.debug('GRUMBLE' * np.isnan(tracker.delta_xy[i]).any())
        i = int(i)
        tracker = self.tracker
        if np.isnan(tracker.measurements['xy'][i]).any():
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
