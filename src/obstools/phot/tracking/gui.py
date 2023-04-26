

# std
from motley.utils import vbrace
from matplotlib.transforms import AffineDeltaTransform
from collections import defaultdict
from collections import namedtuple

# third-party
import numpy as np
from loguru import logger
from matplotlib.lines import Line2D
from mpl_multitab import MplMultiTab

# local
from pyxides.typing import ListOf
from recipes.dicts import AttrDict as ArtistContainer
from scrawl.image import Image3D
from scrawl.video import VideoFeatureDisplay
from scrawl.moves.callbacks import CallbackManager, mpl_connect

# relative
from ...image.image import ImageContainer, SkyImage
from ..config import CONFIG

# ---------------------------------------------------------------------------- #

# def _draw_if_stale(canvas):
#     if getattr(canvas, 'stale', False):
#         canvas.draw()

# ---------------------------------------------------------------------------- #


class LegendCallbacks(CallbackManager):
    """
    Enables toggling marker / bar / cap visibility by selecting on the legend.
    """

    def __init__(self, figure, art, proxies, states=None, use_blit=False):
        """enable legend picking"""

        n = len(art)
        if states is None:
            states = np.ones(n, bool)

        for item in ('proxies', 'states'):
            if n != len(eval(item)):
                raise ValueError(f'Unequal number of artists and {item}.')

        self.artists = art
        self.proxies = proxies

        # initialize auto-connect
        CallbackManager.__init__(self, figure.canvas)

        # create mapping between the legend artists (markers), and the
        # original (axes) artists
        self.to_orig = {}
        for handel, origart, stat in zip(proxies, self.artists, states):
            # if handel.get_picker() is None:
            #     logging.warning('Setting pick radius=10 for %s' % handel)
            self.to_orig[handel] = origart
            if not stat:
                self.toggle_vis(origart, handel)

    @mpl_connect('pick_event')
    def on_pick(self, event):
        """Pick event handler."""
        self.logger.debug('pick')
        prx = event.artist
        if prx in self.to_orig:
            art = self.to_orig[event.artist]
            self.toggle_vis(art, prx)
            self.logger.debug('toggled', art)

    def toggle_vis(self, art, proxy):
        """
        on the pick event, find the orig line corresponding to the
        legend proxy line, and toggle the visibility.
        """
        fig = self.canvas.figure
        if self.use_blit:
            art.set_animated = True
            proxy.set_animated = True
            background = fig.canvas.copy_from_bbox(fig.bbox)

        # Toggle vis of axes artists
        vis = not art.get_visible()
        art.set_visible(vis)

        # set alpha of legend artist
        proxy.set_alpha(1.0 if vis else 0.2)

        if self.use_blit:
            fig.canvas.restore_region(background)
            art.draw(fig.canvas.renderer)
            proxy.draw(fig.canvas.renderer)
            fig.canvas.blit(fig.bbox)
        else:
            fig.canvas.draw()

# ---------------------------------------------------------------------------- #


class SourceImage(SkyImage):

    def _init_art(self):
        return ArtistContainer(image3d=None, centroids=(), regions=None)

    def show(self, fig, markers, cmap):

        logger.info('Initializing source 3D plots.')
        self.art.image3d = im3 = Image3D(self.data, self.origin, figure=fig, cmap=cmap)

        logger.info('Marking centroids.')
        self.art.centroids = []
        for marker, label in markers.items():
            im3.axi.autoscale(False)
            self.art.centroids.extend(
                im3.axi.plot(0, 0, marker, mfc='none', label=label)
            )
        # # self.art.centroids.
        # im3.axi.plot(0, 0, 'kx', label='Mean Centroid')

        # Regions
        self.art.regions = (Line2D([0], [0]))
        im3.axi.add_line(self.art.regions)

        # blitting
        im3.image.cbar.scroll.add_art(self.art.centroids, self.art.regions)
        # scrolls.append(im3.image.cbar.scroll)

        # legend
        im3.axi.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05))

    # def init_source(self, img, cmap):
    #     # seg = self.tracker.seg

    #     # fig = self.ui.add_tab('Sources', str(lbl))

    #     # im3.fig.tight_layout
    #     for j, (marker, label) in enumerate(self.centroid_style.values()):
    #         im3.axi.autoscale(False)
    #         self.centroid_marks[j, i], = \
    #             im3.axi.plot(0, 0, marker, mfc='none', label=label)
    #         # im3.axi.autoscale(True)

    #     self.centroid_means[i], = cm = \
    #         im3.axi.plot(0, 0, 'kx', label='Mean Centroid')

    #     # Regions
    #     self.im3_regions[i] = im3r = (Line2D([0], [0]))
    #     im3.axi.add_line(im3r)


class SourceImages(ImageContainer, ListOf(SourceImage)):
    pass


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
                 update=True, **kws):

        self.tracker = tracker

        # setup scatter property cycle
        marker_cycle = defaultdict(list)
        weights = iter(tracker.feature_weights.squeeze())

        for stat, style in CONFIG.centroids.items():
            style = dict(zip(('marker', 'color', 'label'), style))
            for key, val in style.items():
                marker_cycle[key].append(val)

        # kws passed to ImageDisplay
        kws.setdefault('clim_every', 0)

        # init video + feature marks
        _n, *shape = tracker.measurements.shape
        VideoFeatureDisplay.__init__(self, data, np.full(shape + 1, np.nan),
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
        #self.sliders.add_art(self.regions, self.label_texts, self.marks)
        self.sliders.link(self.regions, self.label_texts, self.marks)

    def get_coords(self, i):
        # logger.debug('GETCOO', i)
        tracker = self.tracker
        if np.isnan(tracker.delta_xy[i]).any():
            self.logger.debug('No measurements yet for frame {}.', i)
            tracker(self.data[i], i)

        # tracker.get_coords(i)
        if 'avg' in CONFIG.centroids:
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
        self.regions.set_offsets(tracker._origins[i])

        return [*super().update(i, draw), self.regions, self.label_texts]

    # def mark(self, xy, marker_cycle, emboss=1.5, alpha=1, **style):
    #     art = super().mark(xy, marker_cycle, emboss, alpha, **style)
    #     self.ax.plot()
    #     return art

    def legend(self, **kws):
        spacer = self.divider.append_axes('top', 1.25, pad=0.05)
        spacer.set_axis_off()
        art = dict(zip(self.tracker.centroids, zip(*self.marks)))
        self.tracker.legend(spacer,
                            art,
                            bbox_to_anchor=(-0.02, 1), **kws)


class SourceTrackerGUI(TrackerVideo):

    def __init__(self, tracker, hdu, cmap=None, **kws):

        self.ui = ui = MplMultiTab(title=self.__class__.__name__)

        # NOTE: blit does not work if passing figure in to add_tab... ?
        tab = ui.add_tab('Image', hdu.file.name)

        # _, nx, ns, _ = tracker.measurements.shape
        # self.centroid_marks = np.empty((nx, ns), object)
        # self.im3, self.im3_regions, self.centroid_means = np.empty((3, ns), object)

        self.source_images = SourceImages()
        for lbl, (ysec, xsec), seg in tracker.seg.cutouts(
                tracker.seg.data, labelled=True, with_slices=True):
            # create figures (but don't plot yet)
            self.ui.add_tab('Sources', str(lbl))
            self.source_images.append(
                SourceImage(np.zeros(seg.shape),
                            scale=hdu.pixel_scale,
                            origin=(ysec.start - 0.5, xsec.start - 0.5),
                            segmentation=seg)
            )

        # init main display
        super().__init__(tracker, hdu.calibrated, fig=tab.figure, update=False,
                         **kws)

        # Init art for focus tab
        focus = 0
        ui['Sources'].tabs.setCurrentIndex(0)
        self.plot_source(focus, cmap)

        # connect signal to redraw on tab change if needed
        # TODO: blit here?
        # ui.groups.tabs.currentChanged.connect(self._redraw_group_on_change)
        # for g in ui.groups._items.values():
        #     g.tabs.currentChanged.connect(self._redraw_tab_on_change)

        # set data for frame 0
        self.update(0)

    def plot_source(self, lbl, cmap=None):

        img = self.source_images[lbl]
        if img.art.get('image3d'):
            return

        markers = dict(self.centroid_style.values(), kx='Mean Centroid')
        img.show(self.ui['Sources', lbl].figure, markers, cmap)

        # connect the cmaps so that all are the same
        xscroll = img.art.image3d.image.cbar.scroll
        for img1 in self.source_images:
            if other := img1.art.image3d:
                oscroll = other.image.cbar.scroll
                xscroll.on_scroll.add(self.link_cmaps, oscroll)
                oscroll.on_scroll.add(self.link_cmaps, xscroll)

        self.cbar.scroll.on_scroll.add(self.link_cmaps, xscroll)

    def link_cmaps(self, _event, scroll1, scroll2):
        scroll2.set_cmap(scroll1.mappable.get_cmap())
        scroll2.canvas.stale = True

    # def _redraw_group_on_change(self, i):
    #     self.logger.debug('redraw')
    #     mgr = self.ui[i - 1]
    #     _draw_if_stale(mgr[mgr.tabs.currentIndex()].canvas)

    # def _redraw_tab_on_change(self, i):
    #     self.logger.debug('redraw')
    #     mgr = self.ui[self.ui.groups.tabs.currentIndex() - 1]
    #     _draw_if_stale(mgr[i].canvas)

    def show(self):
        self.ui.show()

    def set_cmap(self, cmap):
        self.cbar.set_cmap(cmap)
        for im3 in self.im3:
            im3.set_cmap(cmap)

    def set_image_data(self, image):

        draw_list = super().set_image_data(image)

        self.set_source_images(image)
        # - np.ma.median(self.tracker.seg.mask_sources(image)

        return draw_list

    def set_source_images(self, image):
        for i, img in enumerate(self.tracker.seg.cutouts(image)):
            if im3 := self.source_images[i].art.image3d:
                im3.set_data(img)
                z0 = img.min()
                im3.bars.bars.set_z0(z0)
                im3.bars.bars.set_z(img - z0)

    def update(self, i, draw=False):
        draw_list = super().update(i, draw)
        i = self.frame  # rounded and wrapped

        reg = self.regions
        segments = reg.get_segments()
        colours = reg.cmap(reg._norm(reg._A))
        centroids = self.tracker.measurements[i]
        avg = self.tracker.measure_avg[i]
        # coords = self.get_coords(i)

        for j, img in enumerate(self.source_images):
            if img.art.centroids:
                if j in self.tracker.use_labels:
                    for line, xy in zip(img.art.centroids, centroids[:, j]):
                        line.set_data(xy)

                # This is the avg
                img.art.centroids[-1].set_data(avg[j])

            if img.art.regions:
                # update segment contours for source plot
                img.art.regions.set(data=segments[j].T, color=colours[j])

        return draw_list


SourceTrackerGui = SourceTrackerGUI
ApertureContainer = namedtuple('ApertureContainer', ('stars', 'sky'))
