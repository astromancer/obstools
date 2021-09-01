"""
Plotting mosaics of partially overlapping images
"""

# std
import itertools as itt

# third-party
import numpy as np
import more_itertools as mit
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

# local
from scrawl.imagine import ImageDisplay

# relative
from . import transforms
from .image import ImageContainer, SkyImage


def get_corners(p, fov):
    """Get corners relative to DSS coordinates. xy coords anti-clockwise"""
    c = np.array([[0, 0], fov[::-1]])  # lower left, upper right xy
    # corners = np.c_[c[0], c[:, 1], c[1], c[::-1, 0]].T  # / clockwise yx
    corners = np.c_[c[0], c[::-1, 0], c[1], c[:, 1]].T  # / clockwise xy
    corners = transforms.rigid(corners, p)
    return corners


def ulc(p, fov):
    """
    Get upper left corner given rigid transform parameters and image field of
    vi
    """
    return transforms.rigid([0, fov[0]], p).squeeze()


# def get_ulc(params, fovs):
#     """
#     Get upper left corners of all frames given roto-translation
#     parameters and field of view
#     """
#     ulc = np.empty((len(params), 2))
#     for i, (p, fov) in enumerate(zip(params, fovs)):
#         ulc_ = np.array([[0, fov[0]]])
#         ulc[i] = trf.rigid(ulc_, p)
#     return ulc[:, 0].min(), ulc[:, 1].max()  # xy


class MosaicPlotter(ImageContainer):
    """
    Plot the results from image registration run. This class is designed to work
    with small images that typically span a tens of arcminutes or so and hence
    assumes an affine transform between image pixel coordinates and sky frame
    coordinates. Images are displayed in ICRS coordinates with right ascension
    (RA) decreasing along the x-axis and declination (DEC) increasing along on
    the y-axis. The axes coordinates are always rectilinear and the images are
    rotated, scaled and translated. This allows for images with different
    orientations to be viewed on the same axis. Note that the functionality here
    is subtly different from the provided by `astropy.visualization.WCSAxes`
    which always displays the image "upright" and transforms the grid lines.

    This class also allows one to scroll through images with the mouse wheel in
    order to check the individual alignments. Reset to the original mosaic by
    clicking the center mouse button.
    """

    # TODO: make so that editing params, fovs changes plot live!!
    #  ===>  will need class ParameterDescriptor for this since params is a
    #         mutable array
    # TODO: optional normalize and same clims
    # TODO: use WCSAxes ??

    default_cmap_ref = 'Greys'
    alpha_cycle_value = 0.65

    label_fmt = 'image%i'
    label_props = dict(color='w')

    # @property
    # def fig(self):
    #     if self._fig is None:
    #         self.setup()
    #     return self._fig

    # @property
    # def ax(self):
    #     if self._ax is None:
    #         self.setup()
    #     return self._ax

    @property
    def names(self):
        return list(self.art.keys())

    @classmethod
    def from_register(cls, reg, axes=None, scale='sky', show_ref_image=True):
        """
        Construct from `ImageRegister`
        """
        scale = scale.lower()
        if scale in ('fov', 'sky', 'world'):
            rscale = 1
            oscale = reg.pixel_scale
        elif scale.startswith('pix'):
            rscale = reg.pixel_scale
            oscale = 1
        else:
            raise ValueError(f'Invalid scale: {scale!r}.')

        # image offsets are in units of pixels by default. convert to units of
        # `fov` (arcminutes)
        images = []
        for image in reg.data:
            new = image.copy()
            new.scale = image.scale / rscale
            new.offset = image.offset * oscale
            images.append(new)

        return cls(images, (), axes, show_ref_image, reg.primary)

    def __init__(self, images, fovs=(), axes=None, show_ref_image=True, ridx=0):
        """
        Initialize with sequence `images` of :class:`SkyImages` or sequence
        image arrays `np.ndarray` and sequence `fovs` of field-of-views 
        """

        #
        ImageContainer.__init__(self, images, fovs)
        self.art = {}  # art
        self.image_label = None

        self._counter = itt.count()
        self._ff = self._fig = self._ax = None
        self._low_lims = (np.inf, np.inf)
        self._up_lims = (-np.inf, -np.inf)

        # scrolling: index -1 represents the blended image "mosaic".
        # Scrolling forward will show each image starting at 0
        self.idx = ridx  # reference
        self.alpha_cycle = []
        self._idx_active = -1
        self.show_ref_image = bool(show_ref_image)

        # setup figure
        if axes is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = axes
            self.fig = axes.figure

        self.fig.tight_layout()

        # connect gui events
        self.fig.canvas.mpl_connect('scroll_event', self._scroll_safe)
        self.fig.canvas.mpl_connect('button_press_event', self.reset)

    def __hash__(self):
        # this is a HACK so that we can connect methods of this class to the
        # mpl callback registry which does not allow non-hashable objects
        return 0

    # def __call__()

    def mosaic(self, names=(), params=(),
               cmap=None, cmap_ref=default_cmap_ref,
               alpha=None, alpha_ref=1,
               **kws):
        """Create a mosaiced image"""

        # choose alpha based on number of images
        n = len(self)
        if alpha is None:
            alpha = max(min(1 / n, 0.5), 0.2)

        # loop images and plot
        cmaps = mit.padded([cmap_ref], cmap)
        alphas = mit.padded([alpha_ref], alpha)
        for image, p, name in itt.zip_longest(self, params, names):
            # self.plot_image(image, fov, p, name, coo, cmap=cmap, **kws)
            self.plot_image(image, None, p, name,
                            cmap=next(cmaps), alpha=next(alphas),
                            **kws)

        # always keep reference image for comparison when scrolling images
        self.alpha_cycle = np.vstack([np.eye(n) * self.alpha_cycle_value,
                                      np.ones(n) * alpha, ])
        # NOTE might have to fix this cycle if idx != 0
        if self.show_ref_image:
            self.alpha_cycle[:-1, 0] = alpha_ref

    def plot_image(self, image=None, fov=None, p=(0, 0, 0), name=None,
                   frame=True, **kws):
        """

        Parameters
        ----------
        image
        fov
        p
        name
        frame
        kws

        Returns
        -------

        """
        if p is None:
            p = image.params if isinstance(image, SkyImage) else (0, 0, 0)

        if not np.isfinite(p).all():
            raise ValueError('Received non-finite parameter value(s)')

        update = True
        if image is None:
            assert self.images
            image = self[self.idx]
            update = False
            # name = name or (self.names[0] if len(self.names) else None)

        # if image.__class__.__name__ != 'SkyImage':
        *offset, angle = p
        image = SkyImage(image, fov, offset, angle)

        # name image
        if name is None:
            name = self.label_fmt % next(self._counter)

        # plot
        # *image.offsets, image.angle = p
        art = self.art[name] = image.plot(self.ax,
                                          frame=frame, set_lims=False,
                                          **kws)

        # if coords is not None:
        #     line, = self.ax.plot(*coords.T, 'x')
        # plot_points.append(line)

        if update:
            self.update_axes_limits(p, image.fov)

        return art

    def update_axes_limits(self, p, fov, delta=0.01):
        """"""
        corners = get_corners(p, fov)
        self._low_lims = np.min([corners.min(0), self._low_lims], 0)
        self._up_lims = np.max([corners.max(0), self._up_lims], 0)
        # expand limits slightly beyond corners for aesthetic
        xlim, ylim = zip(self._low_lims * (1 - delta),
                         self._up_lims * (1 + delta))
        self.ax.set(xlim=xlim, ylim=ylim)

    def mark_sources(self, xy, marker='x', number=True, color='c',
                     xy_offset=(0, 0)):
        """
        Display the coordinates of the sources

        Returns
        -------

        """
        # show markers
        lines = []
        if marker:
            lines.extend(
                self.ax.plot(*xy.T, marker, color=color)
            )

        if number:
            for i, xy in enumerate(xy):
                lines.extend(
                    self.ax.plot(*(xy + xy_offset), ms=[7, 10][i >= 10],
                                 marker=f'${i}$', color=color)
                )
        return lines

    def mark_target(self, name='', xy=None, colour='forestgreen',
                    arrow_size=10, arrow_head_distance=2.5, arrow_offset=(0, 0),
                    text_offset=3, **text_props):

        # TODO: determine arrow_offset automatically by looking for peak
        """

                      ||     ^
                      ||     |
                     _||_    | arrow_size
                     \  /    |
                      \/     v
                             ↕ arrow_head_distance
                      ✷  


        Parameters
        ----------
        name
        colour
        arrow_size
        arrow_head_distance:
            distance in arc seconds from source location to the arrow head
            point for both arrows
        arrow_offset:
            xy offset in arc seconds for arrow point location
        text_offset:
            xy offset in arc seconds


        Returns
        -------

        """
        # assert self.reg.targetCoordsPix is not None

        import numbers
        import matplotlib.patheffects as path_effects

        assert isinstance(arrow_head_distance, numbers.Real), \
            '`arrow_offset` should be float'

        # convert to arcminutes
        # self[self.idx].pixel_scale

        def to_arcmin(val):
            return np.array(val) / 60

        arrow_size = to_arcmin(arrow_size)
        arrow_head_distance = to_arcmin(arrow_head_distance)
        arrow_offset = to_arcmin(arrow_offset)
        text_offset = to_arcmin(text_offset)

        # target indicator arrows
        if xy is None:
            xy = self.fovs[0] / 2

        xy_target = xy + arrow_offset

        arrows = []
        for i in np.eye(2):
            # quick and easy way to create arrows with annotation
            xy = xy_target + arrow_head_distance * i
            ann = self.ax.annotate('', xy, xy + arrow_size * i,
                                   arrowprops=dict(arrowstyle='simple',
                                                   fc=colour)
                                   )
            arrows.append(ann.arrow_patch)

        # text
        txt = self.ax.text(*(xy_target + text_offset), name,
                           color=colour, **text_props)
        # add border around text to make it stand out (like the arrows)
        txt.set_path_effects(
            [path_effects.Stroke(linewidth=2, foreground='black'),
             path_effects.Normal()])

        return txt, arrows

    def set_cmap(self, cmap, cmap_ref=None):
        if cmap_ref is None:
            cmap_ref = next(iter(self.art.values()))['image'].get_cmap()

        images = next(zip(*self.art.values()))
        for image, cmap in zip(images, mit.padded([cmap_ref], cmap)):
            image.set_cmap(cmap)

    def _set_alphas(self):
        """highlight a particular image in the stack"""
        alphas = self.alpha_cycle[self._idx_active]

        # position 0 represents the mosaic
        if self.image_label:
            self.image_label.set_visible(self._idx_active != len(self.art))

        for i, image in enumerate(self):
            for name, artist in image.art.items():
                artist.set_alpha(alphas[i])
                if i == self._idx_active:
                    artist.set_zorder(1)
                else:
                    # set image z-order 0, frame z-order 1
                    artist.set_zorder(name == 'frame')

    def _scroll(self, event):
        """
        This method allows you to scroll through the images in the mosaic
        using the mouse.
        """
        # if self.image_label is None:
        # self.image_label = self.ax.text(0, 0, '', color='w', alpha=0,
        #                                 rotation_mode='anchor',
        #                                 va='top')

        if not (event.inaxes or len(self.art)):  #
            return

        # set alphas
        self._idx_active += [-1, +1][event.button == 'up']  #
        self._idx_active %= len(self)   # wrap

        #
        i = self._idx_active
        image = self[i]
        if self.image_label is None:
            self.image_label = image.label_image(self.ax)

        self.image_label.set_text(f'{i}: {self.names[i]}')
        xy = ulc(self.params[i], 0.98 * self.fovs[i])
        self.image_label.set_position(xy)

        #
        self._set_alphas()

        # if self.image_label is not None:
        #     self.image_label.remove()
        #     self.image_label = None

        # set tiles
        # if self._idx_active != len(self.art):
        #     # position -1 represents the original image

        # redraw
        self.fig.canvas.draw()

    def _scroll_safe(self, event):
        try:
            # print(vars(event))
            self._scroll(event)

        except Exception as err:
            self.logger.exception(
                f'Scroll failed: {len(self)=} {self._idx_active=}'
                )

            self.image_label = None
            self.fig.canvas.draw()

    def reset(self, event):
        # reset original alphas
        if event.button == 2:
            self._idx_active = -1
            self._set_alphas()
            self.image_label.set_visible(False)

        # redraw
        self.fig.canvas.draw()
