import itertools as itt

import numpy as np
from graphing.imagine import ImageDisplay

from obstools.image.registration import (ImageRegistrationDSS,
                                         ImageContainer, SkyImage)
import obstools.image.transforms as trf

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D


def get_corners(p, fov):
    """Get corners relative to DSS coordinates. xy coords anti-clockwise"""
    c = np.array([[0, 0], fov[::-1]])  # lower left, upper right xy
    # corners = np.c_[c[0], c[:, 1], c[1], c[::-1, 0]].T  # / clockwise yx
    # corners = roto_translate_yx(corners, p)
    corners = np.c_[c[0], c[::-1, 0], c[1], c[:, 1]].T  # / clockwise xy
    corners = trf.rigid(corners, p)
    return corners  # return xy ! [:, ::-1]


def _get_ulc(p, fov):
    return trf.rigid([0, fov[0]], p).squeeze()


def get_ulc(params, fovs):
    """
    Get upper left corners of off all frames given roto-translation
    parameters and field of view
    """
    ulc = np.empty((len(params), 2))
    for i, (p, fov) in enumerate(zip(params, fovs)):
        ulc_ = np.array([[0, fov[0]]])
        ulc[i] = trf.rigid(ulc_, p)
    return ulc[:, 0].min(), ulc[:, 1].max()  # xy


def plot_transformed_image(ax, image, fov=None, p=(0, 0, 0), frame=True,
                           set_lims=True, **kws):
    """

    Parameters
    ----------
    ax
    image
    fov
    p
    frame
    set_lims
    kws

    Returns
    -------

    """

    kws.setdefault('hist', False)
    kws.setdefault('sliders', False)
    kws.setdefault('cbar', False)

    # plot
    im = ImageDisplay(image, ax=ax, **kws)
    art = im.imagePlot

    # set extent
    if fov is None:
        fov = image.shape

    extent = np.c_[[0., 0.], fov[::-1]]
    pixel_size = np.divide(fov, image.shape)
    half_pixel_size = pixel_size / 2
    extent -= half_pixel_size[None].T  # adjust to pixel centers...
    art.set_extent(extent.ravel())

    # Rotate the image by setting the transform
    *xy, theta = p  # * fov, p[-1]
    art.set_transform(Affine2D().rotate(theta).translate(*xy) +
                      art.get_transform())

    if frame:
        from matplotlib.patches import Rectangle

        frame_kws = dict(fc='none', lw=0.5, ec='0.5', alpha=kws.get('alpha'))
        if isinstance(frame, dict):
            frame_kws.update(frame)

        ax.add_patch(
            Rectangle(xy - half_pixel_size, *fov[::-1], np.degrees(theta),
                      **frame_kws)
        )

    if set_lims:
        delta = 1 / 100
        c = get_corners(p, fov)
        xlim, ylim = np.vstack([c.min(0), c.max(0)]).T * (1 - delta, 1 + delta)
        im.ax.set(xlim=xlim, ylim=ylim)

    return art


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

    default_frame = dict(lw=1, ec='0.5')

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
    def from_register(cls, reg,  axes=None, keep_ref_image=True):
        """Construct from `ImageRegister`"""
        return cls(reg.data, (), axes, keep_ref_image, reg.idx)

    def __init__(self, images, fovs=(), axes=None, keep_ref_image=True, ridx=0):
        """
        Initialize with sequence `images` of :class:`SkyImages` or sequence
        image arrays `np.ndarray` and sequence `fovs` of field-of-views 
        """

        #
        ImageContainer.__init__(self, images, fovs)

        # self.names = names
        self.params = []

        # todo: _fov_internal = self.reg.image.shape
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
        self.keep_ref_image = bool(keep_ref_image)

        # setup figure
        if axes is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = axes
            self.fig = axes.fig

        self.fig.tight_layout()

        # connect gui events
        self.fig.canvas.mpl_connect('scroll_event', self._scroll_safe)
        self.fig.canvas.mpl_connect('button_press_event', self.reset)

    def __hash__(self):
        # this is a cheat so that we can connect methods of this class to the
        # mpl callback registry which does not allow non-hashable objects
        return 0

    def mosaic(self, params, names=(), **kws):
        """Create a mosaiced image"""

        cmap = kws.pop('cmap_ref', self.default_cmap_ref)
        cmap_other = kws.pop('cmap', None)

        # choose alpha based on number of images
        n = len(self)
        alpha_magic = max(min(1 / n, 0.5), 0.2)
        alpha = kws.setdefault('alpha', alpha_magic)

        # loop images and plot
        for image, p, name in itt.zip_longest(self, params, names):
            # self.plot_image(image, fov, p, name, coo, cmap=cmap, **kws)
            self.plot_image(image, None, p, name, cmap=cmap, **kws)
            cmap = cmap_other

        # always keep reference image for comparison when scrolling images
        self.alpha_cycle = np.vstack([np.eye(n) * self.alpha_cycle_value,
                                      np.ones(n) * alpha, ])
        # NOTE might have to fix this cycle if idx != 0
        if self.keep_ref_image:
            self.alpha_cycle[:-1, 0] = 1

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

        if not np.isfinite(p).all():
            raise ValueError('Received non-finite parameter value(s)')

        update = True
        if image is None:
            assert len(self.images)
            image = self[self.idx]
            update = False
            # name = name or (self.names[0] if len(self.names) else None)

        #
        # if not isinstance(image, SkyImage):
        if not image.__class__.__name__ == 'SkyImage':
            image = SkyImage(image, fov)

        #
        if name is None:
            name = self.label_fmt % next(self._counter)

        # plot
        # image = image / image.max()
        kws.setdefault('frame', self.default_frame)
        art = self.art[name] = image.plot(self.ax, p, **kws)
        self.params.append(p)

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
        s = []
        if marker:
            s.extend(
                self.ax.plot(*xy.T, marker, color=color)
            )

        if number:
            for i, xy in enumerate(xy):
                s.extend(
                    self.ax.plot(*(xy + xy_offset), ms=[7, 10][i >= 10],
                                 marker=f'${i}$', color=color)
                )
        return s

    def mark_target(self, xy, name, colour='forestgreen', arrow_size=10,
                    arrow_head_distance=2.5, arrow_offset=(0, 0),
                    text_offset=3, **text_props):

        # TODO: determine arrow_offset automatically by looking for peak
        """

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
            xy offset in reg pixels


        Returns
        -------

        """
        # assert self.reg.targetCoordsPix is not None

        import numbers
        import matplotlib.patheffects as path_effects

        assert isinstance(arrow_head_distance, numbers.Real), \
            '`arrow_offset` should be float'

        # target indicator arrows
        # pixel_size_arcsec =
        arrow_offset = arrow_offset / self[self.idx].pixel_scale / 60
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

    def label_image(self, name='', p=(0, 0, 0), fov=(0, 0), **kws):
        # default args for init
        _kws = {}
        _kws.update(self.label_props)
        _kws.update(kws)
        return self.ax.text(*_get_ulc(p, fov), name,
                            rotation=np.degrees(p[-1]),
                            rotation_mode='anchor',
                            va='top',
                            **_kws)

    # def label(self, indices, xy_offset=(0, 0), **text_props):
    #     texts = {}
    #     params = np.array(self.reg.params[1:])
    #     fovs = np.array(self.reg.fovs[1:])
    #     for name, idx in indices.items():
    #         xy = np.add(get_ulc(params[idx], fovs[idx])[::-1],
    #                     xy_offset)
    #         angle = np.degrees(params[idx, -1].mean())
    #         texts[name] = self.ax.text(*xy, name,
    #                                    rotation=angle,
    #                                    rotation_mode='anchor',
    #                                    **text_props)
    #     return texts

    # def label_images(self):
    #     for i, im in enumerate(self.reg.images):
    #         name = f'{i}: {self.names[i]}'
    #         p = self.reg.params[i]
    #         xy = np.atleast_2d([0, im.shape[0]])  # self.reg.fovs[i][0]
    #         xy = trf.rigid(xy, p).squeeze()
    #
    #         # print(xy.shape)
    #         assert xy.shape[0] == 2
    #
    #         self.image_labels.append(
    #                 self.ax.text(*xy, name, color='w', alpha=0,
    #                              rotation=np.degrees(p[-1]),
    #                              rotation_mode='anchor',
    #                              va='top')
    #         )

    def _set_alphas(self):
        """highlight a particular image in the stack"""
        alphas = self.alpha_cycle[self._idx_active]

        # position 0 represents the mosaic
        if self.image_label:
            self.image_label.set_visible(self._idx_active != len(self.art))

        for i, artists in enumerate(self.art.values()):
            for j, art in enumerate(artists):
                art.set_alpha(alphas[i])
                if i == self._idx_active:
                    art.set_zorder(-(i == -1))
                else:
                    # set image z-order 0, frame z-order 1
                    art.set_zorder(j)

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
        n = len(self)
        self._idx_active += [-1, +1][event.button == 'up']  #
        self._idx_active %= (n + 1)  # wrap

        #
        i = self._idx_active
        if self.image_label is None:
            self.image_label = self.label_image()

        txt = self.image_label
        if i < n:
            txt.set_text(f'{i}: {self.names[i]}')
            xy = _get_ulc(self.params[i], 0.98 * self.fovs[i])
            txt.set_position(xy)

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
            import traceback

            print('Scroll failed:')
            traceback.print_exc()
            print('len(self)', len(self))
            print('self._idx_active', self._idx_active)

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
