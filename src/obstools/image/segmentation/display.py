"""
Display segmented images and image cutouts.
"""

# std
import sys
import numbers
from collections.abc import MutableMapping

# third-party
import numpy as np
import more_itertools as mit
from astropy.utils import lazyproperty
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

# local
import motley
import motley.image
from scrawl.utils import embossed
from recipes.functionals import echo
from recipes.pprint import formatters as fmt
from recipes import api, duplicate_if_scalar, pprint, string


STAT_FMT = {
    'flux': ('Flux [ADU]',
             lambda x: pprint.uarray(*x, thousands=' ')),
    # fmt.Measurement(fmt.Decimal(0), thousands=' ', unit='ADU').unicode.starmap
    'com': ('Position (y, x) [px]',
            fmt.Collection(fmt.Decimal(1, short=False), brackets='()').map),
    'areas': ('Area [px²]',
              fmt.Decimal(0).map),
    'roundness': ('Roundness',
                  fmt.Decimal(3).map)
}

# ---------------------------------------------------------------------------- #


class ContourLegendHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        x, y = orig_handle.get_data()
        scale = max(x.ptp(), y.ptp()) / handlebox.height
        line = Line2D((x - x.min()) / scale + x0, (y - y.min()) / scale + y0)
        line.update_from(orig_handle)
        line.set_alpha(1)
        handlebox.add_artist(line)
        return line


def make_contour_legend_proxies(lc, **kws):
    for points, color in zip(lc.get_segments(), lc.get_colors()):
        yield Line2D(*points.T, color=color, **kws)

# ---------------------------------------------------------------------------- #


class SegmentPlotter:
    """
    Helper class for plotting segmented images in various formats.
    """

    @lazyproperty
    def console(self):
        return ConsoleFormatter(self.seg)

    # alias
    terminal = console

    def __init__(self, seg):
        self.seg = seg

    def __call__(self, cmap=None, contours=False, bbox=False, label=True,
                 **kws):
        """
        Plot the segmented image using the `ImageDisplay` class.


        Parameters
        ----------
        cmap: str
            Colourmap name.
        contours: bool
            Should contours be drawn around object perimeters.
        bbox: bool
            Should rectangles be drawn representing the segment bounding boxes.
        label: bool
            Should the segments be labelled with numbers on the image.
        kws:
            passed to `ImageDisplay` class.

        Returns
        -------
        im: `ImageDisplay` instance
        """
        from scrawl.image import ImageDisplay

        # draw segment labels
        # draw_labels = kws.pop('draw_labels', True)
        # draw_rect = kws.pop('draw_rect', False)

        # use categorical colormap (random, muted colours)
        cmap = 'gray' if (self.seg.nlabels == 0) else self.seg.make_cmap()
        # conditional here prevents bork on empty segmentation image

        # plot
        im = ImageDisplay(self.seg.data,
                          cmap=cmap,
                          clim=(0, self.seg.max_label),
                          **{**kws,
                             **dict(sliders=False, hist=False, cbar=False)})

        if contours:
            self.contours(im.ax)

        if bbox:
            self.seg.slices.plot(im.ax)

        if label:
            # add label text (number) on each segment
            self.labels(im.ax, color='w', alpha=0.5,
                        fontdict=dict(weight='bold'))

        return im

    def get_cmap(self, cmap=None):
        # colour map
        if cmap is None:
            return self.seg.make_cmap()

        from matplotlib.pyplot import get_cmap
        return get_cmap(cmap)

    def labels(self, ax=None, offset=(0, 0), emboss=(1, 'k'), **kws):
        # import matplotlib.patheffects as path_effects

        offset = duplicate_if_scalar(offset)

        if ax is None:
            if plt := sys.modules.get('matplotlib.pyplot'):
                ax = plt.gca()
            else:
                raise TypeError('Please provide axes parameter `ax`.')
        #
        kws = {**dict(va='center', ha='center'), **kws}
        texts = []
        for lbl, pos in self._label_positions().items():
            for x, y in pos[:, ::-1] + offset:
                txt = ax.text(x, y, str(lbl), **kws)

                # add border around text to make it stand out (like the arrows)
                texts.append(embossed(txt, *emboss))

        return texts

    def _label_positions(self):
        # compute distance from centroid to nearest pixel of same label.

        # If this distance is large, it means the segment potentially has a
        # hole or is arc shaped, or is disjointed. For these we add the label
        # multiple times, one label for each separate part of the segment.
        # Decide on where to put the labels based on kmeans with 2 clusters.
        # note. this will give bad positions for some shapes, but is good
        # enough for jazz...
        seg = self.seg
        if seg.nlabels == 0:
            return {}

        #
        yx = seg.com(seg.data)
        # yx may contain nans if labels not sequential....
        yx = yx[~np.isnan(yx).any(1)]
        w = np.array(np.where(seg.to_binary(expand=True)))
        splix, = np.where(np.diff(w[0]) >= 1)
        s = np.split(w[1:], splix + 1, 1)
        gy, gx = (np.array(g, 'O')
                  for g in zip(*s))  # each of these is a list of arrays
        magic = [None, ...][bool(len(set(map(len, gy))) - 1)]
        # explicit object arrays avoids DeprecationWarning
        d = ((gy - yx[:, 0, magic]) ** 2 +
             (gx - yx[:, 1, magic]) ** 2) ** 0.5
        # d is an array of arrays!

        pos = {}
        for i, (label, dmin) in enumerate(zip(seg.labels, map(min, d))):
            if dmin > 1:
                from scipy.cluster.vq import kmeans
                # center of mass not close to any pixels. disjointed segment
                # try cluster
                # todo: could try dbscan here for unknown number of clusters
                pos[label], _ = kmeans(np.array((gy[i], gx[i]), float).T, 2)
            else:
                pos[label] = yx[i][None]
        return pos

    # ------------------------------------------------------------------------ #
    # TODO: SegmentedImageContours ??

    def contours(self, ax=None, labels=None, legend=False,
                 linewidth=1.25, emboss=2.5, **kws):
        if ax is None:
            if plt := sys.modules.get('matplotlib.pyplot'):
                ax = plt.gca()
            else:
                raise TypeError('Please proved axes parameter `ax`.')

        lines = self.get_contours(labels, emboss, **kws)
        ax.add_collection(lines)

        if legend:
            # legend defaults
            kws = dict(title='Segments',
                       title_fontproperties={'weight': 'bold'},
                       loc='upper right',
                       bbox_to_anchor=(-0.1, 1),
                       handler_map={Line2D: ContourLegendHandler()})
            if isinstance(legend, dict):
                kws.update(dict)

            labels = self.seg.resolve_labels(labels)
            ax.legend(make_contour_legend_proxies(lines, linewidth=linewidth),
                      map(str, labels),
                      **kws)

        return lines

    # _default_shadow = dict(linewidth=2, foreground='k')

    @api.synonyms({'shadow': 'emboss'})
    def get_contours(self, labels=None, emboss=2.5, alpha=1, **kws):
        """
        Get the collection of lines that trace the circumference of the
        segments.

        Parameters
        ----------
        labels
        kws

        Returns
        -------
        matplotlib.collections.LineCollection

        """
        # NOTE: use PathPatch if you want to be able to hatch the regions.
        #  at the moment you cannot hatch individual paths in PathCollection.

        # boundaries maps labels to boundary line (points array, perimeter)
        boundaries = self.seg.get_boundaries(labels)
        contours, _ = zip(*boundaries.values())

        lines = LineCollection(list(mit.flatten(contours)), alpha=alpha, **kws)
        if 'cmap' in kws:
            lines.set_array(list(mit.flatten(
                [lbl] * len(c) for lbl, c in zip(boundaries, contours))))

        if emboss:
            return embossed(lines, emboss, alpha=alpha)

        if isinstance(emboss, numbers.Real):
            embossed(lines, emboss, alpha=alpha)
            return

        if isinstance(emboss, MutableMapping):
            emboss.setdefault(alpha, alpha)
            return embossed(lines, **emboss)

        raise TypeError(f'Invalid type {type(emboss)} for parameter `emboss`,'
                        f' should be either float of dict.')


class ConsoleFormatter:
    """
    Display images and image cutouts in the console.
    """

    def __init__(self, seg):
        self.seg = seg

    def __call__(self, label=True, frame=True, origin=0):
        """
        A lightweight visualization of the segmented image for display in the
        console. This creates a string with colourised "pixels" using ANSI
        escape sequences. Useful for visualising source cutouts or small
        segmented images. The string representation is printed to stdout and
        returned by this function.


        The string returned by this function, when printed in the console, might
        look something like this, but with the different segments each coloured
        according to its label:

           _________________________________________________
           |                                               |
           |                          ██                   |
           |         ██             ██████        ██       |
           |       ██████             ██        ████       |
           |     ██████████                         ██     |
           |       ██████                                  |
           |         ██                                    |
           |                                ████           |
           |               ██             ████████         |
           |             ██████             ██████         |
           |             ████████             ██           |
           |               ████                            |
           |                 ██                            |
           |                                               |
           |                                               |
           |                                               |
           |             ██                        ██      |
           |           ██████                    ██████    |
           |             ██                    ██████████  |
           |                                     ██████    |
           |                                       ██      |
           |_______________________________________________|


        Parameters
        ----------
        label: bool
            Whether to add region labels (numbers)
        frame: bool
            should a frame be drawn around the image area

        Returns
        -------
        str
        """
        # return self.format_ansi(label, frame, origin).render()
        im = self.format(label, frame, origin)
        print(im)
        return im

    # alias
    console = __call__

    def get_cmap(self, cmap=None):
        # colour map
        if cmap is None:
            return self.seg.make_cmap()

        from matplotlib.pyplot import get_cmap
        return get_cmap(cmap)

    def format(self, label=True, frame=True, origin=0, cmap=None):

        # colour map
        im = motley.image.AnsiImage(self.seg.data,
                                    self.get_cmap(cmap),
                                    origin).format(frame)

        # get positions / str for labels
        if label:
            "TODO"
        #     for lbl, pos in self._label_positions().items():
        #         # TODO: label text colour
        #         label = '%-2i' % lbl
        #         if origin == 0:
        #             pos[:, 0] = nr - 1 - pos[:, 0]

        #         for i in np.ravel_multi_index(
        #                 np.round(pos).astype(int).T, self.shape):
        #             if i in marks:
        #                 marks[i] = marks[i].replace('  ', label)
        #             else:
        #                 marks[i] = markers[lbl].replace('  ', label)

        return im

    def cutouts(self, image, labels=None, extend=1,
                cmap=None, contour=('r', 'B'),
                **kws):
        """
        Cutout image thumbnails displayed as a grid in terminal.

        Parameters
        ----------
        image : np.ndarray
            Image array with sources to display.
        labels : array-like or ellipsis or None
            Segment labels for which cutouts will be imaged.
        extend : int
            The amount of pixels by which to increase the cutout size by on all
            sides of the segment.
        cmap : str, optional
            Colour map, by default 'cmr.voltage_r'.
        contour : str, optional
            Colour for the overlaid contour, by default 'r'.

        """
        ims = self.format_cutouts(image, labels, extend, cmap, contour, **kws)
        print(ims)
        return ims

    def format_cutouts(self, image, labels=..., extend=1,
                       cmap=None, contour=('r', 'B'),
                       statistics=(), **kws):
        """
        Cutout image thumbnails displayed as a grid in terminal. Optional binary
        contours overlaid.

        Parameters
        ----------
        image : np.ndarray
            Image array with sources to display.

        cmap : str, optional
            Colour map, by default 'cmr.voltage_r'.
        contour : str, optional
            Colour for the overlaid contour, by default 'r'.


        """

        #     top : int
        # Number of brightest sources to display images for.

        labels, slices, cutouts = zip(
            *self.seg.cutouts(image, labels=labels,
                              labelled=True, with_slices=True,
                              extend=extend, masked=True)
        )
        cutouts, masks = zip(*((im.data, ~im.mask) for im in cutouts))
        origins = ((x.start, y.start) for y, x in slices)

        # 
        info = {}
        if statistics:
            for stat in statistics:
                header, fmt = STAT_FMT.get(stat, ('', echo))
                result = (func_or_result(image, labels)
                          if callable(func_or_result := getattr(self.seg, stat))
                          else func_or_result)
                info[header] = fmt(result)

        return motley.image.thumbnails_table(
            cutouts, masks, labels, origins, cmap, contour, info=info)


# def source_thumbnails_terminal(image, seg, top,
#                                cmap='cmr.voltage_r', contour='r',
#                                title=None,
#                                label_fmt='{{label:d|B_}: ^{width}}'):
#     """
#     Cutout image thumbnails displayed as a grid in terminal

#     Parameters
#     ----------
#     image : np.ndarray
#         Image array with sources to display.
#     seg : obstools.image.segmentation.SegmentedImage
#         The segmented image of detected sources
#     top : int
#         Number of brightest sources to display images for.
#     image_cmap : str, optional
#         Colour map, by default 'cmr.voltage_r'.
#     contour : str, optional
#         Colour for the overlaid contour, by default 'r'.
#     label_fmt : str, optional
#         Format string for the image titles, by default
#         '{{label:d|Bu}: ^{width}}'. This will produce centre justified lables in
#         bold, underlined text above each image.

#     """
#     #    contour_cmap='hot'):
#     # contours_cmap = seg.get_cmap(contour_cmap)
#     #line_colours  = cmap(np.linspace(0, 1, top))

#     labels = seg.labels[:top]
#     image_stack = np.ma.array(seg.thumbnails(image, labels, True, True))
#     return motley.image.thumbnails(image_stack.data, image_stack.mask,
#                                    cmap, contour,
#                                    title, labels, label_fmt)
