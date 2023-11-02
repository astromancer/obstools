"""
Make finder charts from source name or coordinates.
"""

# third-party
import aplpy
import numpy as np
from loguru import logger

# relative
from .utils import get_dss, get_coordinates, STScIServerError


SERVERS = {'b': ['poss2ukstu_blue', 'poss1_blue'],
           'r': ['poss2ukstu_red', 'poss1_red'],
           'i': ['poss2ukstu_ir'],
           'a': ['all']}


def make_finder(obj_name, coords=None, size=(10, 10), filters='bri'):
    """
    Create a finder chart for source with *obj_name* at alt-az coordinates 
    *coords* with field of view size *size* in arcminutes using the first 
    available image given the preference sequence in *filters*.

    Parameters
    ----------
    obj_name : str
        The name of the astronomical source for which this finder chart is 
        intended.  This name is used for looking up the coordinates if the 
        *coord* parameter is not provided. If the source name contains the 
        alt-az coordinates for the object, these coordinates will be parsed and 
        used unless you have also provided *coords* which takes preference.
        The *obj_name* will also be added verbatim to the plot as a label.
    coords : tuple or astropy.coordinates.SkyCoord, optional
        The object coordinates (RA, DEC), by default None.  If not given 
        resolution of the coordinates will be attempted using *obj_name*.
    size : tuple, optional
        Field of view size in arcminutes, by default (10, 10)
    filters : str, optional
        Filter sequence by preference, by default 'bri'. The sky coverage in
        each filter band is not complete, so successive queries for the sequence
        of filters will be tried if the first fails.

    Returns
    -------
    aplpy.FITSFigure
        A plot of the sky region surrounding your object.

    Raises
    ------
    Exception
        If no image could be retrieved for the given object name or coordinates.
    """
    coo = get_coordinates(coords or obj_name)
    ra, dec = coo.ra.deg, coo.dec.deg

    sequence = (server for band in filters for server in SERVERS[band])
    for server in sequence:
        try:
            logger.debug(
                'Retrieving FITS image from DSS server: %s', server)
            # get image
            hdu = get_dss(server, ra, dec, size=size)
            break
        except STScIServerError as err:
            logger.debug('DSS image retrieval failed with:\n{:s}\n', err)
    else:
        raise Exception('No DSS image could be retrieved.')

    plot = aplpy.FITSFigure(hdu)
    plot.show_grayscale()
    plot.set_theme('publication')
    kws = dict(style='italic', weight='bold', size='large')

    plot.add_label(0.5, 1.03,
                   obj_name,
                   relative=True,
                   layer='text')
    plot.add_label(-0.05, -0.05,
                   server,
                   relative=True, style='italic', weight='bold',
                   layer='labels')

    plot.add_grid()
    plot.grid.set_alpha(0.2)
    plot.grid.set_color('b')

    # add cardinal direction labels
    offset = 4.8 / 60.0
    plot.add_label(ra, dec + offset, "N",
                   color=(0, 0.5, 1), **kws)  # layer='labels'
    plot.add_label(ra + offset / np.abs(np.cos(np.radians(dec))),
                   dec,
                   "E",
                   ha='right', color=(0, 0.5, 1), **kws)  # layer='labels'
    # Draw N-S crosshair
    w, h = size
    for angle, length in zip([0, 90], size):
        plot = draw_line(plot, angle, w, ra, dec, color='g',
                         linewidth=0.5, alpha=1.0)  # layer='crosshair'

    return plot


def draw_line(plot, theta, length, ra, dec, **kws):
    """
    Draw a line centered at *ra*, *dec* of a given *length* at a given *angle*
    """
    # set default keywords
    
    theta = np.radians(theta)
    length = length / 2.0 / 60
    dx = np.sin(theta) * length / np.cos(np.radians(dec))
    dy = np.cos(theta) * length
    coords = np.array([[ra + dx, ra - dx], [dec + dy, dec - dy]])
    #
    plot.show_lines(
        [coords], **{**dict(color='b',
                            linewidth=1,
                            alpha=0.7),
                     **kws}
    )
    return plot
