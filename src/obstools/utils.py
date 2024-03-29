
# std
import re
import numbers
import contextlib
import urllib.request
import operator as op
import functools as ftl
from io import BytesIO

# third-party
import numpy as np
from loguru import logger
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates.name_resolve import NameResolveError
from astropy.coordinates import (EarthLocation, SkyCoord, UnknownSiteException,
                                 jparser)

# local
from recipes import caching

# relative
from . import cachePaths as cached


# from motley.profiling.timers import timer
DMS = '\N{DEGREE SIGN}\N{PRIME}\N{DOUBLE PRIME}'
HMS = 'ʰᵐˢ'

RGX_DSS_ERROR = re.compile(br'(?s)(?i:error).+?<PRE>\s*(.+)\s*</PRE>')


def prod(x):
    """
    Product of a list of numbers; ~40x faster vs np.prod for Python tuples.
    """
    return 1 if (len(x) == 0) else ftl.reduce(op.mul, x)



@caching.to_file(cached.site)
def get_site(name):
    """resolve site name and cache the result"""
    if isinstance(name, EarthLocation):
        return name

    with contextlib.suppress(UnknownSiteException):
        return EarthLocation.of_site(name)

    # try resolve as an address. NOTE this will almost always return a
    # location, even for something that is obviously crap like 'Foo' or 'Moo'
    loc = EarthLocation.of_address(name)
    loc.info.name = name
    return loc


# if raises is warns is silent is None:
#     raises = True   # default behaviour : raise TypeError

# if raises:
#     emit = bork(TypeError)
# elif warns:
#     emit = warnings.warn
# elif silent:
#     emit = _echo  # silently ignore

# emit=bork(TypeError)

def get_coordinates(name_or_coords):
    """
    Get object coordinates from object name or string of coordinates. If the
    coordinates could not be resolved, return None

    Parameters
    ----------
    name_or_coords : str or SkyCoord, optional
        The object name or coordinates (right ascention, declination) as a str 
        that be resolved by SkyCoord, or a SkyCoord object.

    Returns
    -------
    astropy.coordinates.SkyCoord or None
    """
    if isinstance(name_or_coords, SkyCoord):
        return name_or_coords

    try:
        # first try interpret coords.
        if len(name_or_coords) == 2:
            # a 2-tuple.  hopefully ra, dec
            return SkyCoord(*name_or_coords, unit=('h', 'deg'))

        # might also be a single coordinate string
        # eg. '06:14:51.7 -27:25:35.5'
        return SkyCoord(name_or_coords, unit=('h', 'deg'))
    except ValueError:  # as err:
        return get_coords_named(name_or_coords)


def get_coords_named(name):
    """
    Attempts to retrieve coordinates from name, first by parsing the name, or by
    doing SIMBAD Sesame query for the coordinates associated with name.

    Parameters
    ----------
    name : str
        The object name

    Examples
    --------
    >>> get_coords_named('MASTER J061451.7-272535.5')
    >>> get_coords_named('UZ For')
    """

    try:
        coo = resolver(name)
    except NameResolveError as err:  # AttributeError
        logger.warning(
            'Coordinates for object {!r:} could not be retrieved due to the '
            'following exception: \n{:s}', name, str(err)
        )
    else:
        if isinstance(coo, SkyCoord):
            logger.opt(lazy=True).info(
                'The following ICRS J2000.0 coordinates were retrieved:\n\t{:s}',
                lambda: ra_dec_string(coo)
            )
        return coo


@caching.to_file(cached.coo)
def resolver(name):
    """
    Get the target coordinates from object name if known.  This function is
    decorated with the `memoize.to_file` decorator, which caches all previous
    database lookups.  This allows offline usage for repeat queries of the same
    name while also offering a performance improvement for this case.

    Parameters
    ----------
    name : str
        object name

    Returns
    -------
    coords: astropy.coordinates.SkyCoord
    """

    # try parse J coordinates from name.  We do this first, since it is
    # faster than a sesame query

    try:        # EAFP
        return jparser.to_skycoord(name)
    except ValueError as err:
        logger.debug('Could not parse coordinates from name {!r:}.', name)

    try:
        # Attempts a SIMBAD Sesame query with the given object name
        logger.info('Querying SIMBAD database for {!r:}.', name)
        return SkyCoord.from_name(name)
    except NameResolveError as err:
        # check if the name is bad - something like "FLAT" or "BIAS", we want
        # to cache these bad values also to avoid multiple sesame queries for
        # bad values like these
        if str(err).startswith('Unable to find coordinates for name'):
            return None

        # If we are here, it probably means there is something wrong with the
        # connection:
        # NameResolveError: "All Sesame queries failed."
        raise


def convert_skycoords(ra, dec):
    """Try convert ra dec to SkyCoord"""
    if not (ra and dec):
        return

    try:
        return SkyCoord(ra=ra, dec=dec, unit=('h', 'deg'))
    except ValueError:
        logger.warning('Could not interpret coordinates: {:s}; {:s}', ra, dec)


def retrieve_coords_ra_dec(name, verbose=True, **fmt):
    """return SkyCoords and str rep for ra and dec"""
    coords = get_coords_named(name)
    if coords is None:
        return None, None, None

    return coords, *ra_dec_strings(coords, **{**dict(precision=1, pad=1), **fmt})


def ra_dec_string(coords, symbols='αδ', equal=' = ', join='; ', **kws):
    # α '\N{GREEK SMALL LETTER ALPHA}'
    # δ '\N{GREEK SMALL LETTER DELTA}'
    return join.join(map(equal.join, zip(symbols, ra_dec_strings(coords, **kws))))


def ra_dec_strings(coords, **kws):
    kws = {**dict(precision=1, pad=True), **kws}
    return (coords.ra.to_string(unit="h", sep=HMS, **kws),
            coords.dec.to_string(unit="deg", sep=DMS, alwayssign=1, **kws))


def get_skymapper_table(coords, bands, size=(10, 10)):
    # http://skymapper.anu.edu.au/about-skymapper/
    # http://skymapper.anu.edu.au/how-to-access/#public_cutout
    url = 'http://api.skymapper.nci.org.au/public/siap/dr1/query?'

    bands = set(bands.lower())
    assert not (bands - set('uvgriz'))

    # encode payload for the php form
    params = urllib.parse.urlencode(
        dict(POS=coords.to_string().replace(' ', ','),
             BAND=','.join(bands),
             SIZE=','.join(np.divide(size, 60).astype(str)),
             VERB=0,  # verbosity for the table
             INTERSECT='covers',
             RESPONSEFORMAT='TSV',
             )).encode()

    # submit the form
    # req = urllib.request.Request(url)
    raw = urllib.request.urlopen(url, params).read()

    columns, *data = (l.split(b'\t') for l in raw.split(b'\n')[:-1])
    data = np.array(data)
    t = Time(data[:, columns.index(b'mjd_obs')].astype(str), format='mjd')

    logger.opt(lazy=True).info('{}', lambda: 
        f'Found {len(data):d} {bands}-band SkyMapper DR1 images for coordinates {ra_dec_string(coords)} '
        f'spanning dates {t.min().iso.split()[0]} to {t.max().iso.split()[0]}.',
        )

    return columns, data


def get_skymapper(coords, bands, size=(10, 10), combine=True,
                  most_recent_only=False):
    """
    Get combined sky-mapper image
    """

    columns, data = get_skymapper_table(coords, bands, size)

    urls = data[:, columns.index(b'get_fits')].astype(str)
    if most_recent_only:
        t = data[:, columns.index(b'mjd_obs')].astype(float)
        urls = [urls[t.argmin()]]

    # retrieve data possibly from cache
    logger.info('Retrieving images...')
    return [_get_skymapper(url) for url in urls]  # hdus


@caching.to_file(cached.sky)  # memoize for performance
def _get_skymapper(url):
    # get raw image data
    logger.debug(f'Reading data from {url=}')
    raw = urllib.request.urlopen(url).read()

    # load into fits
    fitsData = BytesIO()
    fitsData.write(raw)
    fitsData.seek(0)
    return fits.open(fitsData, ignore_missing_end=True)


# @timer
@caching.to_file(cached.dss, typed={'size': tuple})  # memoize for performance
def get_dss(server, ra, dec, size=(10, 10), epoch=2000):
    """
    Grab a image from STScI server and load as HDUList.
    See [survey description]_.

    Parameters
    ----------
    server
    ra
    dec
    size:   Field of view size in 'arcmin'
    epoch

    Returns
    -------
    `astropy.io.fits.hdu.HDUList`

    `survey description: <http://gsss.stsci.edu/SkySurveys/DSS%20Description.htm>`_
    """

    # , urllib.error, urllib.parse

    # see: http://gsss.stsci.edu/SkySurveys/Surveys.htm
    known_servers = ('all',
                     'poss2ukstu_blue',
                     'poss1_blue',
                     'poss2ukstu_red',
                     'poss1_red',
                     'poss2ukstu_ir',
                     'quickv'
                     )  # TODO: module scope ?
    if server not in known_servers:
        raise ValueError(f'Unknown server: {server!r}. '
                         f'Please use one of the following: {known_servers}')

    # resolve size
    h, w = size  # FIXME: if number

    # make url
    url = 'https://archive.stsci.edu/cgi-bin/dss_search?'

    # encode payload for the php form
    params = urllib.parse.urlencode(
        dict(v=server,
             r=ra, d=dec,
             e=f'J{epoch}',
             h=h, w=w,
             f='fits',
             c='none')
    ).encode()

    # submit the form
    with urllib.request.urlopen(url, params) as html:
        raw = html.read()

    # parse error message
    if error := RGX_DSS_ERROR.search(raw):
        raise STScIServerError(error[1])

    # log
    logger.info("Retrieving {:.1f}′ x {:.1f}′ image for object at J{:.1f} "
                "coordinates RA = {:.3f}; DEC = {:.3f} from {!r:}.",
                h, w, epoch, ra, dec, server)

    # load into fits
    fitsData = BytesIO()
    fitsData.write(raw)
    fitsData.seek(0)
    return fits.open(fitsData, ignore_missing_end=True)


class STScIServerError(Exception):
    pass
