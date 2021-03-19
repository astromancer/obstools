"""
Various functions / models for calculating airmass
"""

import numpy as np
from astropy.constants import R_earth
# import matplotlib.pyplot as plt


RHO0 = 1.225  # Density of air at sea level (kg/m^3)
HMAX = 84852.  # Maximal height for model atmosphere
# Re = 6378100  # Earth radius (m)

YOUNG_COEF1 = [1.002432, 0.148386, 0.0096467]
YOUNG_COEF2 = [1., 0.149864, 0.0102963, 0.000303978]
HARDIE_COEFF = [-0.0008083, -0.002875, -0.0018167, 1]


def altitude(ra, dec, lmst, lat):
    """
    Compute the altitude of an object given

    Parameters
    ----------
    ra, dec: 
        equatorial coordinates in radians
    lat: 
        observer lattitude in radians
    lmst: 
        local mean sidereal time in radians
    """
    # h = lmst - ra  # hour angle

    # a is the altitude
    return np.arcsin(np.sin(lat) * np.sin(dec) +
                     np.cos(lat) * np.cos(dec) * np.cos(lmst - ra))


def refractive_index(h_gp):
    """average atmospheric refractive index"""
    delta = 2.93e-4
    rho = atmosphere(h_gp)

    n = 1. + delta * (rho / RHO0)
    return n


class Atmopshere(object):
    pass


def atmosphere(H_gp):  # class StandardAtmosphere
    """
    US Standard Atmosphere, 1976
    As published by NOAA, NASA, and USAF
    The standard atmosphere is mathematically defined in six layers from 
    sea level to 71 km

    http://scipp.ucsc.edu/outreach/balloon/atmos/1976%20Standard%20Atmosphere.htm

    Parameters
    ----------
    H_gp : Geopotential scale height (m)

    Returns
    -------
    rho : Atmospheric pressure (kg/m^3)
    """
    if isinstance(H_gp, (float, int)):
        H_gp = np.array([H_gp])

    regions = [(0. <= H_gp) & (H_gp <= 11e3),
               (11e3 < H_gp) & (H_gp <= 20e3),
               (20e3 < H_gp) & (H_gp <= 32e3),
               (32e3 < H_gp) & (H_gp <= 47e3),
               (47e3 < H_gp) & (H_gp <= 51e3),
               (51e3 < H_gp) & (H_gp <= 71e3),
               (71e3 < H_gp) & (H_gp <= 84852.)]

    expressions = [lambda x: RHO0 * (1. - x / 44330.94) ** 4.25587615,
                   lambda x: RHO0 * 0.29707755 * np.exp((11e3 - x) / 6341.62),
                   lambda x: RHO0 * (0.978260357 + x / 201019.8) ** -35.163195,
                   lambda x: RHO0 * (0.85699696 + x / 57946.3) ** -13.2011407,
                   lambda x: RHO0 * 0.0011653266 *
                   np.exp((47e3 - x) / 7922.26),
                   lambda x: RHO0 * (0.798988674 - x / 184809.7) ** 11.201141,
                   lambda x: RHO0 * (0.900194103 - x / 198095.96) ** 16.0815975]

    return np.piecewise(H_gp, regions, expressions)


def plane_parallel(z):
    """
    When the zenith angle is small to moderate, a good approximation is given by
    assuming a homogeneous plane-parallel atmosphere (i.e., one in which density
    is constant and Earth’s curvature is ignored). The air mass X  then is
    simply the secant of the zenith angle z:

    .. math::       X = sec(z)

    At a zenith angle of 60°, the air mass is approximately 2. However, because
    the Earth is not flat, this formula is only usable for zenith angles up to
    about 60° to 75°, depending on accuracy requirements. At greater zenith
    angles, the accuracy degrades rapidly, with X = sec z becoming infinite at
    the horizon; the horizon air mass in the more-realistic spherical atmosphere
    is usually less than 40.
    """
    return np.sec(z)


def homogeneous_spherical(z, h=0., h_atm=HMAX):
    """
    Airmass for a non-refracting homogeneous spherical atmosphere with elevated
    observer

    Parameters
    ----------
    z:  float, array 
        apparent zenith distance (radians)

    h: float
        observer altitude in metres 

    h_atm: float
        height of atmosphere in metres

    Returns
    -------
    relative airmass

    Notes
    -----
    <http://en.wikipedia.org/wiki/Airmass#Homogeneous_spherical_atmosphere_with_elevated_observer>_

    References
    ----------
    .. [1] Schoenberg, E. 1929. Theoretische Photometrie, Über die Extinktion
    des Lichtes in der Erdatmosphäre. In Handbuch der Astrophysik. Band II,
    erste Hälfte. Berlin: Springer.
    """

    r = R_earth.value / h_atm
    y = h / h_atm
    cosz = np.cos(z)
    rcosz = (r + y) * cosz

    return np.sqrt(rcosz**2 + 2 * r * (1-y) - y**2 + 1) - rcosz


def Young74(z, h):
    """
    Airmass model derived assuming an isothermal atmosphere. and dropping high
    order terms [1]_.  Isothermal atmosphere with pressure scale hight `h` has
    an exponential density attenuation of the form:

    .. math:: \rho = \rho_0 e^{-y/H}

    In an isothermal atmosphere, 37% of the atmosphere is above the pressure
    scale height. An approximate correction for refraction is also included in
    this model.

    Parameters
    ----------
    z:  float, array
        apparent zenith distance (radians)

    h: float
        observer altitude in metres

    h_atm: float
        height of atmosphere in metres

    Returns
    -------
    relative airmass

    References
    ----------
    .. [1] Young, A. T. 1974. Atmospheric Extinction. Ch. 3.1 in Methods of
    Experimental Physics, Vol. 12 Astrophysics, Part A: Optical and Infrared.
    ed. N. Carleton. New York: Academic Press. ISBN 0-12-474912-1.

    """


# Non-physical (interpolative) models follow
# ----------------------------

def Hardie62(z):
    """
    gives usable results for zenith angles of up to perhaps 85°. As with the
    previous formula, the calculated air mass reaches a maximum, and then
    approaches negative infinity at the horizon [1]_. 

    Parameters
    ----------
    z: float, array true zenith distance (radians)

    Returns
    -------
    relative airmass

    References
    ----------
    .. [1] Hardie, R. H. 1962. In Astronomical Techniques. Hiltner, W. A., ed. Chicago:
    University of Chicago Press, 184–. LCCN 62009113.
    `_ADS: <https://ui.adsabs.harvard.edu/abs/1962aste.book.....H>`_
    """

    secz_1 = np.sez(z) - 1
    return np.polyval(HARDIE_COEFF, secz_1)


def YoungIrvine67(z):
    """
    This gives usable results up to approximately 80°, but the accuracy degrades
    rapidly at greater zenith angles. The calculated air mass reaches a maximum
    of 11.13 at 86.6°, becomes zero at 88°, and approaches negative infinity at
    the horizon [1]_.

    Parameters
    ----------
    z: float, array
        true zenith distance (radians)

    Returns
    -------
    relative airmass

    References
    ----------
    .. [1] Young, A. T., and W. M. Irvine. 1967. Multicolor photoelectric
    photometry of the brighter planets. I. Program and procedure. Astronomical
    Journal 72:945–950. doi: 10.1086/110366
    `_ADS: <https://ui.adsabs.harvard.edu/abs/1967AJ.....72..945Y>`_
    """
    secz = np.sez(z)
    return secz * (1 - 0.0012 * (secz*secz - 1))


def Rozenberg66(z):
    """
    gives reasonable results for high zenith angles, with a horizon air mass of
    40.

    Parameters
    ----------
    z: float, array true zenith distance (radians)

    Returns
    -------
    relative airmass

    References  
    ----------
    .. [1] Rozenberg, G. V. 1966. Twilight: A Study in Atmospheric Optics. New York:
    Plenum Press, 160. Translated from the Russian by R. B. Rodman. LCCN
    65011345.
    """
    cosz = np.cos(z)
    return 1 / (cosz + 0.025 * np.exp(-11 * cosz))


def KastenYoung89(z):
    """
    reasonable results for zenith angles of up to 90°, with an air mass of
    approximately 38 at the horizon. Here the second z term
    is in degrees _[1].

    Parameters
    ----------
    z: float, array zenith distance (radians)

    Returns
    -------
    relative airmass

    References
    ----------
    .. [1] Kasten, F.; Young, A. T. (1989). "Revised optical air mass tables and
    approximation formula". Applied Optics. 28 (22): 4735–4738.
    `ADS <https://ui.adsabs.harvard.edu/abs/1989ApOpt..28.4735K>`
    """
    zd = np.radians(z)
    return (np.cos(z) + 0.50572 * (96.07995 - zd) ** -1.6364) ** -1


def Young94(z):
    """
    in terms of the true zenith angle z t for which he claimed a maximum error
    (at the horizon) of 0.0037 air mass. 

    Parameters
    ----------
    z: float, array true zenith distance (radians)

    Returns
    -------
    relative airmass

    References
    ----------
    .. [1] Young, A. T. 1994. Air mass and refraction. Applied Optics.
    33:1108–1110. doi: 10.1364/AO.33.001108 _ADS:
    `ADS <https://ui.adsabs.harvard.edu/abs/1994ApOpt..33.1108Y>`_
    """

    cosz = np.cos(z)
    return np.polyval(YOUNG_COEF1, cosz) / np.polyval(YOUNG_COEF2, cosz)


def Pickering02(z):
    """
    "The Southern Limits of the Ancient Star Catalog." Pickering, K. A. 2002.  
        DIO 12:1, 20, n. 39. 

    Parameters
    ----------
    z: float, array apparent zenith distance (radians)

    Returns
    -------
    relative airmass

    References
    ----------
    .. [1] Pickering, K. A. (2002). "The Southern Limits of the Ancient Star
    Catalog"  DIO. 12 (1): 20–39.
    `PDF <http://www.dioi.org/vols/wc0.pdf>`_
    """
    a = np.degrees(np.pi / 2 - z)  # apparent altitude in degrees
    gamma = a + 244 / (165 + 47 * a ** 1.1)
    return 1. / np.sin(np.radians(gamma))


def Kivalov07(Z, delh=50):
    """
    References
    ----------

    """

    raise NotImplementedError
    r0 = 6356766  # Earth radius (m)

    def i(z, h):
        n0 = refractive_index(0.)
        n = refractive_index(h)
        rh = r0 + h

        sini = (r0 * n0 / (rh * n)) * np.sin(z)
        return np.arcsin(sini)

    def delM(z, h):
        hm = h + 0.5 * delh  # mean height of layer
        rh = r0 + h  # base of layer
        rhp = r0 + h + delh  # top of layer
        rhm = r0 + hm

        rho = atmosphere(hm)
        im = np.mean([i(z, h), i(z, h + delh)], axis=0)

        cos_delphi = (4 * (rhm * np.cos(im)) ** 2 - (delh * np.sin(im)) ** 2) / (
            4 * (rhm * np.cos(im)) ** 2 + (delh * np.sin(im)) ** 2)
        dM = rho * np.sqrt(rh * rh + rhp * rhp - 2 * rh * rhp * cos_delphi)
        return dM

    H = np.arange(0., Hmax, delh)
    X = np.empty(Z.shape)
    for j, z in enumerate(Z):
        DM = delM(z, H)
        X[j] = sum(DM)
    return X
