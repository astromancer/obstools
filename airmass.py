"""
Various functions / models for calculating airmass
"""

import numpy as np

# import matplotlib.pyplot as plt


rho0 = 1.225  # Density of air at sea level (kg/m^3)
Hmax = 84852.  # Maximal height for model atmosphere


def atmosphere(H_gp):
    """
    US Standard Atmosphere, 1976
    As published by NOAA, NASA, and USAF
    The standard atmosphere is mathematically defined in six layers from sea level to 71 km
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

    expressions = [lambda x: rho0 * (1. - x / 44330.94) ** 4.25587615,
                   lambda x: rho0 * 0.29707755 * np.exp((11e3 - x) / 6341.62),
                   lambda x: rho0 * (0.978260357 + x / 201019.8) ** -35.163195,
                   lambda x: rho0 * (0.85699696 + x / 57946.3) ** -13.2011407,
                   lambda x: rho0 * 0.0011653266 * np.exp((47e3 - x) / 7922.26),
                   lambda x: rho0 * (0.798988674 - x / 184809.7) ** 11.201141,
                   lambda x: rho0 * (0.900194103 - x / 198095.96) ** 16.0815975]

    return np.piecewise(H_gp, regions, expressions)


def refractive_index(H_gp):
    """average atmospheric refractive index"""
    delta = 2.93e-4
    rho = atmosphere(H_gp)
    # rho0 = rho0
    n = 1. + delta * (rho / rho0)
    return n


def KastenYoung89(z):
    """
    F. Kasten and A. T. Young, “Revised optical air mass tables and approximations formula,” Appl. Opt. 28, 4735– 4738 (1989).
    Parameters
    ----------
    z : zenith distance (radians)

    Returns
    -------
    relative airmass
    """
    zd = np.rad2deg(z)
    return (np.cos(z) + 0.50572 * (6.07995 + 90 - zd) ** -1.6364) ** -1


def Young94(z):
    """
    Young, A. T. 1994. Air mass and refraction. Applied Optics. 33:1108–1110. doi: 10.1364/AO.33.001108. Bibcode 1994ApOpt..33.1108Y
    Parameters
    ----------
    z : true zenith distance (radians)

    Returns
    -------
    relative airmass
    """
    coef1 = [1.002432, 0.148386, 0.0096467]
    coef2 = [1., 0.149864, 0.0102963, 0.000303978]
    return np.polyval(coef1, np.cos(z)) / np.polyval(coef2, np.cos(z))


def Pickering02(z):
    """
    Pickering, K. A. 2002. The Southern Limits of the Ancient Star Catalog. DIO 12:1, 20, n. 39. Available as PDF from DIO
    Parameters
    ----------
    z : apparent zenith distance (radians)

    Returns
    -------
    relative airmass
    """
    a = np.rad2deg(np.pi / 2 - z)  # apparent altitude in degrees
    gamma = a + 244 / (165 + 47 * a ** 1.1)
    return 1. / np.sin(np.radians(gamma))


def homosphere(z, yobs=1798, yatm=None):
    """
    Homogeneous spherical atmosphere with elevated observer
    http://en.wikipedia.org/wiki/Airmass#Homogeneous_spherical_atmosphere_with_elevated_observer
    Parameters
    ----------
    z : apparent zenith distance (radians)
    yobs : observer altitude
    yatm : height of atmosphere

    Returns
    -------
    relative airmass
    """
    yatm = yatm if yatm else Hmax
    Re = 6356766  # Earth radius (m)
    rp = Re / yatm
    yp = yobs / yatm

    return np.sqrt(((rp + yp) * np.cos(z)) ** 2 + 2 * rp * (1 - yp) - yp ** 2 + 1) - (rp + yp) * np.cos(z)


def Kivalov07(Z, delh=50):
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


def altitude(ra, dec, lmst, lat):
    """
    Compute the altitude of an object given
    Parameters
    ----------
    ra, dec : equatorial coordinates (radians!!)
    lat: observer lattitude (radians)
    lmst: local mean sidereal time (radians!)
    """
    h = lmst - ra  # hour angle
    sina = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(h)  # a is the altitude
    a = np.arcsin(sina)
    return a
