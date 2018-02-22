import logging
import warnings
import itertools as itt
from pathlib import Path
import multiprocessing as mp

import more_itertools as mit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from grafico.ts import TSplotter
from grafico.imagine import ImageDisplay
# from obstools.psf.psf import GaussianPSF
from obstools.modelling.psf.models_lm import EllipticalGaussianPSF
from motley.profiler.timers import timer
from motley.table import Table
from recipes.misc import is_interactive

from IPython import embed

tsplt = TSplotter()

logger = logging.getLogger('diagnostics')


def new_diagnostics(coords, rcoo, Appars, optstat):
    figs = {}
    # coordinate diagnostics
    fig = plot_coord_scatter(coords, rcoo)
    figs['coords.scatter'] = fig
    fig = plot_coord_walk(coords)
    figs['coords.walk'] = fig
    fig = plot_coord_jumps(coords)
    figs['coords.jump'] = fig

    # aperture diagnostics
    fig = ap_opt_stat_map(optstat)
    figs['opt.stat'] = fig
    fig = plot_appars_walk(Appars.stars, ('a', 'b', 'theta'), 'Star apertures')
    figs['aps.star.walk'] = fig
    fig = plot_appars_walk(Appars.sky, ('a_in', 'b_in', 'a_out'), 'Sky apertures')
    figs['aps.sky.walk'] = fig

    return figs


def ap_opt_stat_map(optstat):

    # TODO: clearer frame numbers for axes and format_coords
    # FIXME: legend being cut off

    nf, ns = optstat.shape
    nq = nf * (ns + 1)
    # make image twice as wide as is tall
    h = int(np.ceil(np.sqrt(nq / 2)))
    w = int(np.ceil(nq / h))
    z = np.full((h, w), np.nan)

    nqq = int(np.ceil(h / (ns + 1)))
    for k in range(ns):
        z[k::(ns + 1)] = np.hstack([optstat[:, k],
                                    np.full(w * nqq - nf, np.nan)]).reshape(-1, w)

    im = ImageDisplay(z, origin='upper', cmap='jet_r', clims=(-3, 1))
    cmap = im.imagePlot.get_cmap()

    proxies = [Rectangle((0, 0), 1, 1, color=c)
               for c in cmap(np.linspace(0, 1, 5))]
    labels = ['error', 'SNR < 1.2. skipped', 'not converged', 'on bound', 'OK']
    im.ax.legend(proxies, labels, loc=2, bbox_to_anchor=(0, 1.2))



    return im.figure


# TODO: plot best model balance for each star

# ====================================================================================================
@timer
def diagnostics(modelDb, locData):
    # np.isnan(flux_ap)
    # problematic = list(filter(None, res))

    # Npar = len(GaussianPSF.params)

    # Diagnostics
    # bad_aic = (np.isnan(AIC) | (abs(AIC) == np.inf))

    # Print fitting summary table
    # FIXME: error if not fitting
    # tbl = fit_summary(modelDb, locData)
    # print(tbl)

    # Check which model is preferred
    lbgb = modelDb.best.ix == modelDb._ix[modelDb.db.bg]  # this is where the pure bg model is the best fit

    badflux = np.isnan(modelDb.best.flux) | lbgb
    fpm = np.ma.masked_where(badflux, modelDb.best.flux)

    # NOTE: validation done individually in phot.psf.aux.StarFit
    par = modelDb.data[modelDb.db.elliptical].params
    alt = modelDb.data[modelDb.db.elliptical].alt

    # TODO: circular??

    badfits = np.isnan(par).any(-1) | np.isinf(par).any(-1)
    # ibad = np.where(badfits)

    pm = np.ma.array(par, copy=True)
    pm[badfits] = np.ma.masked

    paltm = np.ma.array(alt, mask=pm.mask[..., :6])

    return pm, paltm, fpm


# print('Unconvergent: {:%}'.format(np.isnan(psf_par).sum() / psf_par.size))


# ====================================================================================================
def fit_summary(modelDb, locData):
    names, tbl = [], []
    for model in modelDb.gaussians:
        par = modelDb.data[model].params
        badfits = np.isnan(par).any(-1)
        s = badfits.sum(0)  # number of bad fits per star
        f = (s / par.shape[0])  # percentage
        d = map('{:d} ({:.2%})'.format, s, f)
        tbl.append(list(d))
        names.append(modelDb.model_names[modelDb._ix[model]])

    # summary table
    coo = locData.rcoo[modelDb.ix_fit]
    col_headers = list(map('Star {0:d}: ({1[1]:3.1f}, {1[0]:3.1f})'.format, modelDb.ix_fit, coo))
    tbl = Table(tbl,
                title='Fitting summary: Unconvergent', title_props=dict(txt='bold', bg='m'),
                row_headers=names, col_headers=col_headers)

    return tbl


# ====================================================================================================
@timer
def diagnostic_figures(locData, apData, modelDb, fitspath=None, save=True):
    # labels for legends
    nstars = apData.bg.shape[-1]
    ix = modelDb.ix_fit or range(nstars)
    rcoo = locData.rcoo  # finder.Rcoo[ix]
    ir = locData.ir  # finder.ir
    w = locData.window  # finder.window
    star_labels = list(map('{0:d}: ({1[1]:3.1f}, {1[0]:3.1f})'.format, ix, rcoo))

    # #plot some statistics on the parameters!!
    # masked parameters, masked parameter variance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # nans cause Runtimewarning
        pm, paltm, fpm = diagnostics(modelDb, locData)
    fitcoo = pm[..., 1::-1]  # - locData.rcoo[list(modelDb.ix_fit)]
    fitcoo -= np.nanmedian(fitcoo, 0)

    figs = {}
    # plot histograms of parameters
    if pm.size:
        pm[..., :2] -= pm[..., :2].mean(0)  # subtract mean coordinates
        fig = plot_param_hist(pm, EllipticalGaussianPSF.pnames_ordered)
        figs['p.hist.png'] = fig

        pnames = 'sigx, sigy, cov, theta, ellipticity, fwhm'.split(', ')
        fig = plot_param_hist(paltm, pnames)
        figs['p.alt.hist.png'] = fig

    if fitcoo.size:
        fig = plot_coord_scatter(fitcoo, rcoo[ir], w)
        figs['coo.fit.scatter.png'] = fig

        fig = plot_coord_jumps(fitcoo)
        figs['coo.fit.jump.png'] = fig
    elif not locData.find_with_fit:
        #
        fig = plot_coord_scatter(locData.coords, rcoo[ir], w)
        figs['coo.found.scatter.png'] = fig

        fig = plot_coord_jumps(locData.coords)
        figs['coo.found.jump.png'] = fig

    if fpm.size:
        fig = plot_lc_psf(fpm, star_labels)
        figs['lc.psf.png'] = fig

    *figs_lc, fig_bg = plot_lc_aps(apData, star_labels)
    for i, fig in enumerate(figs_lc):
        figs['lc.aps.%i.png' % i] = fig
    figs['lc.bg.png'] = fig_bg

    if save:
        save_figures(figs, fitspath)


@timer
def save_figures(figures, path):
    # create directory for figures to be saved
    # figdir = path.with_suffix('.figs')
    if not path.exists():
        path.mkdir()
    # NOTE existing files will be clobbered

    fnames = [(path / filename).with_suffix('.png')
              for filename in figures.keys()]
    figs = figures.values()

    if is_interactive():
        list(map(saver, figs, fnames))
    else:
        # TODO: figure out why this does not work in ipython
        pool = mp.Pool()
        pool.starmap(saver, zip(figs, fnames))
        pool.close()
        pool.join()


def saver(fig, filename):
    fig.savefig(str(filename))


# ====================================================================================================
@timer
def plot_param_hist(p, names):
    Nstars = p.shape[1]
    p = np.ma.array(p, mask=False)
    div, mod = divmod(p.shape[-1], 2)
    fig, axs = plt.subplots(sum((div, mod)), 2,
                            figsize=(12, 9))
    for i, ax in enumerate(axs.ravel()):
        if mod and i == p.shape[-1]:
            ax.remove()  # remove empty axis
            break

        for pp in p[..., i].T:
            stuff = ax.hist(pp[~pp.mask], bins=50, histtype='step', log=True)
        ax.grid()
        # title
        ax.text(0.5, 0.98, names[i],
                va='top', fontweight='bold', transform=ax.transAxes)
        if i == 1:
            ax.legend(range(Nstars), loc='upper right')
    fig.tight_layout()

    return fig


def plot_density_map(data, range=None, bins=100, filename=None, **kws):
    """
    Plot coordinate scatter. Plot density map in regions with many points for
    efficiency
    """
    from matplotlib.cm import get_cmap

    data = np.asarray(data).squeeze()
    print(data.shape, data.ndim)
    if data.ndim != 2 or 2 not in data.shape:
        raise ValueError('`data` should be an array containing 2 column/row '
                         'vectors.')
    if data.shape[-1] == 2:
        data = data.T

    xdat, ydat = data

    # resolve arguments / defaults
    dthresh = kws.get('density_threshold', 3)  # density threshold
    cmap = get_cmap(kws.pop('cmap', 'jet'))
    # bins
    if np.size(bins) == 1:
        bins = np.array([bins, bins]).squeeze()  # yx bins
    # range
    if range is None:
        range = [(np.floor(np.nanmin(xdat)), np.ceil(np.nanmax(xdat))),
                 (np.floor(np.nanmin(ydat)), np.ceil(np.nanmax(ydat)))]
    range = np.asarray(range)

    # histogram the data
    hh, locx, locy = np.histogram2d(xdat, ydat, bins, range)
    posx = np.digitize(xdat, locx)
    posy = np.digitize(ydat, locy)

    # select points within the histogram
    ind = (posx > 0) & (posx <= bins[0]) & (posy > 0) & (posy <= bins[1])
    hhsub = hh[posx[ind] - 1, posy[ind] - 1]  # values of the histogram where the points are
    xdat1 = xdat[ind][hhsub < dthresh]  # low density points
    ydat1 = ydat[ind][hhsub < dthresh]
    # fill the areas with low density by NaNs
    hh[hh < dthresh] = np.nan

    # plot coordinate scatter
    fig, ax = plt.subplots(figsize=(8, 8),)
                           #subplot_kw=dict(aspect='equal'))

    if not np.isnan(hh).all():
        # plot density map
        ext = range.flatten()
        im = ax.imshow(np.flipud(hh.T), cmap=cmap, extent=ext,
                       interpolation='none', origin='upper')

        from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable
        divider = make_axes_locatable(ax)
        cax =divider.append_axes('right', size=0.25, pad=0.1)

        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.set_ylabel('Density')

    # plot individual points
    ax.plot(xdat1, ydat1, '.', color=cmap(0))

    ax.set_title('Coord scatter')
    ax.set(xlabel='x', ylabel='y')
    ax.grid()

    fig.tight_layout()

    # print('fix coo scatter ' * 100)
    # embed()

    return fig


# @timer
def plot_coord_scatter(coo, rcoo, window=None, filename=None, **kws):
    """
    Plot coordinate scatter. Plot density map in regions with many points for
    efficiency
    """
    from matplotlib.cm import get_cmap
    from matplotlib.patches import Rectangle

    # plot coordinate scatter
    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw=dict(aspect='equal'))
    cmap = get_cmap('jet')

    # flatten data
    xdat, ydat = (coo - rcoo).reshape(-1, 2).T

    # histogram definition
    # w4 = window/4
    # xyrange = np.add(window/2, [[-w4,w4],[-w4,w4]]) # hist  range
    # w2 = window / 2
    if window is None:
        xyrange = np.array([(np.floor(np.nanmin(xdat)), np.ceil(np.nanmax(xdat))),
                            (np.floor(np.nanmin(ydat)), np.ceil(np.nanmax(ydat)))])
    else:
        w2 = window / 2
        xyrange = np.array([[-w2, w2], [-w2, w2]])
        rect = Rectangle((-w2, -w2), window, window,
                         fc='none', lw=1, ls=':', color='r')
        # rect = Rectangle(xyrange[:, 0], xyrange[0].ptp(), xyrange[1].ptp(),
        #                  fc='none', lw=1, ls=':', color='r')
        ax.add_patch(rect)
    bins = [100, 100]  # number of bins
    dthresh = kws.get('density_threshold', 3)  # density threshold

    # histogram the data
    hh, locx, locy = np.histogram2d(xdat, ydat, range=xyrange, bins=bins)
    posx = np.digitize(xdat, locx)
    posy = np.digitize(ydat, locy)

    # select points within the histogram
    ind = (posx > 0) & (posx <= bins[0]) & (posy > 0) & (posy <= bins[1])
    hhsub = hh[posx[ind] - 1, posy[ind] - 1]  # values of the histogram where the points are
    xdat1 = xdat[ind][hhsub < dthresh]  # low density points
    ydat1 = ydat[ind][hhsub < dthresh]
    hh[hh < dthresh] = np.nan  # fill the areas with low density by NaNs

    if not np.isnan(hh).all():
        # plot density map
        ext = xyrange.flatten()
        im = ax.imshow(np.flipud(hh.T), cmap=cmap, extent=ext,
                       interpolation='none', origin='upper')
        cbar = fig.colorbar(im)
        cbar.ax.set_ylabel('Density')

    # plot individual points
    ax.plot(xdat1, ydat1, '.', color=cmap(0))

    # Nstars = coo.shape[1]
    # ax.legend(range(Nstars), loc='upper right')
    # if window:
    # ax.plot(*rcoo[::-1], 'rx', ms=7.5)
    # w2 = window / 2

    # lims = (-w2 - 1, w2 + 1)
    # ax.set_xlim(lims)
    # ax.set_ylim(lims)

    ax.set_title('Coord scatter')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()

    fig.tight_layout()

    # print('fix coo scatter ' * 100)
    # embed()

    return fig


# @timer
def plot_coord_walk(coords):  # coords
    from matplotlib.collections import LineCollection
    # coordinate walk
    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw=dict(aspect='equal'))

    # segments = coords.reshape(-1, 1, 2)
    # segments[-1, -1] = None

    # z = np.random.randn(10,2)
    z = np.vstack([coords, [np.nan] * 2])
    segments = list(mit.pairwise(z))

    lcol = LineCollection(segments)
    lcol.set_array(np.arange(len(coords)))

    ax.add_collection(lcol)
    ax.autoscale_view()
    cbar = fig.colorbar(lcol)

    ax.grid()
    ax.set(title='Coord walk', xlabel='x', ylabel='y')
    cbar.ax.set_ylabel('frame')

    return fig  # , lcol


# @timer
def plot_coord_jumps(coords):  # coords
    # coordinate jumps
    fig, ax = plt.subplots(figsize=(14, 8))
    coords_1 = np.roll(coords, -1, axis=0)
    jumps = np.sqrt(np.square(coords - coords_1).sum(-1))

    # index outliers
    l = jumps > 3
    if l.any():
        for w, j in zip(np.where(l)[0], jumps[l]):
            ax.text(w, j, str(w), fontsize=12)
            # FIXME: May have overlapping text here...

    # plot
    ax.plot(jumps, 'x')  # , alpha=0.75)
    # plot invalid
    xinv, = np.where(np.isnan(coords).any(1))
    ax.plot(xinv, np.zeros_like(xinv), 'rx')

    ax.set(title='Coordinate jumps',
           xlabel='frame', ylabel='$\Delta r$')
    ax.grid()
    # nstars = coords.shape[1]
    # ax.legend(range(nstars))

    fig.tight_layout()

    return fig


def plot_appars_walk(appars, names, title=None):

    # TODO: histogram ??

    nstars = appars.shape[1]
    fig, axes = plt.subplots(nstars, 1, sharex=True, figsize=(14, 8))
    if title:
        fig.suptitle(title)

    for i, (ax, pars) in enumerate(zip(axes, appars.swapaxes(0, 1))):
        for x in pars.T:
            ax.plot(x)
            ax.set_title('Group %i' % i) # TODO add segment labels

        xinv, = np.where(np.isnan(pars).any(1))
        ax.plot(xinv, np.zeros_like(xinv), 'rx')

        ax.grid()
        ax.legend(names)

    return fig


def get_proxy_art(art):
    # TODO: maybe move to tsplt???
    proxies = []
    for a in art:
        clr = a.get_children()[0].get_color()
        r = Rectangle((0, 0), 1, 1, color=clr)
        proxies.append(r)
    return proxies


@timer
def plot_lc_psf(fpm, labels):
    # PSF photometry light curves
    fig, art, *rest = tsplt(fpm.T, title='psf flux',
                            draggable=False,
                            show_hist=True)

    # legend
    hax = fig.axes[1]
    proxies = []
    for a in art:
        clr = a.get_children()[0].get_color()
        r = Rectangle((0, 0), 1, 1, color=clr)
        proxies.append(r)
    hax.legend(proxies, labels, loc='upper right', markerscale=3)

    return fig


def plot_aperture_flux(fitspath, proc, tracker):
    from astropy.time import Time

    #TODO: label top / right axes

    timePath = next(fitspath.parent.glob('*.utc'))
    t = Time(np.loadtxt(timePath, str))

    flux, flxStd = proc.Flx.squeeze().T, proc.FlxStd.squeeze().T
    fluxBG, flxBGStd = proc.FlxBG.T, proc.FlxBGStd.T
    star_labels = list(map('{0:d}: ({1[1]:3.1f}, {1[0]:3.1f})'.format,
                           tracker.segm.labels, tracker.rcoo))

    figs = {}
    figs['lc.aps.opt'] = plot_lc(t, flux, flxStd, star_labels, '(Optimal)')
    figs['lc.aps.bg'] = plot_lc(t, fluxBG, flxBGStd, star_labels, '(BG)')

    return figs


def plot_lc(t, flux, flxStd, labels, description=''):
    logger.info('plotting lc aps: %s', description)

    # no more than 200 errorbars so we don't clutter the plot
    errorevery = flxStd.shape[1] // 200
    title = 'Aperture flux %s' % description

    # plot with frame number at bottom
    timescale = 60 * 60 * 24 / (t[1] - t[0]).to('s').value
    fig, art, *rest = tsplt.plot(flux, flxStd,
                                 title=title,
                                 twinx='sexa',
                                 start=t[0].to_datetime(),
                                 timescale=timescale,
                                 errorbar=dict(errorevery=errorevery),
                                 axlabels=('frame #', 'Flux (photons/pixel)'),
                                 draggable=False,
                                 show_hist=True)

    # Plot with UT seconds on bottom
    # relative time in seconds
    # ts = (t - t[0]).to('s')
    # fig, art, *rest = tsplt.plot(ts, flux, flxStd,
    #                              title=title,
    #                              twinx='sexa',
    #                              start=t[0].to_datetime(),
    #                              errorbar=dict(errorevery=errorevery),
    #                              axlabels=('t (s)', 'Flux (photons/pixel)'),
    #                              draggable=False,
    #                              show_hist=True)
    # legend
    hax = fig.axes[1]
    proxies = get_proxy_art(art)
    hax.legend(proxies, labels, loc='upper right', markerscale=3)

    # date text
    datestr = 'UTC on %s' % str(t[0]).split('T')[0]
    ax = fig.axes[0]
    s = ax.text(1, 1.01, datestr, transform=ax.transAxes)

    return fig


# def plot_lc(args):
#     data, s, labels = args
#     print('plotting lc aps', s)
#     fig, art, *rest = tsplt.plot(data,
#                                  title='aperture flux (%.1f*fwhm)' % s,
#                                  draggable=False,
#                                  show_hist=True)
#     # legend
#     hax = fig.axes[1]
#     proxies = get_proxy_art(art)
#     hax.legend(proxies, labels, loc='upper right', markerscale=3)
#
#     return fig
#

@timer
def plot_lc_aps(apdata, labels):
    # from grafico.multitab import MplMultiTab
    ##ui = MplMultiTab()
    figs = []

    with mp.Pool() as pool:
        figs = pool.map(plot_lc,
                        zip(apdata.flux.T, apdata.scale, itt.repeat(labels)))

    # for i, s in enumerate(apdata.scale):
    #     print('plotting lc aps', i, s)
    #     fig, art, *rest = tsplt.plot(apdata.flux[...,i].T,
    #                              title='aperture flux (%.1f*fwhm)' %s,
    #                              draggable=False,
    #                              show_hist=True)
    #     # legend
    #     hax = fig.axes[1]
    #     proxies = get_proxy_art(art)
    #     hax.legend(proxies, labels, loc='upper right', markerscale=3)
    #
    #     figs.append(fig)
    # ui.add_tab(fig, 'Ap %i' %i)
    # ui.show()

    # Background light curves
    fig, art, *rest = tsplt.plot(apdata.bg.T,
                                 title='bg flux (per pix.)',
                                 draggable=False,
                                 show_hist=True)
    # legend
    hax = fig.axes[1]
    proxies = get_proxy_art(art)
    hax.legend(proxies, labels, loc='upper right', markerscale=3)

    pool.join()
    figs.append(fig)

    return figs


from obstools.aps import ApertureCollection


# ====================================================================================================
def from_params(model, params, scale=3, **kws):
    converged = ~np.isnan(params).any(1)
    ap_data = np.array([model.get_aperture_params(p) for p in params])
    coords = ap_data[converged, :2]  #::-1
    sigma_xy = ap_data[converged, 2:4]
    widths, heights = sigma_xy.T * scale * 2
    angles = np.degrees(ap_data[converged, -1])

    aps = ApertureCollection(coords=coords, widths=widths, heights=heights, angles=angles, **kws)
    return aps, ap_data


# def window_panes(coords, window):
#     from matplotlib.patches import Rectangle
#     from matplotlib.collections import PatchCollection
#     from scipy.spatial.distance import cdist
#
#     sdist = cdist(coords, coords)
#     sdist[np.tril_indices(len(coords))] = np.inf  # since the distance matrix is symmetric, ignore lower half
#     ix = np.where(sdist < window / 2)
#     overlapped = np.unique(ix)
#
#     llc = coords[:, ::-1] - window / 2
#     patches = [Rectangle(coo, window, window) for coo in llc]
#     c = np.array(['g'] * len(coords))
#     c[overlapped] = 'r'
#     rcol = PatchCollection(patches, edgecolor=c, facecolor='none',
#                            lw=1, linestyle=':')
#     return rcol

# ====================================================================================================




from grafico.imagine import VideoDisplayA

# def foo(cube, coords, appars):






from grafico.imagine import FitsCubeDisplay


class FrameDisplay(FitsCubeDisplay):
    # TODO: blit
    # TODO: let the home button restore the original config

    # TODO: enable scroll through - ie inherit from VideoDisplayA
    #     - middle mouse to switch between prePlot and current frame
    #     - toggle legend elements

    # TODO: make sure annotation appear on image area or on top of other stars

    def __init__(self, filename, *args, **kwargs):
        # self.foundCoords = found # TODO: let model carry centroid coords?
        FitsCubeDisplay.__init__(self, filename, *args, **kwargs)

        # FIXME:  this is not always appropriate NOTE: won't have to do this if you use wcs
        # self.ax.invert_xaxis()    # so that it matches sky orientation

    def add_aperture_from_model(self, model, params, r_scale_sigma, rsky_sigma, **kws):
        # apertures from elliptical fit

        apColour = 'g'
        aps, ap_data = from_params(model, params, r_scale_sigma,
                                   ec=apColour, lw=1, ls='--', **kws)
        aps.axadd(self.ax)
        # aps.annotate(color=apColour, size='small')

        # apertures based on finder + scaled by fit
        # from obstools.aps import ApertureCollection
        apColMdl = 'c'
        sigma_xy = ap_data[:, 2:4]
        r = np.nanmean(sigma_xy) * r_scale_sigma * np.ones(len(foundCoords))
        aps2 = ApertureCollection(coords=foundCoords[:, ::-1], radii=r,
                                  ec=apColMdl, ls='--', lw=1)
        aps2.axadd(self.ax)
        # aps2.annotate(color=apColMdl, size='small')

        # skyaps
        rsky = np.multiply(np.nanmean(sigma_xy), rsky_sigma)
        # rsky = np.nanmean(sigma_xy) * rsky_sigma * np.ones_like(foundCoords)
        coosky = [foundCoords[:, ::-1]] * 2
        coosky = np.vstack(list(zip(*coosky)))
        apssky = ApertureCollection(coords=coosky, radii=rsky,
                                    ec='b', ls='-', lw=1)
        apssky.axadd(self.ax)

        # Mark aperture centers
        self.ax.plot(*params[:, 1::-1].T, 'x', color=apColour)

    def mark_found(self, xy, style='rx'):
        # Mark coordinates from finder algorithm
        return self.ax.plot(*xy, style)

    def add_windows(self, xy, window, sdist=None, enumerate=True):
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection

        n = len(xy)
        if sdist is None:
            from scipy.spatial.distance import cdist
            sdist = cdist(xy, xy)

        # since the distance matrix is symmetric, ignore lower half
        try:
            sdist[np.tril_indices(n)] = np.inf
        except:
            print('FUCKUP with add_windows ' * 100)
            embed()
            raise
        ix = np.where(sdist < window / 2)
        overlapped = np.unique(ix)

        # corners
        llc = xy - window / 2
        urc = llc + window
        # patches
        patches = [Rectangle(coo, window, window) for coo in llc]
        # colours
        c = np.array(['g'] * n)
        c[overlapped] = 'r'
        rectangles = PatchCollection(patches,
                                     edgecolor=c, facecolor='none',
                                     lw=1, linestyle=':')
        self.ax.add_collection(rectangles)

        # annotation
        text = []
        if enumerate:
            # names = np.arange(n).astype(str)
            for i in range(n):
                txt = self.ax.text(*urc[i], str(i), color=c[i])
                text.append(txt)
                # ax.annotate(str(i), urc[i], (0,0), color=c[i],
                #                  transform=ax.transData)

            # print('enum_'*100)
            # embed()
        return rectangles, text

    def add_detection_outlines(self, outlines):
        from matplotlib.colors import to_rgba
        overlay = np.empty(self.data.shape + (4,))
        overlay[...] = to_rgba('0.8', 0)
        overlay[..., -1][~outlines.mask] = 1
        # ax.hold(True) # triggers MatplotlibDeprecationWarning
        self.ax.imshow(overlay)

    def add_vectors(self, vectors, ref=None):
        if ref is None:
            'cannot plot vectors without reference star'
        Y, X = self.foundCoords[ref]
        V, U = vectors.T
        self.ax.quiver(X, Y, U, V, color='r', scale_units='xy', scale=1, alpha=0.6)

    def add_legend(self):
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle, Ellipse, Circle
        from matplotlib.legend_handler import HandlerPatch, HandlerLine2D

        def handleSquare(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
            w = width  # / 1.75
            xy = (xdescent, ydescent - width / 3)
            return Rectangle(xy, w, w)

        def handleEllipse(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
            return Ellipse((xdescent + width / 2, ydescent + height / 2), width / 1.25, height * 1.25, angle=45, lw=1)

        def handleCircle(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
            w = width / 2
            return Circle((xdescent + width / 2, ydescent + width / 8), width / 3.5, lw=1)

        apColCir = 'c'
        apColEll = 'g'
        apColSky = 'b'

        # Sky annulus
        pnts = [0], [0]
        kws = dict(c=apColSky, marker='o', ls='None', mfc='None')
        annulus = (Line2D(*pnts, ms=6, **kws),
                   Line2D(*pnts, ms=14, **kws))

        # Fitting
        apEll = Ellipse((0, 0), 0.15, 0.15, ls='--', ec=apColEll, fc='none', lw=1)  # aperture
        rect = Rectangle((0, 0), 1, 1, ec=apColEll, fc='none', ls=':')  # window
        xfit = Line2D(*pnts, mec=apColEll, marker='x', ms=3, ls='none')  # fit position

        # Circular aps
        apCir = Circle((0, 0), 1, ls='--', ec=apColCir, fc='none', lw=1)
        xcom = Line2D(*pnts, mec=apColCir, marker='x', ms=6, ls='none')  # CoM markers

        proxies = (((apEll, xfit, rect), 'Elliptical aps. ($%g\sigma$)' % 3),
                   (apCir, 'Circular aps. ($%g\sigma$)' % 3),
                   (annulus, 'Sky annulus'),
                   # (rect, 'Fitting window'),
                   # ( xfit, 'Fit position'),
                   (xcom, 'Centre of Mass'))

        # proxies = [apfit, annulus, rect, xfit, xcom] # markers in nested tuples are plotted over each other in the legend YAY!
        # labels = [, 'Sky annulus', 'Fitting window', 'Fit position', 'Centre of Mass']

        handler_map = {  # Line2D : HandlerDelegateLine2D(),
            rect: HandlerPatch(handleSquare),
            apEll: HandlerPatch(handleEllipse),
            apCir: HandlerPatch(handleCircle)
        }

        leg1 = self.ax.legend(*zip(*proxies),  # proxies, labels,
                              framealpha=0.5, ncol=2, loc=3,
                              handler_map=handler_map,
                              bbox_to_anchor=(0., 1.02, 1., .102),
                              # bbox_to_anchor=(0, 0.5),
                              # bbox_transform=fig.transFigure,
                              )
        leg1.draggable()

        fig = self.figure
        fig.subplots_adjust(top=0.83)
        figsize = fig.get_size_inches() + [0, 1]
        fig.set_size_inches(figsize)


from grafico.imagine import FitsCubeDisplay


def displayCube(fitsfile, coords, rvec=None):
    # TODO: colour the specific stars used to track differently

    def updater(aps, i):
        cxx = coords[i, None]
        if rvec is not None:
            cxx = cxx + rvec  # relative positions of all other stars
        else:
            cxx = cxx[None]  # make 2d else aps doesn't work
        # print('setting:', aps, i, cxx)
        aps.coords = cxx[:, ::-1]

    im = FitsCubeDisplay(fitsfile, {}, updater,
                         autoscale_figure=False, sidebar=False)
    im.ax.invert_xaxis()  # so that it matches sky orientation
    return im


# def display_frame_coords(data, foundCoords, params=None, model=None, window=None,
#                          vectors=None, ref=None, outlines=None, save=False,
#                          **kws):
#
#     #imd = ImageDisplay(data, origin='llc')
#
#     if params is not None:
#
#
#
#
#     if window:
#
#
#     if outlines is not None:
#
#
#     if vectors is not None:
#
#
#     fig.tight_layout()
#     if save:
#         fig.savefig(save)
#
#     return fig


# ====================================================================================================
def plot_mean_residuals(modelDb):
    # Plot mean residuals
    from mpl_toolkits.axes_grid1 import AxesGrid

    db = modelDb
    names = {m: m.__class__.__bases__[0].__name__ for m in db.models}

    fig = plt.figure()
    fig.suptitle('Mean Residuals', fontweight='bold')
    grid_images = AxesGrid(fig, 111,  # similar to subplot(212)
                           nrows_ncols=(len(db.gaussians), len(db.ix_fit)),
                           axes_pad=0.1,
                           label_mode="L",  # THIS DOESN'T FUCKING WORK!
                           # share_all = True,
                           cbar_location="right",
                           cbar_mode="edge",
                           cbar_size="7.5%",
                           cbar_pad="0%")

    for i, model in enumerate(db.gaussians):
        name = names[model]
        for j, res in enumerate(db.resData[name]):
            ax = grid_images.axes_column[j][i]
            if i == 0:
                ax.set_title('Star %i' % db.ix_fit[i])
            im = ax.imshow(res)
            ax.set_ylabel(name)
        cbax = grid_images.cbar_axes[i]
        ax.figure.colorbar(im, cax=cbax)

    return fig


# ====================================================================================================
# TODO: plot class
@timer
def plot_q_mon(mon_q_file, save=False):  # fitspath
    from astropy.time import Time

    tq, *qsize = np.loadtxt(str(mon_q_file), delimiter=',', unpack=True)

    fig, ax = plt.subplots(figsize=(16, 8), tight_layout=True)
    # x.plot(tm, memo[0], label='free')
    labels = ['find', 'fit', 'bg', 'phot']
    for i, qs in enumerate(qsize):
        t = Time(tq, format='unix').plot_date
        ax.plot_date(t, qs, '-', label=labels[i])  # np.divide(qs, 5e3)
    ax.set_ylabel('Q size')
    ax.set_xlabel('UT')
    ax.grid()
    ax.legend()

    if save:
        filepath = Path(mon_q_file)
        outpath = filepath.with_suffix('.png')
        fig.savefig(str(outpath))

    return fig

    # plot queue occupancy if available


#    if monitor_qs:
#        plot_q_mon()


# if monitor_cpu:
# t, *occ = np.loadtxt(monitor, delimiter=',', unpack=True)
# fig, ax, *stuff = tsplt.plot(t, occ, errorbar={'ls':'-', 'marker':None},
# show_hist=True,
# labels=['cpu%d'%i for i in range(Ncpus)])
# fig.savefig(monitor+'.png')

@timer
def plot_monitor_data(mon_cpu_file, mon_mem_file):
    from astropy.time import Time

    fig, ax1 = plt.subplots(figsize=(16, 8))
    fig.subplots_adjust(top=0.94,
                        left=0.05,
                        right=0.85,
                        bottom=0.05)

    # plot CPU usage
    tc, *occ = np.loadtxt(str(mon_cpu_file), delimiter=',', unpack=True)
    Ncpus = len(occ)

    labels = ['cpu%i' % i for i in range(Ncpus)]
    cmap = plt.get_cmap('gist_heat')
    cols = cmap(np.linspace(0, 1, Ncpus))
    for i, o in enumerate(occ):
        t = Time(tc, format='unix').plot_date
        ax1.plot_date(t, o, '.', color=cols[i], label=labels[i], lw=1)  # np.divide(qs, 5e3)
    ax1.plot(t, np.mean(occ, 0), 'k-', label='cpu mean')

    ax1.set_xlabel('UT')
    ax1.set_ylabel('Usage (%)')
    ax1.grid()
    leg1 = ax1.legend(bbox_to_anchor=(1.05, 1), loc=2,
                      borderaxespad=0., frameon=True)
    ax1.add_artist(leg1)

    # plot memory usage
    tm, *mem = np.loadtxt(str(mon_mem_file), delimiter=',', unpack=True)

    print('Max memory usage: %.3f Gb' % mem[0].ptp())

    ax2 = ax1.twinx()
    labels = ['used', 'free']
    cols = ['c', 'g']
    for i, m in enumerate(mem):
        t = Time(tm, format='unix').plot_date
        ax2.plot_date(t, m, '-', color=cols[i], label=labels[i])

    ax2.set_ylabel('RAM (Gb)')
    ax2.set_ylim(0)
    leg2 = ax2.legend(bbox_to_anchor=(1.05, 0), loc=3,
                      borderaxespad=0., frameon=True)

    # fig.savefig(monitor+'.png')
    return fig


# ====================================================================================================

if __name__ == '__main__':
    path = Path(
            '/home/hannes/work/mensa_sample_run4/')  # /media/Oceanus/UCT/Observing/data/July_2016/FO_Aqr/SHA_20160708.0041.log
    qfiles = list(path.rglob('phot.q.dat'))
    qfigs = list(map(plot_q_mon, qfiles))

    cpufiles, memfiles = zip(*zip(*map(path.rglob, ('phot.cpu.dat', 'phot.mem.dat'))))
    monfigs = list(map(plot_monitor_data, cpufiles, memfiles))
    nlabels = [f.parent.name for f in qfiles]
    wlabels = ['Queues', 'Performance']

    ui = MplMultiTab2D(figures=[qfigs, monfigs], labels=[wlabels, nlabels])
    ui.show()
