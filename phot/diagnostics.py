import numpy as np
from pathlib import Path

from obstools.psf.models import GaussianPSF

from decor.profile import timer


#====================================================================================================
@timer
def diagnostics(psf_par, psf_par_alt, flux_psf, AIC, Rcoo_fit, window):
    
    #np.isnan(flux_ap)
    #problematic = list(filter(None, res))
    
    Npar = len(GaussianPSF.params)
    
    #Diagnostics
    #bad_aic = (np.isnan(AIC) | (abs(AIC) == np.inf))
    
    #Check which model is preferred by comparing AIC
    modsel = (AIC.argmin(-1) == 1)      #1 is the pure bg model
    #bgpref = modsel & ~np.isnan(AIC).any(-1)
    badflux = np.isnan(flux_psf) | modsel
    fpm = np.ma.masked_where(badflux, flux_psf)
    
    #center coordinates outside of window grid - unphysical
    #badcoo = np.any(np.abs(psf_par[...,:2] - Rcoo_fit[::-1] - window/2) >= window/2, -1)
    #negative parameters ==> bad fit!
    ix = list(range(Npar))
    ix.pop(GaussianPSF._pnames_ordered.index('b'))       #parameter b is allowed to be negative
    negpars = np.any(psf_par[...,ix] < 0, -1)
    nans = np.isnan(psf_par).any(-1)
    badfits = negpars | nans
    #ibad = np.where(badfits)

    pm = np.ma.array(psf_par, copy=True)
    pm[badfits] = np.ma.masked

    paltm = np.ma.array(psf_par_alt, mask=pm.mask[..., :6])
    
    return pm, paltm, fpm

#print('Unconvergent: {:%}'.format(np.isnan(psf_par).sum() / psf_par.size))


#====================================================================================================
@timer
def diagnostic_figures(fitspath, flux_ap, flux_psf, psf_par, psf_par_alt, AIC,
                       Rcoo, scale, window, ix_fit):
    #plot some statistics on the parameters!!
    Rcoo_fit = Rcoo[list(ix_fit)]
    
    #create directory for figures to be saved
    figdir = fitspath.with_suffix('.figs')
    if not figdir.exists():
        figdir.mkdir()
    
    #masked parameters, masked parameter variance
    pm, paltm, fpm = diagnostics(psf_par, psf_par_alt, flux_psf, AIC, Rcoo_fit, window)
    fitcoo = pm[...,1::-1] - Rcoo_fit
    
    #plot histograms of parameters
    pm[..., :2] -= pm[..., :2].mean(0)      #subtract mean coordinates
    plot_param_hist(pm, GaussianPSF._pnames_ordered)
    pnames = 'sigx, sigy, cov, theta, ellipticity, fwhm'.split(', ')
    plot_param_hist(paltm, pnames)
    
    #NOTE existing files will be clobbered
    plot_coord_scatter(fitcoo, Rcoo[0], window, str(figdir/'coo.scatter.png'))
    #plot_coord_walk(fitcoo, str(figdir/'coo.walk.png'))
    plot_coord_jump(fitcoo, str(figdir/'coo.jump.png'))

    plot_lcs(fpm, flux_ap, scale, str(figdir/'lc.ap.png'))
    

#====================================================================================================
@timer
def plot_param_hist(p, names):
    Nstars = p.shape[1]
    p = np.ma.array(p, mask=False)
    div, mod = divmod(p.shape[-1], 2)
    fig, axs = plt.subplots(sum((div, mod)), 2,
                            figsize=(12,9))
    for i, ax in enumerate(axs.ravel()):
        if mod and i == p.shape[-1]:
            ax.remove()     #remove empty axis
            break
        
        for pp in p[...,i].T:       
            stuff = ax.hist(pp[~pp.mask], bins=50, histtype='step', log=True)
        ax.grid()
        #title    
        ax.text(0.5, 0.98, names[i],
                va='top', fontweight='bold', transform=ax.transAxes)
        if i==1:
            ax.legend(range(Nstars), loc='upper right')
    fig.tight_layout()
    
    
#@timer
def plot_coord_scatter(coo, rcoo, window, filename=None, **kws):
    '''
    Plot coordinate scatter. Plot density map in regions with many points for
    efficiency
    '''
    from matplotlib.cm import get_cmap
    from matplotlib.patches import Rectangle
    
    #plot coordinate scatter
    fig, ax = plt.subplots()
    cmap = get_cmap('jet')
    
    #flatten data
    xdat, ydat = coo.reshape(-1, 2).T
    
    #histogram definition
    #w4 = window/4
    #xyrange = np.add(window/2, [[-w4,w4],[-w4,w4]]) # hist  range
    w2 = window / 2
    xyrange = np.array([[-w2,w2],[-w2,w2]])
    bins = [100,100] # number of bins
    dthresh = kws.get('density_threshold', 3)    #density threshold

    # histogram the data
    hh, locx, locy = np.histogram2d(xdat, ydat, range=xyrange, bins=bins)
    posx = np.digitize(xdat, locx)
    posy = np.digitize(ydat, locy)

    #select points within the histogram
    ind = (posx > 0) & (posx <= bins[0]) & (posy > 0) & (posy <= bins[1])
    hhsub = hh[posx[ind] - 1, posy[ind] - 1] # values of the histogram where the points are
    xdat1 = xdat[ind][hhsub < dthresh] # low density points
    ydat1 = ydat[ind][hhsub < dthresh]
    hh[hh < dthresh] = np.nan          # fill the areas with low density by NaNs
    
    if not np.isnan(hh).all():
        #plot density map
        ext = xyrange.flatten()
        im = ax.imshow(np.flipud(hh.T), cmap=cmap, extent=ext,
                    interpolation='none', origin='upper')
        fig.colorbar(im)
    
    #plot individual points
    ax.plot(xdat1, ydat1, '.', color=cmap(0))

    #Nstars = coo.shape[1]
    #ax.legend(range(Nstars), loc='upper right')
    #if window:
    #ax.plot(*rcoo[::-1], 'rx', ms=7.5)
    w2 = window / 2
    rect = Rectangle((-w2,-w2), window, window,
                        fc='none', lw=1, ls=':', color='r')
    ax.add_patch(rect)
    lims = (-w2 - 1, w2 + 1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    ax.set_title('Coord scatter (all)')
    ax.set_xlabel('x0')
    ax.set_ylabel('y0')
    ax.grid()
    
    fig.tight_layout()
    
    if filename:
        fig.savefig(filename)

    return fig
    
#@timer    
def plot_coord_walk(coords, filename): #coords
    from matplotlib.collections import LineCollection
    #coordinate walk
    fig, ax = plt.subplots()

    segments = coords.reshape(-1,2,2)
    segments[-1, -1] = None
    lcol = LineCollection(segments)
    lcol.set_array(np.arange(len(coords)))

    ax.add_collection(lcol)
    ax.autoscale_view()
    fig.colorbar(lcol)
    
    if filename:
        fig.savefig(filename)
    
    return fig

#@timer
def plot_coord_jump(coords, filename=None):    #coords
    #coordinate jumps
    fig, ax = plt.subplots()
    coords_1 = np.roll(coords, -1, axis=0)
    jumps = np.sqrt(np.square(coords - coords_1).sum(-1))
    
    ax.plot(jumps, 'x', alpha=0.75)
    ax.set_title('Coordinate jumps')
    Nstars = coords.shape[1]
    ax.legend(range(Nstars))
    ax.grid()
    
    fig.tight_layout()
    
    if filename:
        fig.savefig(filename)

    return fig

@timer    
def plot_lcs(fpm, flux_ap, scale, filename=None):
    from grafico.lc import lcplot
    
    #plot light curves
    #PSF photometry light curves

    
    fig, art, *rest = lcplot(fpm.T, title='psf flux', draggable=False, 
                             show_hist=True)
    fig.savefig(filename)
    
    
    
    #from grafico.multitab import MplMultiTab
    ##ui = MplMultiTab()
    #for i, s in enumerate(scale):
        #fig, art, *rest = lcplot(flux_ap[...,i].T,
                                #title='aperture flux (%.1f*fwhm)' %s,
                                #draggable=True,)
                                ##show_hist=True)
        #ui.add_tab(fig, 'Ap %i' %i)
    
    
    #ui.show()
    
    #fig, art, *rest = lcplot(flux_ap[...,0].T,
                             #title='aperture flux (%s*fwhm)' % scale[0],
                             #draggable=False,
                             #show_hist=True)
    #fig.savefig(filename)


#====================================================================================================
from matplotlib import pyplot as plt
def display_frame_coords(data, coords, window, vectors=None, ref=None, outlines=None,
                         **kws):
    
    from grafico.imagine import ImageDisplay #FITSCubeDisplay,
    from obstools.aps import ApertureCollection

    fig, ax = plt.subplots()
    imd = ImageDisplay(ax, data)
    imd.connect()

    aps = ApertureCollection(coords=coords[:,::-1], radii=7, 
                             ec='darkorange', lw=1, ls='--',
                             **kws)
    aps.axadd(ax)
    aps.annotate(color='orangered', size='small')
    
    if window:
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection
        
        llc = coords[:,::-1] - window/2
        patches = [Rectangle(coo-window/2, window, window) for coo in llc] 
        rcol = PatchCollection(patches, edgecolor='r', facecolor='none',
                               lw=1, linestyle=':')
        ax.add_collection(rcol)
        
    
    if outlines is not None:
        from matplotlib.colors import to_rgba
        overlay = np.empty(data.shape+(4,))
        overlay[...] = to_rgba('0.8', 0)
        overlay[...,-1][~outlines.mask] = 1
        ax.hold(True)
        ax.imshow(overlay)
    
    if vectors is not None:
        if ref is None:
            'cannot plot vectors without reference star'
        Y, X = coords[ref]
        V, U = vectors.T
        ax.quiver(X, Y, U, V, color='r', scale_units='xy', scale=1, alpha=0.6)

    return fig
    

#====================================================================================================
#TODO: plot class
@timer
def plot_q_mon(mon_q_file, save=False): #fitspath
    from astropy.time import Time
    
    tq, *qsize = np.loadtxt(str(mon_q_file), delimiter=',', unpack=True)

    fig, ax = plt.subplots(figsize=(16,8), tight_layout=True)
    #x.plot(tm, memo[0], label='free')
    labels = ['find', 'fit', 'bg', 'phot' ]
    for i, qs in enumerate(qsize):
        t = Time(tq, format='unix').plot_date
        ax.plot_date(t, qs, '-', label=labels[i]) #np.divide(qs, 5e3)
    ax.set_ylabel('Q size')
    ax.set_xlabel('UT')
    ax.grid()
    ax.legend()
    
    if save:
        filepath = Path(mon_q_file)
        outpath = filepath.with_suffix('.png')
        fig.savefig(str(outpath))
    
    return fig


    #plot queue occupancy if available
#    if monitor_qs:
#        plot_q_mon()

    
    #if monitor_cpu:
        #t, *occ = np.loadtxt(monitor, delimiter=',', unpack=True)
        #fig, ax, *stuff = lcplot(t, occ, errorbar={'ls':'-', 'marker':None}, 
                            #show_hist=True, 
                            #labels=['cpu%d'%i for i in range(Ncpus)])
        #fig.savefig(monitor+'.png')

@timer
def plot_monitor_data(mon_cpu_file, mon_mem_file):
    from astropy.time import Time
    
    fig, ax1 = plt.subplots(figsize=(16,8))
    fig.subplots_adjust(top=0.94,
                        left=0.05,
                        right=0.85,
                        bottom=0.05)

    #plot CPU usage
    tc, *occ = np.loadtxt(str(mon_cpu_file), delimiter=',', unpack=True)
    Ncpus = len(occ)

    labels = ['cpu%i'%i for i in range(Ncpus)]
    cmap = plt.get_cmap('gist_heat')
    cols = cmap(np.linspace(0,1,Ncpus))
    for i, o in enumerate(occ):
        t = Time(tc, format='unix').plot_date
        ax1.plot_date(t, o, '.', color=cols[i], label=labels[i], lw=1) #np.divide(qs, 5e3)
    ax1.plot(t, np.mean(occ, 0), 'k-', label='cpu mean')

    ax1.set_xlabel('UT')
    ax1.set_ylabel('Usage (%)')
    ax1.grid()
    leg1 = ax1.legend(bbox_to_anchor=(1.05, 1), loc=2,
                    borderaxespad=0., frameon=True)
    ax1.add_artist(leg1)


    #plot memory usage
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
    
    #fig.savefig(monitor+'.png')
    return fig

#====================================================================================================

if __name__ == '__main__':
    
    path = Path('/home/hannes/work/mensa_sample_run4/') #/media/Oceanus/UCT/Observing/data/July_2016/FO_Aqr/SHA_20160708.0041.log
    qfiles = list(path.rglob('phot.q.dat'))
    qfigs = list(map(plot_q_mon, qfiles))

    cpufiles, memfiles = zip(*zip(*map(path.rglob, ('phot.cpu.dat', 'phot.mem.dat'))))
    monfigs = list(map(plot_monitor_data, cpufiles, memfiles))
    nlabels = [f.parent.name for f in qfiles]
    wlabels = ['Queues', 'Performance']

    ui = MplMultiTab2D(figures=[qfigs, monfigs], labels=[wlabels, nlabels])
    ui.show()



