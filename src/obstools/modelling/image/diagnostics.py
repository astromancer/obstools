"""
Diagnostic tools for modelling images
"""

# std libs
import time

# third-party libs
import numpy as np
import matplotlib.pyplot as plt

# local libs
from recipes import pprint
from scrawl.imagine import ImageDisplay


def plot_modelled_image(model, image, params, seg=None, residual_mask=False,
                        overlay_segments=True):
    """
    Plot image, segmentation, model values, and residuals for a modelled image.

    Parameters
    ----------
    model: `obstools.modelling.core.Model` or Callable
        The model
    image: array
        Modelled data
    params: array
        Parameter vector
    seg: `SegmentedImage`
        Segmented image
    residual_mask: array
        Mask for the residual image.  It is sometimes desirable to mask
        certain elements in the residual (eg. bright foreground objects) in
        order to see the details at an appropriate contrast.
    overlay_segments: bool
        Whether to plot the segmented image in its own axes, or overlay the
        segmentation contours on the model image

    Returns
    -------

    """
    ovr = int(overlay_segments)
    n = 3 if overlay_segments else 4

    fig, axes = plt.subplots(n, 1, figsize=(13, 6),
                             sharex='all', sharey='all',
                             # note THESE VALUES FOR SALTICAM
                             gridspec_kw=dict(top=0.95,
                                              bottom=0.1,
                                              left=0.02,
                                              right=0.97,
                                              hspace=0.3,
                                              wspace=0.2))
    # image
    ImageDisplay(image, ax=axes[0], title='Image')

    if params is not None:
        # model
        # np.ma.MaskedArray(model(params),  np.ma.getmask(image)
        ImageDisplay(model(params), ax=axes[2 - ovr], title='Model')

        # residuals
        if residual_mask is not None and residual_mask is not False:
            image = np.ma.MaskedArray(image, residual_mask)

        residuals = model.residuals(params, image)
        gof = model.redchi(params, image)

        # display
        im = ImageDisplay(residuals, ax=axes[3 - ovr],
                          title='Residual')
        # info text
        info = {'dof': str(model.dof),
                'χ²ᵣ': pprint.sci(gof, compact=True, unicode=True)}
        s = ': '.join((model.__class__.__name__,
                       ';\t'.join(map(' = '.join, info.items()))))
        #
        im.ax.text(0, -0.45, s.expandtabs(),
                   transform=im.ax.transAxes, fontsize=12)

    if seg := seg or getattr(model, 'seg', None):
        if overlay_segments:
            axes[1].add_collection(seg.get_contours())
            seg.draw_labels(axes[1])
        else:
            seg.display(ax=axes[1], label=True, title='Segmentation')

    return fig


def image_fit_report(mdl, image, p0=None):
    # fit
    t0 = time.time()
    r = mdl.fit(image, p0=p0)
    δt = time.time() - t0

    if r is None:
        raise ValueError("Fit didn't converge")

    fig = plot_modelled_image(mdl, image, r)

    # Print!
    print(mdl.name, repr(mdl))
    print('Parameters (%i)' % mdl.dof)
    print('Optimization took: %3.2f s' % δt)
    print(r)
    # '\n'.join(map(numeric, r))  # mdl.format_params(r, precision=3))
    print()
    print('chi2', mdl.redchi(r, image))

    return r, fig



def plot_cross_section(model, p, data, grid=None, std=None, yscale=1,
                     modRes=500):
    # TODO suppress UserWarning: Warning: converting a masked element to nan
    import matplotlib.pyplot as plt
    from recipes.pprint import nrs

    # fig = mdl.plot_fit_results(image, params[mdl.name], modRes)
    # figs.append(fig)

    # plot fit result
    fig, axes = plt.subplots(3, 1, figsize=(10, 8),
                             gridspec_kw=dict(hspace=0,
                                              height_ratios=(3, 1, 0.2)),
                             sharex=True)

    axMdl, axResi, axTxt = axes

    # scale data
    data = data * yscale
    if std is not None:
        std = std * yscale
    if grid is None:
        grid = self.static_grid

    gsc = grid * self._xscale

    # model
    model_colour = 'darkgreen'
    p = p.squeeze()
    x_mdl = np.linspace(grid[0], grid[-1], modRes)
    dfit = self(p, x_mdl) * np.sqrt(yscale)
    line_mdl, = axMdl.plot(x_mdl * self._xscale, dfit, '-',
                           color=model_colour, label='model',
                           zorder=100)

    # plot fitted data
    data_colour = 'royalblue'
    ebMdl = axMdl.errorbar(gsc, data, std,
                           fmt='o', color=data_colour, zorder=10,
                           label='Median $\pm$ (MAD)')

    # residuals
    res = data - self(p, grid) * np.sqrt(yscale)
    ebRes = axResi.errorbar(gsc, res, std, fmt='o', color=data_colour)
    ebars = [ebMdl, ebRes]

    # get the breakpoints
    if isinstance(p, Parameters):
        p = p.view((float, p.npar))

    self._check_params(p)
    if self.fit_breakpoints:
        bp = p[-self.npoly:]
    else:
        bp = self.breakpoints

    # breakpoints
    breakpoints = bp * self._xscale
    lineCols = []
    for ax in axes[:2]:
        lines_bp = ax.vlines(breakpoints, 0, 1,
                             linestyle=':', color='0.2',
                             transform=ax.get_xaxis_transform(),
                             label='Break points')
        lineCols.append(lines_bp)

    #
    axMdl.set_title(('%s Fit' % self.name))  # .title())
    axMdl.set_ylabel('Counts (ADU)')
    axMdl.grid()

    axResi.set_ylabel('Residuals')
    axResi.grid()

    # ylims
    w = data.ptp() * 0.2
    axMdl.set_ylim(data.min() - w, data.max() + w)

    w = res.ptp()
    ymin = res.min() - w
    ymax = res.max() + w
    if std is not None:
        ymin = min(ymin, (res - std).min() * 1.2)
        ymax = max(ymax, (res + std).max() * 1.2)
    axResi.set_ylim(ymin, ymax)

    # GoF statistics
    # Remove all spines, ticks, labels, etc for `axTxt`
    axTxt.set_axis_off()

    def gof_text(stat, name, xpos=0):
        v = stat(p, data, grid, std)
        s = sci_repr(v, latex=True).strip('$')
        txt = f'${name} = {s}$'
        # print(txt)
        return axTxt.text(xpos, 0, txt, fontsize=14, va='top',
                          transform=axTxt.transAxes)

    funcs = self.redchi, self.rsq, self.aic, self.aicc
    names = (r'\chi^2_{\nu}', 'R^2', 'AIC', 'AIC_c')
    positions = (0, 0.25, 0.5, 0.75)
    texts = []
    for f, name, xpos in zip(funcs, names, positions):
        txt = gof_text(f, name, xpos)
        texts.append(txt)

    # turn the ticks back on for the residual axis
    for tck in axResi.xaxis.get_major_ticks():
        tck.label1On = True

    fig.tight_layout()

    art = ebars, line_mdl, lineCols, texts
    return art