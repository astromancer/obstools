"""
Diagnostic tools for modelling images
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from graphical.imagine import ImageDisplay

from recipes import pprint


def plot_modelled_image(model, image, params, segm=None):
    # Plot!!
    fig, axes = plt.subplots(4, 1, figsize=(13, 6),
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

    # segmentation
    if segm is None:
        segm = model.segm
    segm.display(ax=axes[1], label=True, title='Segmentation')

    if params is not None:
        # model
        ImageDisplay(model(params), ax=axes[2], title='Model')

        # residuals
        residuals = model.residuals(params, np.ma.getdata(image))
        gof = model.redchi(params, image)

        # display
        im = ImageDisplay(residuals, ax=axes[3],
                          title='Residual')
        # info text
        info = {'dof': str(model.dof),
                'χ²ᵣ': pprint.sci(gof, compact=True, unicode=True)}
        s = ': '.join((model.__class__.__name__,
                       ';\t'.join(map(' = '.join, info.items()))))
        #
        im.ax.text(0, -0.45, s.expandtabs(),
                   transform=im.ax.transAxes, fontsize=12)

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
