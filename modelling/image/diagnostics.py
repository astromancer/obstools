import time

import matplotlib.pyplot as plt

from graphical.imagine import ImageDisplay


def plot_modelled_image(model, image, params):
    # Plot!!
    fig, axes = plt.subplots(4, 1, figsize=(13, 6),
                             sharex='all', sharey='all',
                             # THESE VALUES FOR SALTICAM
                             gridspec_kw=dict(top=0.97,
                                              bottom=0.02,
                                              left=0.02,
                                              right=0.97,
                                              hspace=0.2,
                                              wspace=0.2))
    # image
    ImageDisplay(image, ax=axes[0], title='Image')

    # segmentation
    model.segm.display(ax=axes[1], label=True, title='Segmentation')

    if params is not None:
        # model
        ImageDisplay(model(params), ax=axes[2], title='Model')

        # residuals
        residuals = model.residuals(params, image)
        ImageDisplay(residuals, ax=axes[3],
                     title='Residual')

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
    # '\n'.join(map(numeric_repr, r))  # mdl.format_params(r, precision=3))
    print()
    print('chi2', mdl.redchi(r, image))

    return r, fig
