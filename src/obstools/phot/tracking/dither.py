
# std
import re
import math
import operator as op

# third-party
import numpy as np
from loguru import logger

# local
from recipes import pprint
from recipes.logging import LoggingMixin
from motley.table import Table
from motley.formatters import Conditional, Decimal, Numeric

# relative
from . import CONFIG


# ---------------------------------------------------------------------------- #

def _sanitize_data(xy, detect_freq_min):

    assert 0 < detect_freq_min <= 1

    n, n_sources, _ = xy.shape
    nans = np.isnan(np.ma.getdata(xy))

    # mask nans.  masked
    xy = np.ma.MaskedArray(xy, nans)

    bad = (nans | np.ma.getmask(xy)).any(-1)
    ignore_frames = nans.all((1, 2))
    n_ignore = ignore_frames.sum()
    n_use = n - n_ignore
    if n_ignore == n:
        raise ValueError('All points are masked!')

    # Due to varying image quality and or camera/telescope drift,
    # some  sources (those near edges of the frame, or variable ones) may
    # not be detected in many frames. Cluster centroids are not an accurate
    # estimator of relative position for these sources since it's an
    # incomplete sample. Only sources that are detected in at least
    # `detect_freq_min` fraction of the frames will be used to calculate
    # frame xy offset.

    # Any measure of centrality for cluster centers is only a good estimator
    # of the relative positions of sources when the camera offsets are
    # taken into account.

    if n_ignore:
        logger.info('Ignoring {:d}/{:d} ({:.1%}) nan values in position '
                    'measurements.', n_ignore, n, n_ignore / n)

    # filter
    good = ~ignore_frames
    use_sources = _filter_sources(n_sources, bad, n_use, detect_freq_min)
    return _select_data(xy, good, use_sources, nans), good, use_sources


def _filter_sources(n_sources, bad, n_use, detect_freq_min):
    if not detect_freq_min:
        return np.ones(n_sources, bool)

    n_detections_per_source = np.zeros(n_sources, int)
    w = np.where(~bad)[1]
    u = np.unique(w)
    n_detections_per_source[u] = np.bincount(w)[u]

    f_det = (n_detections_per_source / n_use)
    use = f_det >= detect_freq_min
    i_use, = np.where(use)
    if not len(i_use):
        raise ValueError(
            'Detected frequency for all sources appears to be too low. There '
            'are {n_sources} objects across {n} images. Their detection '
            'frequencies are: {fdet}.'
        )

    if np.any(~use):
        logger.info('Ignoring {:d}/{:d} sources with low (<={:.0%}) detection '
                    'frequency for frame shift measurement.',
                    n_sources - len(i_use), n_sources, detect_freq_min)

    return use


def _select_data(xy, good, use_sources, nans):
    xyc = xy[good][:, use_sources]
    nansc = nans[good][:, use_sources]
    if nansc.any():
        # prevent warnings
        xyc[nansc] = 0
        xyc[nansc] = np.ma.masked
    return xyc


# ---------------------------------------------------------------------------- #

class PointSourceDitherModel(LoggingMixin):

    def __init__(self, d_cut=None, detect_freq_min=0.9, centre_func=np.mean):
        """
        Measure coordinate positions of point sources accross many frames
        accounting for image dither.

        Parameters
        ----------
        d_cut:  float
            centre distance cutoff for clipping outliers
        detect_freq_min: float
            Required detection frequency of individual sources in order for
            them to be used
        centre_func : _type_, optional
            _description_, by default np.mean
        """
        # maximum distance allowed for point measurements from centre of
        # distribution. This allows tracking sources with low snr, but not
        # computing their centroids which will have larger scatter.
        if d_cut:
            assert d_cut > 0
            self.d_cut = float(d_cut) 
        else:
            self.d_cut = None
        

        self.d_frq = float(detect_freq_min) if detect_freq_min else 0
        assert 0 < self.d_frq <= 1
            
        assert callable(centre_func)
        self.centre_func = centre_func

    def fit(self, xy, weights=None, report=True):
        """
        Measure the centre positions  of detected sources from the individual
        location measurements in xy. Use the locations of the most-often
        detected individual objects.

        Parameters
        ----------
        xy:     array, shape (n_points, n_sources, 2)
            Coordinate points to fit.
        report: bool
            Whether to report on the measured results.

        Returns
        -------
        xy, centres, σxy, δxy, outlier_indices
        """

        n, n_sources, _ = xy.shape
        xys, good, use_sources = _sanitize_data(xy, self.d_frq)

        # Compute cluster centres
        # first estimate of relative positions comes from unshifted cluster centers
        # delay centre compute for fainter sources until after re-centering

        # ensure output same size as input
        centres = np.ma.masked_all((n_sources, 2))
        σxy = np.empty((n_sources, 2))
        δxy = np.ma.masked_all((n, 2))
        # compute positions of all sources with frame offsets measured from best
        # and brightest sources
        centres[use_sources], σxy[use_sources], δxy[good], out = \
            self._fit(xys, weights=weights)

        if len(out):
            # fix outlier indices
            idxf, idxs = np.where(out)
            idxg, = np.where(good)
            idxu, = np.where(use_sources)
            outlier_indices = (idxg[idxf], idxu[idxs])

        # pprint!
        if report:
            try:
                #                                  counts
                self.report(xy, centres, σxy, δxy, None, self.d_frq)

            except Exception as err:
                self.logger.exception('Report failed')

        return xy, centres, σxy, δxy, outlier_indices

    def _fit(self, xy, centres=None, weights=None):
        """
        _summary_

        Parameters
        ----------
        xy : _type_
            _description_
        centres : _type_, optional
            if None (the default) will be computed using `centre_func`.
        axis : int, optional
            _description_, by default 1

        Returns
        -------
        _type_
            _description_
        """

        # compute centes if not given
        if centres is None:
            centres = self.centre_func(xy, axis=0)

        # ensure we have at least some centres
        assert not np.all(np.ma.getmask(centres))

        # main comutation
        centres, sigma_xy, delta_xy = self.compute_centres_offsets(
            xy, centres, weights)

        # break out here  without removing any points if no outlier clipping
        # requested (`d_cut is None`) or there are too few points for the
        # concept of "outlier" to be meaningful
        if (self.d_cut is None) or len(xy) < 10:
            return centres, sigma_xy, delta_xy.squeeze(), ()

        # remove outliers here
        return self._clip_outliers(xy, centres, weights)

    def compute_centres_offsets(self, xy, centres, weights, axis=-2):

        # xy position offset in each frame  (mean combined across sources)
        delta_xy = self.compute_offset(xy, centres, weights, axis=axis,
                                       keepdims=True)

        # shifted cluster centers (all sources)
        xy_shifted = xy - delta_xy

        # Compute cluster centres of shifted point clusters
        centres = self.centre_func(xy_shifted, axis=0)

        return centres, xy_shifted.std(0), delta_xy

    def compute_offset(self, xy, centres, weights=None, **kws):
        """
        Calculate the xy offset of coordinate points `xy` from reference
        `centre` for sources.


        Parameters
        ----------
        xy : array
            Measured points.
        centres : 
            Coordinate centre point for sources.
        weights : array, optional
            Weights for each source for average, by default None

        Returns
        -------
        array
            xy delta. Array can be added to reference coordinates to get the
            source positions for the frame.
        """
        # shift calculated as snr weighted mean of individual CoM shifts
        kws.setdefault('axis', 0)
        return np.ma.average(xy - centres, weights=weights, **kws)

    def _clip_outliers(self, xy, centres, weights):
        # remove outliers

        n, n_sources, _ = xy.shape
        n_points = n * n_sources

        outliers = np.zeros(xy.shape[:-1], bool)
        xym = np.ma.MaskedArray(xy, copy=True)

        for _ in range(5):
            #
            centres, sigma_xy, delta_xy = self.compute_centres_offsets(
                xy, centres, weights)

            # flag outliers
            # compute position residuals after recentre
            dr = np.ma.sqrt(((xym - centres - delta_xy) ** 2).sum(-1))
            out = (dr >= self.d_cut)
            out = np.ma.getdata(out) | np.ma.getmask(out)
            n_out = out.sum()
            if n_out / n_points > 0.5:
                raise ValueError('Too many outliers!!')

            # no new outliers
            if (outliers == out).all():
                if outliers.any():
                    self.logger.info('Ignoring {:d}/{:d} ({:.1%}) values with |δr| > {:.3f}',
                                 n_out, n_points, (n_out / n_points), self.d_cut)
                else:
                    self.logger.info('No outliers detected for position measures.')
                return centres, sigma_xy, delta_xy.squeeze(), outliers

            # mask outliers
            xym[out] = np.ma.masked

        raise ValueError('Emergency stop!')

    def report(self, xy, centres, σ_xy, counts=None,
               detect_frac_min=None, count_thresh=None):
        # report on position measurement

        # from obstools.stats import mad
        # TODO: probably mask nans....
        n_points, n_sources, _ = xy.shape
        n_points_tot = n_points * n_sources

        if np.ma.is_masked(xy):
            bad = xy.mask.any(-1)
            good = np.logical_not(bad)
            points_per_source = good.sum(0)
            sources_per_image = good.sum(1)
            n_bad = bad.sum()
            no_detection, = np.where(np.equal(sources_per_image, 0))
            if len(no_detection):
                self.logger.debug('There are no sources in frames: {!s}', no_detection)

            if n_bad:
                extra = (f'\nn_masked = {n_bad}/{n_points_tot} '
                         f'({n_bad / n_points_tot :.1%})')
        else:
            points_per_source = np.tile(n_points, n_sources)
            extra = ''

        # check if we managed to reduce the variance
        # sigma0 = xy.std(0)
        # var_reduc = (sigma0 - σ_xy) / sigma0
        # mad0 = mad(xy, axis=0)
        # mad1 = mad((xy - xy_offsets[:, None]), axis=0)
        # mad_reduc = (mad0 - mad1) / mad0

        # # overall variance report
        # s0 = xy.std((0, 1))
        # s1 = (xy - xy_offsets[:, None]).std((0, 1))
        # # Fractional variance change
        # self.logger.info('Differencing change overall variance by {!r:}',
        #             np.array2string((s0 - s1) / s0, precision=3))

        # FIXME: percentage format in total wrong
        # TODO: align +- values
        col_headers = ['x', 'y', 'n']  # '(px.)'
        n_min = (detect_frac_min or -math.inf) * n_points
        formatters = {
            'n': Conditional('y', op.le, n_min,
                             Decimal.as_percentage_of(total=n_points,
                                                      precision=0))}

        # get array with number ± std representations
        columns = [pprint.uarray(centres, σ_xy, 2), points_per_source]
        # FIXME: don't print uncertainties if less than 6 measurement points

        # x, y = pprint.uarray(centres, σ_xy, 2)
        # columns = {
        #     'x': Column(x, unit='pixels'),
        #     'y': Column(y, unit='pixels'),
        #     'n': Column(points_per_source,
        #                 fmt=Conditional('y', op.le, n_min,
        #                                     Decimal.as_percentage_of(total=n_points,
        #                                                                  precision=0)),
        #                 total='{}')
        # }

        if counts is not None:
            # columns['counts'] = Column(
            #     counts, unit='ADU',
            #     fmt=Conditional('y', op.le, n_min,
            #                         Percentage(total=n_points,
            #                                        precision=0))
            #     )

            formatters['counts'] = Conditional(
                'c', op.ge, (count_thresh or math.inf),
                Numeric(thousands=' ', precision=1, shorten=False))
            columns.append(counts)
            col_headers += ['counts']

        # add variance columns
        # col_headers += ['r_σx', 'r_σy'] + ['mr_σx', 'mr_σy']
        # columns += [var_reduc[:, ::-1], mad_reduc[:, ::-1]]

        #
        tbl = Table.from_columns(*columns,
                                 units=['pixels', 'pixels', ''],
                                 col_headers=col_headers,
                                 totals=[-1],
                                 formatters=formatters,
                                 **CONFIG.table)

        # fix formatting with percentage in total.
        # TODO Still need to think of a cleaner solution for this
        tbl.data[-1, 0] = re.sub(r'\(\d{3,4}%\)', '', tbl.data[-1, 0])
        # tbl.data[-1, 0] = tbl.data[-1, 0].replace('(1000%)', '')

        self.logger.info('\n{:s}{:s}', tbl, extra)

        return tbl
