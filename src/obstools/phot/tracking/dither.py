
# third-party
import numpy as np
from loguru import logger

# local
from recipes.logging import LoggingMixin

# relative
from ...image.registration import report_measurements


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
    if detect_freq_min is None:
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

        assert callable(centre_func)
        self.d_cut = d_cut
        self.d_frq = detect_freq_min
        self.centre_func = centre_func

    def __call__(self, xy, report=True):
        """
        Measure the centre positions  of detected sources from the individual
        location measurements in xy. Use the locations of the most-often
        detected individual objects.

        Parameters
        ----------
        xy:     array, shape (n_points, n_sources, 2)


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
        centres[use_sources], σxy[use_sources], δxy[good], out = self.fit(xys)

        if out:
            # fix outlier indices
            idxf, idxs = np.where(out)
            idxg, = np.where(good)
            idxu, = np.where(use_sources)
            outlier_indices = (idxg[idxf], idxu[idxs])

        # pprint!
        if report:
            try:
                #                                          counts
                report_measurements(xy, centres, σxy, δxy, None, self.d_frq)

            except Exception as err:
                self.logger.exception('Report failed')

        return xy, centres, σxy, δxy, outlier_indices

    def fit(self, xy, centres=None, axis=1):
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
            centres = self.centre_func(xy)

        # ensure we have at least some centres
        assert not np.all(np.ma.getmask(centres))

        # main comutation
        centres, sigma_xy, delta_xy = self._compute_centres_offsets(xy, centres)

        # break out here  without removing any points if no outlier clipping
        # requested (`d_cut is None`) or there are too few points for the
        # concept of "outlier" to be meaningful
        if (self.d_cut is None) or len(xy) < 10:
            return centres, sigma_xy, delta_xy.squeeze(), ()
        
        # remove outliers here
        return self._clip_outliers(xy)

    def _compute_centres_offsets(self, xy, centres, weights):

        # xy position offset in each frame  (mean combined across sources)
        delta_xy = self.compute_offsets(xy, centres, weights, keepdims=True)

        # shifted cluster centers (all sources)
        xy_shifted = xy - delta_xy

        # Compute cluster centres of shifted point clusters
        centres = self.centre_func(xy_shifted)

        return centres, xy_shifted.std(0), delta_xy

    def compute_offsets(self, xy, centres, weights=None, **kws):
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
            xy delta
        """
        # shift calculated as snr weighted mean of individual CoM shifts
        kws.setdefault('axis', 1)
        return np.ma.average(xy - centres, weights=weights, **kws)

    def _clip_outliers(self, xy):
        # remove outliers

        n, n_sources, _ = xy.shape
        n_points = n * n_sources

        outliers = np.zeros(xy.shape[:-1], bool)
        xym = np.ma.MaskedArray(xy, copy=True)

        for _ in range(5):
            #
            centres, sigma_xy, delta_xy = self._compute_centres_offsets(xy, centres)

            # flag outliers
            # compute position residuals after recentre
            dr = np.ma.sqrt(np.ma.square(xym - centres - delta_xy).sum(-1))
            out = (dr >= self.d_cut)
            out = np.ma.getdata(out) | np.ma.getmask(out)
            n_out = out.sum()
            if n_out / n_points > 0.5:
                raise ValueError('Too many outliers!!')

            # no new outliers
            if (outliers == out).all():
                self.logger.info('Ignoring {:d}/{:d} ({:.1%}) values with |δr| > {:.3f}',
                                 n_out, n_points, (n_out / n_points), self.d_cut)

                return centres, sigma_xy, delta_xy.squeeze(), outliers

            # mask outliers
            xym[out] = np.ma.masked

        raise ValueError('Emergency stop!')
