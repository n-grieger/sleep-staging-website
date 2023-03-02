import numpy as np
from scipy import stats


def _handle_zeros_in_scale(scale, copy=True, constant_mask=None):
    """Set scales of near constant features to 1.
    The goal is to avoid division by very small or zero values.
    Near constant features are detected automatically by identifying
    scales close to machine precision unless they are precomputed by
    the caller and passed with the `constant_mask` kwarg.
    Typically for standard scaling, the scales are the standard
    deviation while near constant features are better detected on the
    computed variances which are closer to machine precision by
    construction.
    """
    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == 0.0:
            scale = 1.0
        return scale
    elif isinstance(scale, np.ndarray):
        if constant_mask is None:
            # Detect near constant values to avoid dividing by a very small
            # value that could lead to surprising results and numerical
            # stability issues.
            constant_mask = scale < 10 * np.finfo(scale.dtype).eps

        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[constant_mask] = 1.0
        return scale


class RobustScaler:
    def __init__(self, *, quantile_range=(25.0, 75.0), ):
        self.quantile_range = quantile_range

    def fit(self, X):
        q_min, q_max = self.quantile_range
        self.center_ = np.nanmedian(X, axis=0)

        quantiles = []
        for feature_idx in range(X.shape[1]):
            column_data = X[:, feature_idx]
            quantiles.append(np.nanpercentile(column_data, self.quantile_range))

        quantiles = np.transpose(quantiles)

        self.scale_ = quantiles[1] - quantiles[0]
        self.scale_ = _handle_zeros_in_scale(self.scale_, copy=False)
        adjust = stats.norm.ppf(q_max / 100.0) - stats.norm.ppf(q_min / 100.0)
        self.scale_ = self.scale_ / adjust

        return self

    def transform(self, X):
        X -= self.center_
        X /= self.scale_
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)
