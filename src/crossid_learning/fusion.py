"""Validation-fitted logistic fusion, serialized as explicit numeric parameters."""
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .reranking import FEATURE_NAMES


def fit_fusion(features, labels, metadata):
    x = features.reshape(-1, features.shape[-1])
    y = np.asarray(labels).reshape(-1)
    if x.shape[1] != len(FEATURE_NAMES) or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError('Invalid fusion feature matrix.')
    if set(y) != {0, 1}:
        raise ValueError('Calibration needs retrieved positives and negatives.')
    scaler = StandardScaler().fit(x)
    classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=20261001).fit(scaler.transform(x), y)
    return {'feature_names': FEATURE_NAMES.copy(), 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
            'coef': classifier.coef_[0].tolist(), 'intercept': float(classifier.intercept_[0]),
            'metadata': metadata}


def apply_fusion(features, artifact):
    if artifact['feature_names'] != FEATURE_NAMES:
        raise ValueError('Fusion feature schema changed.')
    values = (features - np.asarray(artifact['mean']))/np.asarray(artifact['scale'])
    if not np.isfinite(values).all():
        raise ValueError('Fusion features are not finite.')
    return expit(values @ np.asarray(artifact['coef']) + artifact['intercept'])
