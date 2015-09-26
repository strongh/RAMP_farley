from regressor import Regressor
from ts_feature_extractor import FeatureExtractor
from sklearn.cross_validation import cross_val_score
import xray
import numpy as np
from sklearn.cross_validation import ShuffleSplit


def check_model(X_xray, cv_is):
    check_index = 200
    fe = FeatureExtractor()
    X_array = fe.transform(X_xray, n_burn_in, n_lookahead, cv_is)
    X1 = fe.transform(X_xray, n_burn_in, n_lookahead, cv_is)
    check_xray = X_xray.copy(deep=True)
    check_xray['tas'][n_burn_in + check_index:] += \
        np.random.normal(0.0, 10.0, check_xray['tas'][n_burn_in + check_index:].shape)
    X2 = fe.transform(check_xray, n_burn_in, n_lookahead, cv_is)
    first_modified_index = np.argmax(np.not_equal(X1, X2)[:,0])
    if first_modified_index < check_index:
        message = "The feature extractor looks into the feature by {} months".format(
            check_index - first_modified_index)
        raise AssertionError(message)

filename = 'COLA_data/resampled_tas_Amon_CCSM4_piControl_r3i1p1_000101-012012.nc'
resampled_xray = xray.open_dataset(filename, decode_times=False)

random_state = 61
n_burn_in = 120
n_lookahead = 6
skf = ShuffleSplit(resampled_xray['time'].shape[0] - n_burn_in - n_lookahead,
                   n_iter=2,
                   test_size=0.5, random_state=random_state)

check_model(resampled_xray, list(skf)[0])

valid_range = range(n_burn_in, resampled_xray['time'].shape[0] - n_lookahead)
y_array = resampled_xray['target'][valid_range].values
y = y_array.reshape((y_array.shape[0], 1))
fe = FeatureExtractor()
X = fe.transform(resampled_xray, n_burn_in, n_lookahead, list(skf)[0])
reg = Regressor()
scores = np.sqrt(-cross_val_score(reg, X=X, y=y, scoring='mean_squared_error', cv=skf))
print(scores.mean())
