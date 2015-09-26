import numpy as np
# [0, 1, 2, 3, 4, 5, 6, 18, 30, 42, 54, 66, 78]: 
en_lat_bottom = -5
en_lat_top = 5
en_lon_left = 360-170
en_lon_right = 360-120
 
class FeatureExtractor(object):
    
    def __init__(self):
        pass
 
    def transform(self, temperatures_xray, n_burn_in, n_lookahead, skf_is):
        self.n_lookahead = n_lookahead
        """Combine two variables: the montly means corresponding to the month of the target and 
        the current mean temperature in the El Nino 3.4 region."""
        # This is the range for which features should be provided. Strip
        # the burn-in from the beginning and the prediction look-ahead from
        # the end.
        self.valid_range = range(n_burn_in, temperatures_xray['time'].shape[0] - n_lookahead)
        features = []
        for latitude in xrange(-5, 5, 10):
            for longitude in xrange(180, 300, 5):
                for lag in range(0, 19):
                    features.append(self.make_lag_feature(temperatures_xray['tas'], latitude, longitude, lag))
        X = np.vstack(features)
        # all world temps
        all_temps = temperatures_xray['tas'].values
        time_steps, lats, lons = all_temps.shape
        all_temps = all_temps.reshape((time_steps, lats * lons))
        all_temps = all_temps[n_burn_in:-n_lookahead, :]
        # differences
        all_diffs = np.zeros_like(all_temps)
        all_diffs[1:-1,:] = all_temps[:-2,:] - all_temps[2:,:]
        all_diffs[1,:] = all_temps[0,:] - all_temps[1,:]
        all_diffs[-1,:] = all_temps[-2,:] - all_temps[-1,:]                             
        # return feature matrix
#        return np.hstack([X.T, all_temps, all_diffs])
        return np.hstack([X.T, all_temps])
    
    def get_enso_mean(self, tas):
        """The array of mean temperatures in the El Nino 3.4 region at all time points."""
        return tas.loc[:, en_lat_bottom:en_lat_top, en_lon_left:en_lon_right].mean(dim=('lat','lon'))
 
    def get_equatorial_mean(self, tas, x):
        """The array of mean temperatures in the El Nino 3.4 region at all time points."""
        return tas.loc[:, -5:5, x:x+5].mean(dim=('lat','lon'))
 
    def make_feature(self, enso):
        enso_matrix = enso.values.reshape((-1,12))
        count_matrix = np.ones(enso_matrix.shape)
        enso_monthly_mean = (enso_matrix.cumsum(axis=0) / count_matrix.cumsum(axis=0)).ravel()
        enso_monthly_mean_rolled = np.roll(enso_monthly_mean, self.n_lookahead - 12)
        enso_monthly_mean_valid = enso_monthly_mean_rolled[self.valid_range]
        enso_valid = enso.values[self.valid_range]
        return np.array([enso_valid, enso_monthly_mean_valid])
 
    def make_eq_feature(self, tas, longitude, lag):
        enso = self.get_equatorial_mean(tas, longitude)
        lagged = np.roll(enso, self.n_lookahead - lag)
        return self.make_feature(enso) 
 
    def make_lag_feature(self, tas, latitude, longitude, lag):
        enso = tas.loc[:, latitude:latitude+5, longitude:longitude+5].median(dim=('lat','lon'))
        lagged = np.roll(enso, self.n_lookahead - lag)
        return self.make_feature(enso)

    def make_ll_feature_old(self, tas, latitude, longitude, lag):
        enso = tas.loc[:, latitude:latitude+10, longitude:longitude+10].mean(dim=('lat','lon'))
        lagged = np.roll(enso, self.n_lookahead - lag)
        return self.make_feature(enso)
