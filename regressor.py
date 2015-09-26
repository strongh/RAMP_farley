from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import FastICA
from sklearn.decomposition import KernelPCA
 
from sklearn.linear_model import RandomizedLasso
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcess

from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import TruncatedSVD

from sklearn.ensemble import BaggingRegressor

class Regressor(BaseEstimator):
    def __init__(self):
#         self.clf = GradientBoostingRegressor(n_estimators=200, max_features="sqrt", max_depth=5)
#         self.clf = LinearRegression() 
         self.clf = BaggingRegressor(LinearRegression())
#         self.clf = GaussianProcess(theta0=4)
#         self.sp = RandomizedLasso()       
         self.sp = SparseRandomProjection(n_components=5)
#         self.sp = TruncatedSVD()
 #        self.sp = KernelPCA(n_components=3, tol=0.0001, kernel="poly")
    # self.clf = ExtraTreesRegressor(n_estimators=200, max_features="sqrt", max_depth=5)

    def fit(self, X, y):
#        print(self.sp)

#        Xr = self.sp.fit_transform(X, y)
        self.clf.fit(X, y.ravel())
 
    def predict(self, X):
#        Xr = self.sp.transform(X)
        return self.clf.predict(X)
