from sklearn.gaussian_process import GaussianProcess
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        self.gp = GaussianProcess(corr='absolute_exponential')

    def fit(self, X, y):
        y = y + np.random.randn(len(y)) * (1/20)
        self.gp.fit(X, y)

    def predict(self, X):
        return self.gp.predict(X)
