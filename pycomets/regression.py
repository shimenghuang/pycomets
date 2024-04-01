import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sksurv.linear_model import CoxPHSurvivalAnalysis


class RegressionMethod():

    def __init__(self, model):
        self.model = model
        self.model_fitted = None

    def fit(self):
        raise NotImplementedError("Abstract method")

    def predict(self):
        raise NotImplementedError("Abstract method")

    def residuals(self):
        raise NotImplementedError("Abstract method")


class LM(RegressionMethod):

    def __init__(self, **kwargs):
        model = LinearRegression(**kwargs)
        super().__init__(model)
        self.resid_type = "vanilla"

    def fit(self, Y, X):
        self.model_fitted = self.model.fit(X=X, y=Y)

    def predict(self, X):
        return self.model_fitted.predict(X=X)

    def residuals(self, Y, X):
        if self.model_fitted is None:
            raise ValueError("Model not fitted yet!")
        return Y - self.model_fitted.predict(X=X)


class RF(RegressionMethod):

    def __init__(self, **kwargs):
        model = RandomForestRegressor(**kwargs)
        super().__init__(model)
        self.resid_type = "vanilla"

    def fit(self, Y, X):
        self.model_fitted = self.model.fit(X=X, y=Y)

    def predict(self, X):
        return self.model_fitted.predict(X=X)

    def residuals(self, Y, X):
        if self.model_fitted is None:
            raise ValueError("Model not fitted yet!")
        return Y - self.model_fitted.predict(X=X)


class RFC(RegressionMethod):
    """
    Binary classification. TODO: error if Y is more than two classes.
    """

    def __init__(self, **kwargs):
        model = RandomForestClassifier(**kwargs)
        super().__init__(model)
        self.resid_type = "vanilla"

    def fit(self, Y, X):
        self.model_fitted = self.model.fit(X=X, y=Y)

    def predict(self, X):
        return self.model_fitted.predict(X=X)

    def residuals(self, Y, X):
        if self.model_fitted is None:
            raise ValueError("Model not fitted yet!")
        return Y - self.model_fitted.predict_proba(X=X)[:, 1]


class CoxPH(RegressionMethod):

    def __init__(self, **kwargs):
        model = CoxPHSurvivalAnalysis(**kwargs)
        super().__init__(model)
        self.resid_type = "score"

    def fit(self, Y, X):
        self.model_fitted = self.model.fit(X=X, y=Y)

    def predict(self, X):
        return self.model_fitted.predict(X=X)

    def residuals(self, Y, X):
        chfs = self.model_fitted.predict_cumulative_hazard_function(X)
        return np.array([Y[idx][0] - chfs[idx](Y[idx][1])
                         for idx in np.arange(Y.shape[0])])
