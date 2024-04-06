import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sksurv.linear_model import CoxPHSurvivalAnalysis


class DefaultMultiRegression():

    def __init__(self, model, dim):
        self.dim = dim
        self.model = model
        self.models_fitted = None

    def fit(self, Y, X):
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        self.models_fitted = [self.model.fit(
            Y=Y[:, ii], X=X) for ii in np.arange(self.dim)]
        return self

    def predict(self, X):
        return np.column_stack(
            [mod.predict(X=X) for mod in self.models_fitted])

    def residuals(self, Y, X):
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        return np.column_stack([self.models_fitted[ii].residuals(
            Y=Y[:, ii], X=X) for ii in np.arange(self.dim)])


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
        return self

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
        return self

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
        return self

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
        return self

    def predict(self, X):
        return self.model_fitted.predict(X=X)

    def residuals(self, Y, X):
        chfs = self.model_fitted.predict_cumulative_hazard_function(X)
        return np.array([Y[idx][0] - chfs[idx](Y[idx][1])
                         for idx in np.arange(Y.shape[0])])
