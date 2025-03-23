import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sksurv.linear_model import CoxPHSurvivalAnalysis
from xgboost import XGBRegressor, XGBClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from .helper import _get_valid_args


class DefaultMultiRegression:
    def __init__(self, model, dim):
        self.dim = dim
        self.model = model
        self.models_fitted = None

    def fit(self, Y, X):
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        self.models_fitted = [
            self.model.fit(Y=Y[:, ii], X=X) for ii in np.arange(self.dim)
        ]
        return self

    def predict(self, X):
        return np.column_stack([mod.predict(X=X) for mod in self.models_fitted])

    def residuals(self, Y, X):
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        return np.column_stack(
            [
                self.models_fitted[ii].residuals(Y=Y[:, ii], X=X)
                for ii in np.arange(self.dim)
            ]
        )


class RegressionMethod:
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
    Binary classification.
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
        if self.model_fitted is None:
            raise ValueError("Model not fitted yet!")
        chfs = self.model_fitted.predict_cumulative_hazard_function(X)
        return np.array(
            [Y[idx][0] - chfs[idx](Y[idx][1]) for idx in np.arange(Y.shape[0])]
        )


class KRR(RegressionMethod):
    def __init__(self, **kwargs):
        kwargs_kr = _get_valid_args(KernelRidge.__init__, kwargs)
        kwargs_cv = _get_valid_args(GridSearchCV.__init__, kwargs)
        model = GridSearchCV(KernelRidge(**kwargs_kr), **kwargs_cv)
        super().__init__(model)
        self.resid_type = "vanilla"

    def fit(self, Y, X):
        self.model_fitted = self.model.fit(X=X, y=Y).best_estimator_
        return self

    def predict(self, X):
        return self.model_fitted.predict(X=X)

    def residuals(self, Y, X):
        if self.model_fitted is None:
            raise ValueError("Model not fitted yet!")
        return Y - self.model_fitted.predict(X=X)


class XGB(RegressionMethod):
    def __init__(self, **kwargs):
        kwargs_xgb = _get_valid_args(XGBRegressor.__init__, kwargs)
        kwargs_cv = _get_valid_args(GridSearchCV.__init__, kwargs)
        model = GridSearchCV(XGBRegressor(**kwargs_xgb), **kwargs_cv)
        super().__init__(model)
        self.resid_type = "vanilla"

    def fit(self, Y, X):
        self.model_fitted = self.model.fit(X=X, y=Y).best_estimator_
        return self

    def predict(self, X):
        return self.model_fitted.predict(X=X)

    def residuals(self, Y, X):
        if self.model_fitted is None:
            raise ValueError("Model not fitted yet!")
        return Y - self.model_fitted.predict(X=X)


class XGBC(RegressionMethod):
    def __init__(self, **kwargs):
        kwargs_xgb = _get_valid_args(XGBClassifier.__init__, kwargs)
        kwargs_cv = _get_valid_args(GridSearchCV.__init__, kwargs)
        model = GridSearchCV(XGBClassifier(**kwargs_xgb), **kwargs_cv)
        super().__init__(model)
        self.resid_type = "vanilla"

    def fit(self, Y, X):
        self.model_fitted = self.model.fit(X=X, y=Y).best_estimator_
        return self

    def predict(self, X):
        return self.model_fitted.predict(X=X)

    def residuals(self, Y, X):
        if self.model_fitted is None:
            raise ValueError("Model not fitted yet!")
        return Y - self.model_fitted.predict_proba(X=X)[:, 1]
