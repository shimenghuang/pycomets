import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor, XGBClassifier
from .utils import _get_valid_args, _safe_atleast_2d
import copy


class DefaultMultiRegression:
    def __init__(self, model, dim):
        self.dim = dim
        self.model = model
        self.models_fitted = []

    def fit(self, Y, X):
        Y = _safe_atleast_2d(Y)
        for ii in range(self.dim):
            mod = copy.deepcopy(self.model)
            self.models_fitted.append(mod.fit(Y=Y[:, ii], X=X))
        return self

    def predict(self, X):
        return np.column_stack([mod.predict(X=X) for mod in self.models_fitted])

    def residuals(self, Y, X):
        Y = _safe_atleast_2d(Y)
        return np.column_stack(
            [
                self.models_fitted[ii].residuals(Y=Y[:, ii], X=X)
                for ii in np.arange(self.dim)
            ]
        )


class RegressionMethod:
    """
    Example:
        mod = DefaultMultiRegression(LM(), X.shape[1])
        mod.fit(Y=X, X=Z)
        mod.predict(X=Z)
        mod.residuals(Y=X, X=Z)
    """
    def __init__(self, model):
        self.model = model
        self.model_fitted = None

    def fit(self):
        raise NotImplementedError("Abstract method")

    def predict(self):
        raise NotImplementedError("Abstract method")

    def residuals(self):
        raise NotImplementedError("Abstract method")


class LM(RegressionMethod, BaseEstimator):
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


class RF(RegressionMethod, BaseEstimator):
    def __init__(self, **kwargs):
        self.resid_type = "vanilla"
        model = RandomForestRegressor(**kwargs)
        super().__init__(model)

    def fit(self, Y, X):
        self.model_fitted = self.model.fit(X=X, y=Y)
        return self

    def predict(self, X):
        return self.model_fitted.predict(X=X)

    def residuals(self, Y, X):
        if self.model_fitted is None:
            raise ValueError("Model not fitted yet!")
        return Y - self.model_fitted.predict(X=X)


class RFC(RegressionMethod, BaseEstimator):
    """
    Binary classification.
    """

    def __init__(self, **kwargs):
        self.resid_type = "vanilla"
        model = RandomForestClassifier(**kwargs)
        super().__init__(model)

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
        self.resid_type = "score"
        try:
            from sksurv.linear_model import CoxPHSurvivalAnalysis
        except ImportError as e:
            raise ImportError("To use `CoxPH`, please install the 'surv' extra, e.g., pip install pycomets[surv]") from e
        model = CoxPHSurvivalAnalysis(**kwargs)
        super().__init__(model)
        
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


class KRR(RegressionMethod, BaseEstimator):
    def __init__(self, **kwargs):
        self.resid_type = "vanilla"
        self.kwargs_kr = _get_valid_args(KernelRidge.__init__, kwargs)
        self.kwargs_cv = _get_valid_args(GridSearchCV.__init__, kwargs)
        model = GridSearchCV(KernelRidge(**self.kwargs_kr), **self.kwargs_cv)
        super().__init__(model)

    def fit(self, Y, X):
        self.model_fitted = self.model.fit(X=X, y=Y).best_estimator_
        return self

    def predict(self, X):
        return self.model_fitted.predict(X=X)

    def residuals(self, Y, X):
        if self.model_fitted is None:
            raise ValueError("Model not fitted yet!")
        return Y - self.model_fitted.predict(X=X)


class XGB(RegressionMethod, BaseEstimator):
    def __init__(self, param_grid, **kwargs):
        self.resid_type = "vanilla"
        self.param_grid = param_grid 
        self.kwargs_xgb = _get_valid_args(XGBRegressor.__init__, kwargs)
        self.kwargs_cv = _get_valid_args(GridSearchCV.__init__, kwargs)
        model = GridSearchCV(XGBRegressor(**self.kwargs_xgb), self.param_grid, **self.kwargs_cv)
        super().__init__(model)

    def fit(self, Y, X):
        self.model_fitted = self.model.fit(X=X, y=Y).best_estimator_
        return self

    def predict(self, X):
        return self.model_fitted.predict(X=X)

    def residuals(self, Y, X):
        if self.model_fitted is None:
            raise ValueError("Model not fitted yet!")
        return Y - self.model_fitted.predict(X=X)


class XGBC(RegressionMethod, BaseEstimator):
    def __init__(self, param_grid, **kwargs):
        self.resid_type = "vanilla"
        self.param_grid = param_grid
        self.kwargs_xgb = _get_valid_args(XGBClassifier.__init__, kwargs)
        self.kwargs_cv = _get_valid_args(GridSearchCV.__init__, kwargs)
        model = GridSearchCV(XGBClassifier(**self.kwargs_xgb), self.param_grid, **self.kwargs_cv)
        super().__init__(model)
        
    def fit(self, Y, X):
        self.model_fitted = self.model.fit(X=X, y=Y).best_estimator_
        return self

    def predict(self, X):
        return self.model_fitted.predict(X=X)

    def residuals(self, Y, X):
        if self.model_fitted is None:
            raise ValueError("Model not fitted yet!")
        return Y - self.model_fitted.predict_proba(X=X)[:, 1]
