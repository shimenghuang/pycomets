from sklearn.linear_model import LinearRegression


class RegressionMethod():

    def __init__(self, model):
        self.model = model

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

    def fit(self, Y, X):
        self.model.fit(X=X, y=Y)

    def residuals(self, Y, X):
        return Y - self.model.predict(X=X)
