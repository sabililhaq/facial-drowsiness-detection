import pandas as pd
from sklearn.decomposition import PCA

FEATURE_COLUMNS = []
for i in range(468):
    FEATURE_COLUMNS.extend([f"x{i+1}", f"y{i+1}", f"z{i+1}"])

class MyPipeline(object):
    def __init__(self, model, normalize=True, n_pca=None):
        self.model = model
        self.normalize = normalize
        self.n_pca = n_pca
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y):
        if self.normalize:
            self.mean_ = X.mean()
            self.std_ = X.std(ddof=0).replace(0, 1)
            X = self._apply_normalization(X)
        if self.n_pca is not None:
            n_components = min(self.n_pca, X.shape[0], X.shape[1])
            self.pca = PCA(n_components=n_components).fit(X)
            X = self.pca.transform(X)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.normalize:
            X = self._apply_normalization(X)
        if self.n_pca is not None:
            X = self.pca.transform(X)
        return self.model.predict(X)

    def _apply_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        mean = self.mean_ if self.mean_ is not None else df.mean()
        std = self.std_ if self.std_ is not None else df.std(ddof=0).replace(0, 1)
        return (df - mean) / std