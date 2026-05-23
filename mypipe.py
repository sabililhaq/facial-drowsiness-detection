import pandas as pd
from sklearn.decomposition import PCA

class MyPipeline(object):
    def __init__(self, model, normalize = True, n_pca = None):
        self.model = model
        self.normalize = normalize
        self.n_pca = n_pca

    def fit(self, X, y):
        if self.normalize:
            X = self.__zscore_normalization(X)
        if self.n_pca is not None:
            self.pca = PCA(n_components=self.n_pca).fit(X)
            X = self.pca.transform(X)
        self.model.fit(X, y)

    def predict(self, X):
        if self.normalize:
            X = self.__zscore_normalization(X)
        if self.n_pca is not None:
            X = self.pca.transform(X)
        return self.model.predict(X)

    def __zscore_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        df_normalized = (df-df.mean())/df.std(ddof=0)
        return df_normalized