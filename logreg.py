import numpy as np

class LogisticRegression(object):
    def __init__(self, learning_rate=0.1, max_iter=100, C = 0.1):
        self.learning_rate  = learning_rate
        self.max_iter       = max_iter
        self.C              = C
        self.losses         = []
    
    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1] + 1)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        for _ in range(self.max_iter):
            errors = (self.__sigmoid(X @ self.theta)) - y
            N = X.shape[1]
            delta_grad = self.learning_rate * ((self.C * (X.T @ errors)) + np.sum(self.theta))
            self.theta -= delta_grad / N
            preds = np.round(self.__sigmoid(X @ self.theta)).astype(np.int8)
            self.losses.append(self.__loss(preds, y))
        return self

    def predict_proba(self, X):
        return self.__sigmoid((X @ self.theta[1:]) + self.theta[0])
    
    def predict(self, X):
        return np.round(self.predict_proba(X))
        
    def __sigmoid(self, z):
        return 1.0/(1 + np.exp(-z))
    
    def __loss(self, predictions, labels):
        predictions = np.clip(predictions, 1e-8, 1-(1e-8))
        return -np.mean(labels*np.log(predictions)+(1-labels)*np.log(1-predictions))

    def get_params(self):
        try:
            params = dict()
            params['intercept'] = self.theta[0]
            params['coef'] = self.theta[1:]
            return params
        except:
            raise Exception('Latih model terlebih dahulu!')