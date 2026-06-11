import numpy as np 
from sklearn.base import TransformerMixin, BaseEstimator

def engineer_features(X):
    X = X.copy()
    X["volatile_acidity_exp"] = np.exp(-X["volatile acidity"])
    X["residual_sugar_log"] = np.log1p(X["residual sugar"])
    X["density_sin"] = np.sin(X["density"])
    X["sulphates_log"] = np.log1p(X["sulphates"])
    X["alcohol_log"] = np.log1p(X["alcohol"])
    return X

class InteractionTransformer(TransformerMixin,BaseEstimator):
    def __init__(self, pairs):
        self.pairs = pairs

    def fit(self,X,y):
        return self 
    
    def transform(self,X):
        X = X.copy()
        for f1,f2 in self.pairs:
            X[f"{f1}x{f2}"] = X[f1]*X[f2]
        return X 