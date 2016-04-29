import numpy as np
from utils import rmse

def rmse_scorrer(estimator, X, y_true):
    y_pred = estimator.predict(X)
    return rmse(y_true, y_pred)