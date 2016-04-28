import numpy as np


def rmse_scorrer(estimator, X, y_true):
    y_pred = estimator.predict(X)
    output_errors = np.average((y_true - y_pred) ** 2, axis=0)
    return -np.sqrt(output_errors)