import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def labels(y_pred):
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 2:
        return np.argmax(y_pred, axis=1)
    return y_pred

def score(y_true, y_pred, task="classification", metric="accuracy"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if task == "classification":
        yp = labels(y_pred)
        if metric == "accuracy":
            return accuracy_score(y_true, yp)
        if metric == "f1_macro":
            return f1_score(y_true, yp, average="macro")
        if metric == "f1_weighted":
            return f1_score(y_true, yp, average="weighted")
        if metric == "auc":
            if y_pred.ndim == 1:
                return roc_auc_score(y_true, y_pred)
            if y_pred.ndim == 2 and y_pred.shape[1] == 2:
                return roc_auc_score(y_true, y_pred[:, 1])
            return roc_auc_score(y_true, y_pred, multi_class="ovr")
        raise ValueError("Unsupported classification metric")

    if task == "regression":
        yp = y_pred.ravel()
        if metric == "mae":
            return -mean_absolute_error(y_true, yp)
        if metric == "rmse":
            return -np.sqrt(mean_squared_error(y_true, yp))
        if metric == "r2":
            return r2_score(y_true, yp)
        raise ValueError("Unsupported regression metric")

    raise ValueError("task must be classification or regression")
