import numpy as np
from pycfd import cfd_test_predictions, cfd_test_full_predictions

def test_pairwise_prediction():
    y = np.array([0,1,0,1,1,0,1,0])
    a = np.array([0,1,0,0,1,0,1,0])
    b = np.array([0,0,0,1,1,0,1,1])
    f = np.array([0,1,0,1,1,0,1,0])
    res = cfd_test_predictions(y, a, b, f, n_bootstrap=10, n_permutations=10)
    assert hasattr(res, "cfd_index")

def test_full_prediction():
    y = np.array([0,1,0,1,1,0,1,0])
    preds = {
        "A": np.array([0,1,0,0,1,0,1,0]),
        "B": np.array([0,0,0,1,1,0,1,1]),
        "C": np.array([0,1,0,1,0,0,1,0]),
    }
    f = np.array([0,1,0,1,1,0,1,0])
    res = cfd_test_full_predictions(y, preds, f, n_bootstrap=10, n_permutations=10)
    assert hasattr(res, "cfd_index")
