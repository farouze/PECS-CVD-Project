import numpy as np

def strength_from_index(cfd_index):
    if cfd_index <= 0:
        return "None"
    if cfd_index < 1:
        return "Very weak"
    if cfd_index < 5:
        return "Weak"
    if cfd_index < 15:
        return "Moderate"
    if cfd_index < 30:
        return "Strong"
    return "Very strong"

def validate_domains(domains, y):
    if not isinstance(domains, dict):
        raise TypeError("domains must be a dictionary")
    if len(domains) < 2:
        raise ValueError("At least two domains are required")
    y = np.asarray(y)
    out = {}
    for name, X in domains.items():
        X = np.asarray(X)
        if X.shape[0] != len(y):
            raise ValueError(f"{name} has {X.shape[0]} samples but y has {len(y)}")
        out[str(name)] = X
    return out, y
