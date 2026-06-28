import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ALPHA_GRID = np.logspace(-5, 5, 11)


def ridge_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alphas: np.ndarray = ALPHA_GRID,
) -> dict:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    ridge = RidgeCV(alphas=alphas, fit_intercept=True, gcv_mode="svd")
    ridge.fit(X_tr, y_train)

    y_pred = ridge.predict(X_te)

    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot)

    sr = spearmanr(y_pred, y_test)
    rho = float(sr.statistic if hasattr(sr, "statistic") else sr.correlation)

    return {"r2": r2, "rho": rho, "alpha": float(ridge.alpha_), "scaler": scaler, "model": ridge}
