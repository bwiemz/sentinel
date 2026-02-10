"""Batch operations for building cost matrices in one call.

When the C++ extension with the 'batch' submodule is available, all operations
run in a single Python→C++ boundary crossing.  Otherwise, pure-Python fallbacks
replicate the per-element logic exactly.
"""

from __future__ import annotations

import numpy as np

from sentinel.tracking._accel import _HAS_CPP_BATCH, _sentinel_core

# Large sentinel for infeasible (gated-out) entries
_INF = float("inf")


# ---------------------------------------------------------------------------
# Tier 1 — KF cost matrix
# ---------------------------------------------------------------------------

def batch_kf_cost_matrix(
    states: list[np.ndarray],
    covariances: list[np.ndarray],
    measurements: list[np.ndarray],
    H: np.ndarray,
    R: np.ndarray,
    gate: float = 1e5,
) -> np.ndarray:
    """Build (T, D) Mahalanobis cost matrix for KF tracks vs detections.

    Gated entries (distance > *gate*) are set to +inf.

    Args:
        states: T state vectors, each (n,).
        covariances: T covariance matrices, each (n, n).
        measurements: D measurement vectors, each (m,).
        H: (m, n) observation matrix (shared across all tracks).
        R: (m, m) measurement noise (shared).
        gate: Chi-squared gating threshold.

    Returns:
        (T, D) numpy array of squared Mahalanobis distances.
    """
    T = len(states)
    D = len(measurements)
    if T == 0 or D == 0:
        return np.full((T, D), _INF)

    X = np.ascontiguousarray(np.vstack(states), dtype=np.float64)
    Z = np.ascontiguousarray(np.vstack(measurements), dtype=np.float64)
    P_list = [np.ascontiguousarray(p, dtype=np.float64) for p in covariances]
    H_ = np.ascontiguousarray(H, dtype=np.float64)
    R_ = np.ascontiguousarray(R, dtype=np.float64)

    if _HAS_CPP_BATCH:
        return np.asarray(
            _sentinel_core.batch.batch_kf_gating_matrix(X, P_list, Z, H_, R_, gate)
        )

    # Pure-Python fallback
    Ht = H_.T
    cost = np.full((T, D), _INF)
    for i in range(T):
        S = H_ @ P_list[i] @ Ht + R_
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            continue
        z_pred = H_ @ X[i]
        for j in range(D):
            y = Z[j] - z_pred
            d = float(y @ S_inv @ y)
            if d <= gate:
                cost[i, j] = d
    return cost


# ---------------------------------------------------------------------------
# Tier 1 — IoU matrix
# ---------------------------------------------------------------------------

def batch_iou_matrix(
    bboxes_a: np.ndarray,
    bboxes_b: np.ndarray,
) -> np.ndarray:
    """Build (T, D) IoU matrix.

    Args:
        bboxes_a: (T, 4) bounding boxes [x1, y1, x2, y2].
        bboxes_b: (D, 4) bounding boxes [x1, y1, x2, y2].

    Returns:
        (T, D) IoU values in [0, 1].
    """
    A = np.ascontiguousarray(bboxes_a, dtype=np.float64)
    B = np.ascontiguousarray(bboxes_b, dtype=np.float64)
    T = A.shape[0]
    D = B.shape[0]

    if T == 0 or D == 0:
        return np.zeros((T, D))

    if _HAS_CPP_BATCH:
        return np.asarray(_sentinel_core.batch.batch_iou_matrix(A, B))

    # Pure-Python fallback (vectorized per-row)
    iou = np.zeros((T, D))
    for i in range(T):
        x1 = np.maximum(A[i, 0], B[:, 0])
        y1 = np.maximum(A[i, 1], B[:, 1])
        x2 = np.minimum(A[i, 2], B[:, 2])
        y2 = np.minimum(A[i, 3], B[:, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_a = max(0.0, A[i, 2] - A[i, 0]) * max(0.0, A[i, 3] - A[i, 1])
        area_b = np.maximum(0.0, B[:, 2] - B[:, 0]) * np.maximum(0.0, B[:, 3] - B[:, 1])
        union = area_a + area_b - inter
        mask = union > 0
        iou[i, mask] = inter[mask] / union[mask]
    return iou


# ---------------------------------------------------------------------------
# Tier 1 — Camera combined cost
# ---------------------------------------------------------------------------

def batch_camera_cost_matrix(
    states: list[np.ndarray],
    covariances: list[np.ndarray],
    measurements: list[np.ndarray],
    H: np.ndarray,
    R: np.ndarray,
    bboxes_a: np.ndarray,
    bboxes_b: np.ndarray,
    alpha: float,
    gate: float = 1e5,
) -> np.ndarray:
    """Combined camera cost: alpha * maha + (1 - alpha) * (1 - IoU).

    Gated entries (Mahalanobis > gate) are set to +inf.
    """
    T = len(states)
    D = len(measurements)
    if T == 0 or D == 0:
        return np.full((T, D), _INF)

    X = np.ascontiguousarray(np.vstack(states), dtype=np.float64)
    Z = np.ascontiguousarray(np.vstack(measurements), dtype=np.float64)
    P_list = [np.ascontiguousarray(p, dtype=np.float64) for p in covariances]
    H_ = np.ascontiguousarray(H, dtype=np.float64)
    R_ = np.ascontiguousarray(R, dtype=np.float64)
    A = np.ascontiguousarray(bboxes_a, dtype=np.float64)
    B = np.ascontiguousarray(bboxes_b, dtype=np.float64)

    if _HAS_CPP_BATCH:
        return np.asarray(
            _sentinel_core.batch.batch_camera_cost_matrix(
                X, P_list, Z, H_, R_, A, B, alpha, gate
            )
        )

    # Pure-Python fallback
    maha = batch_kf_cost_matrix(states, covariances, measurements, H, R, gate)
    iou = batch_iou_matrix(A, B)
    cost = np.full((T, D), _INF)
    mask = np.isfinite(maha)
    cost[mask] = alpha * maha[mask] + (1.0 - alpha) * (1.0 - iou[mask])
    return cost


# ---------------------------------------------------------------------------
# Tier 2 — EKF cost matrix
# ---------------------------------------------------------------------------

def _wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def batch_ekf_cost_matrix(
    states: list[np.ndarray],
    covariances: list[np.ndarray],
    measurements: list[np.ndarray],
    jacobians: list[np.ndarray],
    h_predictions: list[np.ndarray],
    R: np.ndarray,
    angular_indices: list[int] | None = None,
    gate: float = 1e5,
) -> np.ndarray:
    """Build (T, D) cost matrix for EKF tracks with per-track Jacobians.

    Args:
        states: T state vectors (may differ in dim for different EKF types).
        covariances: T covariance matrices.
        measurements: D measurement vectors, each (m,).
        jacobians: T Jacobian matrices H_i, each (m, n_i).
        h_predictions: T predicted measurement vectors h(x_i), each (m,).
        R: (m, m) shared measurement noise.
        angular_indices: Which measurement components are angles (wrapped).
        gate: Chi-squared gating threshold.

    Returns:
        (T, D) numpy array.  Gated entries = +inf.
    """
    T = len(states)
    D = len(measurements)
    if T == 0 or D == 0:
        return np.full((T, D), _INF)

    angular_idx = angular_indices or []
    Z = np.ascontiguousarray(np.vstack(measurements), dtype=np.float64)
    R_ = np.ascontiguousarray(R, dtype=np.float64)

    if _HAS_CPP_BATCH:
        st = [np.ascontiguousarray(s, dtype=np.float64) for s in states]
        cv = [np.ascontiguousarray(c, dtype=np.float64) for c in covariances]
        jc = [np.ascontiguousarray(j, dtype=np.float64) for j in jacobians]
        hp = [np.ascontiguousarray(h, dtype=np.float64) for h in h_predictions]
        return np.asarray(
            _sentinel_core.batch.batch_ekf_gating_matrix(
                st, cv, Z, jc, hp, R_, angular_idx, gate
            )
        )

    # Pure-Python fallback
    cost = np.full((T, D), _INF)
    for i in range(T):
        Hi = jacobians[i]
        S = Hi @ covariances[i] @ Hi.T + R_
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            continue
        z_pred = h_predictions[i]
        for j in range(D):
            y = Z[j] - z_pred
            for k in angular_idx:
                if k < len(y):
                    y[k] = _wrap_angle(y[k])
            d = float(y @ S_inv @ y)
            if d <= gate:
                cost[i, j] = d
    return cost


# ---------------------------------------------------------------------------
# Tier 3 — JPDA batch likelihood
# ---------------------------------------------------------------------------

def batch_gaussian_likelihood(
    innovations: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """Compute Gaussian likelihood for N innovations against shared S.

    Args:
        innovations: (N, m) matrix of innovation vectors.
        S: (m, m) innovation covariance.

    Returns:
        (N,) array of likelihood values.
    """
    innovations = np.ascontiguousarray(innovations, dtype=np.float64)
    S_ = np.ascontiguousarray(S, dtype=np.float64)
    N = innovations.shape[0]
    if N == 0:
        return np.array([])

    sign, log_det_S = np.linalg.slogdet(S_)
    if sign <= 0:
        return np.zeros(N)

    try:
        S_inv = np.linalg.inv(S_)
    except np.linalg.LinAlgError:
        return np.zeros(N)

    if _HAS_CPP_BATCH:
        return np.asarray(
            _sentinel_core.batch.batch_gaussian_likelihood(
                innovations, S_inv, log_det_S
            )
        )

    # Pure-Python fallback
    d = innovations.shape[1]
    log_norm = -0.5 * (d * np.log(2 * np.pi) + log_det_S)
    result = np.empty(N)
    for i in range(N):
        y = innovations[i]
        mahal = float(y @ S_inv @ y)
        result[i] = np.exp(log_norm - 0.5 * mahal)
    return result


# ---------------------------------------------------------------------------
# Tier 4 — Batch predict
# ---------------------------------------------------------------------------

def batch_kf_predict(
    states: list[np.ndarray],
    covariances: list[np.ndarray],
    F: np.ndarray,
    Q: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Predict T KF states in one call.

    Args:
        states: T state vectors, each (n,).
        covariances: T covariance matrices, each (n, n).
        F: (n, n) transition matrix (shared).
        Q: (n, n) process noise (shared).

    Returns:
        (states_pred, covs_pred) — lists of predicted states and covariances.
    """
    T = len(states)
    if T == 0:
        return [], []

    X = np.ascontiguousarray(np.vstack(states), dtype=np.float64)
    P_list = [np.ascontiguousarray(p, dtype=np.float64) for p in covariances]
    F_ = np.ascontiguousarray(F, dtype=np.float64)
    Q_ = np.ascontiguousarray(Q, dtype=np.float64)

    if _HAS_CPP_BATCH:
        X_pred, P_pred = _sentinel_core.batch.batch_kf_predict(X, P_list, F_, Q_)
        X_pred = np.asarray(X_pred)
        return [X_pred[i] for i in range(T)], list(P_pred)

    # Pure-Python fallback
    Ft = F_.T
    states_pred = []
    covs_pred = []
    for i in range(T):
        states_pred.append(F_ @ X[i])
        covs_pred.append(F_ @ P_list[i] @ Ft + Q_)
    return states_pred, covs_pred
