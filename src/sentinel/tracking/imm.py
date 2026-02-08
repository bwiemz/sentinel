"""Interacting Multiple Model (IMM) filter for maneuvering target tracking.

Runs Constant-Velocity (CV) and Constant-Acceleration (CA) models in parallel,
switching between them based on measurement likelihood. Handles the CV→CA
dimension mismatch (4D↔6D) via zero-padding and truncation.

Usage:
    imm = IMMFilter(dt=0.1, mode="camera")   # or "radar"
    imm.predict()
    imm.update(z)
    pos = imm.position
    is_man = imm.is_maneuvering
"""

from __future__ import annotations

import numpy as np

from sentinel.tracking.filters import (
    ConstantAccelerationEKF,
    ConstantAccelerationKF,
    ExtendedKalmanFilter,
    KalmanFilter,
)


class IMMFilter:
    """Interacting Multiple Model (IMM) estimator.

    Runs two models in parallel:
    - Model 0: Constant Velocity (CV) -- 4D state [x, vx, y, vy]
    - Model 1: Constant Acceleration (CA) -- 6D state [x, vx, ax, y, vy, ay]

    Each cycle:
    1. Mixing: Compute mixed initial conditions for each model
    2. Predict: Run each model's predict
    3. Update: Run each model's update, compute likelihoods
    4. Mode probability update: Bayesian update of mode weights
    5. Combination: Weighted combination of states and covariances

    Args:
        dt: Time step in seconds.
        mode: "camera" for pixel-space KF, "radar" for polar EKF.
        transition_prob: Probability of staying in same mode per step (0-1).
    """

    def __init__(
        self,
        dt: float = 0.1,
        mode: str = "radar",
        transition_prob: float = 0.98,
    ):
        self.dt = dt
        self._mode = mode

        # Create filters
        if mode == "camera":
            self._filters = [
                KalmanFilter(dim_state=4, dim_meas=2, dt=dt),
                ConstantAccelerationKF(dim_state=6, dim_meas=2, dt=dt),
            ]
        else:
            self._filters = [
                ExtendedKalmanFilter(dim_state=4, dim_meas=2, dt=dt),
                ConstantAccelerationEKF(dim_state=6, dim_meas=2, dt=dt),
            ]

        self._n_models = 2
        self._dims = [f.dim_state for f in self._filters]  # [4, 6]

        # Mode probabilities (initially favor CV)
        self.mu = np.array([0.9, 0.1])

        # Markov transition matrix
        p = transition_prob
        self.TPM = np.array(
            [
                [p, 1 - p],
                [1 - p, p],
            ]
        )

        # Combined state (in max-dimension space for output)
        self._combined_x = np.zeros(6)
        self._combined_P = np.eye(6) * 1000.0

    def predict(self) -> np.ndarray:
        """IMM predict: mix → predict each model."""
        self._mixing()
        for f in self._filters:
            f.predict()
        self._combine()
        return self._combined_x[:4].copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        """IMM update: update each model → update mode probs → combine."""
        likelihoods = np.zeros(self._n_models)
        for i, f in enumerate(self._filters):
            likelihoods[i] = self._compute_likelihood(f, z)
            f.update(z)

        # Update mode probabilities
        c = likelihoods @ self.mu
        if c > 1e-300:
            self.mu = (likelihoods * self.mu) / c
        # Prevent numerical collapse
        self.mu = np.clip(self.mu, 1e-6, 1.0 - 1e-6)
        self.mu /= self.mu.sum()

        self._combine()
        return self._combined_x[:4].copy()

    def _mixing(self) -> None:
        """Compute mixing probabilities and mix initial conditions."""
        # Predicted mode probabilities
        c_bar = self.TPM.T @ self.mu  # (n_models,)

        # Mixing probabilities: mu_ij = TPM[i,j] * mu[i] / c_bar[j]
        mixing_probs = np.zeros((self._n_models, self._n_models))
        for j in range(self._n_models):
            if c_bar[j] > 1e-300:
                for i in range(self._n_models):
                    mixing_probs[i, j] = self.TPM[i, j] * self.mu[i] / c_bar[j]

        # Mix states and covariances for each target model
        for j in range(self._n_models):
            dim_j = self._dims[j]

            # Mixed state
            x_mix = np.zeros(dim_j)
            for i in range(self._n_models):
                x_i = self._expand_state(self._filters[i].x, self._dims[i], dim_j)
                x_mix += mixing_probs[i, j] * x_i

            # Mixed covariance
            P_mix = np.zeros((dim_j, dim_j))
            for i in range(self._n_models):
                x_i = self._expand_state(self._filters[i].x, self._dims[i], dim_j)
                P_i = self._expand_covariance(self._filters[i].P, self._dims[i], dim_j)
                dx = x_i - x_mix
                P_mix += mixing_probs[i, j] * (P_i + np.outer(dx, dx))

            self._filters[j].x = x_mix
            self._filters[j].P = P_mix

        self.mu = c_bar

    def _combine(self) -> None:
        """Combine model estimates into a single output."""
        # Combined state in 6D space
        self._combined_x = np.zeros(6)
        for i in range(self._n_models):
            x_i = self._expand_state(self._filters[i].x, self._dims[i], 6)
            self._combined_x += self.mu[i] * x_i

        # Combined covariance
        self._combined_P = np.zeros((6, 6))
        for i in range(self._n_models):
            x_i = self._expand_state(self._filters[i].x, self._dims[i], 6)
            P_i = self._expand_covariance(self._filters[i].P, self._dims[i], 6)
            dx = x_i - self._combined_x
            self._combined_P += self.mu[i] * (P_i + np.outer(dx, dx))

    @staticmethod
    def _expand_state(x: np.ndarray, from_dim: int, to_dim: int) -> np.ndarray:
        """Expand/contract state between CV (4D) and CA (6D).

        CV:  [x, vx,     y, vy    ]
        CA:  [x, vx, ax, y, vy, ay]

        4→6: insert zeros for ax, ay
        6→4: drop ax, ay
        """
        if from_dim == to_dim:
            return x.copy()
        if from_dim == 4 and to_dim == 6:
            return np.array([x[0], x[1], 0.0, x[2], x[3], 0.0])
        if from_dim == 6 and to_dim == 4:
            return np.array([x[0], x[1], x[3], x[4]])
        return x[:to_dim].copy()

    @staticmethod
    def _expand_covariance(P: np.ndarray, from_dim: int, to_dim: int) -> np.ndarray:
        """Expand/contract covariance between CV (4D) and CA (6D)."""
        if from_dim == to_dim:
            return P.copy()

        if from_dim == 4 and to_dim == 6:
            # Map indices: 0→0, 1→1, 2→3, 3→4
            P6 = np.zeros((6, 6))
            idx_map = [0, 1, 3, 4]
            for i, ii in enumerate(idx_map):
                for j, jj in enumerate(idx_map):
                    P6[ii, jj] = P[i, j]
            # Fill acceleration diagonals with reasonable uncertainty
            P6[2, 2] = 100.0
            P6[5, 5] = 100.0
            return P6

        if from_dim == 6 and to_dim == 4:
            # Map indices: 0→0, 1→1, 3→2, 4→3
            P4 = np.zeros((4, 4))
            idx_6_to_4 = [0, 1, 3, 4]
            for i in range(4):
                for j in range(4):
                    P4[i, j] = P[idx_6_to_4[i], idx_6_to_4[j]]
            return P4

        return P[:to_dim, :to_dim].copy()

    def _compute_likelihood(self, f, z: np.ndarray) -> float:
        """Compute Gaussian measurement likelihood for a filter."""
        if hasattr(f, "h"):
            # EKF: nonlinear measurement
            H = f.H_jacobian(f.x)
            y = z - f.h(f.x)
            if len(y) >= 2:
                y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
        else:
            # KF: linear measurement
            H = f.H
            y = z - H @ f.x

        S = H @ f.P @ H.T + f.R
        dim = len(z)

        det_S = np.linalg.det(S)
        if det_S < 1e-300:
            return 1e-300

        exponent = -0.5 * float(y.T @ np.linalg.solve(S, y))
        # Clamp exponent to prevent underflow
        exponent = max(exponent, -500.0)
        normalization = 1.0 / np.sqrt((2 * np.pi) ** dim * det_S)
        return normalization * np.exp(exponent)

    def gating_distance(self, z: np.ndarray) -> float:
        """Gating distance using the combined state (CV model)."""
        return self._filters[0].gating_distance(z)

    @property
    def predicted_measurement(self) -> np.ndarray:
        return self._filters[0].predicted_measurement

    @property
    def innovation_covariance(self) -> np.ndarray:
        return self._filters[0].innovation_covariance

    @property
    def position(self) -> np.ndarray:
        """Combined position [x, y]."""
        return np.array([self._combined_x[0], self._combined_x[3]])

    @property
    def velocity(self) -> np.ndarray:
        """Combined velocity [vx, vy]."""
        return np.array([self._combined_x[1], self._combined_x[4]])

    @property
    def acceleration(self) -> np.ndarray:
        """Combined acceleration [ax, ay]."""
        return np.array([self._combined_x[2], self._combined_x[5]])

    @property
    def x(self) -> np.ndarray:
        """4D state compatible with CV interface: [x, vx, y, vy]."""
        return np.array(
            [
                self._combined_x[0],
                self._combined_x[1],
                self._combined_x[3],
                self._combined_x[4],
            ]
        )

    @x.setter
    def x(self, value: np.ndarray) -> None:
        """Set state from 4D CV vector."""
        if len(value) == 4:
            self._filters[0].x = value.copy()
            self._filters[1].x = self._expand_state(value, 4, 6)
        elif len(value) == 6:
            self._filters[0].x = self._expand_state(value, 6, 4)
            self._filters[1].x = value.copy()

    @property
    def P(self) -> np.ndarray:
        """4D covariance compatible with CV interface."""
        return self._expand_covariance(self._combined_P, 6, 4)

    @P.setter
    def P(self, value: np.ndarray) -> None:
        """Set covariance from 4D matrix."""
        if value.shape == (4, 4):
            self._filters[0].P = value.copy()
            self._filters[1].P = self._expand_covariance(value, 4, 6)

    @property
    def R(self) -> np.ndarray:
        return self._filters[0].R

    @property
    def is_maneuvering(self) -> bool:
        """True if CA model probability > 0.5."""
        return self.mu[1] > 0.5

    @property
    def mode_probabilities(self) -> np.ndarray:
        """[p_cv, p_ca] mode probabilities."""
        return self.mu.copy()

    @property
    def dim_state(self) -> int:
        return 4  # External interface is 4D (CV-compatible)

    @property
    def dim_meas(self) -> int:
        return self._filters[0].dim_meas

    def set_process_noise_std(self, sigma: float) -> None:
        for f in self._filters:
            f.set_process_noise_std(sigma)

    def set_measurement_noise_std(self, sigma: float) -> None:
        if hasattr(self._filters[0], "set_measurement_noise_std"):
            self._filters[0].set_measurement_noise_std(sigma)
        if hasattr(self._filters[1], "set_measurement_noise_std"):
            self._filters[1].set_measurement_noise_std(sigma)

    def set_measurement_noise(self, sigma_range: float, sigma_azimuth_rad: float) -> None:
        for f in self._filters:
            if hasattr(f, "set_measurement_noise"):
                f.set_measurement_noise(sigma_range, sigma_azimuth_rad)
