"""Kalman filter implementations for target tracking.

Phase 2: Standard linear Kalman filter (constant-velocity model).
Phase 4: Extended Kalman filter for sensor fusion.
Phase 5: C++ backend via pybind11.
"""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    """Standard linear Kalman filter for constant-velocity tracking.

    State vector: [x, vx, y, vy] (position and velocity in 2D pixel space).
    Measurement vector: [x, y] (pixel center of bounding box).

    Args:
        dim_state: State vector dimension (default 4).
        dim_meas: Measurement vector dimension (default 2).
        dt: Time step in seconds (default 1/30 for 30 FPS).
    """

    def __init__(self, dim_state: int = 4, dim_meas: int = 2, dt: float = 1 / 30):
        self.dim_state = dim_state
        self.dim_meas = dim_meas
        self.dt = dt

        # State vector
        self.x = np.zeros(dim_state)

        # State covariance -- high initial uncertainty
        self.P = np.eye(dim_state) * 1000.0
        self.P[0, 0] = 10.0  # Position x - lower uncertainty
        self.P[2, 2] = 10.0  # Position y - lower uncertainty

        # State transition matrix (constant velocity model)
        # [x]   [1 dt 0  0] [x]
        # [vx] = [0  1 0  0] [vx]
        # [y]   [0  0 1 dt] [y]
        # [vy]  [0  0 0  1] [vy]
        self.F = np.eye(dim_state)
        self.F[0, 1] = dt
        self.F[2, 3] = dt

        # Observation matrix: we observe [x, y] from state [x, vx, y, vy]
        self.H = np.zeros((dim_meas, dim_state))
        self.H[0, 0] = 1.0  # Observe x
        self.H[1, 2] = 1.0  # Observe y

        # Process noise covariance (continuous white noise acceleration model)
        self.Q = self._build_process_noise(dt)

        # Measurement noise covariance
        self.R = np.eye(dim_meas) * 10.0

    def predict(self) -> np.ndarray:
        """Predict next state. Returns predicted state vector."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        """Update state with measurement z. Returns updated state vector.

        Args:
            z: Measurement vector [x, y] in pixel coordinates.
        """
        # Innovation (measurement residual)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.dim_state) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.x.copy()

    def gating_distance(self, z: np.ndarray) -> float:
        """Mahalanobis distance for gating / association.

        Returns the squared Mahalanobis distance between the predicted
        measurement and the actual measurement z. Used to gate
        unlikely associations.

        A chi-squared threshold of 9.21 (2 DOF, 99% confidence) is typical.
        """
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return float(y.T @ np.linalg.inv(S) @ y)

    @property
    def predicted_measurement(self) -> np.ndarray:
        """Predicted measurement from current state."""
        return self.H @ self.x

    @property
    def innovation_covariance(self) -> np.ndarray:
        """Innovation covariance matrix S = H*P*H' + R."""
        return self.H @ self.P @ self.H.T + self.R

    @property
    def position(self) -> np.ndarray:
        """Current position estimate [x, y]."""
        return np.array([self.x[0], self.x[2]])

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity estimate [vx, vy]."""
        return np.array([self.x[1], self.x[3]])

    def _build_process_noise(self, dt: float, sigma_a: float = 5.0) -> np.ndarray:
        """Build process noise matrix using continuous white noise acceleration model.

        Args:
            dt: Time step.
            sigma_a: Acceleration noise standard deviation (pixels/s^2).
        """
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        # 2x2 block for one axis
        q_block = sigma_a ** 2 * np.array([
            [dt4 / 4, dt3 / 2],
            [dt3 / 2, dt2],
        ])

        Q = np.zeros((self.dim_state, self.dim_state))
        Q[0:2, 0:2] = q_block  # x, vx
        Q[2:4, 2:4] = q_block  # y, vy
        return Q

    def set_process_noise_std(self, sigma_a: float) -> None:
        """Update process noise with new acceleration standard deviation."""
        self.Q = self._build_process_noise(self.dt, sigma_a)

    def set_measurement_noise_std(self, sigma: float) -> None:
        """Update measurement noise with new standard deviation."""
        self.R = np.eye(self.dim_meas) * (sigma ** 2)
