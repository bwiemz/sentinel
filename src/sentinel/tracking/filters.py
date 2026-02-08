"""Kalman filter implementations for target tracking.

Phase 2: Standard linear Kalman filter (constant-velocity model).
Phase 4: Extended Kalman filter for radar fusion.
Phase 5: Bearing-only EKF for thermal tracking.
Phase 7: Constant-acceleration filters (CA-KF, CA-EKF), 3D filters.
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


class ExtendedKalmanFilter:
    """Extended Kalman Filter for radar tracking in Cartesian coordinates.

    State vector: [x, vx, y, vy] in Cartesian (meters, m/s).
    Measurement vector: [range_m, azimuth_rad] in polar coordinates.

    The state transition model is linear (constant velocity, same as KF).
    The measurement model is nonlinear:
        h(x) = [sqrt(x^2 + y^2), atan2(y, x)]

    Args:
        dim_state: State dimension (default 4).
        dim_meas: Measurement dimension (default 2: range, azimuth).
        dt: Time step in seconds (default 0.1 for radar 10 Hz).
    """

    def __init__(self, dim_state: int = 4, dim_meas: int = 2, dt: float = 0.1):
        self.dim_state = dim_state
        self.dim_meas = dim_meas
        self.dt = dt

        # State vector [x, vx, y, vy]
        self.x = np.zeros(dim_state)

        # State covariance -- high initial uncertainty
        self.P = np.eye(dim_state) * 1000.0
        self.P[0, 0] = 100.0   # Position x (meters)
        self.P[2, 2] = 100.0   # Position y (meters)

        # State transition (constant velocity, same as KF)
        self.F = np.eye(dim_state)
        self.F[0, 1] = dt
        self.F[2, 3] = dt

        # Process noise (CWNA model in meters)
        self.Q = self._build_process_noise(dt)

        # Measurement noise: [sigma_range^2, sigma_azimuth^2]
        self.R = np.diag([25.0, np.radians(1.0) ** 2])  # 5m range, 1 deg azimuth

    def predict(self) -> np.ndarray:
        """Predict next state (linear dynamics). Returns predicted state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        """Update state with polar measurement z = [range_m, azimuth_rad].

        Uses analytical Jacobian and angular wrapping on azimuth residual.
        Joseph-form covariance update for numerical stability.
        """
        # Linearize: Jacobian at current state
        H = self.H_jacobian(self.x)

        # Innovation with angular wrapping on azimuth
        y = z - self.h(self.x)
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi  # normalize azimuth residual

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form)
        I_KH = np.eye(self.dim_state) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.x.copy()

    def h(self, x: np.ndarray) -> np.ndarray:
        """Nonlinear measurement function: state → [range, azimuth]."""
        px, _, py, _ = x
        r = np.sqrt(px * px + py * py)
        az = np.arctan2(py, px)
        return np.array([r, az])

    def H_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Analytical Jacobian of h(x), shape (2, 4).

        dh/dx = [[x/r,   0,  y/r,   0],
                  [-y/r², 0,  x/r², 0]]

        Guarded against r ≈ 0.
        """
        px, _, py, _ = x
        r = max(np.sqrt(px * px + py * py), 1e-6)
        r2 = r * r

        return np.array([
            [px / r,    0.0,  py / r,    0.0],
            [-py / r2,  0.0,  px / r2,   0.0],
        ])

    def gating_distance(self, z: np.ndarray) -> float:
        """Squared Mahalanobis distance in polar measurement space.

        Handles angular wrapping on azimuth residual.
        """
        H = self.H_jacobian(self.x)
        y = z - self.h(self.x)
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
        S = H @ self.P @ H.T + self.R
        return float(y.T @ np.linalg.inv(S) @ y)

    @property
    def predicted_measurement(self) -> np.ndarray:
        """Predicted measurement [range, azimuth] from current state."""
        return self.h(self.x)

    @property
    def innovation_covariance(self) -> np.ndarray:
        """Innovation covariance S = H*P*H' + R at current state."""
        H = self.H_jacobian(self.x)
        return H @ self.P @ H.T + self.R

    @property
    def position(self) -> np.ndarray:
        """Current position estimate [x, y] in meters."""
        return np.array([self.x[0], self.x[2]])

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity estimate [vx, vy] in m/s."""
        return np.array([self.x[1], self.x[3]])

    def _build_process_noise(self, dt: float, sigma_a: float = 1.0) -> np.ndarray:
        """CWNA process noise in meters. Default sigma_a=1.0 m/s^2."""
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        q_block = sigma_a ** 2 * np.array([
            [dt4 / 4, dt3 / 2],
            [dt3 / 2, dt2],
        ])

        Q = np.zeros((self.dim_state, self.dim_state))
        Q[0:2, 0:2] = q_block
        Q[2:4, 2:4] = q_block
        return Q

    def set_process_noise_std(self, sigma_a: float) -> None:
        """Update process noise with new acceleration std (m/s^2)."""
        self.Q = self._build_process_noise(self.dt, sigma_a)

    def set_measurement_noise(self, sigma_range: float, sigma_azimuth_rad: float) -> None:
        """Set measurement noise from range (m) and azimuth (rad) std devs."""
        self.R = np.diag([sigma_range ** 2, sigma_azimuth_rad ** 2])


class BearingOnlyEKF:
    """Extended Kalman Filter for bearing-only tracking (thermal sensor).

    State vector: [x, vx, y, vy] in Cartesian (meters).
    Measurement: [azimuth_rad] (bearing only -- no range).

    Bearing-only tracking is inherently unobservable for range from a
    single measurement. Range converges only through:
    - Multiple bearings over time (target motion)
    - Fusion with range-providing sensors (radar)

    Initial range must be assumed or provided externally.

    Args:
        dim_state: State dimension (default 4).
        dt: Time step in seconds.
    """

    def __init__(self, dim_state: int = 4, dt: float = 0.033):
        self.dim_state = dim_state
        self.dim_meas = 1  # bearing only
        self.dt = dt

        # State [x, vx, y, vy]
        self.x = np.zeros(dim_state)

        # High initial uncertainty, especially in range direction
        self.P = np.eye(dim_state) * 10000.0
        self.P[0, 0] = 1e6   # Very uncertain in x (range)
        self.P[2, 2] = 1e6   # Very uncertain in y

        # Constant velocity transition
        self.F = np.eye(dim_state)
        self.F[0, 1] = dt
        self.F[2, 3] = dt

        # Process noise
        self.Q = self._build_process_noise(dt)

        # Measurement noise: bearing only
        self.R = np.array([[np.radians(0.1) ** 2]])  # 0.1 deg

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        """Update with bearing measurement z = [azimuth_rad]."""
        H = self.H_jacobian(self.x)
        y = z - self.h(self.x)
        # Angular wrapping
        y[0] = (y[0] + np.pi) % (2 * np.pi) - np.pi
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(self.dim_state) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.x.copy()

    def h(self, x: np.ndarray) -> np.ndarray:
        """Measurement function: state → [azimuth_rad]."""
        px, _, py, _ = x
        return np.array([np.arctan2(py, px)])

    def H_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of h(x), shape (1, 4).

        dh/dx = [[-y/r², 0, x/r², 0]]
        """
        px, _, py, _ = x
        r2 = max(px * px + py * py, 1e-6)
        return np.array([[-py / r2, 0.0, px / r2, 0.0]])

    def gating_distance(self, z: np.ndarray) -> float:
        """Squared Mahalanobis distance in bearing space."""
        H = self.H_jacobian(self.x)
        y = z - self.h(self.x)
        y[0] = (y[0] + np.pi) % (2 * np.pi) - np.pi
        S = H @ self.P @ H.T + self.R
        return float(y.T @ np.linalg.inv(S) @ y)

    @property
    def predicted_measurement(self) -> np.ndarray:
        return self.h(self.x)

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x[0], self.x[2]])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.x[1], self.x[3]])

    def _build_process_noise(self, dt: float, sigma_a: float = 1.0) -> np.ndarray:
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        q_block = sigma_a ** 2 * np.array([
            [dt4 / 4, dt3 / 2],
            [dt3 / 2, dt2],
        ])
        Q = np.zeros((self.dim_state, self.dim_state))
        Q[0:2, 0:2] = q_block
        Q[2:4, 2:4] = q_block
        return Q

    def set_measurement_noise_std(self, sigma_azimuth_rad: float) -> None:
        self.R = np.array([[sigma_azimuth_rad ** 2]])


class ConstantAccelerationKF:
    """Kalman filter with constant-acceleration motion model.

    State vector: [x, vx, ax, y, vy, ay] (6D).
    Measurement: [x, y] (pixel center).

    Tracks maneuvering targets better than CV model, at the cost of
    higher noise sensitivity during non-maneuvering flight.

    Args:
        dim_state: State dimension (default 6).
        dim_meas: Measurement dimension (default 2).
        dt: Time step in seconds.
    """

    def __init__(self, dim_state: int = 6, dim_meas: int = 2, dt: float = 1 / 30):
        self.dim_state = dim_state
        self.dim_meas = dim_meas
        self.dt = dt

        self.x = np.zeros(dim_state)

        # State covariance
        self.P = np.eye(dim_state) * 1000.0
        self.P[0, 0] = 10.0   # position x
        self.P[3, 3] = 10.0   # position y

        # State transition: constant acceleration
        # [x ]   [1  dt  dt²/2  0  0   0    ] [x ]
        # [vx] = [0  1   dt     0  0   0    ] [vx]
        # [ax]   [0  0   1      0  0   0    ] [ax]
        # [y ]   [0  0   0      1  dt  dt²/2] [y ]
        # [vy]   [0  0   0      0  1   dt   ] [vy]
        # [ay]   [0  0   0      0  0   1    ] [ay]
        dt2 = dt * dt
        self.F = np.eye(dim_state)
        self.F[0, 1] = dt
        self.F[0, 2] = dt2 / 2
        self.F[1, 2] = dt
        self.F[3, 4] = dt
        self.F[3, 5] = dt2 / 2
        self.F[4, 5] = dt

        # Observation: [x, y] from [x, vx, ax, y, vy, ay]
        self.H = np.zeros((dim_meas, dim_state))
        self.H[0, 0] = 1.0  # x
        self.H[1, 3] = 1.0  # y

        self.Q = self._build_process_noise(dt)
        self.R = np.eye(dim_meas) * 10.0

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(self.dim_state) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.x.copy()

    def gating_distance(self, z: np.ndarray) -> float:
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return float(y.T @ np.linalg.inv(S) @ y)

    @property
    def predicted_measurement(self) -> np.ndarray:
        return self.H @ self.x

    @property
    def innovation_covariance(self) -> np.ndarray:
        return self.H @ self.P @ self.H.T + self.R

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x[0], self.x[3]])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.x[1], self.x[4]])

    @property
    def acceleration(self) -> np.ndarray:
        return np.array([self.x[2], self.x[5]])

    def _build_process_noise(self, dt: float, sigma_j: float = 1.0) -> np.ndarray:
        """CWNA process noise for constant-acceleration model.

        Driven by jerk (derivative of acceleration). sigma_j is jerk std.
        """
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        dt5 = dt4 * dt
        dt6 = dt5 * dt

        # 3x3 block for one axis [pos, vel, acc]
        q_block = sigma_j ** 2 * np.array([
            [dt6 / 36, dt5 / 12, dt4 / 6],
            [dt5 / 12, dt4 / 4,  dt3 / 2],
            [dt4 / 6,  dt3 / 2,  dt2],
        ])

        Q = np.zeros((self.dim_state, self.dim_state))
        Q[0:3, 0:3] = q_block  # x, vx, ax
        Q[3:6, 3:6] = q_block  # y, vy, ay
        return Q

    def set_process_noise_std(self, sigma_j: float) -> None:
        self.Q = self._build_process_noise(self.dt, sigma_j)

    def set_measurement_noise_std(self, sigma: float) -> None:
        self.R = np.eye(self.dim_meas) * (sigma ** 2)


class ConstantAccelerationEKF:
    """Extended Kalman Filter with constant-acceleration motion model.

    State vector: [x, vx, ax, y, vy, ay] (6D) in meters.
    Measurement: [range_m, azimuth_rad] (2D polar).

    For radar tracking of maneuvering targets.

    Args:
        dim_state: State dimension (default 6).
        dim_meas: Measurement dimension (default 2).
        dt: Time step in seconds.
    """

    def __init__(self, dim_state: int = 6, dim_meas: int = 2, dt: float = 0.1):
        self.dim_state = dim_state
        self.dim_meas = dim_meas
        self.dt = dt

        self.x = np.zeros(dim_state)
        self.P = np.eye(dim_state) * 1000.0
        self.P[0, 0] = 100.0
        self.P[3, 3] = 100.0

        # CA transition
        dt2 = dt * dt
        self.F = np.eye(dim_state)
        self.F[0, 1] = dt
        self.F[0, 2] = dt2 / 2
        self.F[1, 2] = dt
        self.F[3, 4] = dt
        self.F[3, 5] = dt2 / 2
        self.F[4, 5] = dt

        self.Q = self._build_process_noise(dt)
        self.R = np.diag([25.0, np.radians(1.0) ** 2])

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        H = self.H_jacobian(self.x)
        y = z - self.h(self.x)
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(self.dim_state) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.x.copy()

    def h(self, x: np.ndarray) -> np.ndarray:
        """Measurement function: [range, azimuth] from state."""
        px, _, _, py, _, _ = x
        r = np.sqrt(px * px + py * py)
        az = np.arctan2(py, px)
        return np.array([r, az])

    def H_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of h(x), shape (2, 6)."""
        px, _, _, py, _, _ = x
        r = max(np.sqrt(px * px + py * py), 1e-6)
        r2 = r * r
        return np.array([
            [px / r,  0.0, 0.0, py / r,  0.0, 0.0],
            [-py / r2, 0.0, 0.0, px / r2, 0.0, 0.0],
        ])

    def gating_distance(self, z: np.ndarray) -> float:
        H = self.H_jacobian(self.x)
        y = z - self.h(self.x)
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
        S = H @ self.P @ H.T + self.R
        return float(y.T @ np.linalg.inv(S) @ y)

    @property
    def predicted_measurement(self) -> np.ndarray:
        return self.h(self.x)

    @property
    def innovation_covariance(self) -> np.ndarray:
        H = self.H_jacobian(self.x)
        return H @ self.P @ H.T + self.R

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x[0], self.x[3]])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.x[1], self.x[4]])

    @property
    def acceleration(self) -> np.ndarray:
        return np.array([self.x[2], self.x[5]])

    def _build_process_noise(self, dt: float, sigma_j: float = 1.0) -> np.ndarray:
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        dt5 = dt4 * dt
        dt6 = dt5 * dt
        q_block = sigma_j ** 2 * np.array([
            [dt6 / 36, dt5 / 12, dt4 / 6],
            [dt5 / 12, dt4 / 4,  dt3 / 2],
            [dt4 / 6,  dt3 / 2,  dt2],
        ])
        Q = np.zeros((self.dim_state, self.dim_state))
        Q[0:3, 0:3] = q_block
        Q[3:6, 3:6] = q_block
        return Q

    def set_process_noise_std(self, sigma_j: float) -> None:
        self.Q = self._build_process_noise(self.dt, sigma_j)

    def set_measurement_noise(self, sigma_range: float, sigma_azimuth_rad: float) -> None:
        self.R = np.diag([sigma_range ** 2, sigma_azimuth_rad ** 2])


class KalmanFilter3D:
    """3D Kalman filter for position tracking in Cartesian space.

    State vector: [x, vx, y, vy, z, vz] (6D).
    Measurement: [x, y, z] (3D Cartesian).

    Args:
        dt: Time step in seconds.
    """

    def __init__(self, dt: float = 0.1):
        self.dim_state = 6
        self.dim_meas = 3
        self.dt = dt

        self.x = np.zeros(6)
        self.P = np.eye(6) * 1000.0
        self.P[0, 0] = 100.0
        self.P[2, 2] = 100.0
        self.P[4, 4] = 100.0

        # Constant velocity transition: [x,vx,y,vy,z,vz]
        self.F = np.eye(6)
        self.F[0, 1] = dt
        self.F[2, 3] = dt
        self.F[4, 5] = dt

        # Observe [x, y, z]
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 2] = 1.0
        self.H[2, 4] = 1.0

        self.Q = self._build_process_noise(dt)
        self.R = np.eye(3) * 25.0

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(self.dim_state) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.x.copy()

    def gating_distance(self, z: np.ndarray) -> float:
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return float(y.T @ np.linalg.inv(S) @ y)

    @property
    def predicted_measurement(self) -> np.ndarray:
        return self.H @ self.x

    @property
    def innovation_covariance(self) -> np.ndarray:
        return self.H @ self.P @ self.H.T + self.R

    @property
    def position(self) -> np.ndarray:
        """[x, y, z] in meters."""
        return np.array([self.x[0], self.x[2], self.x[4]])

    @property
    def velocity(self) -> np.ndarray:
        """[vx, vy, vz] in m/s."""
        return np.array([self.x[1], self.x[3], self.x[5]])

    def _build_process_noise(self, dt: float, sigma_a: float = 1.0) -> np.ndarray:
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        q_block = sigma_a ** 2 * np.array([
            [dt4 / 4, dt3 / 2],
            [dt3 / 2, dt2],
        ])
        Q = np.zeros((6, 6))
        Q[0:2, 0:2] = q_block
        Q[2:4, 2:4] = q_block
        Q[4:6, 4:6] = q_block
        return Q

    def set_process_noise_std(self, sigma_a: float) -> None:
        self.Q = self._build_process_noise(self.dt, sigma_a)

    def set_measurement_noise_std(self, sigma: float) -> None:
        self.R = np.eye(3) * (sigma ** 2)


class ExtendedKalmanFilter3D:
    """3D Extended Kalman Filter for radar tracking.

    State vector: [x, vx, y, vy, z, vz] (6D) in meters.
    Measurement: [range_m, azimuth_rad, elevation_rad] (3D polar).

    h(x) = [sqrt(x^2 + y^2 + z^2), atan2(y, x), atan2(z, sqrt(x^2+y^2))]

    Args:
        dt: Time step in seconds.
    """

    def __init__(self, dt: float = 0.1):
        self.dim_state = 6
        self.dim_meas = 3
        self.dt = dt

        self.x = np.zeros(6)
        self.P = np.eye(6) * 1000.0
        self.P[0, 0] = 100.0
        self.P[2, 2] = 100.0
        self.P[4, 4] = 100.0

        # Constant velocity transition
        self.F = np.eye(6)
        self.F[0, 1] = dt
        self.F[2, 3] = dt
        self.F[4, 5] = dt

        self.Q = self._build_process_noise(dt)
        self.R = np.diag([25.0, np.radians(1.0) ** 2, np.radians(1.0) ** 2])

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        H = self.H_jacobian(self.x)
        y = z - self.h(self.x)
        # Angular wrapping on azimuth and elevation
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(self.dim_state) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.x.copy()

    def h(self, x: np.ndarray) -> np.ndarray:
        """Measurement function: [range, azimuth, elevation]."""
        px, _, py, _, pz, _ = x
        r_xy = np.sqrt(px * px + py * py)
        r = np.sqrt(px * px + py * py + pz * pz)
        az = np.arctan2(py, px)
        el = np.arctan2(pz, r_xy)
        return np.array([r, az, el])

    def H_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Analytical Jacobian of h(x), shape (3, 6).

        Row 0: d(range)/d(state)
        Row 1: d(azimuth)/d(state)
        Row 2: d(elevation)/d(state)
        """
        px, _, py, _, pz, _ = x
        r_xy_sq = max(px * px + py * py, 1e-12)
        r_xy = np.sqrt(r_xy_sq)
        r_sq = r_xy_sq + pz * pz
        r = max(np.sqrt(r_sq), 1e-6)

        H = np.zeros((3, 6))
        # d(range)/dx
        H[0, 0] = px / r
        H[0, 2] = py / r
        H[0, 4] = pz / r

        # d(azimuth)/dx: same as 2D
        H[1, 0] = -py / r_xy_sq
        H[1, 2] = px / r_xy_sq

        # d(elevation)/dx
        # el = atan2(z, r_xy)
        # del/dpx = -pz*px / (r^2 * r_xy)
        # del/dpy = -pz*py / (r^2 * r_xy)
        # del/dpz = r_xy / r^2
        r_xy_safe = max(r_xy, 1e-6)
        H[2, 0] = -pz * px / (r_sq * r_xy_safe)
        H[2, 2] = -pz * py / (r_sq * r_xy_safe)
        H[2, 4] = r_xy_safe / r_sq

        return H

    def gating_distance(self, z: np.ndarray) -> float:
        H = self.H_jacobian(self.x)
        y = z - self.h(self.x)
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi
        S = H @ self.P @ H.T + self.R
        return float(y.T @ np.linalg.inv(S) @ y)

    @property
    def predicted_measurement(self) -> np.ndarray:
        return self.h(self.x)

    @property
    def innovation_covariance(self) -> np.ndarray:
        H = self.H_jacobian(self.x)
        return H @ self.P @ H.T + self.R

    @property
    def position(self) -> np.ndarray:
        """[x, y, z] in meters."""
        return np.array([self.x[0], self.x[2], self.x[4]])

    @property
    def velocity(self) -> np.ndarray:
        """[vx, vy, vz] in m/s."""
        return np.array([self.x[1], self.x[3], self.x[5]])

    def _build_process_noise(self, dt: float, sigma_a: float = 1.0) -> np.ndarray:
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        q_block = sigma_a ** 2 * np.array([
            [dt4 / 4, dt3 / 2],
            [dt3 / 2, dt2],
        ])
        Q = np.zeros((6, 6))
        Q[0:2, 0:2] = q_block
        Q[2:4, 2:4] = q_block
        Q[4:6, 4:6] = q_block
        return Q

    def set_process_noise_std(self, sigma_a: float) -> None:
        self.Q = self._build_process_noise(self.dt, sigma_a)

    def set_measurement_noise(
        self, sigma_range: float, sigma_azimuth_rad: float, sigma_elevation_rad: float,
    ) -> None:
        self.R = np.diag([
            sigma_range ** 2, sigma_azimuth_rad ** 2, sigma_elevation_rad ** 2,
        ])


class ExtendedKalmanFilterWithDoppler:
    """Extended Kalman Filter with Doppler (radial velocity) measurement.

    State vector: [x, vx, y, vy] (4D) in meters/m/s.
    Measurement: [range_m, azimuth_rad, radial_velocity_mps] (3D).

    The radial velocity measurement:
        v_r = (x*vx + y*vy) / sqrt(x^2 + y^2)

    Doppler provides direct velocity observability, enabling faster
    convergence and better track accuracy for radar targets.

    Args:
        dim_state: State dimension (default 4).
        dt: Time step in seconds.
    """

    def __init__(self, dim_state: int = 4, dt: float = 0.1):
        self.dim_state = dim_state
        self.dim_meas = 3
        self.dt = dt

        self.x = np.zeros(dim_state)
        self.P = np.eye(dim_state) * 1000.0
        self.P[0, 0] = 100.0
        self.P[2, 2] = 100.0

        # Constant velocity transition
        self.F = np.eye(dim_state)
        self.F[0, 1] = dt
        self.F[2, 3] = dt

        self.Q = self._build_process_noise(dt)
        # Measurement noise: [range, azimuth, radial_velocity]
        self.R = np.diag([25.0, np.radians(1.0) ** 2, 0.25])  # 5m, 1deg, 0.5m/s

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        """Update with measurement z = [range, azimuth_rad, radial_vel_mps]."""
        H = self.H_jacobian(self.x)
        y = z - self.h(self.x)
        # Angular wrapping on azimuth
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(self.dim_state) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.x.copy()

    def h(self, x: np.ndarray) -> np.ndarray:
        """Measurement function: [range, azimuth, radial_velocity]."""
        px, vx, py, vy = x
        r = max(np.sqrt(px * px + py * py), 1e-6)
        az = np.arctan2(py, px)
        v_r = (px * vx + py * vy) / r
        return np.array([r, az, v_r])

    def H_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Analytical Jacobian of h(x), shape (3, 4).

        Row 0: d(range)/d(state)     = [px/r, 0, py/r, 0]
        Row 1: d(azimuth)/d(state)   = [-py/r², 0, px/r², 0]
        Row 2: d(v_r)/d(state):
            dv_r/dpx = vx/r - px*(px*vx + py*vy)/r³
            dv_r/dvx = px/r
            dv_r/dpy = vy/r - py*(px*vx + py*vy)/r³
            dv_r/dvy = py/r
        """
        px, vx, py, vy = x
        r = max(np.sqrt(px * px + py * py), 1e-6)
        r2 = r * r
        r3 = r2 * r
        dot_pv = px * vx + py * vy

        H = np.zeros((3, 4))
        # Range row
        H[0, 0] = px / r
        H[0, 2] = py / r
        # Azimuth row
        H[1, 0] = -py / r2
        H[1, 2] = px / r2
        # Radial velocity row
        H[2, 0] = vx / r - px * dot_pv / r3
        H[2, 1] = px / r
        H[2, 2] = vy / r - py * dot_pv / r3
        H[2, 3] = py / r
        return H

    def gating_distance(self, z: np.ndarray) -> float:
        H = self.H_jacobian(self.x)
        y = z - self.h(self.x)
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
        S = H @ self.P @ H.T + self.R
        return float(y.T @ np.linalg.inv(S) @ y)

    @property
    def predicted_measurement(self) -> np.ndarray:
        return self.h(self.x)

    @property
    def innovation_covariance(self) -> np.ndarray:
        H = self.H_jacobian(self.x)
        return H @ self.P @ H.T + self.R

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x[0], self.x[2]])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.x[1], self.x[3]])

    def _build_process_noise(self, dt: float, sigma_a: float = 1.0) -> np.ndarray:
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        q_block = sigma_a ** 2 * np.array([
            [dt4 / 4, dt3 / 2],
            [dt3 / 2, dt2],
        ])
        Q = np.zeros((self.dim_state, self.dim_state))
        Q[0:2, 0:2] = q_block
        Q[2:4, 2:4] = q_block
        return Q

    def set_process_noise_std(self, sigma_a: float) -> None:
        self.Q = self._build_process_noise(self.dt, sigma_a)

    def set_measurement_noise(
        self, sigma_range: float, sigma_azimuth_rad: float, sigma_velocity: float,
    ) -> None:
        self.R = np.diag([
            sigma_range ** 2, sigma_azimuth_rad ** 2, sigma_velocity ** 2,
        ])
