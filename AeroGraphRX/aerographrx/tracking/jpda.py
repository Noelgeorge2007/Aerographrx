"""Graph-Enhanced JPDA Tracking (Algorithm 1).

Implements Joint Probabilistic Data Association (JPDA) tracking with
graph-based association weighting for networked signal sources.
"""

import numpy as np
from scipy import stats


class JPDATracker:
    """Graph-Enhanced JPDA tracker for multi-target tracking.

    Implements Algorithm 1 combining JPDA with graph-based association
    probabilities for correlated source tracking.

    Args:
        F: State transition matrix of shape (state_dim, state_dim).
        H: Measurement matrix of shape (meas_dim, state_dim).
        Q: Process noise covariance of shape (state_dim, state_dim).
        R: Measurement noise covariance of shape (meas_dim, meas_dim).
        gate_threshold: Mahalanobis distance gating threshold (default: 16.0).
    """

    def __init__(self, F, H, Q, R, gate_threshold=16.0):
        """Initialize JPDA tracker.

        Args:
            F: State transition matrix (6x6 for constant velocity model).
            H: Measurement matrix.
            Q: Process noise covariance.
            R: Measurement noise covariance.
            gate_threshold: Mahalanobis gating threshold.
        """
        self.F = F  # State transition
        self.H = H  # Measurement matrix
        self.Q = Q  # Process noise
        self.R = R  # Measurement noise
        self.gate_threshold = gate_threshold

        # Initialize state
        state_dim = F.shape[0]
        self.x = np.zeros((state_dim, 1))  # State vector
        self.P = np.eye(state_dim)  # State covariance

        self.n_targets = 1

    def predict(self):
        """Predict step: x_pred = F @ x, P_pred = F @ P @ F^T + Q.

        Updates internal state and covariance estimates.

        Returns:
            Tuple of (x_pred, P_pred).
        """
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        return x_pred, P_pred

    def gate(self, measurements, S):
        """Mahalanobis distance gating.

        Determines which measurements are plausible for association.

        Args:
            measurements: Measurement matrix of shape (n_meas, meas_dim).
            S: Measurement covariance (innovation covariance) of shape
               (meas_dim, meas_dim).

        Returns:
            Boolean gate matrix of shape (n_meas,) where True indicates
            measurement is within gate.
        """
        n_meas = measurements.shape[0]
        gated = np.zeros(n_meas, dtype=bool)

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Singular matrix, use pseudoinverse
            S_inv = np.linalg.pinv(S)

        for j in range(n_meas):
            measurement = measurements[j:j+1, :].T
            x_pred = self.F @ self.x
            predicted_meas = self.H @ x_pred
            innovation = measurement - predicted_meas  # (meas_dim, 1)

            # Mahalanobis distance
            maha_dist = float(innovation.T @ S_inv @ innovation)

            # Check against threshold
            gated[j] = maha_dist <= self.gate_threshold

        return gated

    def compute_association_probabilities(self, innovations, S, temporal_weights):
        """Compute JPDA association probabilities (Eq. 17).

        beta_j = W_j * N(nu_j; 0, S) / sum_i(W_i * N(nu_i; 0, S))

        Args:
            innovations: Innovation vectors of shape (n_meas, meas_dim).
            S: Innovation covariance of shape (meas_dim, meas_dim).
            temporal_weights: Temporal weights for each measurement of shape
                             (n_meas,).

        Returns:
            Association probabilities of shape (n_meas,).
        """
        n_meas = innovations.shape[0]
        meas_dim = innovations.shape[1]

        if n_meas == 0:
            return np.array([])

        # Compute likelihoods
        try:
            S_inv = np.linalg.inv(S)
            det_S = np.linalg.det(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            det_S = np.linalg.det(S + 1e-6 * np.eye(meas_dim))

        if det_S <= 0:
            det_S = 1e-10

        # Normalization factor
        norm = np.sqrt((2 * np.pi) ** meas_dim * det_S)

        likelihoods = np.zeros(n_meas)
        for j in range(n_meas):
            nu = innovations[j:j+1, :].T
            exponent = -0.5 * float(nu.T @ S_inv @ nu)
            likelihoods[j] = np.exp(exponent) / norm

        # Weighted likelihoods
        weighted_likelihoods = temporal_weights * likelihoods

        # Normalize to get probabilities
        total = np.sum(weighted_likelihoods)
        if total > 1e-10:
            probabilities = weighted_likelihoods / total
        else:
            probabilities = np.ones(n_meas) / n_meas

        return probabilities

    def update(self, measurements, graph):
        """JPDA update step (Algorithm 1).

        Steps:
            1. Predict next state
            2. Gate measurements
            3. Compute graph-weighted association probabilities
            4. Compute combined innovation
            5. Update state and covariance

        Args:
            measurements: Measurement matrix of shape (n_meas, meas_dim).
            graph: Graph adjacency matrix of shape (n_targets, n_targets)
                   or scalar weights for single target.

        Returns:
            Dictionary containing:
                - 'x': Updated state
                - 'P': Updated covariance
                - 'gated_measurements': Boolean mask of gated measurements
                - 'association_probabilities': Association probabilities
        """
        # Step 1: Predict
        x_pred, P_pred = self.predict()

        # Predicted measurement and innovation covariance
        z_pred = self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R

        # Step 2: Gate measurements
        gated_mask = self.gate(measurements, S)
        gated_measurements = measurements[gated_mask]

        if len(gated_measurements) == 0:
            # No gated measurements: use prediction
            self.x = x_pred
            self.P = P_pred
            return {
                'x': self.x,
                'P': self.P,
                'gated_measurements': gated_mask,
                'association_probabilities': np.array([])
            }

        # Step 3: Compute innovations for gated measurements
        innovations = gated_measurements - z_pred.T  # (n_gated, meas_dim)

        # Step 4: Graph-weighted association probabilities
        # Use graph structure to weight measurements
        n_gated = gated_measurements.shape[0]
        if isinstance(graph, np.ndarray) and graph.size > 1:
            # Multi-target: use graph weights
            temporal_weights = np.ones(n_gated)
            for i in range(n_gated):
                if i < graph.shape[0]:
                    temporal_weights[i] = np.sum(graph[i, :])
            temporal_weights = temporal_weights / np.sum(temporal_weights)
        else:
            # Single target: uniform weights
            temporal_weights = np.ones(n_gated) / n_gated

        beta = self.compute_association_probabilities(innovations, S, temporal_weights)

        # Step 5: Combined innovation (weighted average)
        combined_innovation = np.sum(beta[:, np.newaxis] * innovations, axis=0)

        # Update state and covariance
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        K = P_pred @ self.H.T @ S_inv  # Kalman gain

        self.x = x_pred + K @ combined_innovation.reshape(-1, 1)
        self.P = P_pred - K @ S @ K.T

        return {
            'x': self.x,
            'P': self.P,
            'gated_measurements': gated_mask,
            'association_probabilities': beta
        }

    def track(self, all_measurements, graph_sequence, n_steps):
        """Run full JPDA tracking over time series.

        Args:
            all_measurements: Sequence of measurement sets, list of arrays
                             where each element is (n_meas_t, meas_dim).
            graph_sequence: Sequence of graph adjacency matrices, one per timestep.
            n_steps: Total number of tracking steps.

        Returns:
            Dictionary containing:
                - 'track_history': List of states over time
                - 'covariance_history': List of covariances over time
                - 'measurements_history': List of measurement sets
                - 'gate_history': List of gating masks
                - 'continuity_metric': Track continuity score
        """
        track_history = []
        covariance_history = []
        gate_history = []

        for t in range(n_steps):
            # Get measurements and graph for this timestep
            if t < len(all_measurements):
                measurements = all_measurements[t]
            else:
                measurements = np.empty((0, self.H.shape[1]))

            if t < len(graph_sequence):
                graph = graph_sequence[t]
            else:
                graph = np.eye(self.n_targets)

            # Update
            result = self.update(measurements, graph)

            # Store history
            track_history.append(result['x'].copy())
            covariance_history.append(result['P'].copy())
            gate_history.append(result['gated_measurements'])

        # Compute continuity metric
        # Count number of timesteps with measurements
        timesteps_with_meas = sum(1 for m in all_measurements if len(m) > 0)
        continuity_metric = timesteps_with_meas / max(1, n_steps)

        return {
            'track_history': track_history,
            'covariance_history': covariance_history,
            'measurements_history': all_measurements,
            'gate_history': gate_history,
            'continuity_metric': continuity_metric
        }


class NNTracker:
    """Nearest-Neighbor (NN) baseline tracker.

    Simple association method using Euclidean distance for comparison
    with JPDA.

    Args:
        F: State transition matrix of shape (state_dim, state_dim).
        H: Measurement matrix of shape (meas_dim, state_dim).
        Q: Process noise covariance of shape (state_dim, state_dim).
        R: Measurement noise covariance of shape (meas_dim, meas_dim).
        distance_threshold: Maximum Euclidean distance for association
                          (default: 10.0).
    """

    def __init__(self, F, H, Q, R, distance_threshold=10.0):
        """Initialize NN tracker.

        Args:
            F: State transition matrix.
            H: Measurement matrix.
            Q: Process noise covariance.
            R: Measurement noise covariance.
            distance_threshold: Association distance threshold.
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.distance_threshold = distance_threshold

        # Initialize state
        state_dim = F.shape[0]
        self.x = np.zeros((state_dim, 1))
        self.P = np.eye(state_dim)

    def track(self, all_measurements, n_steps):
        """Run NN tracking over time series.

        Args:
            all_measurements: Sequence of measurement sets, list of arrays
                             where each element is (n_meas_t, meas_dim).
            n_steps: Total number of tracking steps.

        Returns:
            Dictionary containing:
                - 'track_history': List of states over time
                - 'covariance_history': List of covariances over time
                - 'association_history': List of associations
                - 'continuity_metric': Track continuity score
        """
        track_history = []
        covariance_history = []
        association_history = []

        for t in range(n_steps):
            # Predict
            x_pred = self.F @ self.x
            P_pred = self.F @ self.P @ self.F.T + self.Q

            # Predicted measurement
            z_pred = self.H @ x_pred

            # Get measurements for this timestep
            if t < len(all_measurements):
                measurements = all_measurements[t]
            else:
                measurements = np.empty((0, self.H.shape[1]))

            # Nearest neighbor association
            if len(measurements) > 0:
                distances = np.linalg.norm(
                    measurements - z_pred.T, axis=1
                )
                best_idx = np.argmin(distances)

                if distances[best_idx] <= self.distance_threshold:
                    # Associate with nearest measurement
                    z = measurements[best_idx:best_idx+1, :].T
                    associated = True
                else:
                    # No association
                    self.x = x_pred
                    self.P = P_pred
                    associated = False
            else:
                # No measurements
                self.x = x_pred
                self.P = P_pred
                associated = False

            # Kalman update if associated
            if associated:
                S = self.H @ P_pred @ self.H.T + self.R
                try:
                    S_inv = np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    S_inv = np.linalg.pinv(S)

                K = P_pred @ self.H.T @ S_inv
                innovation = z - (self.H @ x_pred)
                self.x = x_pred + K @ innovation
                self.P = P_pred - K @ S @ K.T
            else:
                self.x = x_pred
                self.P = P_pred

            # Store history
            track_history.append(self.x.copy())
            covariance_history.append(self.P.copy())
            association_history.append(associated)

        # Compute continuity metric
        timesteps_with_association = sum(association_history)
        continuity_metric = timesteps_with_association / max(1, n_steps)

        return {
            'track_history': track_history,
            'covariance_history': covariance_history,
            'association_history': association_history,
            'continuity_metric': continuity_metric
        }
