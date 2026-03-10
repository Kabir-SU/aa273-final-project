import numpy as np
from scipy.linalg import block_diag

# State vector layout (18 states total):
#   Own drone : [px, vx, py, vy, pz, vz]    indices 0-5
#   Target 1  : [x,  vx, y,  vy, z,  vz]    indices 6-11
#   Target 2  : [x,  vx, y,  vy, z,  vz]    indices 12-17
#
# We use a constant-velocity (CV) model for targets rather than the 9-state
# constant-acceleration model. Target accelerations are UNOBSERVABLE from
# range/bearing measurements alone, so the CA model causes filter divergence.

class ConsensusEKF:
    """Consensus EKF with constant-velocity target model (18-state)."""

    def __init__(self, 
                 init_mu, 
                 init_cov, 
                 dt=0.01, 
                 m=20, 
                 num_targets=2, 
                 landmark_track=False, 
                 landmark_pos=np.array([0., 0., 0.])
        ):
        
        self.dt = dt
        self.m = m
        self.g = 9.81
        self.num_targets = num_targets
        # 18 states!
        self.num_states = 6 + 6 * num_targets
        self.track_landmark = landmark_track

        self.state_est = init_mu.copy()
        self.prior_state = init_mu.copy()
        self.cov_est = init_cov.copy()
        self.prior_cov = init_cov.copy()

        # Measurement noise:
        # Gryo standard deviation directly impact dead reckoning
        sigma_gyro = 0.01
        sigma_yaw_pitch = sigma_gyro**2
        # Dead reckoning input uncertainty
        R_dr = np.diag([0., sigma_yaw_pitch**2, sigma_yaw_pitch**2])
        
        # Altimeter, Range, Angle measurement standard deviations
        altimeter_sigma, range_std, angle_std = 0.1, 0.5, 0.5
        # Measurement uncertainty
        R_meas = np.diag([range_std**2, np.deg2rad(angle_std)**2, np.deg2rad(angle_std)**2])
        
        R_target = R_meas + R_dr
        R_alt = np.array([[altimeter_sigma**2]])
        self.R = block_diag(R_target, R_target, R_alt)
        
        # If handling the leader, different uncertainties are used
        # If using the true sensor specs, the filter becomes smug/overconfident
        # Therefore the following are tuned such that the leader is not overconfident
        if self.track_landmark:
            # Handlind uncertainty from dead reckoning input
            sigma_ang = np.deg2rad(3.)
            R_dr = np.diag([0., sigma_ang**2, sigma_ang**2])
            # Tuned Altimeter, Range, Angle measurement standard deviations
            altimeter_sigma, range_std, angle_std = 0.5, 1.0, 1.0
            # Measurement uncertainty
            R_meas = np.diag([range_std**2, np.deg2rad(angle_std)**2, np.deg2rad(angle_std)**2])
            R_target = R_meas + R_dr
            # Landmark measurement uncertainty is placed much higher as it is the global anchor
            R_landmark = np.diag([10.0**2, np.deg2rad(3.0)**2, np.deg2rad(3.0)**2]) + R_dr
            
            self.R = block_diag(R_landmark, R_target, R_target, R_alt)

        # Process noise
        drone_accel_var = 10.0
        target_accel_var = 5.0
        q0 = np.array([
            [self.dt**4 / 4, self.dt**3 / 2],
            [self.dt**3 / 2, self.dt**2]
        ])

        # Define the drone Q and target Q
        Q_own = 10*drone_accel_var**2 * block_diag(q0, q0, q0)
        Q_tgt = 10*target_accel_var**2 * block_diag(q0, q0, q0)

        self.Q = block_diag(Q_own, Q_tgt, Q_tgt)

        self.state_est_time_hist = [init_mu.copy()]
        self.cov_est_time_hist = [init_cov.copy()]

        self.A = self.get_dynamics_jacobian()
        self.landmark_pos = landmark_pos

    def get_dynamics_jacobian(self):
        cv = np.array([[1., self.dt], [0., 1.]])
        block_cv = block_diag(cv, cv, cv)
        A = np.zeros((self.num_states, self.num_states))
        for k in range(1 + self.num_targets):
            s = 6 * k
            A[s:s+6, s:s+6] = block_cv
        return A

    def dynamics(self, state, attitude_state, control):
        phi, theta, psi = attitude_state[:3]
        F = control[0]

        cphi = np.cos(phi)
        sphi = np.sin(phi)
        cth = np.cos(theta)
        sth = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)

        ax = (-cphi * sth * cpsi - sphi * spsi) * F / self.m
        ay = -(-cphi * sth * spsi - sphi * cpsi) * F / self.m
        az = -self.g + (cphi * cth) * F / self.m

        own = state[:6]
        new_own = np.zeros(6)
        for i, a in enumerate([ax, ay, az]):
            new_own[2*i] = own[2*i] + own[2*i+1] * self.dt
            new_own[2*i+1] = own[2*i+1] + a * self.dt

        new_targets = []
        for i in range(self.num_targets):
            t = state[6 + 6*i: 6 + 6*(i+1)]
            nt = np.zeros(6)
            for j in range(3):
                nt[2*j] = t[2*j] + t[2*j+1] * self.dt
                nt[2*j+1] = t[2*j+1]
            new_targets.append(nt)

        return np.concatenate([new_own] + new_targets)

    def measurements(self, state, attitude_state):
        theta, psi = attitude_state[1], attitude_state[2]
        px, py, pz = state[0], state[2], state[4]
        meas = np.zeros(3 * self.num_targets + 1)
        if self.track_landmark:
            meas = np.zeros(3 * (self.num_targets + 1) + 1)
        for i in range(self.num_targets):
            t = state[6 + 6*i: 6 + 6*(i+1)]
            dx = t[0] - px
            dy = t[2] - py
            dz = t[4] - pz
            eps = 1e-8
            r = np.linalg.norm([dx, dy, dz]) + eps
            az = np.arctan2(dy, dx) - psi
            el = np.arcsin(np.clip(dz / r, -1., 1.)) - theta
            meas[3*i:3*i+3] = [r, az, el]
        if self.track_landmark:
            z_landmark = self.landmark_measurement(state, attitude_state)
            meas[6:9] = z_landmark
        altimeter_meas = pz
        meas[-1] = altimeter_meas
        return meas

    def get_measurement_jacobian(self, state):
        H = np.zeros((3 * self.num_targets + 1, self.num_states))
        if self.track_landmark:
            H = np.zeros((3 * (self.num_targets + 1) + 1, self.num_states))
        
        px, py, pz = state[0], state[2], state[4]
        for i in range(self.num_targets):
            t = state[6 + 6*i: 6 + 6*(i+1)]
            dx = t[0] - px
            dy = t[2] - py
            dz = t[4] - pz
            eps = 1e-8
            r = np.linalg.norm([dx, dy, dz]) + eps
            s = np.linalg.norm([dx, dy]) + eps
            row = 3 * i
            ct = 6 + 6 * i

            # Target position partials
            H[row, ct] = dx/r
            H[row, ct+2] = dy/r
            H[row, ct+4] = dz/r
            H[row+1, ct] = -dy/s**2
            H[row+1, ct+2] = dx/s**2
            H[row+2, ct] = -dx*dz/(r**2*s)
            H[row+2, ct+2] = -dy*dz/(r**2*s)
            H[row+2, ct+4] = s/r**2

            # Own drone position partials (cols 0, 2, 4)
            H[row, 0] = -dx/r
            H[row, 2] = -dy/r
            H[row, 4] = -dz/r
            H[row+1, 0] = dy/s**2; H[row+1, 2] = -dx/s**2
            H[row+2, 0] = dx*dz/(r**2*s)
            H[row+2, 2] = dy*dz/(r**2*s)
            H[row+2, 4] = -s/r**2
        
        if self.track_landmark:
            row = 3 * self.num_targets
            H_L = self.get_landmark_jacobian(state)
            H[row:row+3, :6] = H_L

        H[-1, 4] = 1
        return H

    def landmark_measurement(self, state, attitude_state):
        px, py, pz = state[0], state[2], state[4]
        theta, psi = attitude_state[1], attitude_state[2]
        dx = self.landmark_pos[0] - px
        dy = self.landmark_pos[1] - py
        dz = self.landmark_pos[2] - pz
        eps = 1e-8
        r = np.linalg.norm([dx, dy, dz]) + eps
        az = np.arctan2(dy, dx) - psi
        el = np.arcsin(np.clip(dz / r, -1., 1.)) - theta
        return np.array([r, az, el])

    def get_landmark_jacobian(self, state):
        px, py, pz = state[0], state[2], state[4]
        dx = self.landmark_pos[0] - px
        dy = self.landmark_pos[1] - py
        dz = self.landmark_pos[2] - pz
        eps = 1e-8
        r = np.linalg.norm([dx, dy, dz]) + eps
        s = np.linalg.norm([dx, dy]) + eps
        HL = np.zeros((3, 6))
        HL[0, 0] = -dx/r
        HL[0, 2] = -dy/r
        HL[0, 4] = -dz/r
        HL[1, 0] = dy/s**2
        HL[1, 2] = -dx/s**2
        HL[2, 0] = dx*dz/(r**2*s)
        HL[2, 2] = dy*dz/(r**2*s)
        HL[2, 4] = -s/r**2
        return HL

    def step(
        self, 
        y, 
        neighbor_ests, 
        attitude_state, 
        control, 
        y_landmark=None, 
        landmark_pos=np.array([0., 0., 0.])
    ):
        if self.track_landmark:
            # If it's the leader reform the correct y measurement vector
            y_new = np.zeros(3 * (self.num_targets + 1) + 1)
            y_new[:6] = y[:6]
            y_new[6:9] = y_landmark
            y_new[-1] = y[-1]
            y = y_new

        x_bar = self.prior_state.copy()
        P_bar = self.prior_cov.copy()
        # Target measurement update
        H = self.get_measurement_jacobian(x_bar)
        # Calculate gamma (equivalent to gain for consensus)
        eps = 10 * self.dt
        gamma = eps / (1 + np.linalg.norm(P_bar, 'fro'))
        # Kalman gain
        K = P_bar @ H.T @ np.linalg.inv(H @ P_bar @ H.T + self.R)
        innov = y - self.measurements(x_bar, attitude_state)

        wrap_idxs = self.num_targets + (1 if self.track_landmark else 0)
        for i in range(wrap_idxs):
            innov[3*i+1] = (innov[3*i+1] + np.pi) % (2*np.pi) - np.pi
            innov[3*i+2] = (innov[3*i+2] + np.pi) % (2*np.pi) - np.pi
            
        # Consensus! YEAHHH
        C = gamma * P_bar
        consensus_corr = sum(xj - x_bar for xj in neighbor_ests)
        # Clip to prevent unconverged neighbor estimates from destabilizing the filter
        max_corr = 5.0
        corr_norm = np.linalg.norm(consensus_corr)
        if corr_norm > max_corr:
            consensus_corr = consensus_corr * (max_corr / corr_norm)
            
        # Consensus Update!!
        x_hat = x_bar + K @ innov + C @ consensus_corr

        # Posterior covariance — Joseph form
        IKH = np.eye(self.num_states) - K @ H
        M = IKH @ P_bar @ IKH.T + K @ self.R @ K.T

        # let's update everything now
        self.state_est = x_hat
        self.cov_est = M

        self.prior_state = self.dynamics(x_hat, attitude_state, control)
        self.prior_cov = self.A @ M @ self.A.T + self.Q

        self.state_est_time_hist.append(x_hat.copy())
        self.cov_est_time_hist.append(M.copy())

        return x_hat, M

    def get_state_est_time_hist(self):
        return np.array(self.state_est_time_hist)

    def get_cov_est_time_hist(self):
        return np.array(self.cov_est_time_hist)