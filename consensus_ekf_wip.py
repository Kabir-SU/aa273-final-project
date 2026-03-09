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

    def __init__(self, init_mu, init_cov, dt=0.01, m=20, num_targets=2):
        self.dt = dt
        self.m = m
        self.g = 9.81
        self.num_targets = num_targets
        self.num_states = 6 + 6 * num_targets   # 18

        self.state_est   = init_mu.copy()
        self.prior_state = init_mu.copy()
        self.cov_est     = init_cov.copy()
        self.prior_cov   = init_cov.copy()

        # Measurement noise: [range, az, el] per target
        range_var = 0.1 ** 2
        angle_var = np.deg2rad(0.1) ** 2
        self.R = block_diag(*([np.diag([range_var, angle_var, angle_var])] * num_targets))

        # Process noise: 3 drone accel channels + 3 per target (as disturbance input)
        drone_accel_var  = 2.0 ** 2
        target_accel_var = 1.0 ** 2
        q_vals = [drone_accel_var] * 3 + [target_accel_var] * 3 * num_targets
        self.Q = np.diag(q_vals)   # (9, 9)

        # Consensus gain
        self.gamma = 1e-4

        self.state_est_time_hist = [init_mu.copy()]
        self.cov_est_time_hist   = [init_cov.copy()]

        self.A = self.get_dynamics_jacobian()
        self.B = self.get_noise_mapping_matrix()

    def get_dynamics_jacobian(self):
        cv = np.array([[1., self.dt], [0., 1.]])
        block_cv = block_diag(cv, cv, cv)   # (6, 6)
        A = np.zeros((self.num_states, self.num_states))
        for k in range(1 + self.num_targets):
            s = 6 * k
            A[s:s+6, s:s+6] = block_cv
        return A

    def get_noise_mapping_matrix(self):
        b1d = np.array([[0.5 * self.dt**2], [self.dt]])
        b6  = block_diag(b1d, b1d, b1d) # (6, 3)
        B   = block_diag(*([b6] * (1 + self.num_targets))) # (18, 9)
        return B

    def dynamics(self, state, attitude_state, control):
        phi, theta, psi = attitude_state[:3]
        F = control[0]

        cphi = np.cos(phi); sphi = np.sin(phi)
        cth  = np.cos(theta); sth  = np.sin(theta)
        cpsi = np.cos(psi); spsi = np.sin(psi)

        ax = (-cphi * sth * cpsi - sphi * spsi) * F / self.m
        ay = -(-cphi * sth * spsi - sphi * cpsi) * F / self.m
        az = -self.g + (cphi * cth) * F / self.m

        own = state[:6]
        new_own = np.zeros(6)
        for i, a in enumerate([ax, ay, az]):
            new_own[2*i]   = own[2*i] + own[2*i+1] * self.dt
            new_own[2*i+1] = own[2*i+1] + a * self.dt

        new_targets = []
        for i in range(self.num_targets):
            t = state[6 + 6*i: 6 + 6*(i+1)]
            nt = np.zeros(6)
            for j in range(3):
                nt[2*j]   = t[2*j] + t[2*j+1] * self.dt
                nt[2*j+1] = t[2*j+1]
            new_targets.append(nt)

        return np.concatenate([new_own] + new_targets)

    def measurements(self, state, attitude_state):
        theta, psi = attitude_state[1], attitude_state[2]
        px, py, pz = state[0], state[2], state[4]
        meas = np.zeros(3 * self.num_targets)
        for i in range(self.num_targets):
            t  = state[6 + 6*i: 6 + 6*(i+1)]
            dx = t[0] - px;  dy = t[2] - py;  dz = t[4] - pz
            eps = 1e-8
            r  = np.linalg.norm([dx, dy, dz]) + eps
            az = np.arctan2(dy, dx) - psi
            el = np.arcsin(np.clip(dz / r, -1., 1.)) - theta
            meas[3*i:3*i+3] = [r, az, el]
        return meas

    def get_measurement_jacobian(self, state):
        H  = np.zeros((3 * self.num_targets, self.num_states))
        px, py, pz = state[0], state[2], state[4]
        for i in range(self.num_targets):
            t  = state[6 + 6*i: 6 + 6*(i+1)]
            dx = t[0] - px;  dy = t[2] - py;  dz = t[4] - pz
            eps = 1e-8
            r  = np.linalg.norm([dx, dy, dz]) + eps
            s  = np.linalg.norm([dx, dy]) + eps
            row = 3 * i
            ct  = 6 + 6 * i

            # Target position partials
            H[row,   ct];   H[row,   ct]   =  dx/r
            H[row,   ct+2] =  dy/r;         H[row,   ct+4] =  dz/r
            H[row+1, ct]   = -dy/s**2;      H[row+1, ct+2] =  dx/s**2
            H[row+2, ct]   = -dx*dz/(r**2*s)
            H[row+2, ct+2] = -dy*dz/(r**2*s)
            H[row+2, ct+4] =  s/r**2

            # Own drone position partials (cols 0, 2, 4)
            H[row,   0] = -dx/r;    H[row,   2] = -dy/r;    H[row,   4] = -dz/r
            H[row+1, 0] =  dy/s**2; H[row+1, 2] = -dx/s**2
            H[row+2, 0] =  dx*dz/(r**2*s)
            H[row+2, 2] =  dy*dz/(r**2*s)
            H[row+2, 4] = -s/r**2
        return H

    def landmark_measurement(self, state, attitude_state, landmark_pos=np.array([0., 0., 0.])):
        px, py, pz = state[0], state[2], state[4]
        theta, psi = attitude_state[1], attitude_state[2]
        dx = landmark_pos[0] - px
        dy = landmark_pos[1] - py
        dz = landmark_pos[2] - pz
        eps = 1e-8
        r  = np.linalg.norm([dx, dy, dz]) + eps
        az = np.arctan2(dy, dx) - psi
        el = np.arcsin(np.clip(dz / r, -1., 1.)) - theta
        return np.array([r, az, el])

    def get_landmark_jacobian(self, state, landmark_pos=np.array([0., 0., 0.])):
        px, py, pz = state[0], state[2], state[4]
        dx = landmark_pos[0] - px
        dy = landmark_pos[1] - py
        dz = landmark_pos[2] - pz
        eps = 1e-8
        r  = np.linalg.norm([dx, dy, dz]) + eps
        s  = np.linalg.norm([dx, dy]) + eps
        HL = np.zeros((3, self.num_states))
        HL[0, 0] = -dx/r;      HL[0, 2] = -dy/r;      HL[0, 4] = -dz/r
        HL[1, 0] =  dy/s**2;   HL[1, 2] = -dx/s**2
        HL[2, 0] =  dx*dz/(r**2*s)
        HL[2, 2] =  dy*dz/(r**2*s)
        HL[2, 4] = -s/r**2
        return HL

    def get_consensus_gain(self):
        return self.gamma * np.eye(self.num_states) # Defined gamma as scalar "gain" val for more pronounced behavior

    def step(self, y, neighbor_ests, attitude_state, control, y_landmark=None):
        x_bar = self.prior_state.copy()
        P_bar = self.prior_cov.copy()

        # Landmark update (leader only)
        if y_landmark is not None:
            HL  = self.get_landmark_jacobian(x_bar)
            RL  = np.diag([0.1**2, np.deg2rad(0.1)**2, np.deg2rad(0.1)**2])
            K_L = P_bar @ HL.T @ np.linalg.inv(HL @ P_bar @ HL.T + RL)
            innov_l = y_landmark - self.landmark_measurement(x_bar, attitude_state)
            innov_l[1] = (innov_l[1] + np.pi) % (2*np.pi) - np.pi
            innov_l[2] = (innov_l[2] + np.pi) % (2*np.pi) - np.pi
            x_bar = x_bar + K_L @ innov_l
            P_bar = (np.eye(self.num_states) - K_L @ HL) @ P_bar

        # Target measurement update
        H = self.get_measurement_jacobian(x_bar)
        K = P_bar @ H.T @ np.linalg.inv(H @ P_bar @ H.T + self.R)
        innov = y - self.measurements(x_bar, attitude_state)
        for i in range(self.num_targets):
            innov[3*i+1] = (innov[3*i+1] + np.pi) % (2*np.pi) - np.pi
            innov[3*i+2] = (innov[3*i+2] + np.pi) % (2*np.pi) - np.pi

        x_post = x_bar + K @ innov

        # Consensus correction
        C = self.get_consensus_gain()
        consensus_corr = sum(xj - x_bar for xj in neighbor_ests)
        # Clip to prevent unconverged neighbor estimates from destabilizing the filter
        max_corr = 5.0
        corr_norm = np.linalg.norm(consensus_corr)
        if corr_norm > max_corr:
            consensus_corr = consensus_corr * (max_corr / corr_norm)
        x_hat = x_post + C @ consensus_corr

        # Posterior covariance — Joseph form
        IKH = np.eye(self.num_states) - K @ H
        M   = IKH @ P_bar @ IKH.T + K @ self.R @ K.T

        self.state_est = x_hat
        self.cov_est   = M

        self.prior_state = self.dynamics(x_hat, attitude_state, control)
        self.prior_cov   = self.A @ M @ self.A.T + self.B @ self.Q @ self.B.T

        self.state_est_time_hist.append(x_hat.copy())
        self.cov_est_time_hist.append(M.copy())

        return x_hat, M

    def get_state_est_time_hist(self):
        return np.array(self.state_est_time_hist)

    def get_cov_est_time_hist(self):
        return np.array(self.cov_est_time_hist)