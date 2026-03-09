import numpy as np
from scipy.linalg import block_diag

# TODO: we need to somehow implement a version for the leader where it also get's landmark estimates
# we also need to add the leader's position somewhere in this, I haven't figured that out yet, though we
# may be able to just say it's a measurement that the leader passes along, it may also be baked into
# leader's own covariance when the consensus stuff happens so maybe do some work on that

class ConsensusEKF:
    """Consensus EKF"""
    def __init__(self, init_mu, init_cov, dt=0.01, m=20, num_targets=2):
        self.state_est = init_mu
        self.prior_state = init_mu
        self.cov_est = init_cov
        self.prior_cov = init_cov
        self.dt = dt
        self.m = m
        self.g = 9.81
        self.num_targets = num_targets
        self.num_states = 6 + 9*num_targets
        
        self.R = np.eye(6) # TODO: Assign actual values
        self.R = np.diag([0.01, np.deg2rad(0.1)**2, np.deg2rad(0.1)**2, 0.01, np.deg2rad(0.1)**2, np.deg2rad(0.1)**2])
        self.Q = np.eye(9) # TODO: Process noise vector has 9 elements (3 for own drone, 3 for each target)
        
        self.state_est_time_hist = [init_mu]
        self.cov_est_time_hist = [init_cov]
        
        # Precompute constant Jacobians
        self.A = self.get_dynamics_jacobian()
        self.B = self.get_noise_mapping_matrix()

    def landmark_measurement_model(self, state, landmark_pos=np.array([0, 0, 0])):
        """Calculates r, az, el relative to a specific stationary landmark"""
        # Extract drone position from state
        px, py, pz = state[0], state[2], state[4]
        
        # Explicitly calculate the difference vector
        dx = landmark_pos[0] - px
        dy = landmark_pos[1] - py
        dz = landmark_pos[2] - pz

        r = np.sqrt(dx**2 + dy**2 + dz**2) 
        return r, dx, dy, dz

    def get_landmark_jacobian(self, state, landmark_pos=np.array([0, 0, 0])):
        """Jacobian mapping drone position to landmark measurements"""
        px, py, pz = state[0], state[2], state[4]
        dx = landmark_pos[0] - px
        dy = landmark_pos[1] - py
        dz = landmark_pos[2] - pz
        
        r = np.linalg.norm([dx, dy, dz])
        s = np.linalg.norm([dx, dy]) # horizontal distance
        
        H_L = np.zeros((3, self.num_states))
        # Range derivatives
        H_L[0, 0:6:2] = [-dx/r, -dy/r, -dz/r] 
        # Azimuth derivatives
        H_L[1, 0:3:2] = [dy/s**2, -dx/s**2]
        # Elevation derivatives
        H_L[2, 0:6:2] = [dx*dz/(r**2*s), dy*dz/(r**2*s), -s/r**2]
        
        return H_L


    def get_dynamics_jacobian(self):
        """Dynamics Jacobian used for covariance propagation"""
        # Form drone block
        A_drone_block = np.array([
            [1., self.dt],
            [0., 1.]
        ])
        A_drone = block_diag(*([A_drone_block] * 3))
        
        # Form target block
        A_target_block = np.array([
            [1, self.dt, 0.5 * self.dt**2],
            [0, 1, self.dt],
            [0, 0, 1]
        ])
        A_target = block_diag(*([A_target_block] * 6))
        
        # Make the dynamics matrix for the drone and targets 
        A = np.zeros((24, 24))
        A[6:, 6:] = A_target
        A[:6, :6] = A_drone
        
        return A
    
    def get_noise_mapping_matrix(self):
        """This maps the noises in accelerations to the states in  Ax + Bw
        """
        B_drone_1d = np.array([
            [0.5 * self.dt**2],
            [self.dt]
        ])
        B_drone = block_diag(B_drone_1d, B_drone_1d, B_drone_1d)

        # accel random walk
        B_target_1d = np.array([
            [0.0],
            [0.0],
            [1.0]
        ])
        B_target_single = block_diag(B_target_1d, B_target_1d, B_target_1d)

        B = block_diag(B_drone, B_target_single, B_target_single)
        
        return B
    
    def get_unoptimal_consensus_gain(self, cov):
        """Returns the non_optimal consensus gain"""
        gamma = 0.1
        eps = self.dt
        return gamma * ((eps * cov) / (1 + np.linalg.norm(cov)))
    
    def dynamics(self, state, attitude_state, control):
        """Dyanmics function used in predict step
        
        state is the entire EKF state
        attitude state is from dead reckoning (phi, theta, psi, p, q, r)
        control! (F, tau_x, tau_y, tau_z)
        
        """
        # Extract necessary states
        drone_state = state[:6]
        target_states = [state[6:15], state[15:]]
        phi, theta, psi = attitude_state[:3]
        F = control[0]
        
        # Precompute sines and cosines
        cphi = np.cos(phi)
        cth = np.cos(theta)
        cpsi = np.cos(psi)
        sphi = np.sin(phi)
        sth = np.sin(theta)
        spsi = np.sin(psi)

        # Get the acceleration from the control input dynamics
        a_vec = np.array([
            (-cphi * sth * cpsi - sphi * spsi) * F / self.m,
            (cphi*sth*spsi - sphi*cpsi) * F / self.m,
            (cphi * cth) * F / self.m - self.g,
        ])
        
        # Update the drone states!
        new_state = np.zeros(6)
        for i in range(3):
            new_state[2*i] = drone_state[2*i] + drone_state[2*i + 1]*self.dt
            new_state[2*i + 1] = drone_state[2*i + 1] + a_vec[i]*self.dt
            
        # Update the target states!
        for i in range(self.num_targets):
            target_state = target_states[i]
            new_target_state = np.zeros(9)

            for j in range(3):
                pos = target_state[3*j]
                vel = target_state[3*j + 1]
                acc = target_state[3*j + 2]

                new_target_state[3*j] = pos + vel * self.dt + 0.5 * acc * self.dt**2
                new_target_state[3*j + 1] = vel + acc * self.dt
                new_target_state[3*j + 2] = acc
                
            new_state = np.concatenate([new_state, new_target_state])
            
        return new_state
    
    def measurements(self, state, attitude_state):
        theta, psi = attitude_state[1:3]
        px, py, pz = state[0], state[2], state[4]
        target_states = [state[6:15], state[15:]]
        
        meas = np.zeros(3*self.num_targets)
        for target_idx in range(self.num_targets):
            target = target_states[target_idx]
            
            # Target blocks are 9 elements: x, x_dot, x_ddot, y, y_dot, y_ddot, z, z_dot, z_ddot
            # NOT x, x, x... they are structured by dimension [x, vx, ax, y, vy, ay, z, vz, az]
            x_rel = target[0] - px
            y_rel = target[3] - py
            z_rel = target[6] - pz
            
            eps = 1e-8
            r = np.linalg.norm([x_rel, y_rel, z_rel]) + eps
            az = np.atan2(y_rel, x_rel) - psi
            el = np.arcsin(z_rel / r) - theta
            
            meas[3*target_idx:3*target_idx + 3] = np.array([r, az, el])
        
        return meas
    
    def get_measurement_jacobian(self, state):
        H = np.zeros((3*self.num_targets, self.num_states))
        px, py, pz = state[0], state[2], state[4]
        target_states = [state[6:15], state[15:]]
        for i in range(self.num_targets):
            target = target_states[i]

            # Corrected for relative measurements
            dx = target[0] - px
            dy = target[3] - py
            dz = target[6] - pz
            
            eps = 1e-8
            r = np.linalg.norm([dx, dy, dz]) + eps
            s = np.linalg.norm([dx, dy]) + eps
            
            target_block = np.array([
                [dx/r, 0., 0., dy/r, 0., 0., dz/r, 0., 0.],
                [-dy / s**2, 0., 0., dx / s**2, 0., 0., 0., 0., 0.],
                [-dx*dz / (r**2 * s), 0., 0., -dy*dz / (r**2 * s), 0., 0., s / r**2, 0., 0.],
            ])
            
            H[3*i:3*i + 3, 6 + 9*i: 6 + 9*(i+1)] = target_block
            
            drone_block = np.array([
                [-dx/r, 0., -dy/r, 0., -dz/r, 0.],
                [dy / s**2, 0., -dx / s**2, 0., 0., 0.],
                [dx*dz / (r**2 * s), 0., dy*dz / (r**2 * s), 0., -s / r**2, 0.],
            ])
            # The drone position states are 0(px), 2(py), 4(pz). The drone_block assumes they are contiguous 0,1,2.
            # We must assign them properly.
            H[3*i:3*i + 3, 0] = drone_block[:, 0]  # px
            H[3*i:3*i + 3, 2] = drone_block[:, 2]  # py
            H[3*i:3*i + 3, 4] = drone_block[:, 4]  # pz
            
        return H
            
    
    def step(self, y, neighbor_ests, attitude_state, control, y_landmark=None):
        """Consesus filter step function
        
        The consensus filter operates slightly different to a standard EKF/KF, which
        would have a predict step and a following update step. 
        
        Instead, the consensus filter computes the state estimate `x_hat` first based on the
        measurement residual and consensus gain, along with the covariance estimate `sigma_hat`.
        
        After doing this, the error covariance matrix `M` is calculated, which is
        then used to get the prior mean `x_bar` and covariance `sigma_bar` for the following state.
        
        `y`: range-azimuth-elevation measurement for each other agent in the network
        `neighbor_ests`: A list of state estimates provided by other agents in the network
        """
        x_bar_prior = self.prior_state
        sigma_bar_prior = self.prior_cov

        if y_landmark is not None:
            HL = self.get_landmark_jacobian(x_bar_prior)
            RL = np.diag([0.01, 0.001, 0.001])
            K_L = sigma_bar_prior @ HL.T @ np.linalg.inv(HL @ sigma_bar_prior @ HL.T + RL)
            r_l, dx, dy, dz = self.landmark_measurement_model(x_bar_prior)
            z_l_pred = np.array([r_l, np.arctan2(dy, dx) - attitude_state[2], np.arcsin(dz/r_l) - attitude_state[1]])
            
            # Update the prior state with absolute truth
            x_bar_prior = x_bar_prior + K_L @ (y_landmark - z_l_pred)
            sigma_bar_prior = (np.eye(self.num_states) - K_L @ HL) @ sigma_bar_prior
            
        # Following Paper's notation generally for suboptimal Consensus Implmentation
        R = self.R
        Q = self.Q
        
        # Masurement jacobian
        H = self.get_measurement_jacobian(x_bar_prior)
        # Kalman Gain
        K = sigma_bar_prior @ H.T @ np.linalg.inv(R + H @ sigma_bar_prior @ H.T)
        # Measurement Prior
        Hx = self.measurements(x_bar_prior, attitude_state)

        # Splitting up local/global state estimates for now
        x_post_local = x_bar_prior + K @ (y - Hx)

        # Consensus Gain
        C = self.get_unoptimal_consensus_gain(sigma_bar_prior)

        consensus_corr = np.zeros(self.num_states)
        for xj in neighbor_ests:
            consensus_corr += (xj - x_bar_prior)
        
        # State Estimate
        x_hat = x_post_local + C @ consensus_corr
        self.state_est = x_hat
        
        # Unsure what F is frankly it's in the paper though
        F = np.eye((K@H).shape[0]) - K @ H
        # Posterior covariance
        M = F @ sigma_bar_prior @ F.T + K @ R @ K.T
        self.cov_est = M
        
        # Dynamics Jacobian
        A = self.A
        # Noise Mapping matrix
        B = self.B
        # The next timestep's prior covariance
        P_bar_post = A @ M @ A.T + B @ Q @ B.T
        # The next timestep's prior state estimate
        x_bar_post = self.dynamics(x_hat, attitude_state, control)
        self.prior_cov = P_bar_post
        self.prior_state = x_bar_post
        
        self.state_est_time_hist.append(x_hat)
        self.cov_est_time_hist.append(M)
        
        return x_hat, M
        
    def get_state_est_time_hist(self):
        return np.array(self.state_est_time_hist)
    
    def get_cov_est_time_hist(self):
        return np.array(self.cov_est_time_hist)
