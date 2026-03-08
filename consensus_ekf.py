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
        self.Q = np.eye(24) # TODO: Assign actual values
        
        self.state_est_time_hist = [init_mu]
        self.cov_est_time_hist = [init_cov]
            
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
        eps = self.dt
        return eps / (1 + np.linalg.norm(cov)) * cov
    
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
        target_states = [state[6:15], state[15:]]
        
        meas = np.zeros(3*self.num_targets)
        for target_idx in range(self.num_targets):
            target = target_states[target_idx]
            
            x_rel, y_rel, z_rel = target[0], target[3], target[6]
            pos = np.array([x_rel, y_rel, z_rel])
            
            r = np.linalg.norm(pos)
            az = np.atan2(y_rel, x_rel) - psi
            el = np.arcsin(z_rel / r) - theta
            
            meas[3*target_idx:3*target_idx + 3] = np.array([r, az, el])
        
        return meas
    
    def get_measurement_jacobian(self, state):
        H = np.zeros((3*self.num_targets, self.num_states))
        target_states = [state[6:15], state[15:]]
        for i in range(self.num_targets):
            target = target_states[i]
            x = target[0]
            y = target[3]
            z = target[6]
            
            r = np.linalg.norm([x, y, z])
            s = np.linalg.norm([x, y])
            
            target_block = np.array([
                [x/r, 0., 0., y/r, 0., 0., z/r, 0., 0.],
                [-y / s**2, 0., 0., x / s**2, 0., 0., 0., 0., 0.],
                [-x*z / (r**2 * s), 0., 0., -y*z / (r**2 * s), 0., 0., s / r**2, 0., 0.],
            ])
            
            H[3*i:3*i + 3, 6 + 9*i: 6 + 9*(i+1)] = target_block
            
        return H
            
    
    def step(self, y, neighbor_ests, attitude_state, control):
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
        
        # Following Paper's notation generally for suboptimal Consensus Implmentation
        R = self.R
        Q = self.Q
        
        # Masurement jacobian
        H = self.get_measurement_jacobian(x_bar_prior)
        # Consensus Gain
        C = self.get_unoptimal_consensus_gain(sigma_bar_prior)
        # Measurement Prior
        Hx = self.measurements(x_bar_prior, attitude_state)
        
        # Kalman Gain
        K = sigma_bar_prior @ H.T @ np.linalg.inv(R + H @ sigma_bar_prior @ H.T)
        # State Estimate
        x_hat = x_bar_prior + K @ (y - Hx) + C @ np.sum([(xj - x_bar_prior) for xj in neighbor_ests], axis=0)
        self.state_est = x_hat
        
        # Unsure what F is frankly it's in the paper though
        F = np.eye((K@H).shape[0]) - K @ H
        # Posterior covariance
        M = F @ sigma_bar_prior @ F.T + K @ R @ K.T
        self.cov_est = M
        
        # Dynamics Jacobian
        A = self.get_dynamics_jacobian()
        # Noise Mapping matrix
        B = self.get_noise_mapping_matrix()
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
        