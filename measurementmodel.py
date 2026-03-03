import numpy as np

class MeasurementModel:
    def __init__(self, range_std=0.1, angle_std=0.1, gyro_std=0.01):
        self.R = np.diag([range_std**2, np.deg2rad(angle_std)**2, np.deg2rad(angle_std)**2])
        self.R_gyro = np.diag([gyro_std**2, gyro_std**2, gyro_std**2])
        self.gyro_bias = np.array([0.001, -0.002, 0.001]) # small static bias

    def get_relative_measurement(self, observer_state, target_pos, add_noise=False):
        px, py, pz = observer_state[0:3]
        phi, theta, psi = observer_state[6:9]
        dx = target_pos[0] - px
        dy = target_pos[1] - py
        dz = target_pos[2] - pz

        r = np.sqrt(dx**2 + dy**2 + dz**2)
        alpha = np.arctan2(dy, dx) - psi
        beta = np.arcsin(dz/r) - theta
        z = np.array([r, alpha, beta])

        if add_noise:
            z += np.random.multivariate_normal(np.zeros(3), self.R)
            
        return z

    def get_gyroscope_measurement(self, state, add_noise=False):
        
        p, q, r = state[9:12]
        gyro_true = np.array([p, q, r])
        
        if add_noise:
            gyro_meas = gyro_true + self.gyro_bias + np.random.multivariate_normal(np.zeros(3), self.R_gyro)
        else:
            gyro_meas = gyro_true + self.gyro_bias 
            
        return gyro_meas