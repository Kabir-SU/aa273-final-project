import numpy as np

class MeasurementModel:
    def __init__(self, range_std=0.1, angle_std=0.1):
        self.R = np.diag([range_std**2, np.deg2rad(angle_std)**2, np.deg2rad(angle_std)**2])

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
        
        