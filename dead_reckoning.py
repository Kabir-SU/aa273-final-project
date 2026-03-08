import numpy as np

class DeadReckon:
    def __init__(self, inital_attitude, initial_rates):
        self.attitude_state = np.concatenate([inital_attitude, initial_rates])
        self.dt = 0.01
        self.attitude_time_hist = [self.attitude_state]
        
    def step(self, gyro_meas):
        wx, wy, wz = gyro_meas
        phi, theta, psi, p, q, r = self.attitude_state
        
        new_state = np.array([
            phi + wx*self.dt,
            theta + wy*self.dt,
            psi + wz*self.dt,
            wx,
            wy,
            wz,
        ])
        
        self.attitude_state = new_state
        self.attitude_time_hist.append(self.attitude_state)
        
    def get_time_hist(self):
        return np.array(self.attitude_time_hist)
        