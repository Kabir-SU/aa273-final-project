import numpy as np

class Drone:
    def __init__(self):
        self.dt = 0.01  # seconds
        # initialize time at 0s
        self.time = 0
        
        # All of these are WAGs
        self.J_x = 10  # kg m^s
        self.J_y = 10  # kg m^s
        self.J_z = 5  # kg m^s
        self.m = 20 # kg
        
        self.g = 9.81 # m/s^2
        
        # Drone has 12 states:
        # px, py, pz vx, vy, vz (transaltion position and velocity)
        # phi, theta, psi, p, q, r (orientation and body rates)
        self.state = np.zeros(12)
        
        # Drone has 4 allocated control inputs
        # F - thrust
        # Tx, Ty, Tz - torque around each body axis
        self.control_input = np.zeros(4)
        
        self.state_time_history = []
        self.control_time_history = []
        self.time_history = [0.]
        
    def set_init_condition(self, state):
        self.state = state
        self.state_time_history.append(state)
        
    def step_dynamics(self, control_input=None):
        """Wrapper function to step all dynamics by one timestep"""
        # Update internal storage of control input
        if control_input is None:
            control_input = self.spiral_trajectory_controller(R0=0.0)
        
        self.control_input = control_input
        
        # Check to make sure thrust can't be negative (unphysical)
        if control_input[0] < 0:
            self.control_input[0] = 0.
        
        self.control_time_history.append(self.control_input)
        # Forward propagate using rk4
        self.state = self.rk4(self.dydt)
        self.state_time_history.append(self.state)
        # Add time
        self.time += self.dt
        self.time_history.append(self.time)
        
    def rk4(self, dydt):
        """RK4 Dynamics Propagation"""
        k1 = dydt(self.state)
        k2 = dydt(self.state + 0.5 * self.dt * k1)
        k3 = dydt(self.state + 0.5 * self.dt * k2)
        k4 = dydt(self.state + self.dt * k3)
        return self.state + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    def dydt(self, y):
        """Continuous time state derivative dynamics"""
        # Extract states
        vx, vy, vz = y[3:6]
        phi, theta, psi = y[6:9]
        p, q, r = y[9:12]
        
        # Extract control inputs
        F, Tx, Ty, Tz = self.control_input
        
        xdot, ydot, zdot = vx, vy, vz
        
        xddot = ((-np.cos(phi) * np.sin(theta) * np.cos(psi) - np.sin(phi) * np.sin(psi)) * F) / self.m
        yddot = ((-np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) * F) / self.m
        zddot = -self.g + (np.cos(phi) * np.cos(theta) * F) / self.m
    
        phidot, thetadot, psidot = p, q, r
        
        phiddot = Tx / self.J_x
        thetaddot = Ty / self.J_y
        psiddot = Tz / self.J_z
        
        return np.array(
            [
                xdot, ydot, zdot, # Position derivative
                xddot, yddot, zddot, # Velocity derivative
                phidot, thetadot, psidot, # Attitude derivative
                phiddot, thetaddot, psiddot, # Body Rate derivative
            ]
        )
        
    def get_state_time_history(self):
        return np.array(self.state_time_history)
    
    def get_times(self):
        return np.array(self.time_history)
    
    def get_control_history(self):
        return np.array(self.control_time_history)
        
    def spiral_trajectory_controller(self, R0):
        t = self.time
        px, py, pz = self.state[:3]
        vx, vy, vz = self.state[3:6]
        phi, theta, psi = self.state[6:9]
        p, q, r = self.state[9:12]
        
        # radius growth rate
        k = 0.02 # m/s
        R_new = R0 + k*t
        
        omega = 0.6 # rad/s (spin rate)
        psi = omega * t
        
        # Desired X and Y
        x_des = R_new * np.cos(psi)
        y_des = R_new * np.sin(psi)
        
        # Desired X and Y vels
        x_des_dot = k * np.cos(psi) -R_new * omega * np.sin(psi)
        y_des_dot = k* np.sin(psi) + R_new * omega * np.cos(psi)
        
        # Ascend Rate
        v_z_des = 0.1 # m/s
        z_des = 10.0 + v_z_des * t
        
        # Gains & Control Law
        kp_xy, kd_xy = 0.8, 1.6
        ax_cmd = kp_xy * (x_des - px) + kd_xy * (x_des_dot - vx)
        ay_cmd = kp_xy * (y_des - py) + kd_xy * (y_des_dot - vy)
        
        kp_z, kd_z = 3.0 ,4.0
        az_cmd = kp_z * (z_des - pz) + kd_z * (v_z_des - vz)
        
        phi_des = -ay_cmd / self.g
        theta_des = -ax_cmd / self.g
        F = self.m * (self.g + az_cmd)
        
        # Limit tilt
        tilt_lim = 0.35  # 20 deg
        phi_des   = np.clip(phi_des,   -tilt_lim, tilt_lim)
        theta_des = np.clip(theta_des, -tilt_lim, tilt_lim)
        
        kp_ang, kd_ang = 25.0, 10.0
        Tx = self.J_x * (kp_ang*(phi_des - phi)     + kd_ang*(0.0 - p))
        Ty = self.J_y * (kp_ang*(theta_des - theta) + kd_ang*(0.0 - q))
        Tz = 0.0
        
        
        return np.array([F, Tx, Ty, Tz])
        
        
        