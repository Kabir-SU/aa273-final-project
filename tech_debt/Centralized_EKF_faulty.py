#Centralized approach
import numpy as np
from scipy.linalg import block_diag
from measurementmodel import MeasurementModel

dt = 0.001
m = 20
g = -9.81
jphi, jtheta, jpsi = 10, 10, 5
    
def truedynamics(state, u): 
    #Compute the simulated next state using euler approximation on a single drone dynamics model
    F, tauphi, tautheta, taupsi = u
    px, py, pz, vx, vy, vz, phi, theta, psi, dphi, dtheta, dpsi = state
    step = np.zeros_like(state)
    step[0:3] = [px + dt*vx, py + dt*vy, pz + dt*vz]
    vxstep = vx + dt*F/m*(-np.cos(phi)*np.sin(theta)*np.cos(psi) - np.sin(phi)*np.sin(psi))
    vystep = vy + dt*F/m*(-np.cos(phi)*np.sin(theta)*np.sin(psi) + np.sin(phi)*np.cos(psi))
    vzstep = vz + dt*(g - np.cos(phi)*np.cos(theta)*F)/m
    step[3:6] = [vxstep, vystep, vzstep]
    step[6:9] = [phi + dt*dphi, theta + dt*dtheta, psi + dt*dpsi]
    step[9:] = [dphi + dt*tauphi/jphi, dtheta + dt*tautheta/jtheta, dpsi + dt*taupsi/jpsi]
    return step

def dynamics_model_joint(state, u):
    #Compute the simulated next state for the full 3 drone state model
    nextstate = np.zeros_like(state)
    for i in range(0, 3):
        stepstate = truedynamics(state[i*12 : (i+1)*12], u[i*4 : (i+1)*4])
        nextstate[i*12: (i+1)*12] = stepstate[:]
    return nextstate

def getG_single(state, u):
    #Compute the state dynamics jacobian for a single drone
    F, tauphi, tautheta, taupsi = u
    px, py, pz, vx, vy, vz, phi, theta, psi, dphi, dtheta, dpsi = state

    cphi = np.cos(phi);   sphi = np.sin(phi)
    cth  = np.cos(theta); sth  = np.sin(theta)
    cpsi = np.cos(psi);   spsi = np.sin(psi)

    G = np.eye(12)
    G[0,3] = dt
    G[1,4] = dt
    G[2,5] = dt
    
    dA_dphi = (sphi*sth*cpsi - cphi*spsi)
    dA_dtheta = (-cphi*cth*cpsi)
    dA_dpsi   = (cphi*sth*spsi - sphi*cpsi)
    dB_dphi = (sphi*sth*spsi + cphi*cpsi)
    dB_dtheta = -cphi*cth*spsi
    dB_dpsi   = -cphi*sth*cpsi - sphi*spsi

    G[3,6] = dt*(F/m)*dA_dphi
    G[3,7] = dt*(F/m)*dA_dtheta
    G[3,8] = dt*(F/m)*dA_dpsi

    G[4,6] = dt*(F/m)*dB_dphi
    G[4,7] = dt*(F/m)*dB_dtheta
    G[4,8] = dt*(F/m)*dB_dpsi

    G[5,6] = dt*(F/m)*(sphi*cth)
    G[5,7] = dt*(F/m)*(cphi*sth)

    G[6,9]  = dt
    G[7,10] = dt
    G[8,11] = dt
    return G

def get_G_full(state, u):
    A = getG_single(state[0:12], u[0:4])
    B = getG_single(state[12:24], u[4:8])
    C = getG_single(state[24:36], u[8:12]) 
    return block_diag(A, B, C)

def measurement_model_full(state, meas_model, add_noise = False):
    z = []
    for observer in range(3):
        obs_state = state[observer*12:(observer+1)*12]
        for target in range(3):
            if observer == target:
                continue
            target_pos = state[target*12:target*12+3]
            z_ij = meas_model.get_relative_measurement(
                        obs_state,
                        target_pos,
                        add_noise=add_noise)
            z.append(z_ij)
    return np.concatenate(z)

def getH_full(state):
    H = np.zeros((18,36))
    row = 0
    for observer in range(3):
        px_i = state[observer*12 + 0]
        py_i = state[observer*12 + 1]
        pz_i = state[observer*12 + 2]
        for target in range(3):
            if observer == target:
                continue
            px_j = state[target*12 + 0]
            py_j = state[target*12 + 1]
            pz_j = state[target*12 + 2]
            dx = px_j - px_i
            dy = py_j - py_i
            dz = pz_j - pz_i
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            rho2 = dx**2 + dy**2
            rho = np.sqrt(rho2)

            # range derivatives
            dr_dxi = -dx/r
            dr_dyi = -dy/r
            dr_dzi = -dz/r

            dr_dxj = dx/r
            dr_dyj = dy/r
            dr_dzj = dz/r

            # azimuth/angle derivatives
            da_dxi = dy/rho2
            da_dyi = -dx/rho2
            da_dxj = -dy/rho2
            da_dyj = dx/rho2
            da_dpsi = -1

            #  elevation derivatives
            root = np.sqrt(1 - (dz/r)**2)
            db_dxi = -(dx*dz)/(r**3 * root)
            db_dyi = -(dy*dz)/(r**3 * root)
            db_dzi = (rho2)/(r**3 * root)
            db_dxj = (dx*dz)/(r**3 * root)
            db_dyj = (dy*dz)/(r**3 * root)
            db_dzj = -(rho2)/(r**3 * root)
            db_dtheta = -1

            # set nonzero entries
            oi = observer*12
            tj = target*12

            H[row,oi+0] = dr_dxi
            H[row,oi+1] = dr_dyi
            H[row,oi+2] = dr_dzi
            H[row,tj+0] = dr_dxj
            H[row,tj+1] = dr_dyj
            H[row,tj+2] = dr_dzj

            H[row+1,oi+0] = da_dxi
            H[row+1,oi+1] = da_dyi
            H[row+1,tj+0] = da_dxj
            H[row+1,tj+1] = da_dyj
            H[row+1,oi+8] = da_dpsi

            H[row+2,oi+0] = db_dxi
            H[row+2,oi+1] = db_dyi
            H[row+2,oi+2] = db_dzi
            H[row+2,tj+0] = db_dxj
            H[row+2,tj+1] = db_dyj
            H[row+2,tj+2] = db_dzj
            H[row+2,oi+7] = db_dtheta

            row += 3
    return H

#design starting covariances
pos_var = 1.0
vel_var = 0.25
angle_var = 0.03
rate_var = 0.01

cov_single = np.diag([
    pos_var, pos_var, pos_var,
    vel_var, vel_var, vel_var,
    angle_var, angle_var, angle_var,
    rate_var, rate_var, rate_var
])
cov0 = block_diag(cov_single, cov_single, cov_single)

#set up EKF
sim_time = 10  # seconds
num_steps = int(sim_time/dt)

initial_state = np.zeros(12*3)
initial_state[0:3] = [0, 0, 0] #leader
initial_state[12:15] = [-7, -5, 10] #follower 1
initial_state[24:27] = [7, 5, 10] #follower 2

mu0 = initial_state
measurements = []
true_states = []
state_estimations = []
cov_estimations = []
state_estimations.append(mu0)
cov_estimations.append(cov0)
true_states.append(initial_state)

#process noise block
process_pos = 1e-4
process_vel = 1e-3
process_angle = 1e-4
process_rate = 1e-3

Q_single = np.diag([
    process_pos, process_pos, process_pos,
    process_vel, process_vel, process_vel,
    process_angle, process_angle, process_angle,
    process_rate, process_rate, process_rate
])

Q = block_diag(Q_single, Q_single, Q_single)

#measurement noise block
meas_model = MeasurementModel()

R_full = block_diag(
    meas_model.R,
    meas_model.R,
    meas_model.R,
    meas_model.R,
    meas_model.R,
    meas_model.R
)

for t in range(num_steps):
    #True Dynamics
    u = np.zeros(12)
    u[0:4] = [0.5, 2*np.sin(0.2*t*dt), 2*np.cos(0.2*t*dt), 0.5]
    u[4:8] = [0.5, 2*np.sin(0.2*t*dt), 2*np.cos(0.2*t*dt), 0.5]
    u[8:12] = [0.5, 2*np.sin(0.2*t*dt), 2*np.cos(0.2*t*dt), 0.5]

    state_new = dynamics_model_joint(true_states[t], u)
    state_new += np.random.multivariate_normal(np.zeros(36), Q)
    true_states.append(state_new)

    #EKF prediction
    mu_next = dynamics_model_joint(state_estimations[t], u)
    G = get_G_full(state_estimations[t], u)
    cov_next = G@cov_estimations[t]@G.T + Q

    z = measurement_model_full(true_states[t+1], meas_model, add_noise=True)
    z_mu = measurement_model_full(mu_next, meas_model, add_noise=False)
    y = z - z_mu
    for i in range(1, len(y), 3):
        y[i] = (y[i] + np.pi) % (2 * np.pi) - np.pi
        y[i+1] = (y[i+1] + np.pi) % (2 * np.pi) - np.pi

    H = getH_full(mu_next)
    S = H @ cov_next @ H.T + R_full
    K = cov_next @ H.T @ np.linalg.inv(S)
    mu_updated = mu_next + K @ y
    cov_updated = (np.eye(36) - K @ H) @ cov_next

    state_estimations.append(mu_updated)
    cov_estimations.append(cov_updated)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Convert lists to numpy arrays for easier indexing
true_states_arr = np.array(true_states)
est_states_arr = np.array(state_estimations)

fig = plt.figure(figsize=(14, 6))

# --- Plot 1: 3D Trajectories ---
ax1 = fig.add_subplot(121, projection='3d')
colors = ['r', 'g', 'b']

for i in range(3):
    # Slice indices for position (px, py, pz)
    idx = i * 12
    ax1.plot(true_states_arr[:, idx], true_states_arr[:, idx+1], true_states_arr[:, idx+2], 
             linestyle='--', color=colors[i], alpha=0.5, label=f'Drone {i} True')
    ax1.plot(est_states_arr[:, idx], est_states_arr[:, idx+1], est_states_arr[:, idx+2], 
             color=colors[i], label=f'Drone {i} EKF')

ax1.set_xlabel('X Position (m)')
ax1.set_ylabel('Y Position (m)')
ax1.set_zlabel('Z Position (m)')
ax1.set_title('3D Flight Paths: True vs. Estimated')
ax1.legend()

# --- Plot 2: Position Error Over Time ---
ax2 = fig.add_subplot(122)
time_axis = np.linspace(0, sim_time, len(est_states_arr))

for i in range(3):
    idx = i * 12
    # Euclidean distance error for position
    error = np.sqrt(np.sum((true_states_arr[:, idx:idx+3] - est_states_arr[:, idx:idx+3])**2, axis=1))
    ax2.plot(time_axis, error, color=colors[i], label=f'Drone {i} Error')

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Position Error (m)')
ax2.set_title('Total Position Estimation Error')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()