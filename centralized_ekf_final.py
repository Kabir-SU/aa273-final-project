#Centralized approach
import numpy as np
from scipy.linalg import block_diag
from measurementmodel import MeasurementModel

dt = 0.01
m = 20
g = -9.81
jphi, jtheta, jpsi = 10, 10, 5
    
def truedynamics(state, attitude_state, u): 
    #Compute the simulated next state using euler approximation on a single drone dynamics model
    F, _, _, _ = u
    px, py, pz, vx, vy, vz = state
    phi, theta, psi, dphi, dtheta, dpsi = attitude_state
    step = np.zeros_like(state)
    step[0:3] = [px + dt*vx, py + dt*vy, pz + dt*vz]
    vxstep = vx + dt*F/m*(-np.cos(phi)*np.sin(theta)*np.cos(psi) - np.sin(phi)*np.sin(psi))
    vystep = vy + dt*F/m*(np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi))
    vzstep = vz + dt*(g + np.cos(phi)*np.cos(theta)*F/m)
    step[3:6] = [vxstep, vystep, vzstep]
    return step

def dynamics_model_joint(state, attitude_state, u):
    #Compute the simulated next state for the full 3 drone state model
    nextstate = np.zeros_like(state)
    for i in range(0, 3):
        nextstate[i*6: (i+1)*6] = truedynamics(
            state[i*6 : (i+1)*6], 
            attitude_state[i*6 : (i+1)*6], 
            u[i*4 : (i+1)*4]
        )
    return nextstate

def getG_single(state):
    """Compute the state dynamics jacobian for a single drone"""

    G = np.eye(6)
    G[0,3] = dt
    G[1,4] = dt
    G[2,5] = dt
    return G

def get_G_full(state):
    A = getG_single(state[:6])
    B = getG_single(state[6:12])
    C = getG_single(state[12:]) 
    
    G = block_diag(A, B, C)
    assert(G.shape[0] == 18)
    return G

def measurement_model_full(state, meas_model, landmark_pos, stacked_attitude_state, add_noise = False):
    z = []
    for idx in range(3):
        obs_state = state[idx*6:(idx+1)*6]
        attitude_state = stacked_attitude_state[idx*6:(idx+1)*6]
        for target in range(3):
            if idx == target:
                continue
            target_pos = state[target*6:target*6+3]
            obs_state_with_attitude = np.concatenate([obs_state, attitude_state])
            z_ij = meas_model.get_relative_measurement(
                        obs_state_with_attitude,
                        target_pos,
                        add_noise=add_noise)
            z.extend(np.asarray(z_ij).ravel())
        if idx == 0:
            # add the landmark measurement
            z_ij = meas_model.get_relative_measurement(
                        obs_state_with_attitude,
                        landmark_pos,
                        add_noise=add_noise)
            z.extend(np.asarray(z_ij).ravel())
            
        alt_meas = meas_model.get_altimeter_measurement(
            obs_state,
            add_noise=add_noise,
        )
        z.append(alt_meas)
    return np.array(z)

import numpy as np

def getH_full(state, landmark_pos):
    def meas_jacobian_from_delta(d):
        dx, dy, dz = d
        r2 = d @ d
        r = np.sqrt(r2)

        rho2 = dx**2 + dy**2
        rho = np.sqrt(rho2)

        eps = 1e-12
        r = max(r, eps)
        rho2 = max(rho2, eps)

        u = dz / r
        u = np.clip(u, -1 + eps, 1 - eps)
        root = np.sqrt(1 - u**2)
        root = max(root, eps)

        J = np.array([
            [dx/r, dy/r, dz/r],
            [-dy/rho2, dx/rho2, 0.0],
            [-(dx*dz)/(r**3*root), -(dy*dz)/(r**3*root), rho2/(r**3*root)]
        ])
        return J

    H = np.zeros((24, 18))
    row = 0

    for obs in range(3):
        p_obs = state[6*obs: 6*obs + 3]

        # Add relative target measurements
        for tgt in range(3):
            if obs == tgt:
                continue

            p_tgt = state[6*tgt: 6*tgt + 3]

            J = meas_jacobian_from_delta(p_tgt - p_obs)

            H[row:row+3, 6*obs: 6*obs + 3] = -J
            H[row:row+3, 6*tgt: 6*tgt + 3] =  J
            row += 3

        # Add relative landmark measurement
        if obs == 0:
            J = meas_jacobian_from_delta(landmark_pos - p_obs)
            H[row:row+3, 6*obs: 6*obs + 3] = -J
            row += 3
        
        # Add altimeter
        H[row, 6*obs + 2] = 1
        row +=1
        
    assert(H.shape == (24, 18))
    return H

def make_Q_single(dt, q):
    Q1 = q * np.array([
        [dt**4/4, dt**3/2],
        [dt**3/2, dt**2]
    ])

    Q = np.zeros((6, 6))
    Q[np.ix_([0, 3], [0, 3])] = Q1
    Q[np.ix_([1, 4], [1, 4])] = Q1
    Q[np.ix_([2, 5], [2, 5])] = Q1
    return Q