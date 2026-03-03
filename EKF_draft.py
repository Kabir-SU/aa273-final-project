#Centralized approach
import numpy as np

dt = 0.01
m = 20
g = 9.81
jphi, jtheta, jpsi = 10, 10, 5

def truedynamics(state, u): 
    F, tauphi, tautheta, taupsi = u
    px, py, pz, vx, vy, vz, phi, theta, psi, dphi, dtheta, dpsi = state
    step = np.zeros_like(state)
    step[0:3] = [px + dt*vx, py + dt*vy, pz + dt*vz]
    vxstep = vx + dt*F/m*(-np.cos(phi)*np.sin(theta)*np.cos(psi) - np.sin(phi)*np.sin(psi))
    vystep = vy + dt*F/m*(-np.cos(phi)*np.sin(theta)*np.sin(psi) + np.sin(phi)*np.cos(psi))
    vzstep = vz + dt*(g - np.cos(phi)*np.cos(theta)*F)/m
    step[3:6] = [vxstep, vystep, vzstep]
    step[6:9] = [phi + dt*dphi, theta + dt*dtheta, psi + dt*dpsi]
    step[9:] = [phi + dt*tauphi/jphi, theta + dt*tautheta/jtheta, psi + dt*taupsi/jpsi]
    return step

def dynamics_model_joint(state, u):
    nextstate = np.zeros_like(state)
    for i in range(0, 3):
        stepstate = truedynamics(state[i*12 : (i+1)*12], u[i*4 : (i+1)*4])
        nextstate[i*12: (i+1)*12] = stepstate[:]
    return nextstate

def getG_single(state, u):
    F, tauphi, tautheta, taupsi = u
    px, py, pz, vx, vy, vz, phi, theta, psi, dphi, dtheta, dpsi = state

    cphi = np.cos(phi);   sphi = np.sin(phi)
    cth  = np.cos(theta); sth  = np.sin(theta)
    cpsi = np.cos(psi);   spsi = np.sin(psi)

    G = np.eye(12)

    G[0,0] = 1.0; G[0,3] = dt
    G[1,1] = 1.0; G[1,4] = dt
    G[2,2] = 1.0; G[2,5] = dt

    # -----------------
    # Velocity updates
    # -----------------

    # Precompute projection terms
    A = (-cphi*sth*cpsi - sphi*spsi)
    B = (-cphi*sth*spsi + sphi*cpsi)

    # Derivatives of A
    dA_dphi   =  sphi*sth*cpsi - cphi*spsi
    dA_dtheta = -cphi*cth*cpsi
    dA_dpsi   =  cphi*sth*spsi - sphi*cpsi

    # Derivatives of B
    dB_dphi   =  sphi*sth*spsi + cphi*cpsi
    dB_dtheta = -cphi*cth*spsi
    dB_dpsi   = -cphi*sth*cpsi - sphi*spsi

    # vx
    G[3,3] = 1.0
    G[3,6] = dt*(F/m)*dA_dphi
    G[3,7] = dt*(F/m)*dA_dtheta
    G[3,8] = dt*(F/m)*dA_dpsi

    # vy
    G[4,4] = 1.0
    G[4,6] = dt*(F/m)*dB_dphi
    G[4,7] = dt*(F/m)*dB_dtheta
    G[4,8] = dt*(F/m)*dB_dpsi

    # vz
    G[5,5] = 1.0
    G[5,6] = dt*(F/m)*( sphi*cth )
    G[5,7] = dt*(F/m)*( cphi*sth )
    # no psi dependence

    # -----------------
    # Angle updates
    # -----------------
    G[6,6] = 1.0; G[6,9]  = dt
    G[7,7] = 1.0; G[7,10] = dt
    G[8,8] = 1.0; G[8,11] = dt

    # -----------------
    # Angular rate updates
    # -----------------
    G[9,9]   = 1.0
    G[10,10] = 1.0
    G[11,11] = 1.0

    return G
