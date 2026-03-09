import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from measurementmodel import MeasurementModel
from dead_reckoning import DeadReckon
from drone import Drone
from consensus_ekf_wip import ConsensusEKF
import utils

np.random.seed(273)

dt = 0.01
dt_inv = 100
leader = Drone(dt=dt)
follower_1 = Drone(dt=dt)
follower_2 = Drone(dt=dt)
meas_model = MeasurementModel(range_std=0.1, angle_std=0.1)
landmark_pos = np.array([0., 0., 0.])

init_state_leader = np.zeros(12)
init_state_leader[2] = 20.
leader.set_init_condition(init_state_leader)

init_state_follower_1 = np.zeros(12)
init_state_follower_1[0] = -7.
init_state_follower_1[1] = -5.
init_state_follower_1[2] = 10.
follower_1.set_init_condition(init_state_follower_1)

init_state_follower_2 = np.zeros(12)
init_state_follower_2[0] =  7.
init_state_follower_2[1] =  5.
init_state_follower_2[2] = 10.
follower_2.set_init_condition(init_state_follower_2)

sim_time  = 50
num_steps = int(sim_time * dt_inv)

# -----------------------------------------------------------------------
# EKF state layout (18 states):
#   Own:      [px, vx, py, vy, pz, vz]   (indices 0-5)
#   Target k: [x,  vx, y,  vy, z,  vz]   (indices 6+6k .. 11+6k)
#
#   Leader : [own=Ldr(6), T1=F1(6), T2=F2(6)]
#   F1     : [own=F1(6),  T1=Ldr(6), T2=F2(6)]
#   F2     : [own=F2(6),  T1=Ldr(6), T2=F1(6)]
# -----------------------------------------------------------------------
def to_cv6(state12):
    """Extract CV state [px,vx,py,vy,pz,vz] from a 12-state drone vector."""
    return np.array([state12[0], state12[3],
                     state12[1], state12[4],
                     state12[2], state12[5]])

init_mu_leader = np.concatenate([to_cv6(init_state_leader),
                                  to_cv6(init_state_follower_1),
                                  to_cv6(init_state_follower_2)])

init_mu_f1 = np.concatenate([to_cv6(init_state_follower_1),
                               to_cv6(init_state_leader),
                               to_cv6(init_state_follower_2)])

init_mu_f2 = np.concatenate([to_cv6(init_state_follower_2),
                               to_cv6(init_state_leader),
                               to_cv6(init_state_follower_1)])

init_cov = np.eye(18) * 10.0

ekf_leader = ConsensusEKF(init_mu_leader, init_cov.copy(), dt=dt)
ekf_f1     = ConsensusEKF(init_mu_f1,     init_cov.copy(), dt=dt)
ekf_f2     = ConsensusEKF(init_mu_f2,     init_cov.copy(), dt=dt)

# -----------------------------------------------------------------------
# Simulation loop — dynamics + sensor data collection
# -----------------------------------------------------------------------
gyro_h_ldr, gyro_h_f1, gyro_h_f2 = [], [], []
meas_links_leader, meas_links_f1, meas_links_f2 = [], [], []

for i in range(num_steps):
    leader.step_dynamics()
    follower_1.step_dynamics(control_input=follower_1.random_walk_controller())
    follower_2.step_dynamics(control_input=follower_2.random_walk_controller())

    gyro_h_ldr.append(meas_model.get_gyroscope_measurement(leader.state,     add_noise=True))
    gyro_h_f1.append( meas_model.get_gyroscope_measurement(follower_1.state, add_noise=True))
    gyro_h_f2.append( meas_model.get_gyroscope_measurement(follower_2.state, add_noise=True))

    if i % 100 == 0:
        meas_links_leader.append((leader.state[:3].copy(),     landmark_pos.copy()))
        meas_links_f1.append(    (follower_1.state[:3].copy(), leader.state[:3].copy()))
        meas_links_f2.append(    (follower_2.state[:3].copy(), leader.state[:3].copy()))

gyro_h_ldr = np.array(gyro_h_ldr)
gyro_h_f1  = np.array(gyro_h_f1)
gyro_h_f2  = np.array(gyro_h_f2)

dr_ldr = DeadReckon(np.zeros(3), np.zeros(3))
dr_f1  = DeadReckon(np.zeros(3), np.zeros(3))
dr_f2  = DeadReckon(np.zeros(3), np.zeros(3))

for i in range(num_steps):
    dr_ldr.step(gyro_h_ldr[i])
    dr_f1.step( gyro_h_f1[i])
    dr_f2.step( gyro_h_f2[i])

att_ldr = dr_ldr.get_time_hist()   # (N+1, 6)
att_f1  = dr_f1.get_time_hist()
att_f2  = dr_f2.get_time_hist()

times       = leader.get_times()
true_states = leader.get_state_time_history()
labels_att  = ['phi', 'theta', 'psi', 'p', 'q', 'r']

fig_dr, axs_dr = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
fig_dr.suptitle('Leader Dead Reckoning vs True Attitude')
for i, ax in enumerate(axs_dr.flat):
    ax.plot(times, att_ldr[:, i],         label='Dead Reckoning')
    ax.plot(times, true_states[:, 6 + i], label='True')
    ax.set_title(labels_att[i])
    ax.grid(True)
axs_dr[0, 0].legend()
plt.tight_layout()

# Alignment helpers
# All blocks are now 6-element CV states: [px,vx,py,vy,pz,vz]
def align_est_for_leader(ef1, ef2):
    # Leader: [own=Ldr(6), T1=F1(6), T2=F2(6)]
    # F1:     [own=F1(6),  T1=Ldr(6), T2=F2(6)]  -> own=F1 goes to T1, T1=Ldr goes to own
    # F2:     [own=F2(6),  T1=Ldr(6), T2=F1(6)]  -> own=F2 goes to T2, T2=F1 goes to T1
    f1_aligned = np.concatenate([ef1[6:12], ef1[0:6],  ef1[12:18]])
    f2_aligned = np.concatenate([ef2[6:12], ef2[12:18], ef2[0:6]])
    return [f1_aligned, f2_aligned]

def align_est_for_f1(el, ef2):
    # F1: [own=F1(6), T1=Ldr(6), T2=F2(6)]
    # Ldr: [own=Ldr(6), T1=F1(6), T2=F2(6)] -> T1=F1 goes to own, own=Ldr goes to T1
    # F2:  [own=F2(6), T1=Ldr(6), T2=F1(6)] -> T2=F1 goes to own, T1=Ldr stays, own=F2 goes to T2
    ldr_aligned = np.concatenate([el[6:12],  el[0:6],  el[12:18]])
    f2_aligned  = np.concatenate([ef2[12:18], ef2[6:12], ef2[0:6]])
    return [ldr_aligned, f2_aligned]

def align_est_for_f2(el, ef1):
    # F2: [own=F2(6), T1=Ldr(6), T2=F1(6)]
    # Ldr: [own=Ldr(6), T1=F1(6), T2=F2(6)] -> T2=F2 goes to own, own=Ldr goes to T1, T1=F1 goes to T2
    # F1:  [own=F1(6), T1=Ldr(6), T2=F2(6)] -> T2=F2 goes to own, T1=Ldr stays, own=F1 goes to T2
    ldr_aligned = np.concatenate([el[12:18], el[0:6],   el[6:12]])
    f1_aligned  = np.concatenate([ef1[12:18], ef1[6:12], ef1[0:6]])
    return [ldr_aligned, f1_aligned]

control_leader = leader.get_control_history()
control_f1     = follower_1.get_control_history()
control_f2     = follower_2.get_control_history()

for i in range(num_steps):
    # Leader measures F1 and F2
    z_ldr_f1 = meas_model.get_relative_measurement(
        leader.state_time_history[i], follower_1.state_time_history[i][:3], add_noise=True)
    z_ldr_f2 = meas_model.get_relative_measurement(
        leader.state_time_history[i], follower_2.state_time_history[i][:3], add_noise=True)
    y_leader   = np.concatenate([z_ldr_f1, z_ldr_f2])
    y_landmark = meas_model.get_relative_measurement(
        leader.state_time_history[i], landmark_pos, add_noise=True)

    # F1 measures Leader and F2
    z_f1_ldr = meas_model.get_relative_measurement(
        follower_1.state_time_history[i], leader.state_time_history[i][:3], add_noise=True)
    z_f1_f2  = meas_model.get_relative_measurement(
        follower_1.state_time_history[i], follower_2.state_time_history[i][:3], add_noise=True)
    y_f1 = np.concatenate([z_f1_ldr, z_f1_f2])

    # F2 measures Leader and F1
    z_f2_ldr = meas_model.get_relative_measurement(
        follower_2.state_time_history[i], leader.state_time_history[i][:3], add_noise=True)
    z_f2_f1  = meas_model.get_relative_measurement(
        follower_2.state_time_history[i], follower_1.state_time_history[i][:3], add_noise=True)
    y_f2 = np.concatenate([z_f2_ldr, z_f2_f1])

    # Align neighbor estimates (use previous timestep's prior)
    ldr_neighbors = align_est_for_leader(ekf_f1.prior_state, ekf_f2.prior_state)
    f1_neighbors  = align_est_for_f1(ekf_leader.prior_state, ekf_f2.prior_state)
    f2_neighbors  = align_est_for_f2(ekf_leader.prior_state, ekf_f1.prior_state)

    # Dead-reckoned attitudes at this timestep
    ekf_leader.step(y_leader, ldr_neighbors, att_ldr[i], control_leader[i], y_landmark)
    ekf_f1.step(y_f1, f1_neighbors, att_f1[i], control_f1[i])
    ekf_f2.step(y_f2, f2_neighbors, att_f2[i], control_f2[i])

times_m = leader.get_times()[:-1]

ts_m    = leader.get_state_time_history()[:-1]

fig_gyro, axs_gyro = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig_gyro.suptitle('Leader Gyroscope Measurements vs True Body Rates')
rate_labels = [('True $p$', 'Measured $p$', 'Roll Rate (rad/s)',   'red'),
               ('True $q$', 'Measured $q$', 'Pitch Rate (rad/s)', 'green'),
               ('True $r$', 'Measured $r$', 'Yaw Rate (rad/s)',   'blue')]
for k, (tl, ml, yl, col) in enumerate(rate_labels):
    axs_gyro[k].plot(times_m, ts_m[:, 9+k], label=tl, color='black', linewidth=2)
    axs_gyro[k].plot(times_m, gyro_h_ldr[:, k], label=ml, color=col, alpha=0.5)
    axs_gyro[k].set_ylabel(yl); axs_gyro[k].legend(); axs_gyro[k].grid(True)
axs_gyro[2].set_xlabel('Time (s)')
plt.tight_layout()

fig_3d = utils.plot_3d_trajectory_all_drones(
    [leader, follower_1, follower_2],
    ['Leader True', 'Follower 1 True', 'Follower 2 True']
)
ax = fig_3d.gca()

leader_est = ekf_leader.get_state_est_time_hist()
f1_est     = ekf_f1.get_state_est_time_hist()
f2_est     = ekf_f2.get_state_est_time_hist()

# CV state: px=col0, py=col2, pz=col4
ax.plot(leader_est[:, 0], leader_est[:, 2], leader_est[:, 4],
        color='green',  linestyle=':', linewidth=2, label='Leader EKF')
ax.plot(f1_est[:, 0],     f1_est[:, 2],     f1_est[:, 4],
        color='blue',   linestyle=':', linewidth=2, label='Follower 1 EKF')
ax.plot(f2_est[:, 0],     f2_est[:, 2],     f2_est[:, 4],
        color='orange', linestyle=':', linewidth=2, label='Follower 2 EKF')
ax.scatter(*landmark_pos, color='red', marker='X', s=100, label='Stationary Landmark')
ax.legend()

plt.show()