import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from measurementmodel import MeasurementModel
from dead_reckoning import DeadReckon
from drone import Drone
from consensus_ekf_final import ConsensusEKF
import utils

np.random.seed(273)

dt = 0.01
dt_inv = 100
leader = Drone(dt=dt)
follower_1 = Drone(dt=dt)
follower_2 = Drone(dt=dt)
meas_model = MeasurementModel(range_std=0.1, angle_std=0.1)

landmark_pos = np.array([5, 5, 0])

init_state_leader = np.zeros(12)
init_state_leader[2] = 2.
leader.set_init_condition(init_state_leader)

# Create follower drones
follower_1 = Drone(dt=dt)
follower_2 = Drone(dt=dt)

# Follower 1 initial conditions
init_state_follower_1 = np.zeros(12)
init_state_follower_1[0] = -7.
init_state_follower_1[1] = -5.
init_state_follower_1[2] = 5. 
follower_1.set_init_condition(init_state_follower_1)

# Follower 2 initial conditions
init_state_follower_2 = np.zeros(12)
init_state_follower_2[0] = 7.
init_state_follower_2[1] = 5.
init_state_follower_2[2] = 5. 
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

init_cov = np.eye(18) * 1.0

ekf_leader = ConsensusEKF(init_mu_leader, init_cov.copy(), dt=dt, landmark_track=True, landmark_pos=landmark_pos)
ekf_f1     = ConsensusEKF(init_mu_f1, init_cov.copy(), dt=dt)
ekf_f2     = ConsensusEKF(init_mu_f2, init_cov.copy(), dt=dt)

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

times = leader.get_times()
true_states = leader.get_state_time_history()

# Alignment helpers
# All blocks are now 6-element CV states: [px,vx,py,vy,pz,vz]
def align_est_for_leader(ef1, ef2, pf1, pf2):
    # Leader: [own=Ldr(6), T1=F1(6), T2=F2(6)]
    # F1:     [own=F1(6),  T1=Ldr(6), T2=F2(6)]  -> own=F1 goes to T1, T1=Ldr goes to own
    # F2:     [own=F2(6),  T1=Ldr(6), T2=F1(6)]  -> own=F2 goes to T2, T2=F1 goes to T1
    f1_aligned = np.concatenate([ef1[6:12], ef1[0:6],  ef1[12:18]])
    f2_aligned = np.concatenate([ef2[6:12], ef2[12:18], ef2[0:6]])

    idx1 = np.r_[6:12, 0:6, 12:18]
    idx2 = np.r_[6:12, 12:18, 0:6]
    cov1_aligned = pf1[np.ix_(idx1, idx1)]
    cov2_aligned = pf2[np.ix_(idx2, idx2)]

    return ([f1_aligned, f2_aligned], [cov1_aligned, cov2_aligned])


def align_est_for_f1(el, ef2, pl, pf2):
    # F1: [own=F1(6), T1=Ldr(6), T2=F2(6)]
    # Ldr: [own=Ldr(6), T1=F1(6), T2=F2(6)] -> T1=F1 goes to own, own=Ldr goes to T1
    # F2:  [own=F2(6), T1=Ldr(6), T2=F1(6)] -> T2=F1 goes to own, T1=Ldr stays, own=F2 goes to T2
    ldr_aligned = np.concatenate([el[6:12],  el[0:6],  el[12:18]])
    f2_aligned  = np.concatenate([ef2[12:18], ef2[6:12], ef2[0:6]])

    idx_ldr = np.r_[6:12, 0:6, 12:18]
    idx_f2  = np.r_[12:18, 6:12, 0:6]
    cov_ldr_aligned = pl[np.ix_(idx_ldr, idx_ldr)]
    cov2_aligned    = pf2[np.ix_(idx_f2, idx_f2)]

    return ([ldr_aligned, f2_aligned], [cov_ldr_aligned, cov2_aligned])


def align_est_for_f2(el, ef1, pl, pf1):
    # F2: [own=F2(6), T1=Ldr(6), T2=F1(6)]
    # Ldr: [own=Ldr(6), T1=F1(6), T2=F2(6)] -> T2=F2 goes to own, own=Ldr goes to T1, T1=F1 goes to T2
    # F1:  [own=F1(6), T1=Ldr(6), T2=F2(6)] -> T2=F2 goes to own, T1=Ldr stays, own=F1 goes to T2
    ldr_aligned = np.concatenate([el[12:18], el[0:6],   el[6:12]])
    f1_aligned  = np.concatenate([ef1[12:18], ef1[6:12], ef1[0:6]])

    idx_ldr = np.r_[12:18, 0:6, 6:12]
    idx_f1  = np.r_[12:18, 6:12, 0:6]
    cov_ldr_aligned = pl[np.ix_(idx_ldr, idx_ldr)]
    cov1_aligned    = pf1[np.ix_(idx_f1, idx_f1)]

    return ([ldr_aligned, f1_aligned], [cov_ldr_aligned, cov1_aligned])

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
    alt_ldr = meas_model.get_altimeter_measurement(
        leader.state_time_history[i],
        add_noise=True,
    )
    y_leader = np.append(y_leader, alt_ldr)
    

    # F1 measures Leader and F2
    z_f1_ldr = meas_model.get_relative_measurement(
        follower_1.state_time_history[i], leader.state_time_history[i][:3], add_noise=True)
    z_f1_f2  = meas_model.get_relative_measurement(
        follower_1.state_time_history[i], follower_2.state_time_history[i][:3], add_noise=True)
    y_f1 = np.concatenate([z_f1_ldr, z_f1_f2])
    alt_f1 = meas_model.get_altimeter_measurement(
        follower_1.state_time_history[i], 
        add_noise=True
    )
    y_f1 = np.append(y_f1, alt_f1)
    

    # F2 measures Leader and F1
    z_f2_ldr = meas_model.get_relative_measurement(
        follower_2.state_time_history[i], leader.state_time_history[i][:3], add_noise=True)
    z_f2_f1  = meas_model.get_relative_measurement(
        follower_2.state_time_history[i], follower_1.state_time_history[i][:3], add_noise=True)
    y_f2 = np.concatenate([z_f2_ldr, z_f2_f1])
    alt_f2 = meas_model.get_altimeter_measurement(
        follower_2.state_time_history[i], 
        add_noise=True
    )
    y_f2 = np.append(y_f2, alt_f2)

    # Align neighbor estimates (use previous timestep's prior)
    ldr_neigh_ests, ldr_neigh_covs = align_est_for_leader(
        ekf_f1.prior_state, 
        ekf_f2.prior_state, 
        ekf_f1.prior_cov,
        ekf_f2.prior_cov,
    )
    f1_neigh_ests, f1_neigh_covs = align_est_for_f1(
        ekf_leader.prior_state,
        ekf_f2.prior_state,
        ekf_leader.prior_cov,
        ekf_f2.prior_cov,
    )
    f2_neigh_ests, f2_neigh_covs = align_est_for_f2(
        ekf_leader.prior_state,
        ekf_f1.prior_state,
        ekf_leader.prior_cov,
        ekf_f1.prior_cov,
    )

    # Dead-reckoned attitudes at this timestep
    ekf_leader.step(
        y_leader, 
        ldr_neigh_ests,  
        att_ldr[i], 
        control_leader[i], 
        y_landmark, 
        landmark_pos,
    )
    ekf_f1.step(
        y_f1, 
        f1_neigh_ests, 
        att_f1[i], 
        control_f1[i],
    )
    ekf_f2.step(
        y_f2, 
        f2_neigh_ests,
        att_f2[i], 
        control_f2[i]
    )

times_m = leader.get_times()[:-1]

ts_m = leader.get_state_time_history()[:-1]
ts_f1 = follower_1.get_state_time_history()[:-1]
ts_f2 = follower_2.get_state_time_history()[:-1]

leader_est = ekf_leader.get_state_est_time_hist()
f1_est     = ekf_f1.get_state_est_time_hist()
f2_est     = ekf_f2.get_state_est_time_hist()

leader_cov = ekf_leader.get_cov_est_time_hist()
f1_cov = ekf_f1.get_cov_est_time_hist()
f2_cov = ekf_f2.get_cov_est_time_hist()

idx_reorder = [0, 2, 4, 1, 3, 5]

leader_hist = leader_est[:, idx_reorder]
follower_1_hist = f1_est[:, idx_reorder]
follower_2_hist = f2_est[:, idx_reorder]
leader_cov_hist = leader_cov[:, idx_reorder][:, :, idx_reorder]
f1_cov_hist     = f1_cov[:, idx_reorder][:, :, idx_reorder]
f2_cov_hist     = f2_cov[:, idx_reorder][:, :, idx_reorder]

utils.plot_drone_ekf_diagnostics(
    times_m,
    leader_hist,
    leader_cov_hist,
    ts_m,
    drone_name="Leader"
)

utils.plot_drone_ekf_diagnostics(
    times_m,
    follower_1_hist,
    f1_cov_hist,
    ts_f1,
    drone_name="Follower 1"
)

utils.plot_drone_ekf_diagnostics(
    times_m,
    follower_2_hist,
    f2_cov_hist,
    ts_f2,
    drone_name="Follower 2"
)

# utils.plot_drone_trajectory_3d(
#     times_m,
#     leader_hist,
#     ts_m,
#     landmark_pos,
#     drone_name="Leader"
# )

# utils.plot_drone_trajectory_3d(
#     times_m,
#     follower_1_hist,
#     ts_f1,
#     landmark_pos,
#     drone_name="Follower 1"
# )

# utils.plot_drone_trajectory_3d(
#     times_m,
#     follower_2_hist,
#     ts_f2,
#     landmark_pos,
#     drone_name="Follower 2"
# )




####################################################################
##############              NEES and RMSE            ###############
####################################################################
''' We have true_states_leader, true_states_follower_1, and true_states_follower_2 which are
np arrays of the true states in the trajectory. mu_hist contains the estimated state (position and 
velocity) of the three drones. NEES will be calculated over the full state (per drone), and RMSE 
will be calculated separately for position and velocity.
    '''
from scipy.stats import chi2
def plot_metrics(true_state_12, est_state_6, cov_history_6x6, drone):
    num_estimates = est_state_6.shape[0]
    true_state_6 = true_state_12[:num_estimates, :6]
    errors = true_state_6 - est_state_6
    pos_error_sq = np.sum(errors[:, :3]**2, axis=1)
    vel_error_sq = np.sum(errors[:, 3:6]**2, axis=1)  
    rmse_p = np.sqrt(np.mean(pos_error_sq))
    rmse_v = np.sqrt(np.mean(vel_error_sq))
    print(f"{drone}: Position RMSE = {rmse_p:.4f} m, Velocity RMSE = {rmse_v:.4f} m/s")

    nees_hist = []
    for k in range(len(true_state_6)):
        nees = errors[k] @ np.linalg.solve(cov_history_6x6[k], errors[k])
        nees_hist.append(nees)

    plt.figure()
    plt.plot(times[:len(nees_hist)], nees_hist, label = 'NEES')
    plt.axhline(y=6, color='g', linestyle='--', label='Expected Value (6)')
    plt.axhline(y=chi2.ppf(0.975, df=6), color='r', linestyle=':', label='95% Upper Bound')
    plt.axhline(y=chi2.ppf(0.025, df=6), color='b', linestyle=':', label='95% Lower Bound')
    plt.title(f"{drone} Normalized Estimation Error Squared vs Time for Position and Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("NEES score")
    plt.legend()
    plt.grid(True)

# plot_metrics(ts_f1, follower_1_hist[:-1], f1_cov_hist[:-1], "Follower 1")
# plot_metrics(ts_f2, follower_2_hist[:-1], f2_cov_hist[:-1], "Follower 2")
# plot_metrics(ts_m, leader_hist[:-1], leader_cov_hist[:-1], "Leader")

plt.show()