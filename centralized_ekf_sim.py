import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from mpl_toolkits.mplot3d import Axes3D 
from measurementmodel import MeasurementModel
from dead_reckoning import DeadReckon
import centralized_ekf_final as centralized

from drone import Drone
import utils

np.random.seed(10)

# Set timestep
dt = 0.01
dt_inv = 100

# Create leader drone object
leader = Drone(dt=dt)
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


# Sim parameters
sim_time = 60  # seconds
num_steps = int(sim_time * dt_inv)

# Initting gyroscope measurement history
gyro_history_leader = []
gyro_history_f1 = []
gyro_history_f2 = []

# All relative measurements for EKF input
leader_measurements_time_hist = []
follower_1_measurements_time_hist = []
follower_2_measurements_time_hist = []
stacked_measurement_vector_time_hist = []

# Simulation loop
for i in range(num_steps):
    # Take measurements from current state
    # Leader's LIDAR measurements
    z_leader_to_landmark = meas_model.get_relative_measurement(leader.state, landmark_pos, add_noise=True)
    z_leader_to_1 = meas_model.get_relative_measurement(leader.state, follower_1.state[:3], add_noise=True)
    z_leader_to_2 = meas_model.get_relative_measurement(leader.state, follower_2.state[:3], add_noise=True)

    # Follower 1's LIDAR measurements
    z_follower_1_to_leader = meas_model.get_relative_measurement(follower_1.state, leader.state[:3], add_noise=True)
    z_follower_1_to_2 = meas_model.get_relative_measurement(follower_1.state, follower_2.state[:3], add_noise=True)

    # Follower 2's LIDAR measurements
    z_follower_2_to_leader = meas_model.get_relative_measurement(follower_2.state, leader.state[:3], add_noise=True)
    z_follower_2_to_1 = meas_model.get_relative_measurement(follower_2.state, follower_1.state[:3], add_noise=True)

    # All drones' altimeter measurements
    alt_leader = meas_model.get_altimeter_measurement(leader.state, add_noise=True)
    alt_follower_1 = meas_model.get_altimeter_measurement(follower_1.state, add_noise=True)
    alt_follower_2 = meas_model.get_altimeter_measurement(follower_2.state, add_noise=True)

    # Make the beefy measurement vector
    leader_meas = np.concatenate([z_leader_to_1, z_leader_to_2, z_leader_to_landmark, [alt_leader]])
    follower_1_meas = np.concatenate([z_follower_1_to_leader, z_follower_1_to_2, [alt_follower_1]])
    follower_2_meas = np.concatenate([z_follower_2_to_leader, z_follower_2_to_1, [alt_follower_2]])
    stacked_measurement_vector = np.concatenate([leader_meas, follower_1_meas, follower_2_meas])

    # Append measurements to time history
    leader_measurements_time_hist.append(leader_meas)
    follower_1_measurements_time_hist.append(follower_1_meas)
    follower_2_measurements_time_hist.append(follower_2_meas)
    stacked_measurement_vector_time_hist.append(stacked_measurement_vector)

    # Gyros from current state
    gyro_leader = meas_model.get_gyroscope_measurement(leader.state, add_noise=True)
    gyro_follower_1 = meas_model.get_gyroscope_measurement(follower_1.state, add_noise=True)
    gyro_follower_2 = meas_model.get_gyroscope_measurement(follower_2.state, add_noise=True)

    # Append gyros to time history
    gyro_history_leader.append(gyro_leader)
    gyro_history_f1.append(gyro_follower_1)
    gyro_history_f2.append(gyro_follower_2)

    # Step dynamics to next time
    leader.step_dynamics()
    follower_1.step_dynamics(control_input=follower_1.random_walk_controller())
    follower_2.step_dynamics(control_input=follower_2.random_walk_controller())
        
gyro_history_leader = np.array(gyro_history_leader)
gyro_history_f1 = np.array(gyro_history_f1)
gyro_history_f2 = np.array(gyro_history_f2)

####################################################################
##############        ATTITUDE DEAD RECKON           ###############
####################################################################

# Initialize dead reckon class
dead_reckon = DeadReckon(np.zeros(3), np.zeros(3))
dead_reckon_follower1 = DeadReckon(np.zeros(3), np.zeros(3))
dead_reckon_follower2 = DeadReckon(np.zeros(3), np.zeros(3))
# Feed gyro measurements to get attitude history
for i in range(num_steps):
    gyro_meas = gyro_history_leader[i]
    dead_reckon.step(gyro_meas)
    
    gyro_meas = gyro_history_f1[i]
    dead_reckon_follower1.step(gyro_meas)
    
    gyro_meas = gyro_history_f2[i]
    dead_reckon_follower2.step(gyro_meas)

dead_reckon_time_hist_leader = dead_reckon.get_time_hist()
dead_reckon_time_hist_follower1 = dead_reckon_follower1.get_time_hist()
dead_reckon_time_hist_follower2 = dead_reckon_follower2.get_time_hist()

####################################################################
############           CENTRALIZED FILTER              #############
####################################################################

# Initialize EKF with true initial states with some  relatively certain covariance
cent_mu_est = np.zeros(18)
cent_mu_est[:3] = [0, 0, 2]
cent_mu_est[6:9] = [-7, -5, 5]
cent_mu_est[12:15] = [7, 5, 5]
cent_cov_est = np.eye(18) * 1

# Initial all data-storing arrays
centralized_mu_time_hist = [cent_mu_est]
centralized_cov_time_hist = [cent_cov_est]
times = leader.get_times()
true_states = leader.get_state_time_history()
control_hist_leader = leader.get_control_history()
control_hist_follower1 = follower_1.get_control_history()
control_hist_follower2 = follower_2.get_control_history()

# Define Q (crazy high process noise cus the model kinda sucks)
Q_single = centralized.make_Q_single(dt, q=1000.0)
Q = block_diag(Q_single, Q_single, Q_single)

# Define R based on sensor specs and expected dead reckoning drift
sigma_gyro = 0.01
sigma_yaw = sigma_gyro**2 * 3600
sigma_pitch = sigma_yaw
altimeter_sigma = 0.1
R_single = meas_model.R + np.diag([0., sigma_yaw**2, sigma_pitch**2])
R_alt = np.array([[altimeter_sigma**2]])
R_leader = block_diag(R_single, R_single, R_single, R_alt)
R_follower = block_diag(R_single, R_single, R_alt)
R_full = block_diag(R_leader, R_follower, R_follower)

# Loop through all measurements
for i in range(len(times) - 2):
    control = np.concatenate([
        control_hist_leader[i, :], 
        control_hist_follower1[i, :], 
        control_hist_follower2[i, :]
    ])
    
    # Get attitude states from dead reckoning
    attitude_state_leader = dead_reckon_time_hist_leader[i]
    attitude_state_follower1 = dead_reckon_time_hist_follower1[i]
    attitude_state_follower2 = dead_reckon_time_hist_follower2[i]
    stacked_attitude_state = np.concatenate([
        attitude_state_leader,
        attitude_state_follower1,
        attitude_state_follower2,
    ])

    # Predict step
    mu_predict = centralized.dynamics_model_joint(cent_mu_est, stacked_attitude_state, control)
    G = centralized.get_G_full(cent_mu_est)
    cov_predict = G @ cent_cov_est @ G.T + Q

    # Get real measurements
    z = stacked_measurement_vector_time_hist[i + 1]
    
    # Update step
    z_mu = centralized.measurement_model_full(
        mu_predict, 
        meas_model, 
        landmark_pos, 
        stacked_attitude_state, 
        add_noise=False
    )
    
    # Calculate residual
    y = z - z_mu
    
    # Wrap the azimuth elevation angles just in case
    angle_indices = [1, 2, 4, 5, 7, 8, 11, 12, 14, 15, 18, 19, 21, 22]
    for idx in angle_indices:
        y[idx] = (y[idx] + np.pi) % (2*np.pi) - np.pi

    # Measurement Jacobian
    H = centralized.getH_full(mu_predict, landmark_pos)
    
    # Kalman Gain
    S = H @ cov_predict @ H.T + R_full
    K = cov_predict @ H.T @ np.linalg.inv(S)
    
    # Posterior calcluations
    cent_mu_est = mu_predict + K @ y
    I = np.eye(18)
    cent_cov_est = (I - K @ H) @ cov_predict @ (I - K @ H).T + K @ R_full @ K.T
    
    # Append to time hist
    centralized_mu_time_hist.append(cent_mu_est)
    centralized_cov_time_hist.append(cent_cov_est)

# Convert time hist to array
mu_hist = np.array(centralized_mu_time_hist)
P_hist = np.array(centralized_cov_time_hist)

####################################################################
##############              PLOTTING BELOW           ###############
####################################################################

# Get true states for plotting
true_states_leader = leader.get_state_time_history()
true_states_follower_1 = follower_1.get_state_time_history()
true_states_follower_2 = follower_2.get_state_time_history()



# leader
utils.plot_drone_ekf_diagnostics(
    times,
    mu_hist[:, 0:6],
    P_hist[:, 0:6, 0:6],
    true_states_leader,
    drone_name="Leader"
)

# follower 1
utils.plot_drone_ekf_diagnostics(
    times,
    mu_hist[:, 6:12],
    P_hist[:, 6:12, 6:12],
    true_states_follower_1,
    drone_name="Follower 1"
)

# follower 2
utils.plot_drone_ekf_diagnostics(
    times,
    mu_hist[:, 12:18],
    P_hist[:, 12:18, 12:18],
    true_states_follower_2,
    drone_name="Follower 2"
)

utils.plot_drone_trajectory_3d(
    times,
    mu_hist[:, 0:6],
    true_states_leader,
    drone_name="Leader"
)

utils.plot_drone_trajectory_3d(
    times,
    mu_hist[:, 6:12],
    true_states_follower_1,
    drone_name="Follower 1"
)

utils.plot_drone_trajectory_3d(
    times,
    mu_hist[:, 12:18],
    true_states_follower_2,
    drone_name="Follower 2"
)

plt.show()