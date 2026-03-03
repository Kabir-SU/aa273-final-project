import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from measurementmodel import MeasurementModel

from drone import Drone
import utils

np.random.seed(273)

# Set timestep
dt = 0.01
dt_inv = 100

# Create leader drone object
leader = Drone(dt=dt)
meas_model = MeasurementModel(range_std=0.1, angle_std=0.1)
landmark_pos = np.array([0, 0, 0])

init_state_leader = np.zeros(12)
init_state_leader[2] = 20.
leader.set_init_condition(init_state_leader)

# Create follower drones
follower_1 = Drone(dt=dt)
follower_2 = Drone(dt=dt)

# Set initial conditions
init_state_follower_1 = np.zeros(12)
init_state_follower_1[0] = -7.
init_state_follower_1[1] = -5.
init_state_follower_1[2] = 10. 
follower_1.set_init_condition(init_state_follower_1)

init_state_follower_2 = np.zeros(12)
init_state_follower_2[0] = 7.
init_state_follower_2[1] = 5.
init_state_follower_2[2] = 10. 
follower_2.set_init_condition(init_state_follower_2)


# Sim parameters
sim_time = 50  # seconds
num_steps = int(sim_time * dt_inv)

# Initting rel measurement vectors
meas_links_leader = []
meas_links_f1 = []
meas_links_f2 = []

# Initting gyroscope measurement history
gyro_history_leader = []
gyro_history_f1 = []
gyro_history_f2 = []

# Simulation loop
for i in range(num_steps):
    leader.step_dynamics()
    # follower_1.step_dynamics()
    # follower_2.step_dynamics()
    follower_1.step_dynamics(control_input=follower_1.random_walk_controller())
    follower_2.step_dynamics(control_input=follower_2.random_walk_controller())
    
    z_leader = meas_model.get_relative_measurement(leader.state, landmark_pos, add_noise=True)
    z_follower_1 = meas_model.get_relative_measurement(follower_1.state, leader.state[:3], add_noise=True)
    z_follower_2 = meas_model.get_relative_measurement(follower_2.state, leader.state[:3], add_noise=True)
    
    gyro_leader = meas_model.get_gyroscope_measurement(leader.state, add_noise=True)
    gyro_follower_1 = meas_model.get_gyroscope_measurement(follower_1.state, add_noise=True)
    gyro_follower_2 = meas_model.get_gyroscope_measurement(follower_2.state, add_noise=True)
    
    gyro_history_leader.append(gyro_leader)
    gyro_history_f1.append(gyro_follower_1)
    gyro_history_f2.append(gyro_follower_2)

    if i % 100 == 0:
        meas_links_leader.append((leader.state[:3].copy(), landmark_pos))
        meas_links_f1.append((follower_1.state[:3].copy(), leader.state[:3].copy()))
        meas_links_f2.append((follower_2.state[:3].copy(), leader.state[:3].copy()))

gyro_history_leader = np.array(gyro_history_leader)
gyro_history_f1 = np.array(gyro_history_f1)
gyro_history_f2 = np.array(gyro_history_f2)

####################################################################
##############              PLOTTING BELOW           ###############
####################################################################
# All plotting tools should just take the drone objects as inputs!

# Plot Gyroscope Measurements
fig_gyro, axs_gyro = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig_gyro.suptitle('Leader Drone Gyroscope Measurements vs True Body Rates')
times = leader.get_times()[:-1]  # Exclude last time step to match measurement length

true_states = leader.get_state_time_history()[:-1]
true_p, true_q, true_r = true_states[:, 9], true_states[:, 10], true_states[:, 11]

# P (leader)
axs_gyro[0].plot(times, true_p, label='True $p$', color='black', linewidth=2)
axs_gyro[0].plot(times, gyro_history_leader[:, 0], label='Measured $p$', color='red', alpha=0.5)
axs_gyro[0].set_ylabel('Roll Rate (rad/s)')
axs_gyro[0].legend()
axs_gyro[0].grid(True)

# Q (leader)
axs_gyro[1].plot(times, true_q, label='True $q$', color='black', linewidth=2)
axs_gyro[1].plot(times, gyro_history_leader[:, 1], label='Measured $q$', color='green', alpha=0.5)
axs_gyro[1].set_ylabel('Pitch Rate (rad/s)')
axs_gyro[1].legend()
axs_gyro[1].grid(True)

# R (leader)
axs_gyro[2].plot(times, true_r, label='True $r$', color='black', linewidth=2)
axs_gyro[2].plot(times, gyro_history_leader[:, 2], label='Measured $r$', color='blue', alpha=0.5)
axs_gyro[2].set_ylabel('Yaw Rate (rad/s)')
axs_gyro[2].set_xlabel('Time (s)')
axs_gyro[2].legend()
axs_gyro[2].grid(True)

# All States and Controls Plots!
fig1 = utils.plot_states_and_controls(leader, title='Leader Drone')
# fig2 = utils.plot_states_and_controls(leader, title='Follower 1 Drone')
# fig3 = utils.plot_states_and_controls(leader, title='Follower 2 Drone')

fig_2d = utils.plot_aerial_view(
    [leader, follower_1, follower_2], 
    ['Leader', 'Follower 1', 'Follower 2']
)

# 3D Plot
fig_3d = utils.plot_3d_trajectory_all_drones(
    [leader, follower_1, follower_2], 
    ['Leader', 'Follower 1', 'Follower 2']
)
ax = fig_3d.gca()

ax.scatter(landmark_pos[0], landmark_pos[1], landmark_pos[2], 
           color='red', marker='X', s=100, label='Stationary Landmark')

# Line of sight plots
for obs, tar in meas_links_leader:
    ax.plot([obs[0], tar[0]], [obs[1], tar[1]], [obs[2], tar[2]], color='green', linestyle='--', alpha=0.3)
for obs, tar in meas_links_f1:
    ax.plot([obs[0], tar[0]], [obs[1], tar[1]], [obs[2], tar[2]], color='blue', linestyle='--', alpha=0.3)
for obs, tar in meas_links_f2:
    ax.plot([obs[0], tar[0]], [obs[1], tar[1]], [obs[2], tar[2]], color='orange', linestyle='--', alpha=0.3)

ax.legend()
plt.show()