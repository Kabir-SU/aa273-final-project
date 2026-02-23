import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # seemed to help for 3d plot despite being unused?

from drone import Drone

# Set timestep
dt = 0.01
dt_inv = 100

# Create leader drone object
leader = Drone(dt=dt)

# Set initial condition (will throw error if no initial condition is set)
init_state_leader = np.zeros(12)
init_state_leader[2] = 20.  # set initial z to 10 meters
leader.set_init_condition(init_state_leader)

# Sim parameters
sim_time = 50  # seconds
num_steps = sim_time * dt_inv

# Simulation loop
for i in range(num_steps):
    leader.step_dynamics()
    # Add EKF and UKF steps under dynamics propagation

####################################################################
##############              PLOTTING BELOW           ###############
####################################################################
# Extract states from drone
leader_states = leader.get_state_time_history()
leader_times = leader.get_times()
leader_controls = leader.get_control_history()

fig, axs = plt.subplots(3, 3, figsize=(12, 8), sharex=True)

# POSITIONS PLOTS
axs[0, 0].plot(leader_times, leader_states[:, 0], label='Leader')
axs[0, 0].set_ylabel('X (m)')
axs[0, 0].grid()

axs[1, 0].plot(leader_times, leader_states[:, 1], label='Leader')
axs[1, 0].set_ylabel('Y (m)')
axs[1, 0].grid()

axs[2, 0].plot(leader_times, leader_states[:, 2], label='Leader')
axs[2, 0].set_ylabel('Z (m)')
axs[2, 0].set_xlabel('Time (s)')
axs[2, 0].grid()

# ATTITUDE PLOTS
axs[0, 1].plot(leader_times, np.rad2deg(leader_states[:, 6]))
axs[0, 1].set_ylabel('Roll (deg)')
axs[0, 1].grid()

axs[1, 1].plot(leader_times, np.rad2deg(leader_states[:, 7]))
axs[1, 1].set_ylabel('Pitch (deg)')
axs[1, 1].grid()

axs[2, 1].plot(leader_times, np.rad2deg(leader_states[:, 8]))
axs[2, 1].set_ylabel('Yaw (Deg)')
axs[2, 1].set_xlabel('Time (s)')
axs[2, 1].grid()

# CONTROLS PLOTS
axs[0, 2].plot(leader_times[1:], leader_controls[:, 0])
axs[0, 2].set_ylabel('F (N)')
axs[0, 2].grid()

axs[1, 2].plot(leader_times[1:], leader_controls[:, 1])
axs[1, 2].set_ylabel('Tx (Nm)')
axs[1, 2].grid()

axs[2, 2].plot(leader_times[1:], leader_controls[:, 2])
axs[2, 2].set_ylabel('Ty (Nm)')
axs[2, 2].set_xlabel('Time (s)')
axs[2, 2].grid()
plt.tight_layout()

# 2D Aerial View
plt.figure()
plt.plot(leader_states[:,0], leader_states[:,1], label='Leader')
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.axis("equal")
plt.grid()

# 3D Plot
fig3 = plt.figure(figsize=(8,6))
ax = fig3.add_subplot(111, projection='3d')
ax.plot(leader_states[:,0], leader_states[:,1], leader_states[:, 2], label='Leader')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Drone Trajectory')
plt.tight_layout()

plt.show()  # Only call after all plots show they appear simulataneously