import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from drone import Drone

drone = Drone()

dt = 0.01
dt_inv = 100

# Set initial condition
init_state = np.zeros(12)
init_state[2] = 10.  # set initial z to 10 meters
drone.set_init_condition(init_state)

# Simulation loop
for i in range(50 * dt_inv):
    t = dt * i
    control = np.array([40 * 9.81 * np.cos(2*t), 0., 0., 0.])
    drone.step_dynamics()

states = drone.get_state_time_history()
times = drone.get_times()
controls = drone.get_control_history()

fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True)

# POSITIONS PLOTS
axs[0, 0].plot(times, states[:, 0])
axs[0, 0].set_ylabel('X (m)')
axs[0, 0].grid()

axs[1, 0].plot(times, states[:, 1])
axs[1, 0].set_ylabel('Y (m)')
axs[1, 0].grid()

axs[2, 0].plot(times, states[:, 2])
axs[2, 0].set_ylabel('Z (m)')
axs[2, 0].set_xlabel('Time (s)')
axs[2, 0].grid()

# CONTROLS PLOTS
axs[0, 1].plot(times[1:], controls[:, 0])
axs[0, 1].set_ylabel('F (N)')
axs[0, 1].grid()

axs[1, 1].plot(times[1:], controls[:, 1])
axs[1, 1].set_ylabel('Tx (Nm)')
axs[1, 1].grid()

axs[2, 1].plot(times[1:], controls[:, 2])
axs[2, 1].set_ylabel('Ty (Nm)')
axs[2, 1].set_xlabel('Time (s)')
axs[2, 1].grid()
plt.tight_layout()

# 2D Aerial View
plt.figure()
plt.plot(states[:,0], states[:,1])
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.axis("equal")
plt.grid()

# 3D Plot
fig3 = plt.figure(figsize=(8,6))
ax = fig3.add_subplot(111, projection='3d')
ax.plot(states[:,0], states[:,1], states[:, 2])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Drone Trajectory')
plt.tight_layout()
plt.show()




plt.show()