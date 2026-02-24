import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # seemed to help for 3d plot despite being unused?

from drone import Drone
import utils

# Set timestep
dt = 0.01
dt_inv = 100

# Create leader drone object
leader = Drone(dt=dt)
# Set initial condition (will throw error if no initial condition is set)
init_state_leader = np.zeros(12)
init_state_leader[2] = 20.
leader.set_init_condition(init_state_leader)

# Create follower drones
follower_1 = Drone(dt=dt)
follower_2 = Drone(dt=dt)
# Set initial condition
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
num_steps = sim_time * dt_inv

# Simulation loop
for i in range(num_steps):
    leader.step_dynamics()
    follower_1.step_dynamics()
    follower_2.step_dynamics()
    # Add EKF and UKF steps under dynamics propagation

####################################################################
##############              PLOTTING BELOW           ###############
####################################################################
# All plotting tools should just take the drone objects as inputs!

# All States and Controls Plots!
fig1 = utils.plot_states_and_controls(leader, title='Leader Drone')
# fig2 = utils.plot_states_and_controls(leader, title='Follower 1 Drone')
# fig3 = utils.plot_states_and_controls(leader, title='Follower 2 Drone')

# 2D Aerial View
fig_3d = utils.plot_aerial_view(
    [leader, follower_1, follower_2], 
    ['Leader', 'Follower 1', 'Follower 2']
)

# 3D Plot
fig_3d = utils.plot_3d_trajectory_all_drones(
    [leader, follower_1, follower_2], 
    ['Leader', 'Follower 1', 'Follower 2']
)

plt.show()  # Only call after all plots show they appear simulataneously