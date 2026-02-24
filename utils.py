import numpy as np
import matplotlib.pyplot as plt
from drone import Drone

def plot_states_and_controls(drone: Drone, title='Leader Drone'):
    """Plots the states and controls for a single drone in a 9 by 9 grid"""
    states = drone.get_state_time_history()
    controls = drone.get_control_history()
    time = drone.get_times()
    
    fig, axs = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
    fig.suptitle(title)
    # POSITIONS PLOTS
    axs[0, 0].plot(time, states[:, 0], label='Leader')
    axs[0, 0].set_ylabel('X (m)')
    axs[0, 0].grid()

    axs[1, 0].plot(time, states[:, 1], label='Leader')
    axs[1, 0].set_ylabel('Y (m)')
    axs[1, 0].grid()

    axs[2, 0].plot(time, states[:, 2], label='Leader')
    axs[2, 0].set_ylabel('Z (m)')
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 0].grid()

    # ATTITUDE PLOTS
    axs[0, 1].plot(time, np.rad2deg(states[:, 6]))
    axs[0, 1].set_ylabel('Roll (deg)')
    axs[0, 1].grid()

    axs[1, 1].plot(time, np.rad2deg(states[:, 7]))
    axs[1, 1].set_ylabel('Pitch (deg)')
    axs[1, 1].grid()

    axs[2, 1].plot(time, np.rad2deg(states[:, 8]))
    axs[2, 1].set_ylabel('Yaw (Deg)')
    axs[2, 1].set_xlabel('Time (s)')
    axs[2, 1].grid()

    # CONTROLS PLOTS
    axs[0, 2].plot(time[1:], controls[:, 0])
    axs[0, 2].set_ylabel('F (N)')
    axs[0, 2].grid()

    axs[1, 2].plot(time[1:], controls[:, 1])
    axs[1, 2].set_ylabel('Tx (Nm)')
    axs[1, 2].grid()

    axs[2, 2].plot(time[1:], controls[:, 2])
    axs[2, 2].set_ylabel('Ty (Nm)')
    axs[2, 2].set_xlabel('Time (s)')
    axs[2, 2].grid()
   
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # this is so the title doesn't overlap stuff
    return fig

def plot_3d_trajectory_all_drones(drones, labels):
    """Plots 3D trajectory for all drones provided in a list and their lables"""
    size = len(drones)
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(size):
        # 3D Plot
        states = drones[i].get_state_time_history()
        ax.plot(states[:,0], states[:,1], states[:, 2], label=labels[i])

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Drone Trajectories')
    plt.tight_layout()
    plt.legend()
    
    return fig

def plot_aerial_view(drones, labels):
    size = len(drones)
    
    fig = plt.figure()
    for i in range(size):
        states = drones[i].get_state_time_history()
        plt.plot(states[:,0], states[:,1], label=labels[i])
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Aerial Trajectories')
    plt.tight_layout()
    plt.legend()
    plt.grid()