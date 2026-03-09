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
    """Plots Aerial view of all drones provided."""
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

def plot_measurement_links(ax, observer_pos, target_pos, label=None, color='gray'):
    """
    Draws a dashed line representing the range/bearing measurement link.
    """
    ax.plot([observer_pos[0], target_pos[0]], [observer_pos[1], target_pos[1]], [observer_pos[2], target_pos[2]], color=color, linestyle='--', alpha=0.6, label=label)


def plot_drone_ekf_diagnostics(times, mu_hist_drone, P_hist_drone, truth_hist_drone, drone_name="Drone"):
    """
    Plot EKF estimate vs truth with ±3σ bounds, and position error with ±3σ, for a single drone.

    Parameters
    ----------
    times : array-like, shape (T,)
        Time vector.
    mu_hist_drone : array-like, shape (T, 6)
        Estimated state history for one drone: [x, y, z, vx, vy, vz].
    P_hist_drone : array-like, shape (T, 6, 6)
        Covariance history for that same drone state block.
    truth_hist_drone : array-like, shape (T, >=6)
        True state history for one drone. First 6 entries must be [x, y, z, vx, vy, vz].
    drone_name : str
        Label for plot titles.
    """
    mu_hist_drone = np.asarray(mu_hist_drone)
    P_hist_drone = np.asarray(P_hist_drone)
    truth_hist_drone = np.asarray(truth_hist_drone)

    N = min(len(times), len(mu_hist_drone), len(P_hist_drone), len(truth_hist_drone))
    t = np.asarray(times[:N])

    mu = mu_hist_drone[:N]
    P = P_hist_drone[:N]
    truth = truth_hist_drone[:N, :6]

    labels_pos = ['x', 'y', 'z']
    labels_vel = ['vx', 'vy', 'vz']

    # -------------------------
    # Estimate vs truth + covariance
    # -------------------------
    fig1, axs1 = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    fig1.suptitle(f"{drone_name} Estimate vs Truth with Covariance Bounds", fontsize=16)

    for i in range(3):
        sigma = np.sqrt(P[:, i, i])
        est = mu[:, i]
        true = truth[:, i]

        ax = axs1[i, 0]
        ax.plot(t, est, label='Estimate')
        ax.plot(t, true, label='True')
        ax.fill_between(t, est - 3*sigma, est + 3*sigma, alpha=0.25)
        ax.set_ylabel(labels_pos[i])
        ax.grid(True)

    for i in range(3):
        sigma = np.sqrt(P[:, i+3, i+3])
        est = mu[:, i+3]
        true = truth[:, i+3]

        ax = axs1[i, 1]
        ax.plot(t, est, label='Estimate')
        ax.plot(t, true, label='True')
        ax.fill_between(t, est - 3*sigma, est + 3*sigma, alpha=0.15)
        ax.set_ylabel(labels_vel[i])
        ax.grid(True)

    axs1[0, 0].set_title("Position")
    axs1[0, 1].set_title("Velocity")
    axs1[2, 0].set_xlabel("Time (s)")
    axs1[2, 1].set_xlabel("Time (s)")
    axs1[0, 0].legend()

    fig1.tight_layout(rect=[0, 0, 1, 0.96])

    # -------------------------
    # Position error + covariance
    # -------------------------
    fig2, axs2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig2.suptitle(f"{drone_name} Position Error with 3 Sigma Bounds", fontsize=16)

    for i, label in enumerate(labels_pos):
        err = mu[:, i] - truth[:, i]
        sigma = np.sqrt(P[:, i, i])

        axs2[i].plot(t, err, label='Error')
        axs2[i].fill_between(t, -3*sigma, 3*sigma, alpha=0.2)
        axs2[i].axhline(0, color='k', linewidth=1)
        axs2[i].set_ylabel(label)
        axs2[i].grid(True)

    axs2[0].legend()
    axs2[-1].set_xlabel("Time (s)")
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    

def plot_drone_trajectory_3d(times, mu_hist_drone, truth_hist_drone, drone_name="Drone"):
    """
    Plot 3D trajectory of estimated vs true position for a single drone.

    Parameters
    ----------
    times : array-like, shape (T,)
        Time vector. Only used for length matching.
    mu_hist_drone : array-like, shape (T, 6) or (T, >=3)
        Estimated state history for one drone. First 3 entries must be [x, y, z].
    truth_hist_drone : array-like, shape (T, >=3)
        True state history for one drone. First 3 entries must be [x, y, z].
    drone_name : str
        Label for title/legend.
    """
    mu_hist_drone = np.asarray(mu_hist_drone)
    truth_hist_drone = np.asarray(truth_hist_drone)

    N = min(len(times), len(mu_hist_drone), len(truth_hist_drone))
    est = mu_hist_drone[:N, :3]
    truth = truth_hist_drone[:N, :3]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(truth[:, 0], truth[:, 1], truth[:, 2], label='True', linewidth=2)
    ax.plot(est[:, 0], est[:, 1], est[:, 2], '--', label='Estimate', linewidth=2)

    ax.scatter(truth[0, 0], truth[0, 1], truth[0, 2], marker='o', s=50, label='True Start')

    ax.set_title(f"{drone_name} 3D Trajectory: Truth vs Estimate")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()