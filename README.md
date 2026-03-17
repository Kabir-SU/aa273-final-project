# AA273 Final Project: Distributed UAV Swarm Localization

Multi-agent drone localization in a GNSS-denied environment using relative range/bearing measurements, altimeter data, and dead-reckoned attitude estimates. This project compares a **Centralized Extended Kalman Filter (C-EKF)** against a **Consensus Extended Kalman Filter (Consensus EKF)** for a three-drone leader–follower swarm.

## Overview

This repository contains the simulation and estimation code for the AA273 final project on distributed swarm localization. The scenario consists of:

- **Three drones**:
  - 1 leader
  - 2 followers
- **GNSS-denied operation**
- **Relative sensing** between agents using:
  - range
  - azimuth
  - elevation
- **Altimeter measurements** for altitude
- **Gyroscope measurements** used for attitude dead reckoning
- **A stationary landmark** observed by the leader to provide a global reference

The main goal of the project is to study the trade-off between:

- **Centralized estimation**, which fuses all swarm measurements jointly
- **Decentralized consensus estimation**, which distributes computation across agents but sacrifices some global consistency

## Repository Structure

```text
aa273-final-project/
├── centralized_ekf_final.py        # Centralized EKF implementation
├── centralized_ekf_sim.py          # Centralized EKF simulation script / helper
├── consensus_ekf_final.py          # Consensus EKF implementation
├── consensus_ekf_sim_final.py      # Consensus EKF simulation script / helper
├── dead_reckoning.py               # Gyro-based attitude dead reckoning
├── drone.py                        # Drone dynamics and control logic for trajectory generation
├── measurementmodel.py             # Relative measurement and sensor models
├── sim.py                          # General simulation / visualization script to get accustomed to usage
├── utils.py                        # Plotting and helper utilities
└── tech_debt/                      # Unused files
