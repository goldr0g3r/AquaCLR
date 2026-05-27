# Research, TODOs & Learning Path

This document outlines the required research, immediate development backlog, and key frameworks to study to bring the LEGION swarm to reality.

See [README.md](file:///c:/Code/AquaCLR/docs/legion/README.md) for the documentation index.

## 1. Things to Learn (Frameworks & Tools)

To build this system, you must become proficient in the following specific technologies:

1. **ROS 2 (Jazzy/Humble):** Specifically, you need to understand DDS QoS (Quality of Service) profiles. You must configure DDS for high latency and packet loss so that the nodes don't crash when the acoustic link drops.
2. **DESERT Underwater:** Read the documentation for this acoustic network simulation framework. It is the bridge between ROS 2 and physical acoustic modems.
3. **Stonefish Simulator:** Learn how to write XML scenario files for Stonefish to spawn multiple ROVs in a single physics environment.
4. **robot_localization:** Understand how to tune the covariance matrices of an EKF (Extended Kalman Filter) to fuse DVL, IMU, and Depth sensors.
5. **GTSAM:** For the Mother node's global map merging, learn GTSAM (Georgia Tech Smoothing and Mapping) for factor-graph optimization.

## 2. Research Details & Papers to Read

Familiarize yourself with the current academic state-of-the-art (2024/2025):

- **Swarm Control:** Read up on *Virtual Structure* and *Consensus-based* formation control for AUVs. This dictates how the Daughters maintain formation without bumping into each other.
- **USV-AUV Cooperative SLAM:** Search for recent papers on heterogeneous loop closure between surface vessels and underwater vehicles.
- **Physics-Informed Deep Learning:** Keep following literature that integrates optical models (like Jaffe-McGlamery and Sea-Thru) directly into U-Net architectures for real-time restoration (like your AquaCLR implementation).

## 3. Immediate TODOs (M2 & M3 Milestones)

### Hardware & Simulation (M2)
- [ ] Install Stonefish and configure a multi-agent simulation with 1 Mother (USV) and 2 Daughters (BlueROVs).
- [ ] Set up the ROS 2 node graph using namespaces (e.g., `/daughter_1/cmd_vel`, `/mother/usbl`).
- [ ] Extract the AquaCLR TensorRT inference engine into a standalone ROS 2 Jazzy node that publishes to `/camera/image_desnowed`.
- [ ] Configure `robot_localization` EKF for the Daughters using simulated Stonefish DVL/IMU data.

### AI & SLAM (M3)
- [ ] Implement ORB-SLAM3 on the Daughter nodes, subscribing to the AquaCLR cleaned image feed.
- [ ] Design the acoustic compression node: compress Daughter pose estimates into minimal byte arrays to simulate 100 bps acoustic modem constraints.
- [ ] Build the Mother node's global map merger using GTSAM to align the pose graphs of the Daughters.
- [ ] Implement the `Safing Mode` fallback behavior for when Daughters lose acoustic lock with the Mother for > 30 seconds.
