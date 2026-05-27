# LEGION Swarm Architecture Documentation

Welcome to the documentation for the **LEGION** (Large-scale Exploration via Grouped Intelligent Oceanic Nodes) project. This project aims to build a complete autonomous multi-swarm ROV/AUV system utilizing a Mother-Daughter hierarchical link.

## Directory Structure

This documentation suite is broken down into modular, cross-linked files to make studying and implementation easier.

- 📖 **[Architecture & Systems Design](architecture.md)**
  Details the ROS 2 integration, Mother-Daughter topology, acoustic middleware, and node graphs.
- 🔌 **[ROS 2 Node Implementation Reference](ros2_nodes_detailed.md)**
  Full topic schemas, QoS profiles, code skeletons, thruster allocation matrix, and GTSAM integration for every node.
- 📡 **[Communications Protocol Specification](comms_protocol.md)**
  Acoustic/optical link constraints, binary message byte layouts, DESERT integration, ARQ strategy, and clock sync.
- 🔺 **[Formation Control Mathematics](formation_control.md)**
  Virtual structure, consensus protocol, artificial potential fields, and underwater-specific challenges.
- 🧪 **[Simulation & Testing (SIL/HIL/PIL)](simulation_testing.md)**
  Covers the validation pipeline using Gazebo Harmonic, Stonefish, DAVE, and MATLAB/Simulink hydrodynamics.
- 🧠 **[AI, Perception & SLAM](ai_slam_pipeline.md)**
  Deep dive into the neural network pipelines, sonar/optical sensor fusion, and multi-agent SLAM.
- 📋 **[Research, TODOs & Learning Path](todo_research.md)**
  A curated guide of academic papers, frameworks to learn, and the immediate development backlog.

## High-Level Concept

The swarm operates on a **Centralized-Decentralized Hybrid** model:
1. **Mother Node (USV / Main Tethered ROV):** Acts as the primary gateway. It handles high-bandwidth satellite/RF communications to base, heavy computational tasks (like global map merging), and provides localized acoustic positioning (USBL).
2. **Daughter Nodes (Untethered AUVs / Micro-ROVs):** Highly agile, edge-compute equipped units (e.g., NVIDIA Jetson Orin Nano). They perform localized perception, obstacle avoidance, and task-specific AI execution. They communicate with the Mother via low-bandwidth acoustic modems.
