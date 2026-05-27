# Simulation & Testing Pipeline

Because underwater hardware deployments are expensive and carry a high risk of vehicle loss, the LEGION swarm must be validated through rigorous software and hardware in-the-loop testing before touching the water.

See [README.md](file:///c:/Code/AquaCLR/docs/legion/README.md) for the documentation index.

## 1. Simulators (The "Loop" Environments)

As of 2024/2025, the community is transitioning away from Gazebo Classic. The recommended simulators for LEGION are:

### A. DAVE (Aquatic Virtual Environment) / Gazebo Harmonic
- **Best for:** Complex manipulation, high-fidelity hydrodynamics, and ROS 2 Jazzy integration.
- **Why:** DAVE builds on the `uuv_simulator` legacy. It provides physical models for DVLs, Multibeam Sonars, and USBLs out-of-the-box.
- **Role:** Use DAVE to test the physics of the ROV arms and accurate thruster allocation matrices.

### B. Stonefish
- **Best for:** Perception testing and multi-vehicle (swarm) simulations.
- **Why:** Written in C++ with direct OpenGL/GPU acceleration, it simulates optical distortion, turbidity, and sonar acoustics much faster than Gazebo. 
- **Role:** Use Stonefish to generate synthetic data for the AI pipeline and to test swarm collision avoidance.

---

## 2. The Validation Hierarchy

### Step 1: SIL (Software-in-the-Loop)
- **Concept:** Run the exact ROS 2 nodes (Perception, Control, Planning) on your desktop PC, connected to Stonefish/DAVE.
- **Goal:** Verify logic. Does the swarm aggregate? Do the ROS 2 topics route correctly?
- **Setup:** A Docker container running ROS 2 + Simulator + LEGION Codebase.

### Step 2: PIL (Processor-in-the-Loop)
- **Concept:** Compile the ROS 2 control nodes for the target ARM architecture (e.g., Jetson Orin) and run them on the actual physical Jetson board on your desk. The Jetson receives fake sensor data from the Simulator via ethernet.
- **Goal:** Verify compute bounds. Can the Jetson run AquaCLR + SLAM at 15 FPS without thermal throttling?

### Step 3: HIL (Hardware-in-the-Loop)
- **Concept:** Connect the actual flight controller (e.g., Pixhawk / BlueROV hardware) to the simulation. The simulator outputs fake IMU/DVL data as raw electrical signals (or MAVLink packets). The Pixhawk computes PWM outputs, which the simulator reads to spin virtual thrusters.
- **Goal:** Verify electrical integration and real-time RTOS scheduling.

---

## 3. MATLAB / Simulink Modeling

Before writing a single line of C++ control code, the ROV dynamics must be modeled. 
- **Toolbox:** Use the **Marine Systems Simulator (MSS)** toolbox for MATLAB by Thor Fossen.
- **Hydrodynamics:** Model the 6-DOF (Degrees of Freedom) equations of motion. You must calculate the Added Mass matrix ($M_A$) and the Damping matrix ($D$).
- **Thruster Allocation:** Use Simulink to design the PID or MPC (Model Predictive Control) loops that convert a desired XYZ thrust vector into 8 specific PWM signals for the BlueROV thrusters.
- **Export:** Use Simulink Coder to export the verified MPC controller directly into a C++ ROS 2 Node.
