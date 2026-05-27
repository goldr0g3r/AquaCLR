# AI, Perception & SLAM Pipeline

The intelligence of the LEGION swarm relies on fusing multiple imperfect underwater sensors. Vision is short-range but high-resolution; sonar is long-range but low-resolution.

See [README.md](README.md) for the documentation index.

## 1. The Perception Pipeline

On every Daughter node, the raw sensors are passed through a deep learning preprocessing pipeline before they reach the SLAM system.

### Optical Pipeline (AquaCLR + YOLOv8)
1. **Raw Feed:** Optical camera captures raw RGB, heavily degraded by marine snow, backscatter, and color attenuation.
2. **Restoration (AquaCLR):** The physics-informed LEGION-DeSnow network (running on TensorRT) predicts the transmission map $t$ and backscatter $B$, and analytically inverts the image to remove snow and restore contrast.
3. **Detection (YOLOv8):** The cleaned image is passed to a lightweight YOLO model to detect pipelines, obstacles, or docking stations.
4. **Depth Prior:** The transmission map $t$ is used as a proxy for depth uncertainty (pixels with $t \to 0$ are marked as unreliable for SLAM).

### Sonar Pipeline (FLS - Forward Looking Sonar)
- **Raw Feed:** Acoustic imaging sonar (e.g., BlueView).
- **Processing:** Sonar images are noisy (speckle noise). They must be filtered (e.g., Frost filter, or a specialized UNet) before feature extraction.

---

## 2. Sensor Fusion & State Estimation

Because underwater GPS does not exist, the AUV must guess its location by fusing dead-reckoning sensors.

- **DVL (Doppler Velocity Log):** Shoots acoustic beams at the sea floor to measure the vehicle's X/Y/Z velocity over ground. (Crucial for drift-free translation).
- **IMU (Inertial Measurement Unit):** Measures angular velocity and linear acceleration. (Crucial for pitch/roll/yaw).
- **Depth Sensor:** Highly accurate pressure transducer for Z-axis position.
- **EKF Fusion:** The `robot_localization` ROS 2 package runs an Extended Kalman Filter to fuse DVL ($V_x, V_y$), IMU (Orientation), and Depth ($Z$) into a single, smooth odometry estimate (`/odom`).

---

## 3. Underwater SLAM (Simultaneous Localization and Mapping)

Even with DVL, the vehicle's position will slowly drift over hours of operation. SLAM fixes this by recognizing previously seen landmarks (Loop Closure).

### Visual SLAM (ORB-SLAM3)
- **Concept:** Extracts corner features from the AquaCLR-cleaned optical images. If it sees the same rock twice, it corrects the drift.
- **Challenge:** Underwater environments are often featureless (sand) or dynamic (kelp, fish). This is why AquaCLR is so important — marine snow causes false feature-matches that break SLAM.

### Swarm Cooperative SLAM
In the Mother-Daughter topology:
1. Daughters run lightweight V-SLAM locally to map their immediate surroundings.
2. They compress their map keyframes and send them to the Mother via optical/acoustic links.
3. The Mother runs a **Global Pose Graph Optimizer** (e.g., GTSAM). If Daughter 1 and Daughter 2 observe the same underwater structure, the Mother detects this loop closure and aligns both of their maps globally.
