# Formation Control Mathematics

Detailed mathematical treatment of swarm formation control for the LEGION Mother-Daughter topology. Covers virtual structure, consensus, artificial potential fields, and collision avoidance algorithms.

See [README.md](README.md) for the documentation index.

---

## 1. Problem Formulation

Let $N$ Daughter AUVs form a swarm. Each Daughter $i$ has a 3D position $\mathbf{p}_i \in \mathbb{R}^3$ and velocity $\dot{\mathbf{p}}_i \in \mathbb{R}^3$. The goal of formation control is to manoeuvre the swarm so that:

$$\mathbf{p}_i(t) \to \mathbf{p}_i^* \quad \text{as} \quad t \to \infty$$

where $\mathbf{p}_i^*$ is the **desired position** of Daughter $i$ in the formation. In the LEGION system, $\mathbf{p}_i^*$ is dynamically updated by the Mother based on the current mission objective.

---

## 2. Virtual Structure Approach

The virtual structure method treats the entire formation as a single rigid body moving through space. This is the **primary method** used by LEGION because it gives the operator direct, intuitive control over the formation shape and heading.

### 2.1 Formation Definition

The formation is defined by a set of **offset vectors** $\boldsymbol{\delta}_i$ in the body frame of the virtual structure:

$$\mathbf{p}_i^* = \mathbf{p}_c + \mathbf{R}_c \, \boldsymbol{\delta}_i$$

- $\mathbf{p}_c \in \mathbb{R}^3$: position of the virtual structure's **centroid** (commanded by the Mother).
- $\mathbf{R}_c \in SO(3)$: rotation matrix of the virtual structure (yaw-aligned with the mission heading).
- $\boldsymbol{\delta}_i \in \mathbb{R}^3$: fixed offset of Daughter $i$ relative to the centroid.

**Example: Triangular survey formation (3 Daughters)**

$$\boldsymbol{\delta}_1 = \begin{bmatrix} 3 \\ 0 \\ 0 \end{bmatrix}, \quad \boldsymbol{\delta}_2 = \begin{bmatrix} -1.5 \\ 2.6 \\ 0 \end{bmatrix}, \quad \boldsymbol{\delta}_3 = \begin{bmatrix} -1.5 \\ -2.6 \\ 0 \end{bmatrix}$$

This places Daughters in an equilateral triangle with 3 m side length at the same depth.

### 2.2 Formation Controller

Each Daughter runs an independent proportional-derivative (PD) controller to track its desired position:

$$\mathbf{u}_i = K_p \, (\mathbf{p}_i^* - \mathbf{p}_i) + K_d \, (\dot{\mathbf{p}}_i^* - \dot{\mathbf{p}}_i)$$

where $\mathbf{u}_i$ is the velocity command sent to `control_node`, and:

- $K_p$: Proportional gain (position error correction), typically $K_p = 0.8$ for BlueROV2.
- $K_d$: Derivative gain (damping), typically $K_d = 0.2$.
- $\dot{\mathbf{p}}_i^* = \dot{\mathbf{p}}_c + \dot{\mathbf{R}}_c \, \boldsymbol{\delta}_i$: desired velocity from virtual structure kinematics.

### 2.3 Formation Scaling and Rotation

The Mother can dynamically adjust $\boldsymbol{\delta}_i$ to:
- **Scale** the formation (e.g., compress when entering a narrow pipe).
- **Rotate** the formation (e.g., rotate to face an inspection target).

```python
import numpy as np

def compute_desired_positions(
    p_c: np.ndarray,           # (3,) centroid position
    yaw_c: float,              # formation heading in radians
    offsets: np.ndarray,       # (N, 3) offset vectors for each Daughter
    scale: float = 1.0,
) -> np.ndarray:               # (N, 3) desired positions
    c, s = np.cos(yaw_c), np.sin(yaw_c)
    R_c = np.array([[c, -s, 0],
                    [s,  c, 0],
                    [0,  0, 1]], dtype=np.float32)
    return p_c + (scale * offsets) @ R_c.T
```

---

## 3. Consensus-Based Approach

Consensus control is used as a **fallback** when the Mother's acoustic link degrades and Daughters can only communicate with each other via peer-to-peer acoustic or optical links.

### 3.1 Graph Theory Background

Define the communication topology as a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ where:
- $\mathcal{V} = \{1, \ldots, N\}$ is the set of Daughter nodes.
- $(i, j) \in \mathcal{E}$ if Daughter $i$ can currently communicate with Daughter $j$.

The **graph Laplacian** $\mathbf{L} \in \mathbb{R}^{N \times N}$:

$$L_{ij} = \begin{cases} \deg(i) & \text{if } i = j \\ -1 & \text{if } (i,j) \in \mathcal{E} \\ 0 & \text{otherwise} \end{cases}$$

A connected graph has $\text{rank}(\mathbf{L}) = N-1$, guaranteeing consensus convergence.

### 3.2 Consensus Protocol

The consensus protocol drives each Daughter to converge toward the **average** of all Daughters' positions, plus an individual offset:

$$\dot{\mathbf{p}}_i = -\alpha \sum_{j \in \mathcal{N}_i} (\mathbf{p}_i - \boldsymbol{\delta}_i - (\mathbf{p}_j - \boldsymbol{\delta}_j))$$

where $\mathcal{N}_i$ is the set of neighbors of Daughter $i$, and $\alpha > 0$ is the consensus gain. This drives the **relative positions** $(\mathbf{p}_i - \boldsymbol{\delta}_i)$ to converge to a common value, preserving the formation offsets.

**Convergence condition:** The gain $\alpha$ must satisfy:

$$\alpha < \frac{1}{\lambda_{\max}(\mathbf{L})}$$

where $\lambda_{\max}$ is the largest eigenvalue of the Laplacian. For a fully connected 3-node graph, $\lambda_{\max} = 3$, so $\alpha < 0.33$.

```python
def consensus_velocity_command(
    i: int,
    positions: dict[int, np.ndarray],   # {daughter_id: position}
    offsets: dict[int, np.ndarray],      # {daughter_id: formation offset}
    neighbors: list[int],                # IDs of reachable neighbors
    alpha: float = 0.25,
) -> np.ndarray:
    """Compute the consensus velocity command for Daughter i."""
    p_i = positions[i]
    delta_i = offsets[i]
    cmd = np.zeros(3)
    for j in neighbors:
        p_j = positions[j]
        delta_j = offsets[j]
        cmd -= alpha * ((p_i - delta_i) - (p_j - delta_j))
    return cmd
```

---

## 4. Artificial Potential Fields for Collision Avoidance

Formation controllers can generate collision paths in cluttered environments. APF (Artificial Potential Fields) adds a repulsive term that is always active, regardless of the formation mode.

### 4.1 Repulsive Potential

For Daughter $i$ and obstacle $k$ (which may be another Daughter, a wall, or the sea floor):

$$U_{rep,k}(\mathbf{p}_i) = \begin{cases} \frac{1}{2} \eta \left(\frac{1}{d_{ik}} - \frac{1}{d_0}\right)^2 & \text{if } d_{ik} \leq d_0 \\ 0 & \text{if } d_{ik} > d_0 \end{cases}$$

- $d_{ik} = \|\mathbf{p}_i - \mathbf{p}_k\|$: Euclidean distance to the obstacle.
- $d_0$: Safety influence radius (e.g., 2.0 m for inter-Daughter avoidance, 0.5 m for walls).
- $\eta$: Repulsion gain (e.g., $\eta = 5.0$).

The repulsive force on Daughter $i$ is:

$$\mathbf{F}_{rep,k} = -\nabla_{\mathbf{p}_i} U_{rep,k} = \eta \left(\frac{1}{d_{ik}} - \frac{1}{d_0}\right) \frac{1}{d_{ik}^2} \cdot \frac{\mathbf{p}_i - \mathbf{p}_k}{d_{ik}}$$

### 4.2 Combined Control Law

The final velocity command for Daughter $i$ is the sum of the formation term and all repulsive terms:

$$\mathbf{u}_i^{total} = \underbrace{\mathbf{u}_i^{formation}}_{\text{PD or Consensus}} + \sum_{k \neq i} \mathbf{F}_{rep,k} + \mathbf{F}_{rep,walls}$$

```python
def repulsive_force(
    p_i: np.ndarray,
    obstacles: list[np.ndarray],   # List of obstacle positions
    d0: float = 2.0,
    eta: float = 5.0,
) -> np.ndarray:
    """Compute the total APF repulsive velocity contribution for agent i."""
    force = np.zeros(3)
    for p_k in obstacles:
        diff = p_i - p_k
        d = np.linalg.norm(diff)
        if 0.01 < d < d0:  # 0.01 guard: never divide by near-zero
            mag = eta * (1.0 / d - 1.0 / d0) / (d ** 2)
            force += mag * diff / d
    return force
```

### 4.3 Known Limitation: Local Minima

APF can trap agents in local minima (where attractive and repulsive forces cancel). The LEGION mitigations are:

1. **Depth perturbation:** If velocity norm < 0.05 m/s for > 10 s and goal is not reached, apply a small random upward/downward depth impulse to break symmetry.
2. **Formation reconfiguration:** The Mother can command a different $\boldsymbol{\delta}$ set to move the virtual structure around the obstacle.

---

## 5. Formation Switching and Convergence Time Estimation

The Mother may command a **formation switch** mid-mission (e.g., triangle → line for pipe inspection). The transition time can be estimated:

$$T_{conv} \approx \frac{\|\boldsymbol{\delta}_i^{new} - \boldsymbol{\delta}_i^{old}\|_{max}}{v_{max}}$$

where $v_{max}$ is the Daughter's maximum sustainable velocity. For a BlueROV2 Heavy, $v_{max} \approx 1.0$ m/s. A 3 m re-arrangement therefore takes approximately 3 seconds.

The Mother waits $T_{conv} + 1.5 \text{ s}$ (safety margin) before considering the new formation reached. During transition, the Mother's `mission_planner_node` pauses forward centroid motion.

---

## 6. Underwater-Specific Challenges

| Challenge | Impact | Mitigation |
|---|---|---|
| Acoustic delay (0.5–5 s) | Daughters receive stale position data from peers | Use EKF prediction to extrapolate peer positions forward by the expected delay |
| Current drift | Daughters get pushed off formation by currents | DVL provides ground-referenced velocity; EKF corrects position; PD controller handles steady-state current disturbance |
| Packet loss (10–30%) | Consensus graph becomes disconnected | Apply higher $K_p$ in PD mode (virtual structure) as fallback; consensus only activates when Mother link is dead |
| Featureless environments | SLAM position drift corrupts formation | Fuse DVL-based dead reckoning as a backup position source with higher covariance |
| Thruster saturation | APF pushes thrusters beyond PWM limits | Clamp $\|\mathbf{u}_i^{total}\|$ to `MAX_VELOCITY` before sending to `control_node`; scale direction, not just magnitude |
