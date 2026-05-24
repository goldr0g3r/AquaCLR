# Part 1: Introduction and Model Comparison

This multi-part guide details the transition of the AquaCLR physics-informed network from the simplified **Jaffe-McGlamery** underwater image formation model to the **Akkaynak-Treibitz Revised Model (Sea-Thru)**. 

Because AquaCLR intrinsically relies on physical constraints for both inversion and loss calculations, a change in the physical model cascades across the architecture.

## Theoretical Comparison

### 1. The Jaffe-McGlamery Model (Current)
The Jaffe-McGlamery model is a foundational model for underwater and atmospheric scattering. It describes the observed image $I$ as a combination of direct transmission and backscatter:

$$ I(x) = J(x) \cdot t(x) + B \cdot (1 - t(x)) $$

- **$I(x)$**: Observed hazy/snowy image.
- **$J(x)$**: Radiance of the clear scene.
- **$t(x)$**: Transmission map, $t(x) = e^{-c z(x)}$, where $c$ is a single combined attenuation coefficient and $z(x)$ is the depth.
- **$B$**: Backscatter (ambient veiling light).

**Limitations**: It assumes that light attenuates identically whether it travels directly from an object to the sensor (direct signal) or scatters from the water body itself (backscatter).

### 2. The Akkaynak-Treibitz Revised Model (Sea-Thru)
The Sea-Thru model corrects the limitations of Jaffe-McGlamery by distinguishing between the attenuation of the direct signal ($\beta_D$) and the attenuation of the backscatter ($\beta_B$). Furthermore, it explicitly acknowledges that these coefficients are wavelength-dependent (i.e., varying across RGB channels).

$$ I_c(x) = J_c(x) \cdot e^{-\beta_{D, c} z(x)} + B_{inf, c} \cdot (1 - e^{-\beta_{B, c} z(x)}) $$

- **$I_c(x), J_c(x)$**: Observed and true radiance for color channel $c$.
- **$z(x)$**: True physical depth map (range) instead of a generic transmission proxy.
- **$\beta_{D, c}$**: Attenuation coefficient for the direct signal in channel $c$.
- **$\beta_{B, c}$**: Attenuation coefficient for the backscatter in channel $c$.
- **$B_{inf, c}$**: Ambient illumination at infinity for channel $c$.

## How It Affects the Neural Network

Transitioning to Sea-Thru fundamentally alters what the network must learn and predict.

### 1. Network Outputs
- **Current**: Predicts a 1-channel spatial map $t(x)$ and a 3-channel global vector $B$.
- **New**: Must predict a 1-channel spatial depth map $z(x)$, and three 3-channel global (or local, depending on implementation) parameters: $\beta_D$, $\beta_B$, and $B_{inf}$.

### 2. Physical Constraints
- The network will enforce stricter color constancy since $\beta_D$ and $\beta_B$ act differentially on RGB channels.
- The concept of "transmission" $t(x)$ splits into two distinct transmission terms: $t_D(x) = e^{-\beta_D z(x)}$ and $t_B(x) = e^{-\beta_B z(x)}$.

### 3. Out-of-Distribution Robustness
- Sea-Thru offers significantly higher accuracy in restoring true colors at various depths, reducing the "red-depletion" effect seen in standard inversions. The model will become more robust across different water bodies (e.g., coastal green water vs. oceanic blue water).

### 4. Loss Function Geometry
- The total variation (TV) loss, currently applied to $t(x)$, will now apply directly to $z(x)$ to enforce piecewise-smooth geometries, mapping more closely to true 3D scene structures.

In the next sections, we will walk through the exact code modifications required to enact this shift.
