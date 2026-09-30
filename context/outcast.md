paper link: https://arxiv.org/pdf/2204.09341

The authors of *OutCast* have not publicly released their source code or pre-trained weights. However, because the paper's architecture is highly modular, you can reconstruct the inference pipeline using modern PyTorch components.

To recreate this method on your own images, you need to build a pipeline that computes the deterministic screen-space shadows mathematically, and then feeds those calculated masks into a neural network for synthesis.

Here is the technical blueprint to build this pipeline in PyTorch.

1. **Extract Monocular Geometry:** Replaces the need for 3D meshes.
Pass your image ($C_o$) through a state-of-the-art monocular depth estimator like **Depth Anything V2** or **MiDaS** to get a 2.5D depth map ($D$).

Next, compute the surface normals ($N$). You do not need a neural network for this; you can calculate it deterministically in PyTorch. Unproject the depth map into camera-space 3D coordinates, take the spatial gradients (`torch.gradient`) in the X and Y directions, and calculate their cross product to generate the normal map.


2. **Estimate the Original Light Vector:** Finding \omega_o.
To remove existing shadows, you need to know where the sun currently is. You can use an off-the-shelf outdoor illumination estimator (like the one from *Hold-Geoffroy et al.*) to extract the original 3D light vector $\omega_o$.


3. **Compute Screen-Space Shadows:** The core deterministic step.
Write a custom ray-marching function in PyTorch (or write a custom CUDA kernel if you want it to run fast).

For every pixel in your depth map, step along the 2D projection of the target light vector ($\omega_n$). At each step, check if the depth of the current pixel is physically "below" the depth of the pixel you are marching across. If it is, that pixel is occluded.

* Run this for the new light vector to generate the target shadow mask ($S_n$).
* Run this for the original light vector to generate the estimated original shadow mask ($S_o$).


4. **Neural Synthesis:** Bridging graphics and deep learning.
The original paper concatenates all these tensors — $C_o, S_o, S_n, N$, and the light vectors — and feeds them into a U-Net.

**The bottleneck:** Training this U-Net requires a massive paired dataset of outdoor scenes with perfectly aligned lighting variations (which the authors generated synthetically).

**The modern workaround:** Instead of training a U-Net from scratch, use an Image-to-Image Diffusion model (like **Stable Diffusion XL**) integrated with **ControlNet**. You can use your calculated original shadow ($S_o$) as a negative prompt/mask to inpaint and remove the old shadow, and use the new shadow mask ($S_n$) combined with the normal map ($N$) as spatial conditioning for ControlNet to explicitly guide where the new shadows and highlights must be drawn.


By replacing the custom U-Net with a conditioned Diffusion model, you bypass the need to train a network from scratch while still utilizing the exact screen-space logic that makes the *OutCast* approach work.