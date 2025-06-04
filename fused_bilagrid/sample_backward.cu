#include "sample_backward_kernel.cu"

#define COMPUTE_COORDS_GRAD
#include "sample_backward_kernel.cu"

void bilagrid_sample_backward(
    const float* bilagrid,
    const float* coords,
    const float* rgb,
    const float* v_output,
    float* v_bilagrid,
    float* v_coords,
    float* v_rgb,
    int N, int L, int H, int W,
    int m, int h, int w
) {
    // Use 1D grid for better load balancing
    int threads = 256;
    int total = N * m * h * w;
    int blocks = min((total + threads - 1) / threads, 65535);
    
    if (v_coords == nullptr) {
        bilagrid_sample_backward_kernel<<<blocks, threads>>>(
            bilagrid, coords, rgb, v_output,
            v_bilagrid, v_rgb,
            N, L, H, W, m, h, w
        );
    }
    else {
        bilagrid_sample_backward_kernel_cg<<<blocks, threads>>>(
            bilagrid, coords, rgb, v_output,
            v_bilagrid, v_coords, v_rgb,
            N, L, H, W, m, h, w
        );
    }
}
