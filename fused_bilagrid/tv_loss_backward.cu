#include <cuda_runtime.h>

// Constants for optimal thread configuration
const int TV_CUDA_THREADS = 256;
const int TV_MIN_BLOCKS_PER_SM = 4;

__launch_bounds__(TV_CUDA_THREADS, TV_MIN_BLOCKS_PER_SM)
__global__ void tv_loss_backward_kernel(
    const float* __restrict__ bilagrid,   // [N,12,L,H,W]
    const float v_tv_loss,                 // scalar gradient dL/d(tv_loss)
    float* __restrict__ v_bilagrid,        // [N,12,L,H,W]
    int N, int L, int H, int W
) {
    // Process multiple cells per thread for better efficiency
    const size_t total = (size_t)N * 12 * L * H * W;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    
    const float s = v_tv_loss / (6*N);
    const float sx = s / (float)(L * H * (W - 1));
    const float sy = s / (float)(L * (H - 1) * W);
    const float sz = s / (float)((L - 1) * H * W);
    
    // Grid-stride loop
    for (size_t cell_idx = tid; cell_idx < total; cell_idx += stride) {
        // Decode position
        size_t idx = cell_idx;
        const int wi = idx % W; idx /= W;
        const int hi = idx % H; idx /= H;
        const int li = idx % L; idx /= L;
        const int ci = idx % 12; idx /= 12;
        const int ni = idx;
        
        float half_grad = 0.0f;
        const float val = bilagrid[cell_idx];
        
        // X-direction gradients (matching original logic)
        if (wi > 0) {
            float val0 = bilagrid[cell_idx - 1];
            half_grad += (val - val0) * sx;
        }
        if (wi < W - 1) {
            float val0 = bilagrid[cell_idx + 1];
            half_grad += (val - val0) * sx;
        }
        
        // Y-direction gradients
        if (hi > 0) {
            float val0 = bilagrid[cell_idx - W];
            half_grad += (val - val0) * sy;
        }
        if (hi < H - 1) {
            float val0 = bilagrid[cell_idx + W];
            half_grad += (val - val0) * sy;
        }
        
        // Z-direction gradients
        if (li > 0) {
            float val0 = bilagrid[cell_idx - W*H];
            half_grad += (val - val0) * sz;
        }
        if (li < L - 1) {
            float val0 = bilagrid[cell_idx + W*H];
            half_grad += (val - val0) * sz;
        }

        v_bilagrid[cell_idx] = half_grad;
    }
}

void tv_loss_backward(
    const float* bilagrid,
    const float v_tv_loss,
    float* v_bilagrid,
    int N, int L, int H, int W
) {
    const size_t total = (size_t)N * 12 * L * H * W;
    const int blocks = min((int)((total + TV_CUDA_THREADS - 1) / TV_CUDA_THREADS), 2048);
    
    tv_loss_backward_kernel<<<blocks, TV_CUDA_THREADS>>>(
        bilagrid, v_tv_loss, v_bilagrid, N, L, H, W
    );
}