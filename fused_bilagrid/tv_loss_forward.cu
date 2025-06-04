#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

__global__ void tv_loss_forward_kernel(
    const float* __restrict__ bilagrid,  // [N,12,L,H,W]
    float* __restrict__ tv_loss,
    int N, int L, int H, int W
) {
    // Use 1D grid for better load balancing
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int total = N * L * H * W;
    
    float local_sum = 0.0f;
    
    // Grid-stride loop
    for (int idx = tid; idx < total; idx += stride) {
        // Decode position
        int tmp = idx;
        int wi = tmp % W; tmp /= W;
        int hi = tmp % H; tmp /= H;
        int li = tmp % L; tmp /= L;
        int ni = tmp;
        
        // Process all 12 channels
        #pragma unroll 12
        for (int ci = 0; ci < 12; ci++) {
            int base = (ni*12+ci)*L*H*W;
            int cell_idx = base + (li*H+hi)*W+wi;
            
            float val = bilagrid[cell_idx];
            
            // X-direction
            if (wi > 0) {
                float val0 = bilagrid[cell_idx - 1];
                float diff = val - val0;
                local_sum += diff * diff / (L*H*(W-1));
            }
            
            // Y-direction
            if (hi > 0) {
                float val0 = bilagrid[cell_idx - W];
                float diff = val - val0;
                local_sum += diff * diff / (L*(H-1)*W);
            }
            
            // Z-direction
            if (li > 0) {
                float val0 = bilagrid[cell_idx - W*H];
                float diff = val - val0;
                local_sum += diff * diff / ((L-1)*H*W);
            }
        }
    }
    
    local_sum /= (12*N);
    
    // Warp-level reduction
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    local_sum = cg::reduce(warp, local_sum, cg::plus<float>());
    
    // First thread in each warp adds to global result
    if (warp.thread_rank() == 0) {
        atomicAdd(tv_loss, local_sum);
    }
}

void tv_loss_forward(
    const float* bilagrid,
    float* tv_loss,
    int N, int L, int H, int W
) {
    int threads = 256;
    int total = N * L * H * W;
    int blocks = min((total + threads - 1) / threads, 2048);
    
    tv_loss_forward_kernel<<<blocks, threads>>>(
        bilagrid, tv_loss,
        N, L, H, W
    );
}