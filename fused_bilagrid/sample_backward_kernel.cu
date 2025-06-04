#include "config.h"

#ifdef COMPUTE_COORDS_GRAD
__global__ void bilagrid_sample_backward_kernel_cg(
#else
__global__ void bilagrid_sample_backward_kernel(
#endif
    const float* __restrict__ bilagrid,  // [N,12,L,H,W]
    const float* __restrict__ coords,  // [N,m,h,w,2]
    const float* __restrict__ rgb,  // [N,m,h,w,3]
    const float* __restrict__ v_output,  // [N,m,h,w,3]
    float* __restrict__ v_bilagrid,  // [N,12,L,H,W]
    #ifdef COMPUTE_COORDS_GRAD
    float* __restrict__ v_coords,  // [N,m,h,w,2]
    #endif
    float* __restrict__ v_rgb,  // [N,m,h,w,3]
    int N, int L, int H, int W,
    int m, int h, int w
) {
    // Use 1D grid-stride loop
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int total = N * m * h * w;
    
    for (int g_off = idx; g_off < total; g_off += stride) {
        // Decode position
        int tmp = g_off;
        int wi = tmp % w; tmp /= w;
        int hi = tmp % h; tmp /= h;
        int mi = tmp % m; tmp /= m;
        int ni = tmp;

        // Load data
        float sr = rgb[3*g_off+0], sg = rgb[3*g_off+1], sb = rgb[3*g_off+2];
        float gx = coords[2*g_off+0];
        float gy = coords[2*g_off+1];
        float gz = kC2G_r * sr + kC2G_g * sg + kC2G_b * sb;
        
        float x = gx * (W - 1);
        float y = gy * (H - 1);
        float z = gz * (L - 1);

        // Compute bounds
        int x0 = floorf(x), y0 = floorf(y), z0 = floorf(z);
        int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
        x0 = min(max(x0,0), W-1); x1 = min(max(x1,0), W-1);
        y0 = min(max(y0,0), H-1); y1 = min(max(y1,0), H-1);
        z0 = min(max(z0,0), L-1); z1 = min(max(z1,0), L-1);

        // Fractional parts
        float fx = x - x0, fy = y - y0, fz = z - z0;
        
        // Precompute weights
        float w[8];
        w[0] = (1-fx)*(1-fy)*(1-fz);
        w[1] = fx*(1-fy)*(1-fz);
        w[2] = (1-fx)*fy*(1-fz);
        w[3] = fx*fy*(1-fz);
        w[4] = (1-fx)*(1-fy)*fz;
        w[5] = fx*(1-fy)*fz;
        w[6] = (1-fx)*fy*fz;
        w[7] = fx*fy*fz;

        // Corner positions
        int cx[8] = {x0,x1,x0,x1,x0,x1,x0,x1};
        int cy[8] = {y0,y0,y1,y1,y0,y0,y1,y1};
        int cz[8] = {z0,z0,z0,z0,z1,z1,z1,z1};

        // Read upstream gradient
        float dr = v_output[3*g_off+0];
        float dg = v_output[3*g_off+1];
        float db = v_output[3*g_off+2];
        float vr = 0.0f, vg = 0.0f, vb = 0.0f;

        // Process channels efficiently
        #pragma unroll 3
        for (int di = 0; di < 3; di++) {
            float gout = (di==0 ? dr : di==1 ? dg : db);
            
            #pragma unroll 4
            for (int si = 0; si < 4; si++) {
                int ci = di * 4 + si;
                float r_coeff = (si==0 ? sr : si==1 ? sg : si==2 ? sb : 1.f);
                float grad_weight = r_coeff * gout;
                
                #pragma unroll 8
                for (int corner = 0; corner < 8; corner++) {
                    int base = ((ni*12 + ci)*L*H*W);
                    int idx = base + (cz[corner]*H + cy[corner])*W + cx[corner];
                    atomicAdd(v_bilagrid + idx, w[corner] * grad_weight);
                    
                    if (si < 3) {
                        float val = bilagrid[idx];
                        (si == 0 ? vr : si == 1 ? vg : vb) += val * w[corner] * gout;
                    }
                }
            }
        }

        // Spatial derivatives
        #ifdef COMPUTE_COORDS_GRAD
        float dwdx[8] = {
            -(1-fy)*(1-fz),  (1-fy)*(1-fz),
            -fy*(1-fz),      fy*(1-fz),
            -(1-fy)*fz,      (1-fy)*fz,
            -fy*fz,          fy*fz
        };
        float dwdy[8] = {
            -(1-fx)*(1-fz), -fx*(1-fz),
             (1-fx)*(1-fz),  fx*(1-fz),
            -(1-fx)*fz,     -fx*fz,
             (1-fx)*fz,      fx*fz
        };
        #endif
        float dwdz[8] = {
            -(1-fx)*(1-fy), -fx*(1-fy),
            -(1-fx)*fy,     -fx*fy,
             (1-fx)*(1-fy),  fx*(1-fy),
             (1-fx)*fy,      fx*fy
        };

        // Compute gradients
        #ifdef COMPUTE_COORDS_GRAD
        float gx_grad = 0.f, gy_grad = 0.f;
        #endif
        float gz_grad = 0.f;
        
        #pragma unroll 8
        for (int corner = 0; corner < 8; corner++) {
            float trilerp = 0.f;
            #pragma unroll 12
            for (int ci = 0; ci < 12; ci++) {
                const float* vol = bilagrid + ((ni*12 + ci)*L*H*W);
                float v = vol[(cz[corner]*H + cy[corner])*W + cx[corner]];
                int si = ci % 4, di = ci / 4;
                float r_coeff = (si==0 ? sr : si==1 ? sg : si==2 ? sb : 1.f);
                float gout = (di==0 ? dr : di==1 ? dg : db);
                trilerp += v * r_coeff * gout;
            }
            #ifdef COMPUTE_COORDS_GRAD
            gx_grad += dwdx[corner] * (W-1) * trilerp;
            gy_grad += dwdy[corner] * (H-1) * trilerp;
            #endif
            gz_grad += dwdz[corner] * (L-1) * trilerp;
        }
        
        // Save gradients
        #ifdef COMPUTE_COORDS_GRAD
        v_coords[2*g_off+0] = gx_grad * (float)(x0 != x && x1 != x);
        v_coords[2*g_off+1] = gy_grad * (float)(y0 != y && y1 != y);
        #endif
        gz_grad *= (float)(z0 != z && z1 != z);
        v_rgb[3*g_off+0] = vr + kC2G_r * gz_grad;
        v_rgb[3*g_off+1] = vg + kC2G_g * gz_grad;
        v_rgb[3*g_off+2] = vb + kC2G_b * gz_grad;
    }
}
