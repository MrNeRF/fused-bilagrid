#include "config.h"


// template<int compute_coords_grad>
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * m * h * w;
    if (idx >= total) return;

    // decode indices
    int tmp = idx;
    int wi = tmp % w; tmp /= w;
    int hi = tmp % h; tmp /= h;
    int mi = tmp % m; tmp /= m;
    int ni = tmp;

    // grid coords
    int g_off = (((ni*m + mi)*h + hi)*w + wi);
    float sr = rgb[3*g_off+0], sg = rgb[3*g_off+1], sb = rgb[3*g_off+2];
    float gx = coords[2*g_off+0];
    float gy = coords[2*g_off+1];
    float gz = kC2G_r * sr + kC2G_g * sg + kC2G_b * sb;
    float x = gx * (W - 1);
    float y = gy * (H - 1);
    float z = gz * (L - 1);

    // floor + ceil, clamped
    int x0 = floorf(x), y0 = floorf(y), z0 = floorf(z);
    int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
    x0 = min(max(x0,0), W-1); x1 = min(max(x1,0), W-1);
    y0 = min(max(y0,0), H-1); y1 = min(max(y1,0), H-1);
    z0 = min(max(z0,0), L-1); z1 = min(max(z1,0), L-1);

    // fractional parts
    float fx = x - x0, fy = y - y0, fz = z - z0;
    float f000 = (1-fx)*(1-fy)*(1-fz);
    float f001 = fx*(1-fy)*(1-fz);
    float f010 = (1-fx)*fy*(1-fz);
    float f011 = fx*fy*(1-fz);
    float f100 = (1-fx)*(1-fy)*fz;
    float f101 = fx*(1-fy)*fz;
    float f110 = (1-fx)*fy*fz;
    float f111 = fx*fy*fz;

    // read rgb coeffs and upstream gradient
    float dr = v_output[3*g_off+0];
    float dg = v_output[3*g_off+1];
    float db = v_output[3*g_off+2];
    float vr = 0.0, vg = 0.0, vb = 0.0;

    // accumulate bilagrid gradient over 12 channels
    #pragma unroll
    for (int ci = 0; ci < 12; ++ci) {
        // weight from rgb channel
        int si = ci % 4, di = ci / 4;
        float r_coeff = (si==0 ? sr : si==1 ? sg : si==2 ? sb : 1.f);
        float gout = (di==0 ? dr : di==1 ? dg : db);
        float grad_weight = r_coeff * gout;

        // scatter back into the eight corners
        // accounts for >90% of run time, needs optimization
        int base = ((ni*12 + ci)*L*H*W);
        atomicAdd(v_bilagrid + base + (z0*H + y0)*W + x0, f000 * grad_weight);
        atomicAdd(v_bilagrid + base + (z0*H + y0)*W + x1, f001 * grad_weight);
        atomicAdd(v_bilagrid + base + (z0*H + y1)*W + x0, f010 * grad_weight);
        atomicAdd(v_bilagrid + base + (z0*H + y1)*W + x1, f011 * grad_weight);
        atomicAdd(v_bilagrid + base + (z1*H + y0)*W + x0, f100 * grad_weight);
        atomicAdd(v_bilagrid + base + (z1*H + y0)*W + x1, f101 * grad_weight);
        atomicAdd(v_bilagrid + base + (z1*H + y1)*W + x0, f110 * grad_weight);
        atomicAdd(v_bilagrid + base + (z1*H + y1)*W + x1, f111 * grad_weight);

        // gradient w.r.t. rgb coefficients
        if (si < 3) {
            float val =
                ( ( (bilagrid[base + (z0*H + y0)*W + x0]*(1-fx) + bilagrid[base + (z0*H + y0)*W + x1]*fx)*(1-fy)
                    + (bilagrid[base + (z0*H + y1)*W + x0]*(1-fx) + bilagrid[base + (z0*H + y1)*W + x1]*fx)*fy )*(1-fz)
                + ( (bilagrid[base + (z1*H + y0)*W + x0]*(1-fx) + bilagrid[base + (z1*H + y0)*W + x1]*fx)*(1-fy)
                    + (bilagrid[base + (z1*H + y1)*W + x0]*(1-fx) + bilagrid[base + (z1*H + y1)*W + x1]*fx)*fy )*fz
                );
            (si == 0 ? vr : si == 1 ? vg : vb) += val * gout;
        }
    }

    // spatial derivatives for coords
    // ∂w000/∂x = -(1-fy)*(1-fz), ∂w001/∂x = +(1-fy)*(1-fz), etc...
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

    // accumulate gradient into coords (chain through bilagrid values and rgb)
    #ifdef COMPUTE_COORDS_GRAD
    float gx_grad = 0.f, gy_grad = 0.f;
    #endif
    float gz_grad = 0.f;
    #pragma unroll
    for (int corner = 0; corner < 8; ++corner) {
        int xi = (corner & 1) ? x1 : x0;
        int yi = (corner & 2) ? y1 : y0;
        int zi = (corner & 4) ? z1 : z0;
        float trilerp = 0.f;
        // gather the corresponding bilagrid value for each of the 12 channels
        #pragma unroll
        for (int ci = 0; ci < 12; ++ci) {
            const float* vol = bilagrid + ((ni*12 + ci)*L*H*W);
            float v = vol[(zi*H + yi)*W + xi];
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
    // save gradient, with discontinuty masking
    #ifdef COMPUTE_COORDS_GRAD
    v_coords[2*g_off+0] = gx_grad * (float)(x0 != x && x1 != x);
    v_coords[2*g_off+1] = gy_grad * (float)(y0 != y && y1 != y);
    #endif
    gz_grad *= (float)(z0 != z && z1 != z);
    v_rgb[3*g_off+0] = vr + kC2G_r * gz_grad;
    v_rgb[3*g_off+1] = vg + kC2G_g * gz_grad;;
    v_rgb[3*g_off+2] = vb + kC2G_b * gz_grad;;
}
