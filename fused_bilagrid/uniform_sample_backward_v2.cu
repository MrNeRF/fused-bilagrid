#include "config.h"


__global__ void bilagrid_uniform_sample_backward_v2_kernel(
    const float* __restrict__ bilagrid,  // [N,12,L,H,W]
    const float* __restrict__ rgb,  // [N,m,h,w,3]
    const float* __restrict__ v_output,  // [N,m,h,w,3]
    float* __restrict__ v_bilagrid,  // [N,12,L,H,W]
    float* __restrict__ v_rgb,  // [N,m,h,w,3]
    int N, int L, int H, int W,
    int m, int h, int w
) {
    #if 0
    int wi = blockIdx.x * blockDim.x + threadIdx.x;
    int hi = blockIdx.y * blockDim.y + threadIdx.y;
    #else
    // reduces number of threads writing to the same address at a time in atomicAdd
    int wi = threadIdx.x * ((w+blockDim.x-1) / blockDim.x) + blockIdx.x;
    int hi = threadIdx.y * ((h+blockDim.y-1) / blockDim.y) + blockIdx.y;
    #endif

    int idx = blockIdx.z * blockDim.z + threadIdx.z;
    bool inside = (wi < w && hi < h && idx < (N*m));
    if (!inside) return;
    int mi = idx % m;
    int ni = idx / m;

    // grid coords
    int g_off = (((ni*m + mi)*h + hi)*w + wi) * 3;
    float sr = rgb[g_off+0], sg = rgb[g_off+1], sb = rgb[g_off+2];
    float x = (float)wi / (float)(w-1) * (float)(W-1);
    float y = (float)hi / (float)(h-1) * (float)(H-1);
    float z = (kC2G_r * sr + kC2G_g * sg + kC2G_b * sb) * (L-1);

    // floor + ceil, clamped
    int x0 = floorf(x), y0 = floorf(y), z0 = floorf(z);
    int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
    z0 = min(max(z0,0), L-1); z1 = min(max(z1,0), L-1);

    // fractional parts
    float fx = x - x0, fy = y - y0, fz = z - z0;

    // read rgb coeffs and upstream gradient
    float dr = v_output[g_off+0];
    float dg = v_output[g_off+1];
    float db = v_output[g_off+2];
    float vr = 0.0, vg = 0.0, vb = 0.0;

    // spatial derivatives for coords

    float gz_grad = 0.f;

    #pragma unroll
    for (int corner = 0; corner < 8; ++corner) {
        int xi = (corner & 1) ? x1 : x0;
        int yi = (corner & 2) ? y1 : y0;
        int zi = (corner & 4) ? z1 : z0;

        float dfdz = ((corner & 1) ? fx : (1-fx)) *
            ((corner & 2) ? fy : (1-fy)) * ((corner & 4) ? 1 : -1);
        float f = dfdz * ((corner & 4) ? fz : (fz-1));

        float trilerp = 0.f;
        #pragma unroll
        for (int ci = 0; ci < 12; ++ci) {
            int bidx = (((ni*12 + ci)*L + zi)*H + yi)*W + xi;
            int si = ci % 4, di = ci / 4;

            float r_coeff = (si==0 ? sr : si==1 ? sg : si==2 ? sb : 1.f);
            float gout = (di==0 ? dr : di==1 ? dg : db);

            float v = bilagrid[bidx];

            if (si < 3)
                (si == 0 ? vr : si == 1 ? vg : vb) += v * f * gout;

            float grad_weight = r_coeff * gout;
            trilerp += v * grad_weight;
            atomicAdd(v_bilagrid+bidx, f * grad_weight);
        }
        gz_grad += dfdz * (L-1) * trilerp;
    }

    // save gradient, with discontinuity masking
    gz_grad *= (float)(z0 != z && z1 != z);
    v_rgb[g_off+0] = vr + kC2G_r * gz_grad;
    v_rgb[g_off+1] = vg + kC2G_g * gz_grad;
    v_rgb[g_off+2] = vb + kC2G_b * gz_grad;
}


void bilagrid_uniform_sample_backward_v2(
    const float* bilagrid,
    const float* rgb,
    const float* v_output,
    float* v_bilagrid,
    float* v_rgb,
    int N, int L, int H, int W,
    int m, int h, int w
) {
    dim3 block = { 16, 16, 1 };
    dim3 bounds = {
        (w +block.x-1)/block.x,
        (h +block.y-1)/block.y,
        (N*m +block.z-1)/block.z
    };
    bilagrid_uniform_sample_backward_v2_kernel<<<bounds, block>>>(
        bilagrid, rgb, v_output,
        v_bilagrid, v_rgb,
        N, L, H, W, m, h, w
    );
}
