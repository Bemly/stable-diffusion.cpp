#include "common.h"
#include "dequantize.h"

// Tiled fp32 GEMM for Metal devices without simdgroup_matrix (AMD RDNA2 and
// other non-Apple7 families). ggml's kernel_mul_mm needs simdgroup_multiply_accumulate,
// so on those devices every batched matmul falls back to kernel_mul_mv, which
// re-reads the whole weight matrix once per 2-4 columns and is memory bound.
//
// Layout, matching ggml mul_mat:
//   A: [K, M, ne02, ne03], row stride nb01, quantized (q8_0) or f16
//   B: [K, N, ne12, ne13], column stride nb11, f32
//   C: [M, N, ne2, ne3], contiguous f32, row stride nb1
//
// Tiles: BM x BN outputs per threadgroup, BK elements per k step, 256 threads,
// each thread accumulating a 4x8 sub-tile in registers.

constant constexpr int MM_AMD_BM      = 64;
constant constexpr int MM_AMD_BN      = 128;
constant constexpr int MM_AMD_BK      = 32;
// +4 floats keeps the f32 column stride 16B aligned for float4 loads while
// still spreading the fill stores over the smem banks (132 % 32 == 4).
constant constexpr int MM_AMD_SBN     = MM_AMD_BN + 4;

// QT: 0 = q8_0, 1 = f16
template <int QT>
kernel void kernel_mul_mm_amd(
        constant ggml_metal_kargs_mul_mm_amd & args,
        device  const char * srcA,
        device  const char * srcB,
        device        char * dst,
        threadgroup   char * shmem [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort  tiitg [[thread_index_in_threadgroup]]) {
    const int K = args.ne00;
    const int M = args.ne01;
    const int N = args.ne11;

    const int im  = tgpig.z;
    const int i12 = im % args.ne12;
    const int i13 = im / args.ne12;

    const int64_t off_a = (int64_t) (i12/args.r2)*args.nb02 + (int64_t) (i13/args.r3)*args.nb03;
    const int64_t off_b = (int64_t) i12*args.nb12 + (int64_t) i13*args.nb13;
    const int64_t off_d = (int64_t) i12*args.nb2  + (int64_t) i13*args.nb3;

    const int m0 = tgpig.y * MM_AMD_BM;
    const int n0 = tgpig.x * MM_AMD_BN;

    const int tid = tiitg;
    const int tx  = tid % 16; // 8 output columns each -> BN
    const int ty  = tid / 16; // 4 output rows each -> BM

    threadgroup float * sA = (threadgroup float *) shmem;                        // [BK][BM]
    threadgroup float * sB = (threadgroup float *) (shmem + sizeof(float)*MM_AMD_BK*MM_AMD_BM); // [BK][SBN]

    float acc[4][8];
    FOR_UNROLL (int i = 0; i < 4; ++i) {
        FOR_UNROLL (int j = 0; j < 8; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // A tile loader: thread covers row tid/4, eight consecutive k of tid%4
    const int a_row = tid / 4;
    const int a_sub = tid % 4;

    // B tile loader: warp covers 32 consecutive k of one column, 16 columns per warp
    const int b_k   = tid % 32;
    const int b_col = tid / 32;

    for (int k0 = 0; k0 < K; k0 += MM_AMD_BK) {
        {
            const int gm = m0 + a_row;
            const int kk = min(MM_AMD_BK, K - k0);

            if (gm < M && kk == MM_AMD_BK) {
                if (QT == 0) {
                    device const block_q8_0 * blk =
                        (device const block_q8_0 *) (srcA + off_a + (int64_t) gm*args.nb01) + (k0/QK8_0);
                    const float d = (float) blk->d;
                    FOR_UNROLL (int i = 0; i < 8; ++i) {
                        sA[(a_sub*8 + i)*MM_AMD_BM + a_row] = d * (float) blk->qs[a_sub*8 + i];
                    }
                } else {
                    device const half * row = (device const half *) (srcA + off_a + (int64_t) gm*args.nb01);
                    FOR_UNROLL (int i = 0; i < 8; ++i) {
                        sA[(a_sub*8 + i)*MM_AMD_BM + a_row] = (float) row[k0 + a_sub*8 + i];
                    }
                }
            } else {
                for (int i = 0; i < 8; ++i) {
                    const int gk = a_sub*8 + i;
                    float v = 0.0f;
                    if (gm < M && gk < kk) {
                        if (QT == 0) {
                            device const block_q8_0 * blk =
                                (device const block_q8_0 *) (srcA + off_a + (int64_t) gm*args.nb01) + (k0/QK8_0);
                            v = (float) blk->d * (float) blk->qs[gk];
                        } else {
                            device const half * row = (device const half *) (srcA + off_a + (int64_t) gm*args.nb01);
                            v = (float) row[k0 + gk];
                        }
                    }
                    sA[gk*MM_AMD_BM + a_row] = v;
                }
            }
        }

        {
            const int kk = min(MM_AMD_BK, K - k0);
            FOR_UNROLL (int t = 0; t < MM_AMD_BN/8; ++t) {
                const int gn = n0 + b_col + 8*t;
                const float v = (b_k < kk && gn < N)
                    ? *(device const float *) (srcB + off_b + (int64_t) gn*args.nb11 + (int64_t) (k0 + b_k)*sizeof(float))
                    : 0.0f;
                sB[b_k*MM_AMD_SBN + (b_col + 8*t)] = v;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < MM_AMD_BK; ++kk) {
            const float4 a  = *(threadgroup const float4 *) (sA + kk*MM_AMD_BM + 4*ty);
            const float4 b0 = *(threadgroup const float4 *) (sB + kk*MM_AMD_SBN + 8*tx);
            const float4 b1 = *(threadgroup const float4 *) (sB + kk*MM_AMD_SBN + 8*tx + 4);

            acc[0][0] = fma(a.x, b0.x, acc[0][0]); acc[0][1] = fma(a.x, b0.y, acc[0][1]);
            acc[0][2] = fma(a.x, b0.z, acc[0][2]); acc[0][3] = fma(a.x, b0.w, acc[0][3]);
            acc[0][4] = fma(a.x, b1.x, acc[0][4]); acc[0][5] = fma(a.x, b1.y, acc[0][5]);
            acc[0][6] = fma(a.x, b1.z, acc[0][6]); acc[0][7] = fma(a.x, b1.w, acc[0][7]);

            acc[1][0] = fma(a.y, b0.x, acc[1][0]); acc[1][1] = fma(a.y, b0.y, acc[1][1]);
            acc[1][2] = fma(a.y, b0.z, acc[1][2]); acc[1][3] = fma(a.y, b0.w, acc[1][3]);
            acc[1][4] = fma(a.y, b1.x, acc[1][4]); acc[1][5] = fma(a.y, b1.y, acc[1][5]);
            acc[1][6] = fma(a.y, b1.z, acc[1][6]); acc[1][7] = fma(a.y, b1.w, acc[1][7]);

            acc[2][0] = fma(a.z, b0.x, acc[2][0]); acc[2][1] = fma(a.z, b0.y, acc[2][1]);
            acc[2][2] = fma(a.z, b0.z, acc[2][2]); acc[2][3] = fma(a.z, b0.w, acc[2][3]);
            acc[2][4] = fma(a.z, b1.x, acc[2][4]); acc[2][5] = fma(a.z, b1.y, acc[2][5]);
            acc[2][6] = fma(a.z, b1.z, acc[2][6]); acc[2][7] = fma(a.z, b1.w, acc[2][7]);

            acc[3][0] = fma(a.w, b0.x, acc[3][0]); acc[3][1] = fma(a.w, b0.y, acc[3][1]);
            acc[3][2] = fma(a.w, b0.z, acc[3][2]); acc[3][3] = fma(a.w, b0.w, acc[3][3]);
            acc[3][4] = fma(a.w, b1.x, acc[3][4]); acc[3][5] = fma(a.w, b1.y, acc[3][5]);
            acc[3][6] = fma(a.w, b1.z, acc[3][6]); acc[3][7] = fma(a.w, b1.w, acc[3][7]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    FOR_UNROLL (int i = 0; i < 4; ++i) {
        const int gm = m0 + 4*ty + i;
        if (gm >= M) {
            continue;
        }
        device float * d = (device float *) (dst + off_d + (int64_t) gm*sizeof(float));
        FOR_UNROLL (int j = 0; j < 8; ++j) {
            const int gn = n0 + 8*tx + j;
            if (gn < N) {
                *(device float *) ((device char *) d + (int64_t) gn*args.nb1) = acc[i][j];
            }
        }
    }
}

typedef decltype(kernel_mul_mm_amd<0>) mul_mm_amd_q8_0_t;
typedef decltype(kernel_mul_mm_amd<1>) mul_mm_amd_f16_t;

template [[host_name("kernel_mul_mm_amd_q8_0")]] kernel mul_mm_amd_q8_0_t kernel_mul_mm_amd<0>;
template [[host_name("kernel_mul_mm_amd_f16" )]] kernel mul_mm_amd_f16_t  kernel_mul_mm_amd<1>;
