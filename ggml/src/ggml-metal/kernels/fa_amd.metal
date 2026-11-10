// ── RX6800 (RDNA2/gfx1030) 自研 FLASH_ATTN_EXT kernel —— FA-RDNA2 立项 ──
// 设计依据探针实测（scripts/probe-rx6800-metal.swift，结论见 mps CONFIG.md）：
//   - RDNA2 无 simdgroup matrix（Metal 后端 SC 编译失败），全程 VALU；
//   - v_pk_fma_f16 由 MSL 生成（half2 packed ≈ 2.7× f32），Q 以 half2 寄存器阵列持有；
//   - LDS 行距避开 64-bank（4B）周期，K 转置块行距 +1 half4 pad；
//   - simd_sum 依赖链 ~70ns，跨线程归约保持在每块常数次；
//   - KV 免 LDS 直读（IC 命中 ~1.5TB/s 吸收 GQA 重读），decode 每块仅 3 个 barrier；
//   - decode（nq==1）：split-k 部分结果 + reduce 合并，保小模型占用率；
//   - prefill（nq>1）：Q tile 32 行/tg，K 块过 LDS，Q/O 全寄存器化。
// 仅支持：F16 KV、dk==dv∈{64,128,256,512}（256/512 为 P1-A 朴素版，nbc=64）、无 sinks/bias/softcap；其余形状由 host 门控回退。

constant bool HAS_MASK_FA_AMD [[function_constant(FC_FLASH_ATTN_EXT_AMD + 0)]];

// ─────────────────────────── decode vec kernel ───────────────────────────
// tg = 128 线程（4 simdgroup），grid = (split, n_head, n_batch)。
// 相位：S（lane-per-token 全 dk 点积，无跨线程归约）→ 在线 softmax（log2 域，3 barrier）
//       → PV（lane-per-dv 对，half2 读宽）→ 写 split 部分结果，由 reduce kernel 合并。
// scale*log2e 折入 Q，S 全程 log2 域，w = exp2(s - m)。

template <int DK, int DV, int NBC>
kernel void kernel_flash_attn_ext_amd_vec_dk(
        constant ggml_metal_kargs_flash_attn_ext_amd & args,
        device  const char * q,
        device  const char * k,
        device  const char * v,
        device  const char * mask,
        device        char * part,
        threadgroup   float * shm [[threadgroup(0)]],   // [NBC] s/w + [4] sg-max + [4] sg-l
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort  tiisg [[thread_index_in_simdgroup]],
        ushort  sgitg [[simdgroup_index_in_threadgroup]]) {
    const int isplit = tgpig[0];
    const int ih     = tgpig[1];
    const int ib     = tgpig[2];
    const int lane   = sgitg*N_SIMDWIDTH + tiisg;   // 0..127

    const int ikv2 = ih / (args.ne02/args.ne_12_2);
    const int ikv3 = ib / (args.ne03/args.ne_12_3);

    device const char * kp = k + ikv2*args.nb12 + ikv3*args.nb13;
    device const char * vp = v + ikv2*args.nb22 + ikv3*args.nb23;
    // nq==1：mask 行 0（nb30==2 已由 host 门控保证）
    device const half * mp = HAS_MASK_FA_AMD
        ? (device const half *)(mask + (ib % args.ne33)*args.nb33 + (ih % args.ne32)*args.nb32)
        : nullptr;

    // Q → 寄存器（half2 × DK/2），折入 scale*log2e
    const float qs = args.scale * 1.44269504088896f;
    half2 qh[DK/2];
    {
        device const float4 * qv4 = (device const float4 *)(q + ib*args.nb03 + ih*args.nb02);
        FOR_UNROLL (int i = 0; i < DK/4; ++i) {
            float4 f = qv4[i];
            qh[2*i + 0] = half2(f.x*qs, f.y*qs);
            qh[2*i + 1] = half2(f.z*qs, f.w*qs);
        }
    }

    const int kv0 = isplit*args.chunk;
    const int kv1 = MIN((int) args.ne11, kv0 + args.chunk);

    // 128 lane 每轮覆盖 256 个 dv 元素；DV=512 需要 2 轮（否则后半 dv 不写，reduce 读到垃圾）
    constexpr int NPV = (DV/2 + 127)/128;

    float  m_run = -INFINITY;
    float  l_run = 0.0f;
    float2 o_run[NPV];   // lane 拥有 dv 元素对 p = lane + 128*i：{2*p, 2*p+1}（p < DV/2）
    FOR_UNROLL (int i = 0; i < NPV; ++i) {
        o_run[i] = 0.0f;
    }

    threadgroup float * ss = shm;             // [NBC]
    threadgroup float * sm = shm + NBC;       // [4]
    threadgroup float * sl = shm + NBC + 4;   // [4]

    for (int t0 = kv0; t0 < kv1; t0 += NBC) {
        // ── S：每 lane 一个 token 的全 dk 点积（f32 累加，免归约）──
        {
            const int t = t0 + lane;
            float s = -INFINITY;
            if (t < kv1) {
                device const half4 * kh4 = (device const half4 *)(kp + (size_t) t*args.nb11);
                float acc = 0.0f;
                FOR_UNROLL (int i = 0; i < DK/4; ++i) {
                    half4 q4 = half4(qh[2*i + 0], qh[2*i + 1]);
                    float4 p = float4(q4 * kh4[i]);
                    acc += p.x + p.y + p.z + p.w;
                }
                s = HAS_MASK_FA_AMD ? acc + (float) mp[t]*1.44269504088896f : acc;   // mask 在 ln 域，S 在 log2 域
            }
            if (lane < NBC) {
                ss[lane] = s;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── 在线 softmax（分 simdgroup 归约 + 4 路合并）──
        {
            const float my = (lane < NBC) ? ss[lane] : -INFINITY;
            const float msg = simd_max(my);
            if (tiisg == 0) {
                sm[sgitg] = msg;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float m_new = MAX(m_run, MAX(sm[0], MAX(sm[1], MAX(sm[2], sm[3]))));
        float e = 0.0f;
        if (lane < NBC) {
            // s、m 同为 -inf 时 s-m 为 NaN，fmax 取非 NaN 侧 → w=2^-128≈0
            e = exp2(fmax(ss[lane] - m_new, -128.0f));
            ss[lane] = e;   // 覆写为未归一化权重
        }
        {
            const float lsg = simd_sum(e);
            if (tiisg == 0) {
                sl[sgitg] = lsg;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            const float corr = (m_new == -INFINITY) ? 1.0f
                             : ((m_run == -INFINITY) ? 0.0f : exp2(m_run - m_new));
            l_run = l_run*corr + (sl[0] + sl[1] + sl[2] + sl[3]);
            FOR_UNROLL (int i = 0; i < NPV; ++i) {
                o_run[i] *= corr;
            }
            m_run = m_new;
        }

        // ── PV：lane-per-dv 对（half2 读宽，w 从 LDS 广播）──
        if (lane < DV/2) {
            device const half2 * vh2 = (device const half2 *) vp;
            for (int j = 0; j < NBC; ++j) {
                if (t0 + j >= kv1) break;
                device const half2 * vrow = vh2 + (size_t) (t0 + j)*(args.nb21/4);
                const float w = ss[j];
                FOR_UNROLL (int i = 0; i < NPV; ++i) {
                    const int p = lane + 128*i;
                    if (p < DV/2) {
                        o_run[i] += w*float2(vrow[p]);
                    }
                }
            }
        }
        // ss holds this block's weights; the next block overwrites it after this barrier
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── 写 split 部分结果：[b][h][split][DV+2] f32（o、m、l）──
    device float * pb = (device float *)(part +
            ((size_t) ib*args.ne02 + ih)*args.split*(DV + 2)*sizeof(float) +
            (size_t) isplit*(DV + 2)*sizeof(float));
    FOR_UNROLL (int i = 0; i < NPV; ++i) {
        const int p = lane + 128*i;
        if (p < DV/2) {
            pb[2*p + 0] = o_run[i].x;
            pb[2*p + 1] = o_run[i].y;
        }
    }
    if (lane == 0) {
        pb[DV + 0] = m_run;
        pb[DV + 1] = l_run;
    }
}

// ─────────────────────────── split 合并 kernel ───────────────────────────
// tg = DV/2 线程（2 simdgroup），grid = (n_head, n_batch)；lane 拥有 dv 元素对，读 split 部分结果归并。

template <int DV>
kernel void kernel_flash_attn_ext_amd_reduce_dk(
        constant ggml_metal_kargs_flash_attn_ext_amd_reduce & args,
        device  const char * part,
        device        char * dst,
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort  tiisg [[thread_index_in_simdgroup]],
        ushort  sgitg [[simdgroup_index_in_threadgroup]]) {
    const int ih   = tgpig[0];
    const int ib   = tgpig[1];
    const int lane = sgitg*N_SIMDWIDTH + tiisg;

    device const float * pb = (device const float *)(part +
            ((size_t) ib*args.ne02 + ih)*args.split*(DV + 2)*sizeof(float));

    float m_max = -INFINITY;
    for (int s = 0; s < args.split; ++s) {
        m_max = MAX(m_max, pb[s*(DV + 2) + DV]);
    }

    float2 o = 0.0f;
    float  l_tot = 0.0f;
    if (m_max != -INFINITY && lane < DV/2) {
        for (int s = 0; s < args.split; ++s) {
            device const float * ps = pb + s*(DV + 2);
            const float c = exp2(ps[DV] - m_max);
            o += float2(ps[2*lane + 0], ps[2*lane + 1])*c;
            l_tot += ps[DV + 1]*c;
        }
    }

    if (lane < DV/2) {
        device float * d = (device float *)(dst + ib*args.nb03 + ih*args.nb02);
        const float inv = l_tot > 0.0f ? 1.0f/l_tot : 0.0f;
        d[2*lane + 0] = o.x*inv;
        d[2*lane + 1] = o.y*inv;
    }
}

// ─────────────────────────── prefill tile kernel ───────────────────────────
// tg = 256 线程（8 simdgroup），grid = (ceil(nq/NQ), n_head, n_batch)。
// lane = (r_loc = lane/8, sub = lane%8)：每 lane 拥有一个 q 行（寄存器）+ 4 个 dv 四元组（O 寄存器）。
// K 块过 LDS（转置 half4 布局，行距 +1 pad 破 bank 周期）；V 免 LDS 直读；sw 存未归一化权重。

template <int DK, int DV, int NBC, int NQ>
kernel void kernel_flash_attn_ext_amd_tile_dk(
        constant ggml_metal_kargs_flash_attn_ext_amd & args,
        device  const char * q,
        device  const char * k,
        device  const char * v,
        device  const char * mask,
        device        char * dst,
        threadgroup   half  * shm [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort  tiisg [[thread_index_in_simdgroup]],
        ushort  sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int DK4 = DK/4;
    constexpr int KTP = NBC + 1;          // half4 行距 pad，破 64-bank 周期
    constexpr int KPT = NBC/8;            // 每个 sub-lane 负责的 K 位置数
    constexpr int TPB = 256/NBC;          // K 装载时覆盖同 t 的 lane 组大小

    const int itile  = tgpig[0];
    const int ih     = tgpig[1];
    const int ib     = tgpig[2];
    const int lane   = sgitg*N_SIMDWIDTH + tiisg;   // 0..255
    const int r_loc  = lane / 8;                    // 0..NQ-1
    const int sub    = lane % 8;                    // 0..7
    const int r_glob = itile*NQ + r_loc;
    const bool r_valid = r_glob < args.ne01;

    const int ikv2 = ih / (args.ne02/args.ne_12_2);
    const int ikv3 = ib / (args.ne03/args.ne_12_3);

    device const char * kp = k + ikv2*args.nb12 + ikv3*args.nb13;
    device const char * vp = v + ikv2*args.nb22 + ikv3*args.nb23;
    device const char * mp = HAS_MASK_FA_AMD
        ? mask + r_glob*args.nb31 + (ih % args.ne32)*args.nb32 + (ib % args.ne33)*args.nb33
        : nullptr;

    // LDS 布局（半字偏移；float 区 8B 对齐已满足：DK4*KTP*4 为偶数）
    threadgroup half4  * kt  = (threadgroup half4  *) shm;                    // [DK4][KTP]
    threadgroup float  * sw  = (threadgroup float  *)(shm + DK4*KTP*4);       // [NQ][NBC]
    threadgroup float  * mps = sw + NQ*NBC;                                   // [NQ][8]
    threadgroup float  * lps = mps + NQ*8;                                    // [NQ][8]
    threadgroup float  * mst = lps + NQ*8;                                    // [NQ]
    threadgroup float  * cst = mst + NQ;                                      // [NQ]
    threadgroup float  * lst = cst + NQ;                                      // [NQ]

    // Q → 寄存器（折入 scale*log2e；无效行置 0 并以 -inf 屏蔽）
    const float qs = args.scale * 1.44269504088896f;
    half2 qh[DK/2];
    {
        device const float4 * qv4 = (device const float4 *)(q + ib*args.nb03 + ih*args.nb02 + (size_t) r_glob*args.nb01);
        FOR_UNROLL (int i = 0; i < DK/4; ++i) {
            if (r_valid) {
                float4 f = qv4[i];
                qh[2*i + 0] = half2(f.x*qs, f.y*qs);
                qh[2*i + 1] = half2(f.z*qs, f.w*qs);
            } else {
                qh[2*i + 0] = 0.0h;
                qh[2*i + 1] = 0.0h;
            }
        }
    }

    constexpr int OGRPS = DV/32;   // 每 lane 拥有的 dv 四元组数（8 sub-lane × OGRPS × 4 = DV）
    float4 o4[OGRPS];   // lane 拥有 dv 四元组 {sub*OGRPS + jj}
    FOR_UNROLL (int jj = 0; jj < OGRPS; ++jj) {
        o4[jj] = 0.0f;
    }

    if (tiisg < NQ) {
        mst[tiisg] = -INFINITY;
        lst[tiisg] = 0.0f;
        cst[tiisg] = 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t0 = 0; t0 < args.ne11; t0 += NBC) {
        // ── 协作装载 K 块 → LDS 转置 half4（lane: t = lane/TPB, i4 = (lane%TPB)*(DK4/TPB) + k）──
        {
            const int tld = lane/TPB;
            const int i0  = (lane%TPB)*(DK4/TPB);
            const int t   = t0 + tld;
            device const half4 * kh4 = (device const half4 *)(kp + (size_t) t*args.nb11);
            FOR_UNROLL (int kk = 0; kk < DK4/TPB; ++kk) {
                kt[(i0 + kk)*KTP + tld] = (t < args.ne11) ? kh4[i0 + kk] : half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── S：每 lane KPT 个 token 的点积（f32 累加，存寄存器）──
        float s_reg[KPT];
        {
            const int tsub = sub*KPT;
            FOR_UNROLL (int kk = 0; kk < KPT; ++kk) {
                const int t = t0 + tsub + kk;
                float s = -INFINITY;
                if (r_valid && t < args.ne11) {
                    float acc = 0.0f;
                    FOR_UNROLL (int i = 0; i < DK4; ++i) {
                        half4 q4 = half4(qh[2*i + 0], qh[2*i + 1]);
                        float4 p = float4(q4 * kt[i*KTP + tsub + kk]);
                        acc += p.x + p.y + p.z + p.w;
                    }
                    s = acc;
                    if (HAS_MASK_FA_AMD) {
                        s += (float) ((device const half *) mp)[t]*1.44269504088896f;   // ln 域 mask → log2 域
                    }
                }
                s_reg[kk] = s;
            }
        }

        // ── 局部 max → 行合并 ──
        {
            float m_loc = -INFINITY;
            FOR_UNROLL (int kk = 0; kk < KPT; ++kk) {
                m_loc = MAX(m_loc, s_reg[kk]);
            }
            mps[r_loc*8 + sub] = m_loc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float corr = 1.0f;
        if (sub == 0) {
            float mm = mst[r_loc];
            FOR_UNROLL (int j = 0; j < 8; ++j) {
                mm = MAX(mm, mps[r_loc*8 + j]);
            }
            const float c = (mm == -INFINITY) ? 1.0f : ((mst[r_loc] == -INFINITY) ? 0.0f : exp2(mst[r_loc] - mm));
            mst[r_loc] = mm;
            cst[r_loc] = c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        corr = cst[r_loc];   // 全行同步读取新值

        // ── 权重 + 局部 l → sw 覆写；O 重缩放 ──
        {
            float l_loc = 0.0f;
            const int tsub = sub*KPT;
            FOR_UNROLL (int kk = 0; kk < KPT; ++kk) {
                const float w = exp2(fmax(s_reg[kk] - mst[r_loc], -128.0f));
                l_loc += w;
                sw[r_loc*NBC + tsub + kk] = w;
            }
            lps[r_loc*8 + sub] = l_loc;
            FOR_UNROLL (int jj = 0; jj < OGRPS; ++jj) {
                o4[jj] *= corr;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── l 行合并 ──
        if (sub == 0) {
            float ll = lst[r_loc]*cst[r_loc];
            FOR_UNROLL (int j = 0; j < 8; ++j) {
                ll += lps[r_loc*8 + j];
            }
            lst[r_loc] = ll;
        }

        // ── PV：w 从 LDS 广播，V 直读（每 lane 4 个 dv 四元组）──
        {
            const int jmax = MIN(NBC, (int) args.ne11 - t0);
            for (int j = 0; j < jmax; ++j) {
                const float w = sw[r_loc*NBC + j];
                device const half4 * vh4 = (device const half4 *)(vp + (size_t) (t0 + j)*args.nb21);
                FOR_UNROLL (int jj = 0; jj < OGRPS; ++jj) {
                    o4[jj] += w*float4(vh4[sub*OGRPS + jj]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // out = o / l (dst is [DV,H,N,B], Q is [DK,N,H,B], use dst strides)
    if (r_valid) {
        const float l = lst[r_loc];
        const float inv = l > 0.0f ? 1.0f/l : 0.0f;
        device float * d = (device float *)(dst + ib*args.nbb_d + ih*args.nbh_d + (size_t) r_glob*args.nbq_d);
        FOR_UNROLL (int jj = 0; jj < OGRPS; ++jj) {
            *(device float4 *)(d + (sub*OGRPS + jj)*4) = o4[jj]*inv;
        }
    }
}

// ─────────────────────────── 模板实例化 ───────────────────────────
// vec：dk∈{64,128} × nbc∈{64,128}，dk256/512 仅 nbc64（P1-A 朴素版）；tile：dk∈{64,128,256,512}，nbc=64，NQ=32；reduce：dv∈{64,128,256,512}

typedef decltype(kernel_flash_attn_ext_amd_vec_dk<64, 64, 64>)   flash_attn_ext_amd_vec_t;
typedef decltype(kernel_flash_attn_ext_amd_reduce_dk<64>)        flash_attn_ext_amd_reduce_t;
typedef decltype(kernel_flash_attn_ext_amd_tile_dk<64, 64, 64, 32>) flash_attn_ext_amd_tile_t;

template [[host_name("kernel_flash_attn_ext_amd_vec_dk64_nbc64"  )]] kernel flash_attn_ext_amd_vec_t kernel_flash_attn_ext_amd_vec_dk< 64,  64,  64>;
template [[host_name("kernel_flash_attn_ext_amd_vec_dk64_nbc128" )]] kernel flash_attn_ext_amd_vec_t kernel_flash_attn_ext_amd_vec_dk< 64,  64, 128>;
template [[host_name("kernel_flash_attn_ext_amd_vec_dk128_nbc64" )]] kernel flash_attn_ext_amd_vec_t kernel_flash_attn_ext_amd_vec_dk<128, 128,  64>;
template [[host_name("kernel_flash_attn_ext_amd_vec_dk128_nbc128")]] kernel flash_attn_ext_amd_vec_t kernel_flash_attn_ext_amd_vec_dk<128, 128, 128>;
template [[host_name("kernel_flash_attn_ext_amd_vec_dk256_nbc64"  )]] kernel flash_attn_ext_amd_vec_t kernel_flash_attn_ext_amd_vec_dk<256, 256,  64>;
template [[host_name("kernel_flash_attn_ext_amd_vec_dk512_nbc64"  )]] kernel flash_attn_ext_amd_vec_t kernel_flash_attn_ext_amd_vec_dk<512, 512,  64>;

template [[host_name("kernel_flash_attn_ext_amd_reduce_dk64" )]] kernel flash_attn_ext_amd_reduce_t kernel_flash_attn_ext_amd_reduce_dk< 64>;
template [[host_name("kernel_flash_attn_ext_amd_reduce_dk128")]] kernel flash_attn_ext_amd_reduce_t kernel_flash_attn_ext_amd_reduce_dk<128>;
template [[host_name("kernel_flash_attn_ext_amd_reduce_dk256")]] kernel flash_attn_ext_amd_reduce_t kernel_flash_attn_ext_amd_reduce_dk<256>;
template [[host_name("kernel_flash_attn_ext_amd_reduce_dk512")]] kernel flash_attn_ext_amd_reduce_t kernel_flash_attn_ext_amd_reduce_dk<512>;

template [[host_name("kernel_flash_attn_ext_amd_tile_dk64_nbc64" )]] kernel flash_attn_ext_amd_tile_t kernel_flash_attn_ext_amd_tile_dk< 64,  64, 64, 32>;
template [[host_name("kernel_flash_attn_ext_amd_tile_dk128_nbc64")]] kernel flash_attn_ext_amd_tile_t kernel_flash_attn_ext_amd_tile_dk<128, 128, 64, 32>;
template [[host_name("kernel_flash_attn_ext_amd_tile_dk256_nbc64")]] kernel flash_attn_ext_amd_tile_t kernel_flash_attn_ext_amd_tile_dk<256, 256, 64, 32>;
template [[host_name("kernel_flash_attn_ext_amd_tile_dk512_nbc32")]] kernel flash_attn_ext_amd_tile_t kernel_flash_attn_ext_amd_tile_dk<512, 512, 32, 32>;
