static __device__ __forceinline__ int unpack_scales_q45_K_shift(const int* scales, const int ksc) {
    // The assumed layout is:
    //   For ksc == 0: the 4 lower nibble values reside in scales[0] (both lower and upper parts)
    //   For ksc == 1: lower part from scales[2] (no shift) and upper part from scales[0] (shifted by 2)
    //   For ksc == 2: lower part from scales[1] (no shift) and upper part from scales[1] (no shift)
    //   For ksc == 3: lower part from scales[2] (shifted by 4) and upper part from scales[1] (shifted by 2)
    //
    // Preload the three integer values that the original dynamic indexing would use:
    int s0 = scales[0];
    int s1 = scales[1];
    int s2 = scales[2];

    int lower = 0, upper = 0;
    switch (ksc) {
        case 0:
            // ksc == 0: scales[0] for both parts, no shift for lower nibble and no shift for upper nibble.
            lower = s0 >> (4 * 0);   // no shift
            upper = s0 >> (2 * 0);   // no shift
            break;
        case 1:
            // ksc == 1: lower part is from scales[2] with no shift; upper part is from scales[0] shifted by 2 bits.
            lower = s2 >> (4 * 0);   // no shift (since (1 & 0) == 0)
            upper = s0 >> (2 * 1);   // shift by 2 bits
            break;
        case 2:
            // ksc == 2: lower part is from scales[1] with no shift; upper part is from scales[1] with no shift.
            lower = s1 >> (4 * 0);   // no shift (since (2 & 1) == 0)
            upper = s1 >> (2 * 0);   // no shift
            break;
        case 3:
            // ksc == 3: lower part from scales[2] shifted by 4 bits; upper part from scales[1] shifted by 2 bits.
            lower = s2 >> (4 * 1);   // shift by 4 bits (since (3 & 1)==1)
            upper = s1 >> (2 * 1);   // shift by 2 bits
            break;
        default:
            break; // Should not occur.
    }
    // Merge the lower 4-bit portions and upper 2-bit portions as in the original:
    return (lower & 0x0F0F0F0F) | (upper & 0x30303030);
}

// explicit extraction of the desired 8-bit value (using bit shifts) from the 32-bit integer.
static __device__ __forceinline__ half2 unpack_scales_shift(const int4& scales, const int idx) {
    // The constant ds is computed from the first 32 bits of scales,
    // then we apply a sign flip (multiplying by (1.0, -1.0)).
    half2 ds = reinterpret_cast<const half2*>(&scales.x)[0] * make_half2(1.0f, -1.0f);

    // The remaining scales (from scales.y) are treated as an array of ints.
    // Note: the original code uses reinterpret_cast to treat &scales.y as int*.
    const int* sc = reinterpret_cast<const int*>(&scales.y);

    // Determine which "block" of scales to unpack.
    const int k_idx = idx / 4;
    // s_idx selects the 8-bit element from the 32-bit result.
    const int s_idx = idx % 4;
    // Use our refactored function for unpacking the two parts:
    int sc32 = unpack_scales_q45_K_shift(sc, k_idx);
    int m32  = unpack_scales_q45_K_shift(sc, k_idx + 2);

    // Instead of dynamically indexing by reinterpretting the pointer as int8_t*,
    // extract the desired 8-bit value via a bit-shift:
    int sc_val   = (sc32 >> (8 * s_idx)) & 0xFF;
    int msc_val  = (m32 >> (8 * s_idx)) & 0xFF;

    // Multiply by ds. If necessary, cast the integer values to __half.
    // (Assumes that an implicit conversion exists or you use __half(sc_val) if needed.)
    return ds * make_half2(__half(sc_val), __half(msc_val));
}


// copied and adapted from https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/mmvq.cu
template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static __global__ void moe_vec_q(const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
        const int* topk_ids, const int topk, const int ncols, const int nrows, const int token_stride) {
    const auto row = blockIdx.x*blockDim.y + threadIdx.y;

    const auto token = blockIdx.z / topk;
    // const auto expert = reinterpret_cast<const int64_t*>(topk_ids)[blockIdx.z];
    const auto expert = (topk_ids)[blockIdx.z];
    if (expert >= 256 || expert < 0)
        return;

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;

// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = ((const block_q_t  *) vx) + expert * nrows * blocks_per_row;
    const block_q8_1 * y = (const block_q8_1 *) (((const int*)vy) + token * token_stride);

    // const block_q_t  * x = (const block_q_t  *) (vx);
    // const block_q8_1 * y = (const block_q8_1 *) (vy);

    for (auto i = threadIdx.x / (qi/vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index


        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs  = vdr * (threadIdx.x % (qi/vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
        // if (threadIdx.x == 0 && row < 1){
        // printf("block %d, thread %d, adding %f, to %d, %d, %d, token %d, expert %d, bpr %d, off %d\n",
        //         blockIdx.x, threadIdx.y, tmp, blockIdx.z, nrows, row, token, (int)expert, blocks_per_row, (char*)y-(char*)vy);
        // }
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1) {
        tmp += VLLM_SHFL_XOR_SYNC(tmp, mask);
    }

    if (threadIdx.x == 0) {
        dst[blockIdx.z * nrows + row] = tmp;
        // if (row < 1){
        // printf("block %d, thread %d, saving %f, to %d, %d, %d, token %d, expert %d, bpr %d, off %d\n",
        //         blockIdx.x, threadIdx.y, tmp, blockIdx.z, nrows, row, token, (int)expert, blocks_per_row, (char*)y-(char*)vy);
        // }
    }
}


template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static __global__ void moe_vec_q_up(const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
        const int* topk_ids, const int topk, const int ncols, const int nrows, const int token_stride) {
    const auto row = blockIdx.x*blockDim.y + threadIdx.y;

    const auto token = blockIdx.z / topk;
    // const auto expert = reinterpret_cast<const int64_t*>(topk_ids)[blockIdx.z];
    const auto expert = (topk_ids)[blockIdx.z];
    if (expert >= 256 || expert < 0)
        return;

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;

// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = ((const block_q_t  *) vx) + expert * nrows * blocks_per_row;
    const block_q8_1 * y = (const block_q8_1 *) (((const int*)vy) + token * token_stride);
    __shared__ block_q8_1 sblock[8 * 28];
    for (int i = threadIdx.y*blockDim.x + threadIdx.x; i<8*28*sizeof(block_q8_1)/sizeof(int4); i+=blockDim.x*blockDim.y)
    {
        reinterpret_cast<int4*>(sblock)[i] = reinterpret_cast<const int4*>(y)[i];
    }
    __syncthreads();


    for (auto i = threadIdx.x / (qi/vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index


        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs  = vdr * (threadIdx.x % (qi/vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_q_cuda(&x[ibx], &sblock[iby], iqs);
        // if (threadIdx.x == 0 && row < 1){
        // printf("block %d, thread %d, adding %f, to %d, %d, %d, token %d, expert %d, bpr %d, off %d\n",
        //         blockIdx.x, threadIdx.y, tmp, blockIdx.z, nrows, row, token, (int)expert, blocks_per_row, (char*)y-(char*)vy);
        // }
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1) {
        tmp += VLLM_SHFL_XOR_SYNC(tmp, mask);
    }

    if (threadIdx.x == 0) {
        dst[blockIdx.z * nrows + row] = tmp;
        // if (row < 1){
        // printf("block %d, thread %d, saving %f, to %d, %d, %d, token %d, expert %d, bpr %d, off %d\n",
        //         blockIdx.x, threadIdx.y, tmp, blockIdx.z, nrows, row, token, (int)expert, blocks_per_row, (char*)y-(char*)vy);
        // }
    }
}

// static __device__ __forceinline__ int unpack_scales_q45_K(const int * scales, const int ksc) {
//     // scale arrangement after the following two lines:
//     //   - ksc == 0: sc0, sc1, sc2, sc3
//     //   - ksc == 1: sc4, sc5, sc6, sc7
//     //   - ksc == 2:  m0,  m1,  m2,  m3
//     //   - ksc == 3:  m4,  m5,  m6,  m7
//     return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) | // lower 4 bits
//            ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);  // upper 2 bits
// }

// static __device__ __forceinline__ half2 unpack_scales(const int4& scales, const int idx)
// {
//     half2 ds = reinterpret_cast<const half2*>(&scales.x)[0] * make_half2(1.0f, -1.0f);
//     const int* sc = reinterpret_cast<const int*>(&scales.y);
//     const int k_idx = idx/4;
//     const int s_idx = idx%4;
//     int sc32 = unpack_scales_q45_K(sc, k_idx);
//     int m32 = unpack_scales_q45_K(sc, k_idx+2);
//     return ds * make_half2(
//             reinterpret_cast<const int8_t*>(&sc32)[s_idx],
//             reinterpret_cast<const int8_t*>(&m32)[s_idx]
//             );
// }

template<typename scalar_t, int qk>
static __device__ __forceinline__ float vec_dot(const int4& scales, const int& packed, int4& b_vals, const int i)
{
    float tmp = 0.f;

        int2 quants;
        quants.x = packed & 0x0F0F0F0F;
        quants.y = (packed>>4) & 0x0F0F0F0F;

        half2 unpacked_scales[2];
        unpacked_scales[0] = unpack_scales_shift(scales, (threadIdx.x/8)*2);
        unpacked_scales[1] = unpack_scales_shift(scales, (threadIdx.x/8)*2 + 1);

        //thread 0 gets values 0-8 thread 1 gets 8-16 etc..
        //so we need to register shuffle to get the correct values
        //if we want to utilize 16 bit loats
        int4 shfl;

        int i_idx =(threadIdx.x/8) * 64 + 4 * (threadIdx.x%8);

            // if(row == 0 && blockIdx.z == 0 && i < 1)
            // {
            //     printf("thread %d, %d, has vals %f, %f, %f, %f, %f, %f, %f, %f, \n",
            //             threadIdx.x, i_idx,
            //             (float)reinterpret_cast<const scalar_t*>(&b_vals)[0],
            //             (float)reinterpret_cast<const scalar_t*>(&b_vals)[1],
            //             (float)reinterpret_cast<const scalar_t*>(&b_vals)[2],
            //             (float)reinterpret_cast<const scalar_t*>(&b_vals)[3],
            //             (float)reinterpret_cast<const scalar_t*>(&b_vals)[4],
            //             (float)reinterpret_cast<const scalar_t*>(&b_vals)[5],
            //             (float)reinterpret_cast<const scalar_t*>(&b_vals)[6],
            //             (float)reinterpret_cast<const scalar_t*>(&b_vals)[7]
            //             );
            // }

        shfl.x = __shfl_sync(0xFFFFFFFF, b_vals.x, i_idx/8);
        shfl.y = __shfl_sync(0xFFFFFFFF, b_vals.y, i_idx/8);
        shfl.z = __shfl_sync(0xFFFFFFFF, b_vals.z, i_idx/8);
        shfl.w = __shfl_sync(0xFFFFFFFF, b_vals.w, i_idx/8);
        int2 tmp2;
        if(threadIdx.x%2==0)
        {
            tmp2.x = shfl.x;
            tmp2.y = shfl.y;
        }
        else
        {
            tmp2.x = shfl.z;
            tmp2.y = shfl.w;
        }
        i_idx += 32;
            // if(row == 0 && blockIdx.z == 0 && i < 1)
            // {
            //     printf("thread %d, %d, has vals %f, %f, %f, %f, %f, %f, %f, %f, \n",
            //             threadIdx.x, i_idx,
            //             (float)reinterpret_cast<const scalar_t*>(&b_vals)[0],
            //             (float)reinterpret_cast<const scalar_t*>(&b_vals)[1],
            //             (float)reinterpret_cast<const scalar_t*>(&b_vals)[2],
            //             (float)reinterpret_cast<const scalar_t*>(&b_vals)[3],
            //             (float)reinterpret_cast<const scalar_t*>(&b_vals)[4],
            //             (float)reinterpret_cast<const scalar_t*>(&b_vals)[5],
            //             (float)reinterpret_cast<const scalar_t*>(&b_vals)[6],
            //             (float)reinterpret_cast<const scalar_t*>(&b_vals)[7]
            //             );
            // }
        shfl.x = __shfl_sync(0xFFFFFFFF, b_vals.x, i_idx/8);
        shfl.y = __shfl_sync(0xFFFFFFFF, b_vals.y, i_idx/8);
        shfl.z = __shfl_sync(0xFFFFFFFF, b_vals.z, i_idx/8);
        shfl.w = __shfl_sync(0xFFFFFFFF, b_vals.w, i_idx/8);
        b_vals.x = tmp2.x;
        b_vals.y = tmp2.y;
        if(threadIdx.x%2==1)
        {
            b_vals.z = shfl.z;
            b_vals.w = shfl.w;
        }
        else
        {
            b_vals.z = shfl.x;
            b_vals.w = shfl.y;
        }



        const scalar_t* harr = reinterpret_cast<const scalar_t*>(&b_vals);
        const char* a = reinterpret_cast<const char*>(&quants);

            // if(row == 0 && blockIdx.z == 0 && i < 1)
            // {
            //     printf("thread %d, has vals %f, %f, %f, %f, %f, %f, %f, %f, \n",
            //             threadIdx.x, 
            //             (float)reinterpret_cast<const scalar_t*>(&b_vals)[0],
            //             (float)reinterpret_cast<const scalar_t*>(&b_vals)[1],
            //             (float)reinterpret_cast<const scalar_t*>(&b_vals)[2],
            //             (float)reinterpret_cast<const scalar_t*>(&b_vals)[3],
            //             (float)reinterpret_cast<const scalar_t*>(&b_vals)[4],
            //             (float)reinterpret_cast<const scalar_t*>(&b_vals)[5],
            //             (float)reinterpret_cast<const scalar_t*>(&b_vals)[6],
            //             (float)reinterpret_cast<const scalar_t*>(&b_vals)[7]
            //             );
            // }

        for (int j = 0; j<8; j++)
        {
            half2 dsx = unpacked_scales[j/4];
            float dequant = a[j] * (float)dsx.x + (float)dsx.y;
            tmp += dequant * (float)harr[j];
            // if(row == 0 && blockIdx.z == 0 && i < 4)
            // {
            //     int offset = ((const char*)x - (const char*)vx);
            //     printf("thread %d, doing mul %f * %f, from value at idx %d, base %p, curr %p\n", 
            //             threadIdx.x, (float)harr[j], dequant, j, vx, x);
            // }
        }
        return tmp;
}


template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda, bool check>
static __global__ void moe_vec_q4_s(const void * __restrict__ vx, const scalar_t * __restrict__ vy, scalar_t * __restrict__ dst,
        const int* topk_ids, const int topk, const int ncols, const int nrows, const int token_stride) {
    const auto row = blockIdx.x*blockDim.y + threadIdx.y;

    const auto token = blockIdx.z / topk;
    // const auto expert = reinterpret_cast<const int64_t*>(topk_ids)[blockIdx.z];
    const auto expert = (topk_ids)[blockIdx.z];
    if (expert >= 256 || expert < 0)
        return;

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;

// partial sum for each thread
    float tmp = 0.0f;

    const block_q4_K  * x = ((const block_q4_K*) vx) + expert * nrows * blocks_per_row + row * blocks_per_row;
    const scalar_t * y_eff = vy + token*token_stride;
    if constexpr (check)
    {
    for (auto i = 0; i < blocks_per_row; i += 2) {

        const int4 scales = reinterpret_cast<const int4*>(x+i)[0];
        const int packed = ((const int*)((x + i)->qs))[threadIdx.x];
        int4 b_vals = reinterpret_cast<const int4*>(&y_eff[(i+0) * qk])[threadIdx.x];
        const int4 scales2 = reinterpret_cast<const int4*>(x+i+1)[0];
        const int packed2 = ((const int*)((x + i + 1)->qs))[threadIdx.x];
        int4 b_vals2 = reinterpret_cast<const int4*>(&y_eff[(i+1) * qk])[threadIdx.x];

        tmp += vec_dot<scalar_t, qk>(scales, packed, b_vals, i);
        tmp += vec_dot<scalar_t, qk>(scales2, packed2, b_vals2, i + 1);
    }
    }
    else{
    for (auto i = 0; i < blocks_per_row; i += 1) {

        const int4 scales = reinterpret_cast<const int4*>(x+i)[0];
        const int packed = ((const int*)((x + i)->qs))[threadIdx.x];
        int4 b_vals = reinterpret_cast<const int4*>(&y_eff[(i+0) * qk])[threadIdx.x];

        tmp += vec_dot<scalar_t, qk>(scales, packed, b_vals, i);
    }
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1) {
        tmp += VLLM_SHFL_XOR_SYNC(tmp, mask);
    }

    if (threadIdx.x == 0) {
        dst[blockIdx.z * nrows + row] = tmp;
    }
}

template<typename scalar_t>
static void moe_vec_q4_K_s_cuda(const void * vx, const scalar_t * vy, scalar_t * dst, 
        const int* topk_ids, const int top_k, const int tokens,
        const int ncols, const int nrows, const int token_stride, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, tokens*top_k);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    if (top_k == 1)
    {
    moe_vec_q4_s<scalar_t, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1,false>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
    }
    else{
    moe_vec_q4_s<scalar_t, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1,true>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
    }
}

template<typename scalar_t>
static void moe_vec_q4_0_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, 
        const int* topk_ids, const int top_k, const int tokens,
        const int ncols, const int nrows, const int token_stride, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, tokens*top_k);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    moe_vec_q<scalar_t, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template<typename scalar_t>
static void moe_vec_q4_1_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, 
        const int* topk_ids, const int top_k, const int tokens,
        const int ncols, const int nrows, const int token_stride, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, tokens*top_k);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    moe_vec_q<scalar_t, QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template<typename scalar_t>
static void moe_vec_q5_0_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, 
        const int* topk_ids, const int top_k, const int tokens,
        const int ncols, const int nrows, const int token_stride, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, tokens*top_k);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    moe_vec_q<scalar_t, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template<typename scalar_t>
static void moe_vec_q5_1_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, 
        const int* topk_ids, const int top_k, const int tokens,
        const int ncols, const int nrows, const int token_stride, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, tokens*top_k);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    moe_vec_q<scalar_t, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template<typename scalar_t>
static void moe_vec_q8_0_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, 
        const int* topk_ids, const int top_k, const int tokens,
        const int ncols, const int nrows, const int token_stride, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, tokens*top_k);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    moe_vec_q<scalar_t, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template<typename scalar_t>
static void moe_vec_q2_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, 
        const int* topk_ids, const int top_k, const int tokens,
        const int ncols, const int nrows, const int token_stride, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, tokens*top_k);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    moe_vec_q<scalar_t, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template<typename scalar_t>
static void moe_vec_q3_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, 
        const int* topk_ids, const int top_k, const int tokens,
        const int ncols, const int nrows, const int token_stride, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, tokens*top_k);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    moe_vec_q<scalar_t, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template<typename scalar_t>
static void moe_vec_q4_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, 
        const int* topk_ids, const int top_k, const int tokens,
        const int ncols, const int nrows, const int token_stride, cudaStream_t stream) {
    if(top_k == 1)
    {
        const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
        const dim3 block_nums(block_num_y, 1, tokens*top_k);
        const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
        moe_vec_q<scalar_t, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
    }
    else
    {
        constexpr int GGML_CUDA_MMV_Y2 = 8;
        const int block_num_y = (nrows + GGML_CUDA_MMV_Y2 - 1) / GGML_CUDA_MMV_Y2;
        const dim3 block_nums(block_num_y, 1, tokens*top_k);
        const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y2, 1);
        moe_vec_q_up<scalar_t, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
    }
    // {
    //     constexpr int GGML_CUDA_MMV_Y2 = 16;
    //     const int block_num_y = (nrows + GGML_CUDA_MMV_Y2 - 1) / GGML_CUDA_MMV_Y2;
    //     const dim3 block_nums(block_num_y, 1, tokens*top_k);
    //     const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y2, 1);
    //     moe_vec_q_up<scalar_t, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
    //         <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
    // }{
    //     constexpr int GGML_CUDA_MMV_Y2 = 32;
    //     const int block_num_y = (nrows + GGML_CUDA_MMV_Y2 - 1) / GGML_CUDA_MMV_Y2;
    //     const dim3 block_nums(block_num_y, 1, tokens*top_k);
    //     const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y2, 1);
    //     moe_vec_q_up<scalar_t, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
    //         <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
    // }
}


template<typename scalar_t>
static void moe_vec_q5_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, 
        const int* topk_ids, const int top_k, const int tokens,
        const int ncols, const int nrows, const int token_stride, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, tokens*top_k);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    moe_vec_q<scalar_t, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template<typename scalar_t>
static void moe_vec_q6_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, 
        const int* topk_ids, const int top_k, const int tokens,
        const int ncols, const int nrows, const int token_stride, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, tokens*top_k);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    moe_vec_q<scalar_t, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template<typename scalar_t>
static void moe_vec_iq2_xxs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, 
        const int* topk_ids, const int top_k, const int tokens,
        const int ncols, const int nrows, const int token_stride, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, tokens*top_k);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    moe_vec_q<scalar_t, QK_K, QI2_XXS, block_iq2_xxs, 1, vec_dot_iq2_xxs_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template<typename scalar_t>
static void moe_vec_iq2_xs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, 
        const int* topk_ids, const int top_k, const int tokens,
        const int ncols, const int nrows, const int token_stride, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, tokens*top_k);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    moe_vec_q<scalar_t, QK_K, QI2_XS, block_iq2_xs, 1, vec_dot_iq2_xs_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template<typename scalar_t>
static void moe_vec_iq2_s_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, 
        const int* topk_ids, const int top_k, const int tokens,
        const int ncols, const int nrows, const int token_stride, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, tokens*top_k);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    moe_vec_q<scalar_t, QK_K, QI2_S, block_iq2_s, 1, vec_dot_iq2_s_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template<typename scalar_t>
static void moe_vec_iq3_xxs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, 
        const int* topk_ids, const int top_k, const int tokens,
        const int ncols, const int nrows, const int token_stride, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, tokens*top_k);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    moe_vec_q<scalar_t, QK_K, QI3_XXS, block_iq3_xxs, 1, vec_dot_iq3_xxs_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template<typename scalar_t>
static void moe_vec_iq1_s_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, 
        const int* topk_ids, const int top_k, const int tokens,
        const int ncols, const int nrows, const int token_stride, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, tokens*top_k);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    moe_vec_q<scalar_t, QK_K, QI1_S, block_iq1_s, 1, vec_dot_iq1_s_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template<typename scalar_t>
static void moe_vec_iq1_m_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, 
        const int* topk_ids, const int top_k, const int tokens,
        const int ncols, const int nrows, const int token_stride, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, tokens*top_k);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    moe_vec_q<scalar_t, QK_K, QI1_M, block_iq1_m, 1, vec_dot_iq1_m_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template<typename scalar_t>
static void moe_vec_iq4_nl_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, 
        const int* topk_ids, const int top_k, const int tokens,
        const int ncols, const int nrows, const int token_stride, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, tokens*top_k);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    moe_vec_q<scalar_t, QK4_NL, QI4_NL, block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template<typename scalar_t>
static void moe_vec_iq4_xs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, 
        const int* topk_ids, const int top_k, const int tokens,
        const int ncols, const int nrows, const int token_stride, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, tokens*top_k);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    moe_vec_q<scalar_t, QK_K, QI4_XS, block_iq4_xs, 1, vec_dot_iq4_xs_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template<typename scalar_t>
static void moe_vec_iq3_s_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, 
        const int* topk_ids, const int top_k, const int tokens,
        const int ncols, const int nrows, const int token_stride, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, tokens*top_k);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    moe_vec_q<scalar_t, QK_K, QI3_XS, block_iq3_s, 1, vec_dot_iq3_s_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}
