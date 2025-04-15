
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
    //
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

template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
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

    for (auto i = 0; i < blocks_per_row; i ++) {
        // half2 dm = x[i]->dm;
        const int4 scales = reinterpret_cast<const int4*>(&x[i])[0];
        const int packed = ((const int*)((x + i)->qs))[threadIdx.x];

        const int q1 = packed & 0x0F0F0F0F;
        const int q2 = (packed>>4) & 0x0F0F0F0F;

        //thread 0 gets values 0-8 thread 1 gets 8-16 etc..
        //so we need to register shuffle to get the correct values
        //if we want to utilize 16 bit loats
        const int4 b_vals = reinterpret_cast<const int4*>(&y_eff[i * qk])[threadIdx.x];
        const half* harr = reinterpret_cast<const half*>(&b_vals);

        half2 dsx = unpack_scales(scales, (threadIdx.x/8)*2);

        const char* a = reinterpret_cast<const char*>(&q1);

            // if(row == 0 && blockIdx.z == 0)
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

        for (int j = 0; j<4; j++)
        {
            int i_idx =(threadIdx.x/8) * 64 + 4 * (threadIdx.x%8) + j;
            int warp_idx = i_idx / 8;
            int arr_idx = i_idx % 8;
                
            // TODO why the gymnastics needed
            half arr[8];
            for (int idx = 0; idx<8; idx++)
                arr[idx] = __shfl_sync(0xFFFFFFFF, harr[idx], warp_idx);

            const float y_val = (float)(*reinterpret_cast<const scalar_t*>(&arr[arr_idx]));


            float dequant = (float)a[j] * (float)dsx.x + (float)dsx.y;
            // if(row == 0 && blockIdx.z == 0 && i < 4)
            // {
            //     int offset = ((const char*)x - (const char*)vx);
            //     printf("thread %d, doing mul %f * %f, from value at idx %d, warp %d, arr %d, exp %d, offset %d, base %p, curr %p\n", 
            //             threadIdx.x, y_val, dequant, i_idx, warp_idx, arr_idx, expert, offset, vx, x);
            // }
            tmp += dequant * y_val;
        }

        dsx = unpack_scales(scales, (threadIdx.x/8)*2 + 1);
        a = reinterpret_cast<const char*>(&q2);

        for (int j = 0; j<4; j++)
        {
            int i_idx = 32 + (threadIdx.x/8) * 64 + 4 * (threadIdx.x%8) + j;
            int warp_idx = i_idx / 8;
            int arr_idx = i_idx % 8;
            // TODO why the gymnastics needed
            half arr[8];
            for (int idx = 0; idx<8; idx++)
                arr[idx] = __shfl_sync(0xFFFFFFFF, harr[idx], warp_idx);
            const float y_val = (float)(*reinterpret_cast<const scalar_t*>(&arr[arr_idx]));


            float dequant = (float)a[j] * (float)dsx.x + (float)dsx.y;
            // if(row == 0 && blockIdx.z == 0 && i < 4)
            // {
            //     int offset = ((const char*)x - (const char*)vx);
            //     printf("thread %d, doing mul %f * %f, from value at idx %d, warp %d, arr %d, exp %d, offset %d, base %p, curr %p\n", 
            //             threadIdx.x, y_val, dequant, i_idx, warp_idx, arr_idx, expert, offset, vx, x);
            // }
            tmp += dequant * y_val;
        }
            // if(row == 0 && blockIdx.z == 0)
            // {
            //     printf("end thread %d, has vals %f, %f, %f, %f, %f, %f, %f, %f, \n",
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
    moe_vec_q4_s<scalar_t, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
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
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, tokens*top_k);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    moe_vec_q<scalar_t, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
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
