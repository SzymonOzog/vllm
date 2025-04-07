#include <cstdint>
#include "mma.cuh"

#define FAST_MMA 1
static __device__ __forceinline__ half2 unpack_scales(const int4& scales, const int idx)
{
    half2 ds = reinterpret_cast<const half2*>(&scales.x)[0] * make_half2(1.0f, -1.0f);
    const int* sc = reinterpret_cast<const int*>(&scales.y);
    const int k_idx = idx/4;
    const int s_idx = idx%4;
    int sc32 = unpack_scales_q45_K(sc, k_idx);
    int m32 = unpack_scales_q45_K(sc, k_idx+2);
    return ds * make_half2(
            reinterpret_cast<const int8_t*>(&sc32)[s_idx],
            reinterpret_cast<const int8_t*>(&m32)[s_idx]
            );
}

/* Adapted from ./csrc/quantization/gguf/mmq.cuh
   based on ./vllm/model_executor/layers/fused_moe/fused_moe.py */
template <typename scalar_t, int qk, int qr, int qi, bool need_sum,
         typename block_q_t, int mmq_x, int mmq_y, int nwarps, int need_check,
         int vdr, vec_dot_q_mul_mat_cuda_t vec_dot>
         static __device__ __forceinline__ void moe_q_new(
                 const int* __restrict__ vx_qs, 
                 const half2* __restrict__ vx_ds, 
                 const void* __restrict__ vy,
                 scalar_t* __restrict__ dst, const int* __restrict__ sorted_token_ids,
                 const int* __restrict__ expert_ids,
                 const int* __restrict__ num_tokens_post_padded, 
                 const int exp_stride_qs, const int exp_stride_ds,
                 const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y,
                 const int nrows_dst, const int top_k) {

             const int blocks_per_row_x = ncols_x / qk;
             const int blocks_per_col_y = nrows_y / QK8_1;
             const int blocks_per_warp = WARP_SIZE_GGUF / qi;
             const int lane_id = threadIdx.x%32;

             const int ncols_dst = ncols_y * top_k;

             const auto row_dst_0 = blockIdx.x * mmq_y;// + threadIdx.y * 16;
             const int& row_x_0 = row_dst_0;

             const auto col_dst_0 = blockIdx.y * mmq_x;
             const int2 col_dst = reinterpret_cast<const int2*>(&sorted_token_ids[col_dst_0])[lane_id%4];

             // int token_offs[mmq_x / nwarps];
             // for (int i = 0; i < mmq_x; i += nwarps) {
             //     token_offs[i / nwarps] = sorted_token_ids[col_dst_0 + threadIdx.y + i];
             // }

             const int exp_idx = expert_ids[blockIdx.y];
             if (exp_idx > 255 || exp_idx < 0) return;
             if (blockIdx.y * mmq_x > num_tokens_post_padded[0]) return;

             const int* x_qs = (vx_qs + exp_idx * exp_stride_qs);
             const int4* x_ds = reinterpret_cast<const int4*>(vx_ds + exp_idx * exp_stride_ds);

             const block_q8_1* y = (const block_q8_1*)(vy);

             __shared__ int   tile_x_ql[(mmq_y * (WARP_SIZE_GGUF)  + 4 * mmq_y)];
             // __shared__ half2 tile_x_dm[mmq_y * (8) + mmq_y];
             __shared__ int4 tile_x_dm[mmq_y];

             __shared__ int tile_y_qs[mmq_x * WARP_SIZE_GGUF];
             __shared__ half2 tile_y_ds[mmq_x * WARP_SIZE_GGUF / QI8_1];

             float sum[4] = {0.0f};

             for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {

                 for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
                     int i = row_x_0 + i0 + threadIdx.y * 4 + threadIdx.x/8;

                     if (need_check) {
                         i = min(i, nrows_x - 1);
                     }
                     const int* qs = x_qs + i*blocks_per_row_x*QI4_K + ib0*QI4_K;
                     reinterpret_cast<int4 *>(&tile_x_ql[(i-row_x_0) * (WARP_SIZE_GGUF + 4)])[threadIdx.x % 8] = 
                         reinterpret_cast<const int4*>(qs)[threadIdx.x%8];
                 }
                 for (int i0 = threadIdx.y*blockDim.x + threadIdx.x; i0 < mmq_y; i0 += nwarps * 32) {
                     int i = row_x_0 + i0;// + threadIdx.y * 4 + threadIdx.x/8;
                     // int off = threadIdx.x % 8;
                     if (need_check) {
                         i = min(i, nrows_x - 1);
                     }
                     const int4* ds = x_ds + i * blocks_per_row_x + ib0;
                     // if(blockIdx.x == 0 && blockIdx.y == 0)
                     //     printf("loading scales from %d, %d, to %d, %d, ptr %p\n", i, ib0, i0, 0, ds);
                     tile_x_dm[i0] = ds[0];
                     // tile_x_dm[(i-row_x_0)*9 + off] = ds[off];
                 }
                 __syncthreads();
                 // if(lane_id/4 == 0 && blockIdx.x == 0 && (col_dst.x == 0 || col_dst.y==0))
                 // {
                 //     printf("------------\n");
                 //     for (int t = 0; t<mmq_y * (WARP_SIZE_GGUF) + 4 * mmq_y; t++)
                 //     {
                 //         printf("%#010x, ",tile_x_ql[t]);
                 //         if((t+1)%(32 + 4) == 0)
                 //             printf("\n");
                 //     }
                 //     printf("------------\n");
                 // }
                 // if(lane_id/4 == 0 && blockIdx.x == 0 && (col_dst.x == 0 || col_dst.y==0))
                 // {
                 //     printf("------------\n");
                 //     for (int t = 0; t<mmq_y * (8) + mmq_y; t++)
                 //     {
                 //         printf("%f/%f, ",(float)tile_x_dm[t].x, (float)tile_x_dm[t].y);
                 //         if((t+1)%9 == 0)
                 //             printf("\n");
                 //     }
                 //     printf("------------\n");
                 // }

                 const int n_per_r = ((qk * blocks_per_warp) / (qr));
#pragma unroll
                 for (int ir = 0; ir < qr && ib0 * qk + ir * n_per_r < ncols_x; ++ir) {
                     const auto kqs = ir * WARP_SIZE_GGUF + threadIdx.x;
                     const int kbxd = kqs / QI8_1;
                     const auto r0 = ir*WARP_SIZE_GGUF;
                     const auto c0 = threadIdx.y * 4 + threadIdx.x/8;
                     // const auto r = ir*WARP_SIZE_GGUF + threadIdx.x;

#pragma unroll
                     for (int i = 0; i < mmq_x; i += nwarps*4) {
                         // const int col_y_eff = token_offs[i / nwarps] / top_k;
                         const int col_y_eff = sorted_token_ids[col_dst_0 + c0];
                         const int block_x = ib0 * (qk / QK8_1) + r0 + threadIdx.x/(QK8_1/4);
                         if (col_y_eff < ncols_y && block_x < blocks_per_col_y) {
                             const block_q8_1* by0 = &y[col_y_eff * blocks_per_col_y + block_x];
                             // const int index_y =
                             //     (threadIdx.y + i) * WARP_SIZE_GGUF + kqs % WARP_SIZE_GGUF;
                             reinterpret_cast<int4*>(&tile_y_qs[c0 * WARP_SIZE_GGUF])[threadIdx.x%(QK8_1/4)] =
                                 reinterpret_cast<const int4*>(by0->qs)[threadIdx.x%(QK8_1/4)];
                                 // get_int_from_int8_aligned(by0->qs, threadIdx.x % QI8_1);
                         // if(blockIdx.x == 0 && blockIdx.y == 0)
                         // {
                         //     printf("loading val %010x to %d, %d from %d, %d, sv = %d, %d\n", tile_y_qs[index_y], (threadIdx.y + i), kqs%WARP_SIZE_GGUF, col_y_eff, threadIdx.x%QI8_1, col_dst.x, col_dst.y);
                         // }
                         }
                     }

                     if ((lane_id>>2) < n_per_r / QK8_1) {
                         const auto kby = (lane_id>>2) % (WARP_SIZE_GGUF / QI8_1);
                         {
                             const int col_y_eff = col_dst.x / top_k;
                             const int block_x =
                                 ib0 * (qk / QK8_1) + ir * (WARP_SIZE_GGUF / QI8_1) + kby;

                             if (col_y_eff < ncols_y && block_x < blocks_per_col_y) {
                                 const half2* dsi_src = &y[col_y_eff * blocks_per_col_y + block_x].ds;
                                 half2* dsi_dst =
                                     &tile_y_ds[(lane_id%4)*2 * (WARP_SIZE_GGUF / QI8_1) + kby];

                                 if (need_sum) {
                                     *dsi_dst = *dsi_src;
                                 } else {
                                     float* dfi_dst = (float*)dsi_dst;
                                     *dfi_dst = __low2float(*dsi_src);
                                 }
                             }
                         }
                         {
                             const int col_y_eff = col_dst.y / top_k;
                             const int block_x =
                                 ib0 * (qk / QK8_1) + ir * (WARP_SIZE_GGUF / QI8_1) + kby;

                             if (col_y_eff < ncols_y && block_x < blocks_per_col_y) {
                                 const half2* dsi_src = &y[col_y_eff * blocks_per_col_y + block_x].ds;
                                 half2* dsi_dst =
                                     &tile_y_ds[((lane_id%4)*2 + 1) * (WARP_SIZE_GGUF / QI8_1) + kby];

                                 if (need_sum) {
                                     *dsi_dst = *dsi_src;
                                 } else {
                                     float* dfi_dst = (float*)dsi_dst;
                                     *dfi_dst = __low2float(*dsi_src);
                                 }
                             }
                         }
                     }
                     __syncthreads();
                 // if(threadIdx.x ==0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
                 //     {
                 //         printf("-----Y, %d, %d-------\n", threadIdx.x, threadIdx.y);
                 //         for (int t = 0; t<mmq_x * WARP_SIZE_GGUF; t++)
                 //         {
                 //             printf("%010x, ", tile_y_qs[t]);
                 //             if((t+1)%(WARP_SIZE_GGUF) == 0)
                 //                 printf("\n");
                 //         }
                 //         printf("------Y------\n");
                 //     }
                 // if(threadIdx.x ==0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 22)
                 //     {
                 //         printf("-----Y-------\n");
                 //         for (int t = 0; t<mmq_x * WARP_SIZE_GGUF / QI8_1; t++)
                 //         {
                 //             printf("%f/%f, ",(float)tile_y_ds[t].x, (float)tile_y_ds[t].y);
                 //             if((t+1)%(WARP_SIZE_GGUF / QI8_1) == 0)
                 //                 printf("\n");
                 //         }
                 //         printf("------Y------\n");
                 //     }
                     __syncthreads();
                     tile<16, 8, int> A_tiles[2];
                     tile<8, 8, int> B;

#pragma unroll
                     for (int k = ir * 2 * WARP_SIZE_GGUF / qr; k < (ir + 1) * 2 * WARP_SIZE_GGUF / qr;
                             k += 16) 
                     {
                         int row =  threadIdx.y * 16 + (lane_id>>2);
                         int col = lane_id%4 + k/2;
                         half2 dsx[2][2];


#pragma unroll
                         for (int i = 0; i<4; i++)
                         {
                             int row = threadIdx.y * 16 + (lane_id>>2) + (i%2)*8;
                             int col = lane_id%4 + k/2 + (i/2)*4;
                             int packed = tile_x_ql[row*(WARP_SIZE_GGUF+4) + col];
                             A_tiles[0].x[i] = (packed) & 0x0F0F0F0F; A_tiles[1].x[i] = (packed>>4) & 0x0F0F0F0F;
                         }
                         int4 scales1 = tile_x_dm[row];
                         int4 scales2 = tile_x_dm[row+8];
                         dsx[0][0] = unpack_scales(scales1, k/8);//tile_x_dm[row*(9) + k/8];
                         dsx[0][1] = unpack_scales(scales2, k/8);//tile_x_dm[row*(9) + k/8];
                         dsx[1][0] = unpack_scales(scales1, k/8 + 1);//tile_x_dm[row*(9) + k/8];
                         dsx[1][1] = unpack_scales(scales2, k/8 + 1);//tile_x_dm[row*(9) + k/8];

                         // row-=8;
                         // dsx[1][0] = tile_x_dm[row*(9) + k/8 + 1];
                         // row+=8;
                         // dsx[1][1] = tile_x_dm[row*(9) + k/8 + 1];

#pragma unroll
                         for (int k00 = 0; k00 < 2; k00 ++)
                         {
                             tile<16, 8, int> acc;
                             tile<16, 8, int>& A = A_tiles[k00];
                             row = lane_id>>2;
                             col = lane_id%4 + (k%(2*WARP_SIZE_GGUF/qr)) + k00*8;
                             B.x[0] = tile_y_qs[row*WARP_SIZE_GGUF + col];
                             col+=4;
                             B.x[1] = tile_y_qs[row*WARP_SIZE_GGUF + col];

                             half2 dsy[2];
                             dsy[0] = tile_y_ds[(lane_id%4)*2*(WARP_SIZE_GGUF/QI8_1) + col/8];
                             dsy[1] = tile_y_ds[((lane_id%4)*2 + 1)*(WARP_SIZE_GGUF/QI8_1) + col/8];

                             asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                                     : "+r"(acc.x[0]), "+r"(acc.x[1]), "+r"(acc.x[2]), "+r"(acc.x[3])
                                     : "r"(A.x[0]), "r"(A.x[1]), "r"(A.x[2]), "r"(A.x[3]), "r"(B.x[0]), "r"(B.x[1]));

#pragma unroll
                             for (int i = 0; i<4; i++)
                             {
                                 sum[i] += (float)acc.x[i] * (float)dsy[i%2].x * (float)dsx[k00][i/2].x;
                                 sum[i] += (float)dsy[i%2].y * (float)dsx[k00][i/2].y;
                             }
                             // if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
                             // {
                             //     int i = 0;
                             //     printf("loaded mma A %d/%d, k %d, sum %f, acc %f, a %010x b %010x (%d, %d), scales %f, %f\n", row, col, k, sum[i], (float)acc.x[i],
                             //             A.x[0], B.x[0], row, col, (float)dsx[k00][0].x, (float)dsx[k00][0].y);
                             // }
                         }
                     }


                     __syncthreads();
                 }
             }

             int row_dst = row_dst_0 + (lane_id>>2) + threadIdx.y * 16;
             // if (col_dst.x == 0 || col_dst.y == 0)
             //     // if(threadIdx.x == 1 && threadIdx.y == 0 && blockIdx.x == 0 && (col_dst.x == 0 || col_dst.y==0))
             // {
             //     printf("thread %d, %d(%d,%d), saving %f, %f, %f, %f to %d, %d, %d \n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y,
             //             (float)sum[0], (float)sum[1], (float)sum[2], (float)sum[3],
             //             col_dst.x, col_dst.y, row_dst);
             // }
             if (col_dst.x < ncols_dst)
             {
                 // if(col_dst.x*nrows_dst + row_dst == 0 ||col_dst.x*nrows_dst + row_dst + 8 == 0)
                 // {
                 //     printf("thread %d, %d(%d,%d), zero saving saving %f, %f\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, sum[0], sum[1]);
                 // }
                 dst[col_dst.x*nrows_dst + row_dst] = sum[0];
                 dst[col_dst.x*nrows_dst + row_dst + 8] = sum[2];
             }
             if (col_dst.y < ncols_dst)
             {
                 // if(col_dst.y*nrows_dst + row_dst == 0 ||col_dst.y*nrows_dst + row_dst + 8 == 0)
                 // {
                 //     printf("thread %d, %d(%d,%d), zero saving saving %f, %f\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, sum[0], sum[1]);
                 // }
                 dst[col_dst.y*nrows_dst + row_dst] = sum[1];
                 dst[col_dst.y*nrows_dst + row_dst + 8] = sum[3];
             }
         }


#if defined(USE_ROCM)
#define MOE_X_Q4_K 64
#define MOE_Y_Q4_K 128
#define NWARPS_Q4_K 8
#else
#define MOE_X_Q4_K 8
#define MOE_Y_Q4_K 32
#define NWARPS_Q4_K_MOE 2
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q4_K, 2)
#endif
    moe_q4_K(const int* __restrict__ vx_qs,
            const half2* __restrict__ vx_ds,
            const void* __restrict__ vy, 
            scalar_t* __restrict__ dst, const int* sorted_token_ids,
            const int* expert_ids, const int* num_tokens_post_padded,
            const int exp_stride_qs, const int exp_stride_ds, 
            const int ncols_x, const int nrows_x,
            const int ncols_y, const int nrows_y, const int nrows_dst,
            const int top_k) {
        const int mmq_x = MOE_X_Q4_K;
        const int mmq_y = MOE_Y_Q4_K;
        const int nwarps = NWARPS_Q4_K_MOE;

        moe_q_new<scalar_t, QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps, need_check,
            VDR_Q4_K_Q8_1_MMQ, vec_dot_q4_K_q8_1_mul_mat>(
                    vx_qs, vx_ds, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
                    exp_stride_qs, exp_stride_ds, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
    }

template <typename scalar_t>
static void ggml_moe_q4_K_q8_1_cuda_new(
        const void* inp, const int* w_qs, const half2* w_ds,
        scalar_t* dst, const int* sorted_token_ids,
        const int* expert_ids, const int* num_tokens_post_padded,
        const int exp_stride_qs, const int exp_stride_ds, 
        const int ncols_x, const int nrows_x,
        const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
        const int tokens_post_padded, cudaStream_t stream) {
    const int mmq_x = MOE_X_Q4_K;
    const int mmq_y = MOE_Y_Q4_K;
    const int nwarps = NWARPS_Q4_K_MOE;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (tokens_post_padded) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);
    printf("launching gitd %d,%d|%d,%d\n", block_num_x, block_num_y, WARP_SIZE_GGUF, nwarps);

    if (nrows_x % mmq_y == 0) {
        constexpr bool need_check = false;
        moe_q4_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
                w_qs, w_ds, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
                exp_stride_qs, exp_stride_ds, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
    } else {
        constexpr bool need_check = true;
        moe_q4_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
                w_qs, w_ds, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
                exp_stride_qs, exp_stride_ds, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
    }
}
