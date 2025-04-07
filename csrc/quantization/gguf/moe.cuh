#include <cstdint>
#include "mma.cuh"

#define FAST_MMA 1

/* Adapted from ./csrc/quantization/gguf/mmq.cuh
   based on ./vllm/model_executor/layers/fused_moe/fused_moe.py */
template <typename scalar_t, int qk, int qr, int qi, bool need_sum,
          typename block_q_t, int mmq_x, int mmq_y, int nwarps,
          allocate_tiles_cuda_t allocate_tiles, load_tiles_cuda_t load_tiles,
          int vdr, vec_dot_q_mul_mat_cuda_t vec_dot>
static __device__ __forceinline__ void moe_q(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst, const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids,
    const int* __restrict__ num_tokens_post_padded, const int exp_stride,
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

  int token_offs[mmq_x / nwarps];
  for (int i = 0; i < mmq_x; i += nwarps) {
    token_offs[i / nwarps] = sorted_token_ids[col_dst_0 + threadIdx.y + i];
  }

  const int exp_idx = expert_ids[blockIdx.y];
  if (exp_idx > 255 || exp_idx < 0) return;
  if (blockIdx.y * mmq_x > num_tokens_post_padded[0]) return;

  const block_q_t* x = (const block_q_t*)((char*)vx + exp_idx * exp_stride);
  const block_q8_1* y = (const block_q8_1*)(vy);

  int* tile_x_ql = nullptr;
  half2* tile_x_dm = nullptr;
  int* tile_x_qh = nullptr;
  int* tile_x_sc = nullptr;

  allocate_tiles(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc);

  __shared__ int tile_y_qs[mmq_x * WARP_SIZE_GGUF];
  __shared__ half2 tile_y_ds[mmq_x * WARP_SIZE_GGUF / QI8_1];

#if FAST_MMA
  float sum[4] = {0.0f};
#else
  float sum[mmq_y / WARP_SIZE_GGUF][mmq_x / nwarps] = {{0.0f}};
#endif // FAST_MMA

  for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {
    load_tiles(x + row_x_0 * blocks_per_row_x + ib0, tile_x_ql, tile_x_dm,
               tile_x_qh, tile_x_sc, threadIdx.y, nrows_x - row_x_0 - 1,
               threadIdx.x, blocks_per_row_x);
  // __syncthreads();
  // if(threadIdx.x == 0 && threadIdx.y == 1 && blockIdx.x == 0 && blockIdx.y == 0)
  // {
  //     printf("------------\n");
  //     for (int t = 0; t<mmq_y * (WARP_SIZE_GGUF) + 4*mmq_y; t++)
  //     {
  //         printf("%#010x, ",tile_x_ql[t]);
  //         if((t+1)%(36) == 0)
  //             printf("\n");
  //     }
  //     printf("------------\n");
  // }
  // if(threadIdx.x == 0 && threadIdx.y == 1 && blockIdx.x == 0 && blockIdx.y == 0)
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
  //
    const int n_per_r = ((qk * blocks_per_warp) / (qr));
#pragma unroll
    for (int ir = 0; ir < qr && ib0 * qk + ir * n_per_r < ncols_x; ++ir) {
      const auto kqs = ir * WARP_SIZE_GGUF + threadIdx.x;
      const int kbxd = kqs / QI8_1;

#pragma unroll
      for (int i = 0; i < mmq_x; i += nwarps) {
        const int col_y_eff = token_offs[i / nwarps] / top_k;
        const int block_x = ib0 * (qk / QK8_1) + kbxd;
        if (col_y_eff < ncols_y && block_x < blocks_per_col_y) {
          const block_q8_1* by0 = &y[col_y_eff * blocks_per_col_y + block_x];
          const int index_y =
              (threadIdx.y + i) * WARP_SIZE_GGUF + kqs % WARP_SIZE_GGUF;
          tile_y_qs[index_y] =
              get_int_from_int8_aligned(by0->qs, threadIdx.x % QI8_1);
        }
      }
      
      // if (threadIdx.x < n_per_r / QK8_1) {
#if FAST_MMA
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
#else
      if (threadIdx.x < n_per_r / QK8_1) {
        const auto kby = threadIdx.x % (WARP_SIZE_GGUF / QI8_1);
        const int col_y_eff = token_offs[threadIdx.y] / top_k;
        const int block_x =
            ib0 * (qk / QK8_1) + ir * (WARP_SIZE_GGUF / QI8_1) + kby;

        if (col_y_eff < ncols_y && block_x < blocks_per_col_y) {
          const half2* dsi_src = &y[col_y_eff * blocks_per_col_y + block_x].ds;
          half2* dsi_dst =
              &tile_y_ds[threadIdx.y * (WARP_SIZE_GGUF / QI8_1) + kby];

          if (need_sum) {
            *dsi_dst = *dsi_src;
          } else {
            float* dfi_dst = (float*)dsi_dst;
            *dfi_dst = __low2float(*dsi_src);
          }
        }
#endif
      }
      __syncthreads();
      // if(threadIdx.x == 0 && threadIdx.y == 1 && blockIdx.x == 0 && blockIdx.y == 0)
      // {
      //     printf("-----Y-------\n");
      //     for (int t = 0; t<mmq_x * WARP_SIZE_GGUF / QI8_1; t++)
      //     {
      //         printf("%f/%f, ",(float)tile_y_ds[t].x, (float)tile_y_ds[t].y);
      //         if((t+1)%(WARP_SIZE_GGUF / QI8_1) == 0)
      //             printf("\n");
      //     }
      //     printf("------Y------\n");
      // }

#if FAST_MMA
      tile<16, 8, int> A_tiles[2];
      tile<8, 8, int> B;
      // tile<16, 8, int> acc;

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
              A_tiles[0].x[i] = (packed) & 0x0F0F0F0F; 
              A_tiles[1].x[i] = (packed>>4) & 0x0F0F0F0F;
          }
          dsx[0][0] = tile_x_dm[row*(9) + k/8];
          row+=8;
          dsx[0][1] = tile_x_dm[row*(9) + k/8];

          row-=8;
          dsx[1][0] = tile_x_dm[row*(9) + k/8 + 1];
          row+=8;
          dsx[1][1] = tile_x_dm[row*(9) + k/8 + 1];

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
                 //     printf("loaded og mma A %d/%d, k %d, sum %f, acc %f, a %010x b %010x (%d, %d), scales %f, %f\n", row, col, k, sum[i], (float)acc.x[i],
                 //             A.x[0], B.x[0], row, col, (float)dsx[k00][0].x, (float)dsx[k00][0].y);
                 // }
          }
          // if(threadIdx.x == 0 && threadIdx.y == 1 && blockIdx.x == 0 && blockIdx.y == 0)
          //     printf("\n\n");
      }


#else
      // #pragma unroll // unrolling this loop causes too much register pressure
      for (int k = ir * WARP_SIZE_GGUF / qr; k < (ir + 1) * WARP_SIZE_GGUF / qr;
           k += vdr) {
#pragma unroll
        for (int j = 0; j < mmq_x; j += nwarps) {
#pragma unroll
          for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
            sum[i / WARP_SIZE_GGUF][j / nwarps] +=
                vec_dot(tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc, tile_y_qs,
                        tile_y_ds, threadIdx.x + i, threadIdx.y + j, k);
          }
        }
      }
#endif // FAST_MMA
      __syncthreads();
    }
  }

#if FAST_MMA
    int row_dst = row_dst_0 + (lane_id>>2) + threadIdx.y * 16;
    // if (lane_id == 0 )
    //       if(threadIdx.x == 0 && threadIdx.y == 1 && blockIdx.x == 0 && blockIdx.y == 0)
    // {
    //     printf("thread %d, %d(%d,%d), saving %f to %d, %d, %d \n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, (float)sum[0], col_dst.x, col_dst.y, row_dst);
    // }
    if (col_dst.x < ncols_dst)
    {
        dst[col_dst.x*nrows_dst + row_dst] = sum[0];
        dst[col_dst.x*nrows_dst + row_dst + 8] = sum[2];
    }
    if (col_dst.y < ncols_dst)
    {
        dst[col_dst.y*nrows_dst + row_dst] = sum[1];
        dst[col_dst.y*nrows_dst + row_dst + 8] = sum[3];
    }
#else
#pragma unroll
  for (int j = 0; j < mmq_x; j += nwarps) {
    const int col_dst = token_offs[j / nwarps];
    if (col_dst >= ncols_dst) {
      return;
    }

#pragma unroll
    for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
      const auto row_dst = row_dst_0 + threadIdx.x + i;
      if (row_dst >= nrows_dst) {
        continue;
      }
      dst[col_dst * nrows_dst + row_dst] = sum[i / WARP_SIZE_GGUF][j / nwarps];
    }
  }
#endif // FAST_MMA
}

#if defined(USE_ROCM)
  #define MOE_X_Q4_0 64
  #define MOE_Y_Q4_0 128
  #define NWARPS_Q4_0 8
#else
  #define MOE_X_Q4_0 4
  #define MOE_Y_Q4_0 32
  #define NWARPS_Q4_0 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q4_0, 2)
#endif
    moe_q4_0(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q4_0;
  const int mmq_y = MOE_Y_Q4_0;
  const int nwarps = NWARPS_Q4_0;

  moe_q<scalar_t, QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps,
        allocate_tiles_q4_0<mmq_y>, load_tiles_q4_0<mmq_y, nwarps, need_check>,
        VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q4_0_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  int mmq_x = MOE_X_Q4_0;
  int mmq_y = MOE_Y_Q4_0;
  int nwarps = NWARPS_Q4_0;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q4_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q4_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q4_1 64
  #define MOE_Y_Q4_1 128
  #define NWARPS_Q4_1 8
#else
  #define MOE_X_Q4_1 4
  #define MOE_Y_Q4_1 32
  #define NWARPS_Q4_1 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q4_1, 2)
#endif
    moe_q4_1(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q4_1;
  const int mmq_y = MOE_Y_Q4_1;
  const int nwarps = NWARPS_Q4_1;

  moe_q<scalar_t, QK4_1, QR4_1, QI4_1, true, block_q4_1, mmq_x, mmq_y, nwarps,
        allocate_tiles_q4_1<mmq_y>, load_tiles_q4_1<mmq_y, nwarps, need_check>,
        VDR_Q4_1_Q8_1_MMQ, vec_dot_q4_1_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q4_1_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  int mmq_x = MOE_X_Q4_1;
  int mmq_y = MOE_Y_Q4_1;
  int nwarps = NWARPS_Q4_1;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q4_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q4_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q5_0 64
  #define MOE_Y_Q5_0 128
  #define NWARPS_Q5_0 8
#else
  #define MOE_X_Q5_0 4
  #define MOE_Y_Q5_0 32
  #define NWARPS_Q5_0 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q5_0, 2)
#endif
    moe_q5_0(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q5_0;
  const int mmq_y = MOE_Y_Q5_0;
  const int nwarps = NWARPS_Q5_0;

  moe_q<scalar_t, QK5_0, QR5_0, QI5_0, false, block_q5_0, mmq_x, mmq_y, nwarps,
        allocate_tiles_q5_0<mmq_y>, load_tiles_q5_0<mmq_y, nwarps, need_check>,
        VDR_Q5_0_Q8_1_MMQ, vec_dot_q5_0_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q5_0_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MOE_X_Q5_0;
  const int mmq_y = MOE_Y_Q5_0;
  const int nwarps = NWARPS_Q5_0;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q5_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q5_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q5_1 64
  #define MOE_Y_Q5_1 128
  #define NWARPS_Q5_1 8
#else
  #define MOE_X_Q5_1 4
  #define MOE_Y_Q5_1 32
  #define NWARPS_Q5_1 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q5_1, 2)
#endif
    moe_q5_1(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q5_1;
  const int mmq_y = MOE_Y_Q5_1;
  const int nwarps = NWARPS_Q5_1;

  moe_q<scalar_t, QK5_1, QR5_1, QI5_1, true, block_q5_1, mmq_x, mmq_y, nwarps,
        allocate_tiles_q5_1<mmq_y>, load_tiles_q5_1<mmq_y, nwarps, need_check>,
        VDR_Q5_1_Q8_1_MMQ, vec_dot_q5_1_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q5_1_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MOE_X_Q5_1;
  const int mmq_y = MOE_Y_Q5_1;
  const int nwarps = NWARPS_Q5_1;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q5_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q5_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q8_0 64
  #define MOE_Y_Q8_0 128
  #define NWARPS_Q8_0 8
#else
  #define MOE_X_Q8_0 4
  #define MOE_Y_Q8_0 32
  #define NWARPS_Q8_0 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q8_0, 2)
#endif
    moe_q8_0(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q8_0;
  const int mmq_y = MOE_Y_Q8_0;
  const int nwarps = NWARPS_Q8_0;

  moe_q<scalar_t, QK8_0, QR8_0, QI8_0, false, block_q8_0, mmq_x, mmq_y, nwarps,
        allocate_tiles_q8_0<mmq_y>, load_tiles_q8_0<mmq_y, nwarps, need_check>,
        VDR_Q8_0_Q8_1_MMQ, vec_dot_q8_0_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q8_0_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MOE_X_Q8_0;
  const int mmq_y = MOE_Y_Q8_0;
  const int nwarps = NWARPS_Q8_0;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q8_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q8_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q2_K 64
  #define MOE_Y_Q2_K 128
  #define NWARPS_Q2_K 8
#else
  #define MOE_X_Q2_K 4
  #define MOE_Y_Q2_K 32
  #define NWARPS_Q2_K 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q2_K, 2)
#endif
    moe_q2_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q2_K;
  const int mmq_y = MOE_Y_Q2_K;
  const int nwarps = NWARPS_Q2_K;

  moe_q<scalar_t, QK_K, QR2_K, QI2_K, false, block_q2_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q2_K<mmq_y>, load_tiles_q2_K<mmq_y, nwarps, need_check>,
        VDR_Q2_K_Q8_1_MMQ, vec_dot_q2_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q2_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MOE_X_Q2_K;
  const int mmq_y = MOE_Y_Q2_K;
  const int nwarps = NWARPS_Q2_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q2_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q2_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q3_K 64
  #define MOE_Y_Q3_K 128
  #define NWARPS_Q3_K 8
#else
  #define MOE_X_Q3_K 4
  #define MOE_Y_Q3_K 32
  #define NWARPS_Q3_K 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q3_K, 2)
#endif
    moe_q3_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {

  const int mmq_x = MOE_X_Q3_K;
  const int mmq_y = MOE_Y_Q3_K;
  const int nwarps = NWARPS_Q3_K;

  moe_q<scalar_t, QK_K, QR3_K, QI3_K, false, block_q3_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q3_K<mmq_y>, load_tiles_q3_K<mmq_y, nwarps, need_check>,
        VDR_Q3_K_Q8_1_MMQ, vec_dot_q3_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}
template <typename scalar_t>
static void ggml_moe_q3_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MOE_X_Q3_K;
  const int mmq_y = MOE_Y_Q3_K;
  const int nwarps = NWARPS_Q3_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q3_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q3_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
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
    moe_q4_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q4_K;
  const int mmq_y = MOE_Y_Q4_K;
  const int nwarps = NWARPS_Q4_K_MOE;

  moe_q<scalar_t, QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q4_K<mmq_y>, load_tiles_q4_K<mmq_y, nwarps, need_check>,
        VDR_Q4_K_Q8_1_MMQ, vec_dot_q4_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q4_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
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
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q4_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q5_K 64
  #define MOE_Y_Q5_K 128
  #define NWARPS_Q5_K 8
#else
  #define MOE_X_Q5_K 4
  #define MOE_Y_Q5_K 32
  #define NWARPS_Q5_K 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q5_K, 2)
#endif
    moe_q5_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q5_K;
  const int mmq_y = MOE_Y_Q5_K;
  const int nwarps = NWARPS_Q5_K;

  moe_q<scalar_t, QK_K, QR5_K, QI5_K, true, block_q5_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q5_K<mmq_y>, load_tiles_q5_K<mmq_y, nwarps, need_check>,
        VDR_Q5_K_Q8_1_MMQ, vec_dot_q5_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q5_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MOE_X_Q5_K;
  const int mmq_y = MOE_Y_Q5_K;
  const int nwarps = NWARPS_Q5_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q5_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q5_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q6_K 64
  #define MOE_Y_Q6_K 128
  #define NWARPS_Q6_K 8
#else
  #define MOE_X_Q6_K 4
  #define MOE_Y_Q6_K 32
  #define NWARPS_Q6_K 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q6_K, 2)
#endif
    moe_q6_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q6_K;
  const int mmq_y = MOE_Y_Q6_K;
  const int nwarps = NWARPS_Q6_K;

  moe_q<scalar_t, QK_K, QR6_K, QI6_K, false, block_q6_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q6_K<mmq_y>, load_tiles_q6_K<mmq_y, nwarps, need_check>,
        VDR_Q6_K_Q8_1_MMQ, vec_dot_q6_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q6_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MOE_X_Q6_K;
  const int mmq_y = MOE_Y_Q6_K;
  const int nwarps = NWARPS_Q6_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q6_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q6_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}
