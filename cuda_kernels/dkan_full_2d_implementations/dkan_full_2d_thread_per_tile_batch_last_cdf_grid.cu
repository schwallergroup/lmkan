#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
using namespace torch::indexing;

#include "../utilities.h"

template <int tile_size>
__device__ void dkan_full_kernel_2d_thread_per_tile_batch_last_cdf_grid(
    const float*  __restrict__ shared_parameters,
    const float*  __restrict__ grid_points,
    const float*  __restrict__ inverse_chunk_lengths,
    const float*  __restrict__ x,
    float* __restrict__ output,
    int my_tile_in,
    int my_tile_out,
    int N_in,
    int N_out,
    int n_chunks,
    int batch_size,
    int batch_index_from,
    int batch_index_to
) {
    //static_assert(!cdf_grid, "cdf_grid = true is not implemented."); for now just silently ignore cdf_grid = true

    constexpr int half_tile_size = tile_size / 2;
    constexpr int half_size = half_tile_size * tile_size;
    static_assert(tile_size % 2 == 0, "tile_size must be even");

    int thread_idx = threadIdx.x;
    int block_size = blockDim.x;

    int tile_pos = thread_idx % half_size;

    int input_shift = my_tile_in * tile_size;
    int output_shift = my_tile_out * tile_size;

    int first_index, second_index;
    float x_delta_now_1, x_delta_now_2;

    float x_current[tile_size], x_current_next[tile_size];

    float output_current[tile_size];
    float param_1, param_2, param_3, param_4;
    float b_spline_1, b_spline_2, b_spline_3, b_spline_4;
    int chunk_index_now;
    int tile_pos_now;
    int chunk_index_precomputed;

    int full_index = thread_idx % half_size;
    int fast_index = full_index / half_tile_size;
    int slow_index = full_index % half_tile_size;

    int index_now;
    float tmp;

    int fast_index_now, slow_index_now;

    float inverse_chunk_size;
    float chunk_size;

    inverse_chunk_size = static_cast<float>(n_chunks) / static_cast<float>(1.0f);
    chunk_size = 1.0f / static_cast<float>(n_chunks);

    //float output_current_shifted[tile_size];
    //float x_current_shifted[tile_size];

    float func_value;
    float first_projected, second_projected;

    float left_border_first, left_border_second;
    float inverse_chunk_length_first, inverse_chunk_length_second;

    int my_in_warp_index = thread_idx % 32;

    #pragma unroll
    for (int j = 0; j < tile_size; ++j) {
        index_now = j + 2 * slow_index;
        if (index_now >= tile_size) {
            index_now -= tile_size;
        }
        /*if (thread_idx * N_in + input_shift + index_now < batch_size * N_in) {
            x_current_next[j] = x[thread_idx * N_in + input_shift + index_now];
            //x_current_next[j] = __ldcg(x + thread_idx * N_in + input_shift + index_now);
        }*/

        int initial_batch_idx = batch_index_from + thread_idx;
        if (initial_batch_idx < batch_index_to) {
            x_current_next[j] = x[(input_shift + j) * batch_size + initial_batch_idx];
        }
    }

    for (int i = batch_index_from + thread_idx; i < batch_index_to; i += block_size) {
        #pragma unroll
        for (int j = 0; j < tile_size; ++j) {
            //x_current[j] = normcdff(x_current_next[j]);
            //x_current[j] = 1.0f / (1.0f + expf(-1.702f * x_current_next[j]));
            /*func_value = __expf(-fabsf(x_current_next[j]));
            x_current[j] =  (x_current_next[j] > 0.0f)
            ? (1.0f - 0.5f * func_value)
            : (0.5f * func_value);*/
            x_current[j] = x_current_next[j];
        }

        if constexpr (tile_size == 8) {
            ptx_shift_8_inplace(x_current, 2 * slow_index);
        } else if constexpr (tile_size == 16) {
            ptx_shift_16_inplace(x_current, 2 * slow_index);
        } else if constexpr (tile_size == 4) {
            ptx_shift_4_inplace(x_current, 2 * slow_index);
        } else {
            /*#pragma unroll
            for (int j = 0; j < tile_size; ++j) {
                x_current_shifted[j] = x_current[(j + 2 * slow_index) % tile_size];
            }

            #pragma unroll
            for (int j = 0; j < tile_size; ++j) {
                x_current[j] = x_current_shifted[j];
            }*/
            assert(false);
        }

        int next_batch_idx = i + block_size;
        if (next_batch_idx < batch_index_to) {
            #pragma unroll
            for (int j = 0; j < tile_size; ++j) {
                //x_current_next[j] = x[(input_shift + j) * batch_size + i + block_size];
                x_current_next[j] = __ldcg(x + (input_shift + j) * batch_size + next_batch_idx);
            }
        }

        #pragma unroll
        for (int j = 0; j < tile_size; ++j) {
            output_current[j] = 0.0f;
        }


        #pragma unroll
        for (int j = 0; j < half_tile_size; ++j) {
            slow_index_now = j + slow_index;
            if (slow_index_now >= half_tile_size) {
                slow_index_now -= half_tile_size;
            }

            func_value = __expf(-fabsf(x_current[2 * j]));
            first_projected = (x_current[2 * j] > 0.0f)
            ? (1.0f - 0.5f * func_value)
            : (0.5f * func_value);

            func_value = __expf(-fabsf(x_current[2 * j + 1]));
            second_projected = (x_current[2 * j + 1] > 0.0f)
            ? (1.0f - 0.5f * func_value)
            : (0.5f * func_value);

            first_index = __float2int_rz((first_projected) * inverse_chunk_size);
            second_index = __float2int_rz((second_projected) * inverse_chunk_size);

            first_index = max(0, min(first_index, n_chunks - 1));
            second_index = max(0, min(second_index, n_chunks - 1));

            left_border_first = grid_points[first_index * 32 + my_in_warp_index];
            left_border_second = grid_points[second_index * 32 + my_in_warp_index];

            inverse_chunk_length_first = inverse_chunk_lengths[first_index * 32 + my_in_warp_index];
            inverse_chunk_length_second = inverse_chunk_lengths[second_index * 32 + my_in_warp_index];

            x_delta_now_1 = (x_current[2 * j] - left_border_first) * inverse_chunk_length_first;
            x_delta_now_2 = (x_current[2 * j + 1] - left_border_second) * inverse_chunk_length_second;

            b_spline_1 = (1.0f - x_delta_now_1) * (1.0f - x_delta_now_2);
            b_spline_2 = (1.0f - x_delta_now_1) * x_delta_now_2;
            b_spline_3 = x_delta_now_1 * (1.0f - x_delta_now_2);
            b_spline_4 = x_delta_now_1 * x_delta_now_2;

            /*b_spline_1 = (right_border_first - x_current[2 * j]) * (right_border_second - x_current[2 * j + 1]);
            b_spline_2 = (right_border_first - x_current[2 * j]) * (x_current[2 * j + 1] - left_border_second);
            b_spline_3 = (x_current[2 * j] - left_border_first) * (right_border_second - x_current[2 * j + 1]);
            b_spline_4 = (x_current[2 * j] - left_border_first) * (x_current[2 * j + 1] - left_border_second);*/

            chunk_index_precomputed = first_index * (n_chunks + 1) * half_size + second_index * half_size + slow_index_now;

            #pragma unroll
            for (int k = 0; k < tile_size; ++k) {

                fast_index_now = fast_index + k;
                if (fast_index_now >= tile_size) {
                    fast_index_now -= tile_size;
                }
                chunk_index_now = chunk_index_precomputed + fast_index_now * half_tile_size;
                //chunk_index_now = first_index * (n_chunks + 1) * half_size + second_index * half_size + slow_index_now * tile_size + fast_index_now;
                param_1 = shared_parameters[chunk_index_now];
                chunk_index_now += half_size;
                param_2 = shared_parameters[chunk_index_now];
                chunk_index_now += n_chunks * half_size;
                param_3 = shared_parameters[chunk_index_now];
                chunk_index_now += half_size;
                param_4 = shared_parameters[chunk_index_now];

                output_current[k] = __fmaf_rn(param_1 , b_spline_1, output_current[k]);
                output_current[k] = __fmaf_rn(param_2 , b_spline_2, output_current[k]);
                output_current[k] = __fmaf_rn(param_3 , b_spline_3, output_current[k]);
                output_current[k] = __fmaf_rn(param_4 , b_spline_4, output_current[k]);

            }
        }

        if constexpr (tile_size == 8) {
            ptx_shift_8_inplace(output_current, tile_size - fast_index);
        } else if constexpr (tile_size == 16) {
            ptx_shift_16_inplace(output_current, tile_size - fast_index);
        } else if constexpr (tile_size == 4) {
            ptx_shift_4_inplace(output_current, tile_size - fast_index);
        } else {
            /*#pragma unroll
            for (int j = 0; j < tile_size; ++j) {
                output_current_shifted[j] = output_current[(j - fast_index + tile_size) % tile_size];
            }

            #pragma unroll
            for (int j = 0; j < tile_size; ++j) {
                output_current[j] = output_current_shifted[j];
            }*/
            assert(false);
        }

        #pragma unroll
        for (int j = 0; j < tile_size; ++j) {
            atomicAdd(&output[(output_shift + j) * batch_size + i], output_current[j]);
        }
    }
}