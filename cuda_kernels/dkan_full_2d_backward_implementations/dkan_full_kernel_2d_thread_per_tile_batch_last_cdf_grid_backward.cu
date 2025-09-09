#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
using namespace torch::indexing;

#include "../utilities.h"


template <int tile_size>
__device__ void dkan_full_kernel_2d_thread_per_tile_batch_last_cdf_grid_backward(
    const float* __restrict__ shared_parameters,
    const float*  __restrict__ grid_points,
    const float*  __restrict__ inverse_chunk_lengths,
    const float* __restrict__ x,
    const float* __restrict__ output_grad,
    float* __restrict__ shared_parameters_grad,
    float* __restrict__ x_grad,
    int my_tile_in,
    int my_tile_out,
    int N_in,
    int N_out,
    int n_chunks,
    int batch_size,
    int batch_index_from,
    int batch_index_to
) {
    constexpr int half_tile_size = tile_size / 2;
    constexpr int half_size = half_tile_size * tile_size;
    static_assert(tile_size % 2 == 0, "tile_size must be even");

    int thread_idx = threadIdx.x;
    int block_size = blockDim.x;


    int input_shift = my_tile_in * tile_size;
    int output_shift = my_tile_out * tile_size;

    int first_index, second_index;
    float x_delta_now_1, x_delta_now_2;

    float x_current[tile_size], x_current_next[tile_size];
    float output_grad_current[tile_size], output_grad_current_next[tile_size];

    float x_grad_current[tile_size];

    float param_1, param_2, param_3, param_4;
    float b_spline_1, b_spline_2, b_spline_3, b_spline_4;
    int chunk_index_now;

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
        int initial_batch_idx = batch_index_from + thread_idx;
        if (initial_batch_idx < batch_index_to) {
            if ((input_shift + j) * batch_size + initial_batch_idx < N_in * batch_size) {
                x_current_next[j] = x[(input_shift + j) * batch_size + initial_batch_idx];
            }

            if ((output_shift + j) * batch_size + initial_batch_idx < N_out * batch_size) {
                output_grad_current_next[j] = output_grad[(output_shift + j) * batch_size + initial_batch_idx];
            }
        }
    }

    for (int i = batch_index_from + thread_idx; i < batch_index_to; i += block_size) {
        #pragma unroll
        for (int j = 0; j < tile_size; ++j) {
            x_current[j] = x_current_next[j];
            output_grad_current[j] = output_grad_current_next[j];
        }

        if constexpr (tile_size == 8) {
            ptx_shift_8_inplace(x_current, 2 * slow_index);
            ptx_shift_8_inplace(output_grad_current, fast_index);
        } else if constexpr (tile_size == 16) {
            ptx_shift_16_inplace(x_current, 2 * slow_index);
            ptx_shift_16_inplace(output_grad_current, fast_index);
        } else if constexpr (tile_size == 4) {
            ptx_shift_4_inplace(x_current, 2 * slow_index);
            ptx_shift_4_inplace(output_grad_current, fast_index);
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
                if ((input_shift + j) * batch_size + next_batch_idx < N_in * batch_size) {
                    x_current_next[j] = __ldcg(x + (input_shift + j) * batch_size + next_batch_idx);
                }
                if ((output_shift + j) * batch_size + next_batch_idx < N_out * batch_size) {
                    output_grad_current_next[j] = __ldcg(output_grad + (output_shift + j) * batch_size + next_batch_idx);
                }
            }
        }

        #pragma unroll
        for (int j = 0; j < tile_size; ++j) {
            x_grad_current[j] = 0.0f;
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

            // Compute derivatives of b-splines with respect to x inputs
            // Derivatives of b_spline_1 = (1-x_delta_now_1)*(1-x_delta_now_2)
            float db_spline_1_dx_1 = -inverse_chunk_length_first * (1.0f - x_delta_now_2);
            float db_spline_1_dx_2 = -inverse_chunk_length_second * (1.0f - x_delta_now_1);

            // Derivatives of b_spline_2 = (1-x_delta_now_1)*x_delta_now_2
            float db_spline_2_dx_1 = -inverse_chunk_length_first * x_delta_now_2;
            float db_spline_2_dx_2 = inverse_chunk_length_second * (1.0f - x_delta_now_1);

            // Derivatives of b_spline_3 = x_delta_now_1*(1-x_delta_now_2)
            float db_spline_3_dx_1 = inverse_chunk_length_first * (1.0f - x_delta_now_2);
            float db_spline_3_dx_2 = -inverse_chunk_length_second * x_delta_now_1;

            // Derivatives of b_spline_4 = x_delta_now_1*x_delta_now_2
            float db_spline_4_dx_1 = inverse_chunk_length_first * x_delta_now_2;
            float db_spline_4_dx_2 = inverse_chunk_length_second * x_delta_now_1;

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
                atomicAdd(shared_parameters_grad + chunk_index_now, output_grad_current[k] * b_spline_1);
                chunk_index_now += half_size;
                param_2 = shared_parameters[chunk_index_now];
                atomicAdd(shared_parameters_grad + chunk_index_now, output_grad_current[k] * b_spline_2);
                chunk_index_now += n_chunks * half_size;
                param_3 = shared_parameters[chunk_index_now];
                atomicAdd(shared_parameters_grad + chunk_index_now, output_grad_current[k] * b_spline_3);
                chunk_index_now += half_size;
                param_4 = shared_parameters[chunk_index_now];
                atomicAdd(shared_parameters_grad + chunk_index_now, output_grad_current[k] * b_spline_4);

                x_grad_current[2 * j] += output_grad_current[k] * (
                    db_spline_1_dx_1 * param_1 + db_spline_2_dx_1 * param_2 + db_spline_3_dx_1 * param_3 + db_spline_4_dx_1 * param_4
                );
                x_grad_current[2 * j + 1] += output_grad_current[k] * (
                    db_spline_1_dx_2 * param_1 + db_spline_2_dx_2 * param_2 + db_spline_3_dx_2 * param_3 + db_spline_4_dx_2 * param_4
                );
            }
        }

        if constexpr (tile_size == 8) {
            ptx_shift_8_inplace(x_grad_current, tile_size - 2 * slow_index);
        } else if constexpr (tile_size == 16) {
            ptx_shift_16_inplace(x_grad_current, tile_size - 2 * slow_index);
        } else if constexpr (tile_size == 4) {
            ptx_shift_4_inplace(x_grad_current, tile_size - 2 * slow_index);
        } else {
            assert(false);
        }

        #pragma unroll
        for (int j = 0; j < tile_size; ++j) {
            if ((input_shift + j) * batch_size + i < N_in * batch_size) {
                atomicAdd(x_grad + (input_shift + j) * batch_size + i, x_grad_current[j]);
            }
        }
    }
    __syncthreads();
}


template <int tile_size>
__device__ void dkan_full_kernel_2d_thread_per_tile_batch_last_cdf_grid_backward_fast_mode(
    const float* __restrict__ shared_parameters,
    const float*  __restrict__ grid_points,
    const float*  __restrict__ inverse_chunk_lengths,
    const float* __restrict__ x,
    const float* __restrict__ output_grad,
    float* __restrict__ shared_parameters_grad_first,
    float* __restrict__ shared_parameters_grad_second,
    float* __restrict__ shared_parameters_grad_third,
    float* __restrict__ shared_parameters_grad_fourth,
    float* __restrict__ x_grad,
    int my_tile_in,
    int my_tile_out,
    int N_in,
    int N_out,
    int n_chunks,
    int batch_size,
    int batch_index_from,
    int batch_index_to
) {
    constexpr int half_tile_size = tile_size / 2;
    constexpr int half_size = half_tile_size * tile_size;
    static_assert(tile_size % 2 == 0, "tile_size must be even");

    int thread_idx = threadIdx.x;
    int block_size = blockDim.x;
    
    int my_warp_index = thread_idx / 32;
    float* my_shared_parameters_grad;
    if (my_warp_index == 0) {
        my_shared_parameters_grad = shared_parameters_grad_first;
    } else if (my_warp_index == 1) {
        my_shared_parameters_grad = shared_parameters_grad_second;
    } else if (my_warp_index == 2) {
        my_shared_parameters_grad = shared_parameters_grad_third;
    } else if (my_warp_index == 3) {
        my_shared_parameters_grad = shared_parameters_grad_fourth;
    } else {
        assert(false);
    }

    int input_shift = my_tile_in * tile_size;
    int output_shift = my_tile_out * tile_size;

    int first_index, second_index;
    float x_delta_now_1, x_delta_now_2;

    float x_current[tile_size], x_current_next[tile_size];
    float output_grad_current[tile_size], output_grad_current_next[tile_size];

    float x_grad_current[tile_size];

    float param_1, param_2, param_3, param_4;
    float b_spline_1, b_spline_2, b_spline_3, b_spline_4;
    int chunk_index_now;

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
        int initial_batch_idx = batch_index_from + thread_idx;
        if (initial_batch_idx < batch_index_to) {
            if ((input_shift + j) * batch_size + initial_batch_idx < N_in * batch_size) {
                x_current_next[j] = x[(input_shift + j) * batch_size + initial_batch_idx];
            }

            if ((output_shift + j) * batch_size + initial_batch_idx < N_out * batch_size) {
                output_grad_current_next[j] = output_grad[(output_shift + j) * batch_size + initial_batch_idx];
            }
        }
    }

    for (int i = batch_index_from + thread_idx; i < batch_index_to; i += block_size) {
        const unsigned int active_mask = __activemask();
        #pragma unroll
        for (int j = 0; j < tile_size; ++j) {
            x_current[j] = x_current_next[j];
            output_grad_current[j] = output_grad_current_next[j];
        }

        if constexpr (tile_size == 8) {
            ptx_shift_8_inplace(x_current, 2 * slow_index);
            ptx_shift_8_inplace(output_grad_current, fast_index);
        } else if constexpr (tile_size == 16) {
            ptx_shift_16_inplace(x_current, 2 * slow_index);
            ptx_shift_16_inplace(output_grad_current, fast_index);
        } else if constexpr (tile_size == 4) {
            ptx_shift_4_inplace(x_current, 2 * slow_index);
            ptx_shift_4_inplace(output_grad_current, fast_index);
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
                if ((input_shift + j) * batch_size + next_batch_idx < N_in * batch_size) {
                    x_current_next[j] = __ldcg(x + (input_shift + j) * batch_size + next_batch_idx);
                }
                if ((output_shift + j) * batch_size + next_batch_idx < N_out * batch_size) {
                    output_grad_current_next[j] = __ldcg(output_grad + (output_shift + j) * batch_size + next_batch_idx);
                }
            }
        }

        #pragma unroll
        for (int j = 0; j < tile_size; ++j) {
            x_grad_current[j] = 0.0f;
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

            // Compute derivatives of b-splines with respect to x inputs
            // Derivatives of b_spline_1 = (1-x_delta_now_1)*(1-x_delta_now_2)
            float db_spline_1_dx_1 = -inverse_chunk_length_first * (1.0f - x_delta_now_2);
            float db_spline_1_dx_2 = -inverse_chunk_length_second * (1.0f - x_delta_now_1);

            // Derivatives of b_spline_2 = (1-x_delta_now_1)*x_delta_now_2
            float db_spline_2_dx_1 = -inverse_chunk_length_first * x_delta_now_2;
            float db_spline_2_dx_2 = inverse_chunk_length_second * (1.0f - x_delta_now_1);

            // Derivatives of b_spline_3 = x_delta_now_1*(1-x_delta_now_2)
            float db_spline_3_dx_1 = inverse_chunk_length_first * (1.0f - x_delta_now_2);
            float db_spline_3_dx_2 = -inverse_chunk_length_second * x_delta_now_1;

            // Derivatives of b_spline_4 = x_delta_now_1*x_delta_now_2
            float db_spline_4_dx_1 = inverse_chunk_length_first * x_delta_now_2;
            float db_spline_4_dx_2 = inverse_chunk_length_second * x_delta_now_1;

            chunk_index_precomputed = first_index * (n_chunks + 1) * half_size + second_index * half_size + slow_index_now;

            #pragma unroll
            for (int k = 0; k < tile_size; ++k) {
                __syncwarp(active_mask);

                fast_index_now = fast_index + k;
                if (fast_index_now >= tile_size) {
                    fast_index_now -= tile_size;
                }
                chunk_index_now = chunk_index_precomputed + fast_index_now * half_tile_size;
                //chunk_index_now = first_index * (n_chunks + 1) * half_size + second_index * half_size + slow_index_now * tile_size + fast_index_now;
                param_1 = shared_parameters[chunk_index_now];
                //atomicAdd(my_shared_parameters_grad + chunk_index_now, output_grad_current[k] * b_spline_1);
                my_shared_parameters_grad[chunk_index_now] += output_grad_current[k] * b_spline_1;
                chunk_index_now += half_size;
                param_2 = shared_parameters[chunk_index_now];
                //atomicAdd(my_shared_parameters_grad + chunk_index_now, output_grad_current[k] * b_spline_2);
                my_shared_parameters_grad[chunk_index_now] += output_grad_current[k] * b_spline_2;
                chunk_index_now += n_chunks * half_size;
                param_3 = shared_parameters[chunk_index_now];
                //atomicAdd(my_shared_parameters_grad + chunk_index_now, output_grad_current[k] * b_spline_3);
                my_shared_parameters_grad[chunk_index_now] += output_grad_current[k] * b_spline_3;
                chunk_index_now += half_size;
                param_4 = shared_parameters[chunk_index_now];
                //atomicAdd(my_shared_parameters_grad + chunk_index_now, output_grad_current[k] * b_spline_4);
                my_shared_parameters_grad[chunk_index_now] += output_grad_current[k] * b_spline_4;

                x_grad_current[2 * j] += output_grad_current[k] * (
                    db_spline_1_dx_1 * param_1 + db_spline_2_dx_1 * param_2 + db_spline_3_dx_1 * param_3 + db_spline_4_dx_1 * param_4
                );
                x_grad_current[2 * j + 1] += output_grad_current[k] * (
                    db_spline_1_dx_2 * param_1 + db_spline_2_dx_2 * param_2 + db_spline_3_dx_2 * param_3 + db_spline_4_dx_2 * param_4
                );
            }
        }

        if constexpr (tile_size == 8) {
            ptx_shift_8_inplace(x_grad_current, tile_size - 2 * slow_index);
        } else if constexpr (tile_size == 16) {
            ptx_shift_16_inplace(x_grad_current, tile_size - 2 * slow_index);
        } else if constexpr (tile_size == 4) {
            ptx_shift_4_inplace(x_grad_current, tile_size - 2 * slow_index);
        } else {
            assert(false);
        }

        #pragma unroll
        for (int j = 0; j < tile_size; ++j) {
            if ((input_shift + j) * batch_size + i < N_in * batch_size) {
                atomicAdd(x_grad + (input_shift + j) * batch_size + i, x_grad_current[j]);
            }
        }
    }
    __syncthreads();
}