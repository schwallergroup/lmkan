#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
using namespace torch::indexing;

#include "utilities.h"

#include "dkan_full_2d_backward_implementations/dkan_full_kernel_2d_thread_per_tile_batch_last_cdf_grid_backward.cu"

template <int dim, int tile_size, bool cdf_grid, bool batch_last, bool fast_mode>
__global__ void linear_cuda_dkan_full_backward_kernel(
    const float* __restrict__ parameters,
    const float* __restrict__ x,
    const float* __restrict__ output_grad,
    float* __restrict__ parameters_grad,
    float* __restrict__ x_grad,
    int n_chunks,
    int N_in,
    int N_out,
    int batch_size,
    int n_tiles_repeat,
    int n_repetitions
) {
    int full_tile_index;
    int batch_index_from, batch_index_to;
    if (blockIdx.x < n_tiles_repeat * n_repetitions) {
        full_tile_index = blockIdx.x % n_tiles_repeat;
        int chunk_size = batch_size / n_repetitions;
        if (chunk_size == 0) {
            chunk_size = 1;
        }
        int my_repetition_index = blockIdx.x / n_tiles_repeat;
        batch_index_from = chunk_size * my_repetition_index;
        batch_index_to = batch_index_from + chunk_size;
        if (batch_index_to > batch_size) {
            batch_index_to = batch_size;
        }
        if (my_repetition_index == n_repetitions - 1) {
            batch_index_to = batch_size;
        }
    } else {
        full_tile_index = blockIdx.x - n_tiles_repeat * n_repetitions + n_tiles_repeat;
        batch_index_from = 0;
        batch_index_to = batch_size;
    }

    int n_tiles_input = N_in / tile_size;
    int n_tiles_output = N_out / tile_size;

    int my_tile_in, my_tile_out;
    if (n_tiles_input % 4 == 0 && n_tiles_output % 4 == 0) {
        int suptile_index = full_tile_index / (4 * 4);
        int my_tile_index = full_tile_index % (4 * 4);

        int n_suptiles_in = N_in / (tile_size * 4);
        int n_suptiles_out = N_out / (tile_size * 4);

        int my_suptile_in = suptile_index % n_suptiles_in;
        int my_suptile_out = suptile_index / n_suptiles_in;

        my_tile_in = my_suptile_in * 4 + my_tile_index % 4;
        my_tile_out = my_suptile_out * 4 + my_tile_index / 4;
    } else {
        my_tile_in = full_tile_index / n_tiles_output;
        my_tile_out = full_tile_index % n_tiles_output;
    }

    // Shared memory is provided as one contiguous buffer.
    extern __shared__ char buffer[];
    char* shared_mem = buffer;

    // Compute the number of elements per tile.
    const int n_param_tile = (n_chunks + 1) * (n_chunks + 1) * tile_size * (tile_size / 2);

    // First part: shared_parameters
    float* shared_parameters = reinterpret_cast<float*>(shared_mem);

    // Second part: shared_parameters_grad
    float* shared_parameters_grad = shared_parameters + n_param_tile;

    float* shared_parameters_grad_second;
    float* shared_parameters_grad_third;
    float* shared_parameters_grad_fourth;
    if constexpr (fast_mode) {
        shared_parameters_grad_second = shared_parameters_grad + n_param_tile;
        shared_parameters_grad_third = shared_parameters_grad_second + n_param_tile;
        shared_parameters_grad_fourth = shared_parameters_grad_third + n_param_tile;
    }
    
    // Calculate the starting index for grid_points with appropriate padding
    int shared_parameters_size = n_param_tile;
    int parameters_grad_size = n_param_tile;
    int padding = (32 - ((shared_parameters_size + parameters_grad_size) % 32)) % 32;
    int grid_points_offset;
    if constexpr (fast_mode) {
        grid_points_offset = shared_parameters_size + 4 * parameters_grad_size + padding;
    } else {
        grid_points_offset = shared_parameters_size + parameters_grad_size + padding;
    }
    float* grid_points = shared_parameters + grid_points_offset;

    // Calculate the starting index for inverse_chunk_lengths
    int grid_points_size = 32 * (n_chunks + 1);
    int inverse_chunk_lengths_offset = grid_points_offset + grid_points_size;
    float* inverse_chunk_lengths = shared_parameters + inverse_chunk_lengths_offset;

    // Zero out the shared parameters gradients.
    zero_shared_parameters_grad_full_dkan<tile_size>(shared_parameters_grad, n_chunks);

    if constexpr (fast_mode) {
        zero_shared_parameters_grad_full_dkan<tile_size>(shared_parameters_grad_second, n_chunks);
        zero_shared_parameters_grad_full_dkan<tile_size>(shared_parameters_grad_third, n_chunks);
        zero_shared_parameters_grad_full_dkan<tile_size>(shared_parameters_grad_fourth, n_chunks);
    }

    // Copy the parameters (tile) from global memory to shared memory.
    copy_parameters_to_shared_full_dkan<tile_size>(
        parameters,
        shared_parameters,
        n_chunks,
        my_tile_in,
        my_tile_out,
        N_in,
        N_out
    );

    // Invoke the backward computation kernel.
    // This function (which you must implement separately) is responsible for:
    //   - Using the shared parameters, x, and output_grad to compute gradients.
    //   - Writing the gradient with respect to parameters into shared_parameters_grad.
    //   - Writing the gradient with respect to x into x_grad (global memory).
    if constexpr (dim == 2) {
        if constexpr (batch_last) {
            if constexpr (cdf_grid) {
                // Precompute grid points for CDF grid operations
                precompute_grid_points(grid_points, inverse_chunk_lengths, n_chunks);

                if constexpr (fast_mode) {
                    dkan_full_kernel_2d_thread_per_tile_batch_last_cdf_grid_backward_fast_mode<tile_size>(
                        shared_parameters,
                        grid_points,
                        inverse_chunk_lengths,
                        x,
                        output_grad,
                        shared_parameters_grad,
                        shared_parameters_grad_second,
                        shared_parameters_grad_third,
                        shared_parameters_grad_fourth,
                        x_grad,
                        my_tile_in,
                        my_tile_out,
                        N_in,
                        N_out,
                        n_chunks,
                        batch_size,
                        batch_index_from,
                        batch_index_to
                    );
                }
                else {
                    dkan_full_kernel_2d_thread_per_tile_batch_last_cdf_grid_backward<tile_size>(
                        shared_parameters,
                        grid_points,
                        inverse_chunk_lengths,
                        x,
                        output_grad,
                        shared_parameters_grad,
                        x_grad,
                        my_tile_in,
                        my_tile_out,
                        N_in,
                        N_out,
                        n_chunks,
                        batch_size,
                        batch_index_from,
                        batch_index_to
                    );
                }
                
            } else {
                /*dkan_full_kernel_2d_thread_per_tile_batch_last_backward<tile_size>(
                    shared_parameters,
                    x,
                    output_grad,
                    shared_parameters_grad,
                    x_grad,
                    my_tile_in,
                    my_tile_out,
                    N_in,
                    N_out,
                    n_chunks,
                    batch_size
                );*/
                assert(false);
            }
        } else {
            /*dkan_full_kernel_2d_thread_per_four_columns_backward<tile_size, cdf_grid>(
                shared_parameters,
                x,
                output_grad,
                shared_parameters_grad,
                x_grad,
                my_tile_in,
                my_tile_out,
                N_in,
                N_out,
                n_chunks,
                batch_size
            );*/
            assert(false);
        }
    } else {
        static_assert(dim == 2, "dim must be 2");
    }

    // Finally, accumulate the gradients stored in shared_parameters_grad into global memory.

    if constexpr (fast_mode) {
        accumulate_shared_parameters_grad_full_dkan_fast_mode<tile_size>(
            parameters_grad,
            shared_parameters_grad,
            shared_parameters_grad_second,
            shared_parameters_grad_third,
            shared_parameters_grad_fourth,
            n_chunks,
            my_tile_in,
            my_tile_out,
            N_in,
            N_out
        );
    } else {
        accumulate_shared_parameters_grad_full_dkan<tile_size>(
            parameters_grad,
            shared_parameters_grad,
            n_chunks,
            my_tile_in,
            my_tile_out,
            N_in,
            N_out
        );
    }
}

template <int dim, int block_size, bool cdf_grid, bool batch_last, bool fast_mode, int tile_size>
std::tuple<torch::Tensor, torch::Tensor> linear_cuda_dkan_full_tile_size_backward(
    torch::Tensor parameters,
    torch::Tensor x,
    torch::Tensor output_grad,
    int n_tiles_repeat,
    int n_repetitions
) {
    TORCH_CHECK(
        parameters.dtype() == torch::kFloat32 &&
        x.dtype()         == torch::kFloat32 &&
        output_grad.dtype() == torch::kFloat32,
        "linear_cuda_dkan_full_tile_size_backward: Expected all tensors to be float32."
    );

    // Ensure the input tensors are contiguous and on the GPU
    CHECK_INPUT(parameters);
    CHECK_INPUT(x);
    CHECK_INPUT(output_grad);

    // Parameters shape in 2d case: [n_chunks + 1, n_chunks + 1, N_out, N_in / 2]
    // If batch_last is false:
    //   - x shape: [batch_size, N_in]
    // If batch_last is true:
    //   - x shape: [N_in, batch_size]
    int N_in, batch_size;

    if constexpr (batch_last) {
        // For batch_last=true: x has shape [N_in, batch_size]
        N_in = x.size(0);
        batch_size = x.size(1);
    } else {
        // For batch_last=false: x has shape [batch_size, N_in]
        N_in = x.size(1);
        batch_size = x.size(0);
    }

    int N_out = parameters.size(2);

    if (N_in % tile_size != 0) {
        throw std::invalid_argument("N_in must be divisible by tile_size");
    }

    if (N_out % tile_size != 0) {
        throw std::invalid_argument("N_out must be divisible by tile_size");
    }

    int n_chunks = parameters.size(0) - 1;

    // Allocate gradients for parameters and x
    auto parameters_grad = torch::zeros_like(parameters);
    auto x_grad = torch::zeros_like(x);

    int n_tiles_total = (N_in / tile_size) * (N_out / tile_size);
    int n_blocks;

    // Logic to determine n_blocks based on n_tiles_repeat and n_repetitions, similar to forward pass
    if ((n_tiles_repeat == -1) != (n_repetitions == -1)) {
        throw std::invalid_argument("n_tiles_repeat and n_repetitions must either both be provided or both be None.");
    }

    if (n_tiles_repeat == -1) {
         // Default behavior or calculation if not provided, mirroring forward pass if necessary
         // For now, just assume they dictate the block count directly as in forward.
         // This might need adjustment depending on how backward pass parallelization works.
        n_tiles_repeat = n_tiles_total;

        // Get device properties to find number of SMs
        /*cudaDeviceProp prop;
        int device;
        cudaError_t err = cudaGetDevice(&device);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to get current CUDA device.");
        }
        err = cudaGetDeviceProperties(&prop, device); // Get properties for the current device
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to get CUDA device properties.");
        }
        int sm_count = prop.multiProcessorCount;*/
        //int sm_count = 132;
        int desired_grid_size = 512;
        n_repetitions = desired_grid_size / n_tiles_total;

        int upper_bound = 1 + (batch_size - 1) / 32;
        if (n_repetitions > upper_bound) {
            n_repetitions = upper_bound;
        }
        
        if (n_repetitions == 0) {
            n_repetitions = 1;
        }
    }

    n_blocks = n_tiles_total + n_tiles_repeat * (n_repetitions - 1); // Same calculation as forward

    dim3 block_dim(block_size); // Threads per block
    dim3 grid_dim(n_blocks);    // Grid dimensions

    int shared_memory_numel;
    if constexpr (dim == 2) {
        // Size of the first array (shared_parameters)
        int first_array_size = tile_size * tile_size * (n_chunks + 1) * (n_chunks + 1) / 2;

        // Size of the second array (shared_parameters_grad)
        int second_array_size = first_array_size;

        // Calculate padding needed to align to 32-float boundary
        int padding = (32 - ((first_array_size + second_array_size) % 32)) % 32;

        // Size of the third array (grid_points)
        int third_array_size = 32 * (n_chunks + 1);

        // Size of the fourth array (inverse_chunk_lengths)
        int fourth_array_size = 32 * n_chunks;

        // Total shared memory size needed
        if constexpr (fast_mode) {
            shared_memory_numel = first_array_size + 4 * second_array_size + padding + third_array_size + fourth_array_size;
        }
        else {
            shared_memory_numel = first_array_size + second_array_size + padding + third_array_size + fourth_array_size;
        }
    } else {
        static_assert(dim == 2, "dim must be 2");
    }
    std::size_t shared_mem_size = shared_memory_numel * sizeof(float);

    cudaFuncSetAttribute(
        linear_cuda_dkan_full_backward_kernel<dim, tile_size, cdf_grid, batch_last, fast_mode>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_size
    );

    linear_cuda_dkan_full_backward_kernel<dim, tile_size, cdf_grid, batch_last, fast_mode>
        <<<grid_dim, block_dim, shared_mem_size>>>(
            parameters.data_ptr<float>(),
            x.data_ptr<float>(),
            output_grad.data_ptr<float>(),
            parameters_grad.data_ptr<float>(),
            x_grad.data_ptr<float>(),
            n_chunks,
            N_in,
            N_out,
            batch_size,
            n_tiles_repeat,
            n_repetitions
        );

    return std::make_tuple(parameters_grad, x_grad);
}

template <int dim, int block_size, bool cdf_grid, bool batch_last, bool fast_mode>
std::tuple<torch::Tensor, torch::Tensor> linear_cuda_dkan_full_backward(
    torch::Tensor parameters,
    torch::Tensor x,
    torch::Tensor output_grad,
    int tile_size,
    int n_tiles_repeat,
    int n_repetitions
) {
    switch (tile_size) {
        case 4:
            return linear_cuda_dkan_full_tile_size_backward<dim, block_size, cdf_grid, batch_last, fast_mode, 4>(parameters, x, output_grad, n_tiles_repeat, n_repetitions);
        case 8:
            return linear_cuda_dkan_full_tile_size_backward<dim, block_size, cdf_grid, batch_last, fast_mode, 8>(parameters, x, output_grad, n_tiles_repeat, n_repetitions);
        case 16:
            return linear_cuda_dkan_full_tile_size_backward<dim, block_size, cdf_grid, batch_last, fast_mode, 16>(parameters, x, output_grad, n_tiles_repeat, n_repetitions);
        case 32:
            return linear_cuda_dkan_full_tile_size_backward<dim, block_size, cdf_grid, batch_last, fast_mode, 32>(parameters, x, output_grad, n_tiles_repeat, n_repetitions);
        default:
            throw std::invalid_argument("tile_size must be one of {4, 8, 16, 32}");
    }
}

// Inner dispatch: chooses compile-time block size based on runtime value.
template <int dim, bool cdf_grid, bool batch_last, bool fast_mode>
std::tuple<torch::Tensor, torch::Tensor> linear_gpu_dkan_full_backward_inner(
    torch::Tensor parameters,
    torch::Tensor x,
    torch::Tensor output_grad,
    int block_size,
    int tile_size,
    int n_tiles_repeat,
    int n_repetitions
) {
    CHECK_INPUT(parameters);
    CHECK_INPUT(x);
    CHECK_INPUT(output_grad);

    switch (block_size) {
        case 128:
            return linear_cuda_dkan_full_backward<dim, 128, cdf_grid, batch_last, fast_mode>(parameters, x, output_grad, tile_size, n_tiles_repeat, n_repetitions);
        case 256:
            return linear_cuda_dkan_full_backward<dim, 256, cdf_grid, batch_last, fast_mode>(parameters, x, output_grad, tile_size, n_tiles_repeat, n_repetitions);
        case 384:
            return linear_cuda_dkan_full_backward<dim, 384, cdf_grid, batch_last, fast_mode>(parameters, x, output_grad, tile_size, n_tiles_repeat, n_repetitions);
        case 512:
            return linear_cuda_dkan_full_backward<dim, 512, cdf_grid, batch_last, fast_mode>(parameters, x, output_grad, tile_size, n_tiles_repeat, n_repetitions);
        case 640:
            return linear_cuda_dkan_full_backward<dim, 640, cdf_grid, batch_last, fast_mode>(parameters, x, output_grad, tile_size, n_tiles_repeat, n_repetitions);
        case 768:
            return linear_cuda_dkan_full_backward<dim, 768, cdf_grid, batch_last, fast_mode>(parameters, x, output_grad, tile_size, n_tiles_repeat, n_repetitions);
        case 896:
            return linear_cuda_dkan_full_backward<dim, 896, cdf_grid, batch_last, fast_mode>(parameters, x, output_grad, tile_size, n_tiles_repeat, n_repetitions);
        case 1024:
            return linear_cuda_dkan_full_backward<dim, 1024, cdf_grid, batch_last, fast_mode>(parameters, x, output_grad, tile_size, n_tiles_repeat, n_repetitions);
        default:
            throw std::runtime_error("Unsupported block size. Supported block sizes are {128, 256, 384, 512, 640, 768, 896, 1024}.");
    }
}

// Outer runtime dispatcher: decides compile-time flags via the inner helper.
template <int dim>
std::tuple<torch::Tensor, torch::Tensor> linear_gpu_dkan_full_backward(
    torch::Tensor parameters,
    torch::Tensor x,
    torch::Tensor output_grad,
    int block_size,
    bool cdf_grid,
    int tile_size,
    bool batch_last,
    bool fast_mode,
    int n_tiles_repeat,
    int n_repetitions
) {
    if (fast_mode) {
        if (cdf_grid) {
            if (batch_last) {
                return linear_gpu_dkan_full_backward_inner<dim, true, true, true>(parameters, x, output_grad, block_size, tile_size, n_tiles_repeat, n_repetitions);
            } else {
                return linear_gpu_dkan_full_backward_inner<dim, true, false, true>(parameters, x, output_grad, block_size, tile_size, n_tiles_repeat, n_repetitions);
            }
        } else {
            if (batch_last) {
                return linear_gpu_dkan_full_backward_inner<dim, false, true, true>(parameters, x, output_grad, block_size, tile_size, n_tiles_repeat, n_repetitions);
            } else {
                return linear_gpu_dkan_full_backward_inner<dim, false, false, true>(parameters, x, output_grad, block_size, tile_size, n_tiles_repeat, n_repetitions);
            }
        }
    } else {
        if (cdf_grid) {
            if (batch_last) {
                return linear_gpu_dkan_full_backward_inner<dim, true, true, false>(parameters, x, output_grad, block_size, tile_size, n_tiles_repeat, n_repetitions);
            } else {
                return linear_gpu_dkan_full_backward_inner<dim, true, false, false>(parameters, x, output_grad, block_size, tile_size, n_tiles_repeat, n_repetitions);
            }
        } else {
            if (batch_last) {
                return linear_gpu_dkan_full_backward_inner<dim, false, true, false>(parameters, x, output_grad, block_size, tile_size, n_tiles_repeat, n_repetitions);
            } else {
                return linear_gpu_dkan_full_backward_inner<dim, false, false, false>(parameters, x, output_grad, block_size, tile_size, n_tiles_repeat, n_repetitions);
            }
        }
    }
}

// Wrapper function similar to forward pass to handle optional arguments from Python
template <int dim>
std::tuple<torch::Tensor, torch::Tensor> linear_gpu_dkan_full_backward_pybind_wrapper(
    torch::Tensor parameters,
    torch::Tensor x,
    torch::Tensor output_grad,
    int block_size,
    bool cdf_grid,
    int tile_size,
    bool batch_last,
    bool fast_mode,
    std::optional<int> n_tiles_repeat_opt,
    std::optional<int> n_repetitions_opt
) {
    if (fast_mode && (tile_size != 8 || block_size != 128)) {
        throw std::invalid_argument("When fast_mode is true, tile_size must be 8 and block_size must be 128.");
    }
    // Convert optional<int> to int, defaulting to -1 if nullopt (None from Python)
    int n_tiles_repeat = n_tiles_repeat_opt.value_or(-1);
    int n_repetitions = n_repetitions_opt.value_or(-1);

    // Call the original function (now renamed slightly or reused) with the converted int values
    return linear_gpu_dkan_full_backward<dim>(
        parameters,
        x,
        output_grad,
        block_size,
        cdf_grid,
        tile_size,
        batch_last,
        fast_mode,
        n_tiles_repeat,
        n_repetitions
    );
}