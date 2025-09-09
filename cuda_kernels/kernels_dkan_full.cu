#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <optional>
using namespace torch::indexing;

#include "utilities.h"


#include "dkan_full_2d_implementations/dkan_full_2d_thread_per_tile_batch_last_cdf_grid.cu"

template <int dim, int tile_size, bool cdf_grid, bool batch_last>
__global__ void linear_cuda_dkan_full_forward_kernel(
    const float* __restrict__ parameters,
    const float* __restrict__ x,
    float* __restrict__ output,
    int n_chunks,
    int N_in,
    int N_out,
    int batch_size,
    int n_tiles_repeat,
    int n_repetitions
) {

    //N_in horizontal; N_out vertical
    // suptile size is 4 * tile_size

    /*int suprow_size = 4 * N_in;
    int t = blockIdx.x;
    int suprow_index = t / suprow_size;
    t = t % suprow_size;

    int current_height;
    if (suprow_index == )

*/

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
    extern __shared__ char buffer[];

    char* shared_mem = buffer;

    // Reinterpret the char buffer as float* for shared parameters
    float* shared_parameters = reinterpret_cast<float*>(shared_mem);

    // Calculate the starting index for the second array
    int shared_parameters_size = tile_size * tile_size * (n_chunks + 1) * (n_chunks + 1) / 2;
    int padding = (32 - (shared_parameters_size % 32)) % 32;
    int grid_points_offset = shared_parameters_size + padding;
    float* grid_points = shared_parameters + grid_points_offset;

    // Calculate the starting index for the third array (inverse_chunk_lengths)
    int grid_points_size = 32 * (n_chunks + 1);
    int inverse_chunk_lengths_offset = grid_points_offset + grid_points_size;
    float* inverse_chunk_lengths = shared_parameters + inverse_chunk_lengths_offset;

    copy_parameters_to_shared_full_dkan<tile_size>(parameters, shared_parameters, n_chunks, my_tile_in, my_tile_out, N_in, N_out);

    if constexpr (dim == 2) {
        if constexpr (batch_last) {
            if constexpr (cdf_grid) {
                precompute_grid_points(grid_points, inverse_chunk_lengths, n_chunks);
                dkan_full_kernel_2d_thread_per_tile_batch_last_cdf_grid<tile_size>(
                    shared_parameters,
                    grid_points,
                    inverse_chunk_lengths,
                    x,
                    output,
                    my_tile_in,
                    my_tile_out,
                    N_in,
                    N_out,
                    n_chunks,
                    batch_size,
                    batch_index_from,
                    batch_index_to
                );
            } else {
                /*dkan_full_kernel_2d_thread_per_tile_batch_last<tile_size>(
                    shared_parameters,
                    x,
                    output,
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
            /*dkan_full_kernel_2d_thread_per_four_rows<tile_size, cdf_grid>(
                shared_parameters,
                x,
                output,
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
}

template <int dim, int block_size, bool cdf_grid, bool batch_last, int tile_size>
torch::Tensor linear_cuda_dkan_full_tile_size_forward(
    torch::Tensor parameters,
    torch::Tensor x,
    int n_tiles_repeat,
    int n_repetitions
) {
    TORCH_CHECK(
        parameters.dtype() == torch::kFloat32 &&
        x.dtype()         == torch::kFloat32,
        "linear_cuda_dkan_full_tile_size_forward: Expected all tensors to be float32."
    );

    // Ensure the input tensors are contiguous and on the GPU
    CHECK_INPUT(parameters);
    CHECK_INPUT(x);

    // When batch_last is false:
    // x shape: [batch_size, N_in]
    // When batch_last is true:
    // x shape: [N_in, batch_size]
    // parameters shape in 2d case: [n_chunks + 1, n_chunks + 1, N_out, N_in / 2]

    int N_in, N_out, batch_size;

    if constexpr (batch_last) {
        N_in = x.size(0);
        batch_size = x.size(1);
    } else {
        N_in = x.size(1);
        batch_size = x.size(0);
    }

    N_out = parameters.size(2);

    if (N_in % tile_size != 0) {
        throw std::invalid_argument("N_in must be divisible by tile_size");
    }

    if (N_out % tile_size != 0) {
        throw std::invalid_argument("N_out must be divisible by tile_size");
    }

    int n_chunks = parameters.size(0) - 1;

    // Create output tensor with appropriate shape based on batch_last
    torch::Tensor output;
    if constexpr (batch_last) {
        output = torch::zeros({N_out, batch_size}, x.options());
    } else {
        output = torch::zeros({batch_size, N_out}, x.options());
    }

    int n_tiles_total = (N_in / tile_size) * (N_out / tile_size);
    int n_blocks;

    // Check if arguments are provided consistently
    if ((n_tiles_repeat == -1) != (n_repetitions == -1)) {
        throw std::invalid_argument("n_tiles_repeat and n_repetitions must either both be provided or both be None.");
    }

    if (n_tiles_repeat == -1) {
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

    n_blocks = n_tiles_total + n_tiles_repeat * (n_repetitions - 1);

    dim3 block_dim(block_size); // Number of threads per block
    dim3 grid_dim(n_blocks);    // Number of blocks in grid

    int shared_memory_numel;
    if constexpr (dim == 2) {
        // Size of the first array
        int first_array_size = tile_size * tile_size * (n_chunks + 1) * (n_chunks + 1) / 2;

        // Calculate padding needed to align to 32-float boundary
        int padding = (32 - (first_array_size % 32)) % 32;

        // Size of the second array
        int second_array_size = 32 * (n_chunks + 1);

        // Size of the third array
        int third_array_size = 32 * n_chunks;

        // Total shared memory size needed
        shared_memory_numel = first_array_size + padding + second_array_size + third_array_size;
    } else {
        static_assert(dim == 2, "dim must be 2");
    }

    std::size_t shared_mem_size = shared_memory_numel * sizeof(float);


    cudaFuncSetAttribute(
        linear_cuda_dkan_full_forward_kernel<dim, tile_size, cdf_grid, batch_last>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_size
    );

    linear_cuda_dkan_full_forward_kernel<dim, tile_size, cdf_grid, batch_last>
        <<<grid_dim, block_dim, shared_mem_size>>>(
            parameters.data_ptr<float>(),
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            n_chunks,
            N_in,
            N_out,
            batch_size,
            n_tiles_repeat,
            n_repetitions
        );

    return output;
}

template <int dim, int block_size, bool cdf_grid, bool batch_last>
torch::Tensor linear_cuda_dkan_full_forward(
    torch::Tensor parameters,
    torch::Tensor x,
    int tile_size,
    int n_tiles_repeat,
    int n_repetitions
) {
    // Just pick the correct instantiation based on tile_size
    switch (tile_size) {
        case 4:
            return linear_cuda_dkan_full_tile_size_forward<dim, block_size, cdf_grid, batch_last, 4>(parameters, x, n_tiles_repeat, n_repetitions);
        case 8:
            return linear_cuda_dkan_full_tile_size_forward<dim, block_size, cdf_grid, batch_last, 8>(parameters, x, n_tiles_repeat, n_repetitions);
        case 16:
            return linear_cuda_dkan_full_tile_size_forward<dim, block_size, cdf_grid, batch_last, 16>(parameters, x, n_tiles_repeat, n_repetitions);
        case 32:
            return linear_cuda_dkan_full_tile_size_forward<dim, block_size, cdf_grid, batch_last, 32>(parameters, x, n_tiles_repeat, n_repetitions);
        default:
            throw std::invalid_argument("tile_size must be one of {4, 8, 16, 32}");
    }
}


template <int dim, bool cdf_grid, bool batch_last>
torch::Tensor linear_gpu_dkan_full_forward_inner_batch(
    torch::Tensor parameters,
    torch::Tensor x,
    int block_size,
    int tile_size,
    int n_tiles_repeat,
    int n_repetitions
) {
    CHECK_INPUT(parameters);
    CHECK_INPUT(x);

    switch (block_size) {
        case 128:
            return linear_cuda_dkan_full_forward<dim, 128, cdf_grid, batch_last>(parameters, x, tile_size, n_tiles_repeat, n_repetitions);
        case 256:
            return linear_cuda_dkan_full_forward<dim, 256, cdf_grid, batch_last>(parameters, x, tile_size, n_tiles_repeat, n_repetitions);
        case 384:
            return linear_cuda_dkan_full_forward<dim, 384, cdf_grid, batch_last>(parameters, x, tile_size, n_tiles_repeat, n_repetitions);
        case 512:
            return linear_cuda_dkan_full_forward<dim, 512, cdf_grid, batch_last>(parameters, x, tile_size, n_tiles_repeat, n_repetitions);
        case 640:
            return linear_cuda_dkan_full_forward<dim, 640, cdf_grid, batch_last>(parameters, x, tile_size, n_tiles_repeat, n_repetitions);
        case 768:
            return linear_cuda_dkan_full_forward<dim, 768, cdf_grid, batch_last>(parameters, x, tile_size, n_tiles_repeat, n_repetitions);
        case 896:
            return linear_cuda_dkan_full_forward<dim, 896, cdf_grid, batch_last>(parameters, x, tile_size, n_tiles_repeat, n_repetitions);
        case 1024:
            return linear_cuda_dkan_full_forward<dim, 1024, cdf_grid, batch_last>(parameters, x, tile_size, n_tiles_repeat, n_repetitions);
        default:
            throw std::runtime_error(
                "Unsupported block size. Supported block sizes are {128, 256, 384, 512, 640, 768, 896, 1024}."
            );
    }
}

template <int dim, bool cdf_grid>
torch::Tensor linear_gpu_dkan_full_forward_inner(
    torch::Tensor parameters,
    torch::Tensor x,
    int block_size,
    int tile_size,
    bool batch_last,
    int n_tiles_repeat,
    int n_repetitions
) {
    if (batch_last) {
        return linear_gpu_dkan_full_forward_inner_batch<dim, cdf_grid, true>(parameters, x, block_size, tile_size, n_tiles_repeat, n_repetitions);
    } else {
        return linear_gpu_dkan_full_forward_inner_batch<dim, cdf_grid, false>(parameters, x, block_size, tile_size, n_tiles_repeat, n_repetitions);
    }
}

template <int dim>
torch::Tensor linear_gpu_dkan_full_forward(
    torch::Tensor parameters,
    torch::Tensor x,
    int block_size,
    bool cdf_grid,
    int tile_size,
    bool batch_last,
    int n_tiles_repeat,
    int n_repetitions
) {
    if (cdf_grid) {
        return linear_gpu_dkan_full_forward_inner<dim, true>(parameters, x, block_size, tile_size, batch_last, n_tiles_repeat, n_repetitions);
    } else {
        return linear_gpu_dkan_full_forward_inner<dim, false>(parameters, x, block_size, tile_size, batch_last, n_tiles_repeat, n_repetitions);
    }
}

// New wrapper function to handle optional arguments from Python
template <int dim>
torch::Tensor linear_gpu_dkan_full_forward_pybind_wrapper(
    torch::Tensor parameters,
    torch::Tensor x,
    int block_size,
    bool cdf_grid,
    int tile_size,
    bool batch_last,
    std::optional<int> n_tiles_repeat_opt,
    std::optional<int> n_repetitions_opt
) {

    // Convert optional<int> to int, defaulting to -1 if nullopt (None from Python)
    int n_tiles_repeat = n_tiles_repeat_opt.value_or(-1);
    int n_repetitions = n_repetitions_opt.value_or(-1);

    // Call the original function with the converted int values
    return linear_gpu_dkan_full_forward<dim>(
        parameters,
        x,
        block_size,
        cdf_grid,
        tile_size,
        batch_last,
        n_tiles_repeat,
        n_repetitions
    );
}