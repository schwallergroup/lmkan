#ifndef UTILITIES_H
#define UTILITIES_H

#include <torch/extension.h> // To include TORCH_CHECK macros

// Macro to check if the tensor is CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

// Macro to check if the tensor is contiguous
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Macro to check both CUDA and contiguous properties of the tensor
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


__device__ inline void precompute_grid_points(
    float* __restrict__ grid_points,
    float* __restrict__ inverse_chunk_lengths,
    int n_chunks) {

    int thread_idx = threadIdx.x;
    float value;
    float chunk_size = 1.0f / static_cast<float>(n_chunks);

    if (thread_idx < 32) {
        for (int chunk_index = 1; chunk_index < n_chunks; ++chunk_index) {
            value = static_cast<float>(chunk_index) * chunk_size;
            value = (value <= 0.5f)
            ? __logf(2.0f * value)
            : -__logf(2.0f * (1.0f - value));
            grid_points[chunk_index * 32 + thread_idx] = value;
        }
    }

    __syncthreads();

    if (thread_idx < 32) {
        grid_points[0 * 32 + thread_idx] = grid_points[1 * 32 + thread_idx] - (grid_points[2 * 32 + thread_idx] - grid_points[1 * 32 + thread_idx]);
        grid_points[n_chunks * 32 + thread_idx] = grid_points[(n_chunks - 1) * 32 + thread_idx] + (grid_points[(n_chunks - 1) * 32 + thread_idx] - grid_points[(n_chunks - 2) * 32 + thread_idx]);
    }
    __syncthreads();

    // Compute inverse chunk lengths
    if (thread_idx < 32) {
        for (int i = 0; i < n_chunks; ++i) {
            inverse_chunk_lengths[i * 32 + thread_idx] = 1.0f / (grid_points[(i + 1) * 32 + thread_idx] - grid_points[i * 32 + thread_idx]);
        }
    }
    __syncthreads();
}
// Full definition here to avoid problems with instantiation
template <typename scalar_t>
__device__ void copy_parameters_to_shared(
    const scalar_t* __restrict__ parameters,
    scalar_t* shared_parameters,
    int n_params,
    int t
) {
    int thread_idx = threadIdx.x;
    int block_size = blockDim.x;

    // Calculate the base global index for the row to be copied
    const scalar_t* parameters_source_now = parameters + t * n_params;

    // Copy elements from global memory to shared memory
    for (int i = thread_idx; i < n_params; i += block_size) {
        shared_parameters[i] = parameters_source_now[i];
    }

    // Ensure all threads have finished copying before proceeding
    __syncthreads();
}

template <int tile_size>
__device__ void copy_parameters_to_shared_full_dkan(
    const float* __restrict__ parameters,
    float* __restrict__ shared_parameters,
    int n_chunks,
    int my_tile_in,   // now corresponds to the global last dim (N_in/2)
    int my_tile_out,  // now corresponds to the global third dim (N_out)
    int N_in,
    int N_out) {

    int thread_idx = threadIdx.x;
    int block_size = blockDim.x;

    // Tile shape: [n_chunks+1, n_chunks+1, tile_size, tile_size/2]
    // The strides (assuming row-major order) are:
    //   - For the first tile dimension (function index 0): (n_chunks+1)*tile_size*(tile_size/2)
    //   - For the second tile dimension (function index 1): tile_size*(tile_size/2)
    //   - For the third tile dimension: tile_size/2
    const int tile_first_func_index_stride = (n_chunks + 1) * tile_size * (tile_size / 2);
    const int tile_second_func_index_stride = tile_size * (tile_size / 2);
    const int tile_first_tile_index_stride = (tile_size / 2);

    // Global array shape: [n_chunks+1, n_chunks+1, N_out, N_in/2]
    // The strides are:
    //   - Dim0 (first function index): (n_chunks+1)*N_out*(N_in/2)
    //   - Dim1 (second function index): N_out*(N_in/2)
    //   - Dim2 (third dim, corresponding to tile’s 3rd dim): N_in/2
    //   - Dim3 (last dim): 1
    const int full_first_func_index_stride = (n_chunks + 1) * N_out * (N_in / 2);
    const int full_second_func_index_stride = N_out * (N_in / 2);
    const int full_first_tile_index_stride = (N_in / 2); // stride for the third global dimension

    // Total number of elements in the tile.
    const int n_param_copy = (n_chunks + 1) * (n_chunks + 1) * tile_size * (tile_size / 2);

    int full_tile_index;
    int tile_first_func_index, tile_second_func_index, tile_first_tile_index, tile_second_tile_index;
    int source_index;

    // Each thread copies a subset of the tile
    for (int i = thread_idx; i < n_param_copy; i += block_size) {
        full_tile_index = i;

        // Extract the tile indices.
        tile_first_func_index = full_tile_index / tile_first_func_index_stride;
        full_tile_index %= tile_first_func_index_stride;

        tile_second_func_index = full_tile_index / tile_second_func_index_stride;
        full_tile_index %= tile_second_func_index_stride;

        tile_first_tile_index = full_tile_index / tile_first_tile_index_stride;
        tile_second_tile_index = full_tile_index % tile_first_tile_index_stride;

        // Now compute the corresponding global memory index.
        // For the global memory, note that:
        //  - The third dimension (of size N_out) is given by tile_first_tile_index plus an offset my_tile_out * tile_size.
        //  - The fourth dimension (of size N_in/2) is given by tile_second_tile_index plus an offset my_tile_in * (tile_size/2).
        source_index  = tile_first_func_index * full_first_func_index_stride;
        source_index += tile_second_func_index * full_second_func_index_stride;
        source_index += (tile_first_tile_index + my_tile_out * tile_size) * full_first_tile_index_stride;
        source_index += tile_second_tile_index + my_tile_in * (tile_size / 2);

        shared_parameters[i] = parameters[source_index];
        //shared_parameters[i] = static_cast<float>(i);
    }
    __syncthreads();
}

template <int tile_size>
__device__ void zero_shared_parameters_grad_full_dkan(
    float* __restrict__ shared_parameters_grad,
    int n_chunks
) {
    int thread_idx = threadIdx.x;
    int block_size = blockDim.x;

    // Total number of elements in the shared parameters gradient tile.
    const int n_param_grad_elements = (n_chunks + 1) * (n_chunks + 1) * tile_size * (tile_size / 2);

    for (int i = thread_idx; i < n_param_grad_elements; i += block_size) {
        shared_parameters_grad[i] = 0.0f;
    }
    __syncthreads();
}

template <int tile_size>
__device__ void accumulate_shared_parameters_grad_full_dkan_fast_mode(
    float* __restrict__ parameters_grad,
    const float* __restrict__ shared_parameters_grad_first,
    const float* __restrict__ shared_parameters_grad_second,
    const float* __restrict__ shared_parameters_grad_third,
    const float* __restrict__ shared_parameters_grad_fourth,
    int n_chunks,
    int my_tile_in,   // now corresponds to the global last dim (N_in/2)
    int my_tile_out,  // now corresponds to the global third dim (N_out)
    int N_in,
    int N_out) {

    int thread_idx = threadIdx.x;
    int block_size = blockDim.x;

    const int tile_first_func_index_stride = (n_chunks + 1) * tile_size * (tile_size / 2);
    const int tile_second_func_index_stride = tile_size * (tile_size / 2);
    const int tile_first_tile_index_stride = (tile_size / 2);

    const int full_first_func_index_stride = (n_chunks + 1) * N_out * (N_in / 2);
    const int full_second_func_index_stride = N_out * (N_in / 2);
    const int full_first_tile_index_stride = (N_in / 2); // stride for the third global dimension

    // Total number of elements in the tile.
    const int n_param_copy = (n_chunks + 1) * (n_chunks + 1) * tile_size * (tile_size / 2);

    int full_tile_index;
    int tile_first_func_index, tile_second_func_index, tile_first_tile_index, tile_second_tile_index;
    int source_index;

    // Each thread copies a subset of the tile
    for (int i = thread_idx; i < n_param_copy; i += block_size) {
        full_tile_index = i;

        // Extract the tile indices.
        tile_first_func_index = full_tile_index / tile_first_func_index_stride;
        full_tile_index %= tile_first_func_index_stride;

        tile_second_func_index = full_tile_index / tile_second_func_index_stride;
        full_tile_index %= tile_second_func_index_stride;

        tile_first_tile_index = full_tile_index / tile_first_tile_index_stride;
        tile_second_tile_index = full_tile_index % tile_first_tile_index_stride;

        source_index  = tile_first_func_index * full_first_func_index_stride;
        source_index += tile_second_func_index * full_second_func_index_stride;
        source_index += (tile_first_tile_index + my_tile_out * tile_size) * full_first_tile_index_stride;
        source_index += tile_second_tile_index + my_tile_in * (tile_size / 2);

        float value_now = shared_parameters_grad_first[i] + shared_parameters_grad_second[i] + shared_parameters_grad_third[i] + shared_parameters_grad_fourth[i];

        atomicAdd(parameters_grad + source_index, value_now);
    }
    __syncthreads();
}


template <int tile_size>
__device__ void accumulate_shared_parameters_grad_full_dkan(
    float* __restrict__ parameters_grad,
    const float* __restrict__ shared_parameters_grad,
    int n_chunks,
    int my_tile_in,   // now corresponds to the global last dim (N_in/2)
    int my_tile_out,  // now corresponds to the global third dim (N_out)
    int N_in,
    int N_out) {

    int thread_idx = threadIdx.x;
    int block_size = blockDim.x;

    const int tile_first_func_index_stride = (n_chunks + 1) * tile_size * (tile_size / 2);
    const int tile_second_func_index_stride = tile_size * (tile_size / 2);
    const int tile_first_tile_index_stride = (tile_size / 2);

    const int full_first_func_index_stride = (n_chunks + 1) * N_out * (N_in / 2);
    const int full_second_func_index_stride = N_out * (N_in / 2);
    const int full_first_tile_index_stride = (N_in / 2); // stride for the third global dimension

    // Total number of elements in the tile.
    const int n_param_copy = (n_chunks + 1) * (n_chunks + 1) * tile_size * (tile_size / 2);

    int full_tile_index;
    int tile_first_func_index, tile_second_func_index, tile_first_tile_index, tile_second_tile_index;
    int source_index;

    // Each thread copies a subset of the tile
    for (int i = thread_idx; i < n_param_copy; i += block_size) {
        full_tile_index = i;

        // Extract the tile indices.
        tile_first_func_index = full_tile_index / tile_first_func_index_stride;
        full_tile_index %= tile_first_func_index_stride;

        tile_second_func_index = full_tile_index / tile_second_func_index_stride;
        full_tile_index %= tile_second_func_index_stride;

        tile_first_tile_index = full_tile_index / tile_first_tile_index_stride;
        tile_second_tile_index = full_tile_index % tile_first_tile_index_stride;

        source_index  = tile_first_func_index * full_first_func_index_stride;
        source_index += tile_second_func_index * full_second_func_index_stride;
        source_index += (tile_first_tile_index + my_tile_out * tile_size) * full_first_tile_index_stride;
        source_index += tile_second_tile_index + my_tile_in * (tile_size / 2);

        atomicAdd(parameters_grad + source_index, shared_parameters_grad[i]);
    }
    __syncthreads();
}

// Simplified inline PTX cyclic in-place shift for an 8-element array
// using eight temporary registers (t0,...,t7) without redundant moves in stage 3
__device__ inline void ptx_shift_8_inplace(float *array, int fast_index) {
    // https://chatgpt.com/c/67c9e218-48d8-800e-ac36-4a7befb04b6a
    asm volatile(
        "{\n\t"
        "  .reg .pred p4, p2, p1;\n\t"
        "  .reg .u32 bit;\n\t"
        "  .reg .f32 t0, t1, t2, t3, t4, t5, t6, t7;\n\t"

        // Extract predicate bits from fast_index:
        "  bfe.u32 bit, %8, 2, 1;\n\t"  // bit2 => shift-by-4
        "  setp.ne.u32 p4, bit, 0;\n\t"
        "  bfe.u32 bit, %8, 1, 1;\n\t"  // bit1 => shift-by-2
        "  setp.ne.u32 p2, bit, 0;\n\t"
        "  bfe.u32 bit, %8, 0, 1;\n\t"  // bit0 => shift-by-1
        "  setp.ne.u32 p1, bit, 0;\n\t"

        // Stage 1: Conditional shift by 4.
        "  selp.f32 t0, %4, %0, p4;\n\t" // t0 = (p4 ? array[4] : array[0])
        "  selp.f32 t1, %5, %1, p4;\n\t" // t1 = (p4 ? array[5] : array[1])
        "  selp.f32 t2, %6, %2, p4;\n\t" // t2 = (p4 ? array[6] : array[2])
        "  selp.f32 t3, %7, %3, p4;\n\t" // t3 = (p4 ? array[7] : array[3])
        "  selp.f32 t4, %0, %4, p4;\n\t" // t4 = (p4 ? array[0] : array[4])
        "  selp.f32 t5, %1, %5, p4;\n\t" // t5 = (p4 ? array[1] : array[5])
        "  selp.f32 t6, %2, %6, p4;\n\t" // t6 = (p4 ? array[2] : array[6])
        "  selp.f32 t7, %3, %7, p4;\n\t" // t7 = (p4 ? array[3] : array[7])

        // Stage 2: Conditional shift by 2.
        "  selp.f32 %0, t2, t0, p2;\n\t" // array[0] = (p2 ? t2 : t0)
        "  selp.f32 %1, t3, t1, p2;\n\t" // array[1] = (p2 ? t3 : t1)
        "  selp.f32 %2, t4, t2, p2;\n\t" // array[2] = (p2 ? t4 : t2)
        "  selp.f32 %3, t5, t3, p2;\n\t" // array[3] = (p2 ? t5 : t3)
        "  selp.f32 %4, t6, t4, p2;\n\t" // array[4] = (p2 ? t6 : t4)
        "  selp.f32 %5, t7, t5, p2;\n\t" // array[5] = (p2 ? t7 : t5)
        "  selp.f32 %6, t0, t6, p2;\n\t" // array[6] = (p2 ? t0 : t6)
        "  selp.f32 %7, t1, t7, p2;\n\t" // array[7] = (p2 ? t1 : t7)

        // Stage 3: Conditional shift by 1 (directly using the array registers).
        "  selp.f32 t0, %1, %0, p1;\n\t" // t0 = (p1 ? array[1] : array[0])
        "  selp.f32 t1, %2, %1, p1;\n\t" // t1 = (p1 ? array[2] : array[1])
        "  selp.f32 t2, %3, %2, p1;\n\t" // t2 = (p1 ? array[3] : array[2])
        "  selp.f32 t3, %4, %3, p1;\n\t" // t3 = (p1 ? array[4] : array[3])
        "  selp.f32 t4, %5, %4, p1;\n\t" // t4 = (p1 ? array[5] : array[4])
        "  selp.f32 t5, %6, %5, p1;\n\t" // t5 = (p1 ? array[6] : array[5])
        "  selp.f32 t6, %7, %6, p1;\n\t" // t6 = (p1 ? array[7] : array[6])
        "  selp.f32 t7, %0, %7, p1;\n\t" // t7 = (p1 ? array[0] : array[7])

        // Stage 4: Write final result back to array.
        "  mov.f32 %0, t0;\n\t"
        "  mov.f32 %1, t1;\n\t"
        "  mov.f32 %2, t2;\n\t"
        "  mov.f32 %3, t3;\n\t"
        "  mov.f32 %4, t4;\n\t"
        "  mov.f32 %5, t5;\n\t"
        "  mov.f32 %6, t6;\n\t"
        "  mov.f32 %7, t7;\n\t"
        "}\n\t"
        : "+f"(array[0]), "+f"(array[1]), "+f"(array[2]), "+f"(array[3]),
          "+f"(array[4]), "+f"(array[5]), "+f"(array[6]), "+f"(array[7])
        : "r"(fast_index)
        : "memory"
    );
}

// Inline PTX cyclic in-place shift for a 16-element array.
// The shift (modulo 16) is decomposed into conditional shifts of 8, 4, 2, and 1.
// After stage 4 the final result is in the array registers.
__device__ inline void ptx_shift_16_inplace(float *array, int fast_index) {
    asm volatile(
        "{\n\t"
        "    .reg .pred p8, p4, p2, p1;\n\t"
        "    .reg .u32 bit;\n\t"
        "    .reg .f32 t0, t1, t2, t3, t4, t5, t6, t7, "
        "               t8, t9, t10, t11, t12, t13, t14, t15;\n\t"

        // Extract predicate bits from fast_index:
        // p8 controls a shift by 8 (bit 3),
        // p4 controls a shift by 4 (bit 2),
        // p2 controls a shift by 2 (bit 1),
        // p1 controls a shift by 1 (bit 0).
        "    bfe.u32 bit, %16, 3, 1;\n\t"
        "    setp.ne.u32 p8, bit, 0;\n\t"
        "    bfe.u32 bit, %16, 2, 1;\n\t"
        "    setp.ne.u32 p4, bit, 0;\n\t"
        "    bfe.u32 bit, %16, 1, 1;\n\t"
        "    setp.ne.u32 p2, bit, 0;\n\t"
        "    bfe.u32 bit, %16, 0, 1;\n\t"
        "    setp.ne.u32 p1, bit, 0;\n\t"

        // Stage 1: Conditional shift by 8.
        // For indices 0..7: t[i] = (p8 ? array[i+8] : array[i])
        "    selp.f32 t0,  %8,  %0, p8;\n\t"
        "    selp.f32 t1,  %9,  %1, p8;\n\t"
        "    selp.f32 t2,  %10, %2, p8;\n\t"
        "    selp.f32 t3,  %11, %3, p8;\n\t"
        "    selp.f32 t4,  %12, %4, p8;\n\t"
        "    selp.f32 t5,  %13, %5, p8;\n\t"
        "    selp.f32 t6,  %14, %6, p8;\n\t"
        "    selp.f32 t7,  %15, %7, p8;\n\t"
        // For indices 8..15: t[i] = (p8 ? array[i-8] : array[i])
        "    selp.f32 t8,  %0,  %8, p8;\n\t"
        "    selp.f32 t9,  %1,  %9, p8;\n\t"
        "    selp.f32 t10, %2,  %10, p8;\n\t"
        "    selp.f32 t11, %3,  %11, p8;\n\t"
        "    selp.f32 t12, %4,  %12, p8;\n\t"
        "    selp.f32 t13, %5,  %13, p8;\n\t"
        "    selp.f32 t14, %6,  %14, p8;\n\t"
        "    selp.f32 t15, %7,  %15, p8;\n\t"

        // Stage 2: Conditional shift by 4.
        // For each index i, array[i] = (p4 ? t[(i+4) mod 16] : t[i])
        "    selp.f32 %0,  t4,  t0,  p4;\n\t"  // index 0: (0+4 = 4)
        "    selp.f32 %1,  t5,  t1,  p4;\n\t"  // index 1: (1+4 = 5)
        "    selp.f32 %2,  t6,  t2,  p4;\n\t"  // index 2: (2+4 = 6)
        "    selp.f32 %3,  t7,  t3,  p4;\n\t"  // index 3: (3+4 = 7)
        "    selp.f32 %4,  t8,  t4,  p4;\n\t"  // index 4: (4+4 = 8)
        "    selp.f32 %5,  t9,  t5,  p4;\n\t"  // index 5: (5+4 = 9)
        "    selp.f32 %6,  t10, t6,  p4;\n\t"  // index 6: (6+4 = 10)
        "    selp.f32 %7,  t11, t7,  p4;\n\t"  // index 7: (7+4 = 11)
        "    selp.f32 %8,  t12, t8,  p4;\n\t"  // index 8: (8+4 = 12)
        "    selp.f32 %9,  t13, t9,  p4;\n\t"  // index 9: (9+4 = 13)
        "    selp.f32 %10, t14, t10, p4;\n\t"  // index 10: (10+4 = 14)
        "    selp.f32 %11, t15, t11, p4;\n\t"  // index 11: (11+4 = 15)
        "    selp.f32 %12, t0,  t12, p4;\n\t"  // index 12: (12+4 mod16 = 0)
        "    selp.f32 %13, t1,  t13, p4;\n\t"  // index 13: (13+4 mod16 = 1)
        "    selp.f32 %14, t2,  t14, p4;\n\t"  // index 14: (14+4 mod16 = 2)
        "    selp.f32 %15, t3,  t15, p4;\n\t"  // index 15: (15+4 mod16 = 3)

        // Stage 3: Conditional shift by 2.
        // Move from the array registers into temporaries:
        // For each index i, t[i] = (p2 ? array[(i+2) mod 16] : array[i])
        "    selp.f32 t0,  %2,  %0, p2;\n\t"  // index 0: (0+2 = 2)
        "    selp.f32 t1,  %3,  %1, p2;\n\t"  // index 1: (1+2 = 3)
        "    selp.f32 t2,  %4,  %2, p2;\n\t"  // index 2: (2+2 = 4)
        "    selp.f32 t3,  %5,  %3, p2;\n\t"  // index 3: (3+2 = 5)
        "    selp.f32 t4,  %6,  %4, p2;\n\t"  // index 4: (4+2 = 6)
        "    selp.f32 t5,  %7,  %5, p2;\n\t"  // index 5: (5+2 = 7)
        "    selp.f32 t6,  %8,  %6, p2;\n\t"  // index 6: (6+2 = 8)
        "    selp.f32 t7,  %9,  %7, p2;\n\t"  // index 7: (7+2 = 9)
        "    selp.f32 t8,  %10, %8, p2;\n\t"  // index 8: (8+2 = 10)
        "    selp.f32 t9,  %11, %9, p2;\n\t"  // index 9: (9+2 = 11)
        "    selp.f32 t10, %12, %10, p2;\n\t" // index 10: (10+2 = 12)
        "    selp.f32 t11, %13, %11, p2;\n\t" // index 11: (11+2 = 13)
        "    selp.f32 t12, %14, %12, p2;\n\t" // index 12: (12+2 = 14)
        "    selp.f32 t13, %15, %13, p2;\n\t" // index 13: (13+2 = 15)
        "    selp.f32 t14, %0,  %14, p2;\n\t" // index 14: (14+2 mod16 = 0)
        "    selp.f32 t15, %1,  %15, p2;\n\t" // index 15: (15+2 mod16 = 1)

        // Stage 4: Conditional shift by 1.
        // Write the final result directly into the array registers:
        // For each index i, array[i] = (p1 ? t[(i+1) mod 16] : t[i])
        "    selp.f32 %0,  t1,  t0,  p1;\n\t"  // index 0: (0+1 = 1)
        "    selp.f32 %1,  t2,  t1,  p1;\n\t"  // index 1: (1+1 = 2)
        "    selp.f32 %2,  t3,  t2,  p1;\n\t"  // index 2: (2+1 = 3)
        "    selp.f32 %3,  t4,  t3,  p1;\n\t"  // index 3: (3+1 = 4)
        "    selp.f32 %4,  t5,  t4,  p1;\n\t"  // index 4: (4+1 = 5)
        "    selp.f32 %5,  t6,  t5,  p1;\n\t"  // index 5: (5+1 = 6)
        "    selp.f32 %6,  t7,  t6,  p1;\n\t"  // index 6: (6+1 = 7)
        "    selp.f32 %7,  t8,  t7,  p1;\n\t"  // index 7: (7+1 = 8)
        "    selp.f32 %8,  t9,  t8,  p1;\n\t"  // index 8: (8+1 = 9)
        "    selp.f32 %9,  t10, t9,  p1;\n\t"  // index 9: (9+1 = 10)
        "    selp.f32 %10, t11, t10, p1;\n\t"  // index 10: (10+1 = 11)
        "    selp.f32 %11, t12, t11, p1;\n\t"  // index 11: (11+1 = 12)
        "    selp.f32 %12, t13, t12, p1;\n\t"  // index 12: (12+1 = 13)
        "    selp.f32 %13, t14, t13, p1;\n\t"  // index 13: (13+1 = 14)
        "    selp.f32 %14, t15, t14, p1;\n\t"  // index 14: (14+1 = 15)
        "    selp.f32 %15, t0,  t15, p1;\n\t"  // index 15: (15+1 mod16 = 0)
        "}\n\t"
        : "+f"(array[0]),  "+f"(array[1]),  "+f"(array[2]),  "+f"(array[3]),
          "+f"(array[4]),  "+f"(array[5]),  "+f"(array[6]),  "+f"(array[7]),
          "+f"(array[8]),  "+f"(array[9]),  "+f"(array[10]), "+f"(array[11]),
          "+f"(array[12]), "+f"(array[13]), "+f"(array[14]), "+f"(array[15])
        : "r"(fast_index)
        : "memory"
    );
}

// Inline PTX cyclic in-place shift for a 4-element array.
// The shift (modulo 4) is decomposed into conditional shifts by 2 and 1.
__device__ inline void ptx_shift_4_inplace(float *array, int fast_index) {
    asm volatile(
        "{\n\t"
        "    .reg .pred p2, p1;\n\t"
        "    .reg .u32 bit;\n\t"
        "    .reg .f32 t0, t1, t2, t3;\n\t"
        "\n\t"
        // Extract predicate bits from fast_index:
        // Bit 1 controls the shift-by-2 step.
        "    bfe.u32 bit, %4, 1, 1;\n\t"
        "    setp.ne.u32 p2, bit, 0;\n\t"
        // Bit 0 controls the shift-by-1 step.
        "    bfe.u32 bit, %4, 0, 1;\n\t"
        "    setp.ne.u32 p1, bit, 0;\n\t"
        "\n\t"
        // Stage 1: Conditional shift by 2.
        // If p2 is true, swap elements {0,2} and {1,3}.
        "    selp.f32 t0, %2, %0, p2;\n\t" // t0 = (p2 ? array[2] : array[0])
        "    selp.f32 t1, %3, %1, p2;\n\t" // t1 = (p2 ? array[3] : array[1])
        "    selp.f32 t2, %0, %2, p2;\n\t" // t2 = (p2 ? array[0] : array[2])
        "    selp.f32 t3, %1, %3, p2;\n\t" // t3 = (p2 ? array[1] : array[3])
        "\n\t"
        // Stage 2: Conditional shift by 1.
        // If p1 is true, perform a cyclic shift by one: 0<=1, 1<=2, 2<=3, 3<=0.
        "    selp.f32 %0, t1, t0, p1;\n\t" // array[0] = (p1 ? t1 : t0)
        "    selp.f32 %1, t2, t1, p1;\n\t" // array[1] = (p1 ? t2 : t1)
        "    selp.f32 %2, t3, t2, p1;\n\t" // array[2] = (p1 ? t3 : t2)
        "    selp.f32 %3, t0, t3, p1;\n\t" // array[3] = (p1 ? t0 : t3)
        "}\n\t"
        : "+f"(array[0]), "+f"(array[1]), "+f"(array[2]), "+f"(array[3])
        : "r"(fast_index)
        : "memory"
    );
}


#endif // UTILITIES_H
