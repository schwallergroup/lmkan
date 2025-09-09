#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
using namespace torch::indexing;

// Might be worth defining header files for these in the future, but for now it is fine.
#include "kernels_dkan_full.cu"
#include "kernels_dkan_full_backward.cu"  // New backward functions

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  /*m.def("forward_1d", &linear_gpu_forward<1, true>, "1D Linear forward (CUDA)");
  m.def("forward_3d", &linear_gpu_forward<3, true>, "3D Linear forward (CUDA)");
  m.def("forward_1d_bspline", &linear_gpu_forward<1, false>, "1D Linear forward (CUDA) B Spline");
  m.def("forward_2d_bspline", &linear_gpu_forward<2, false>, "2D Linear forward (CUDA) B Spline");
  m.def("forward_3d_bspline", &linear_gpu_forward<3, false>, "3D Linear forward (CUDA) B Spline");
  m.def("forward_dkan_1d", &linear_gpu_dkan_forward<1>, "1D Linear forward DKAN (CUDA)");
  m.def("forward_dkan_2d", &linear_gpu_dkan_forward<2>, "2D Linear forward DKAN (CUDA)");*/
  m.def("forward_dkan_full_2d", &linear_gpu_dkan_full_forward_pybind_wrapper<2>,
        "2D Linear forward DKAN Full (CUDA)",
        py::arg("parameters"),
        py::arg("x"),
        py::arg("block_size"),
        py::arg("cdf_grid"),
        py::arg("tile_size"),
        py::arg("batch_last"),
        py::arg("n_tiles_repeat") = py::none(),
        py::arg("n_repetitions") = py::none());

  // Bind the backward function from kernels_dkan_full_backward.cu
  m.def("backward_dkan_full_2d", &linear_gpu_dkan_full_backward_pybind_wrapper<2>,
        "2D Linear backward DKAN Full (CUDA)",
        py::arg("parameters"),
        py::arg("x"),
        py::arg("output_grad"),
        py::arg("block_size"),
        py::arg("cdf_grid"),
        py::arg("tile_size"),
        py::arg("batch_last"),
        py::arg("fast_mode") = false,
        py::arg("n_tiles_repeat") = py::none(),
        py::arg("n_repetitions") = py::none());
}

int main() {
  // This main function is not used when the module is imported in Python.
}
