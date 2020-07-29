// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "torch/torch.h"

namespace onnxruntime {
namespace cuda {

void torch_reduce_matrix_rows(const float* data, float* output, int m, int n) {
    int device = 0;
    cudaGetDevice(&device);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device);
    torch::Tensor torch_gpu_tensor = torch::zeros(c10::IntArrayRef{m, n}, options);
    cudaMemcpy(torch_gpu_tensor.data_ptr(), data, m * n * sizeof(float), cudaMemcpyDeviceToDevice);
    torch::Tensor torch_result = at::sum(torch_gpu_tensor, c10::IntArrayRef{0});
    cudaMemcpy(output, torch_result.data_ptr(), n * sizeof(float), cudaMemcpyDeviceToDevice);
}

void torch_reduce_matrix_rows(const half* data, half* output, int m, int n) {
    int device = 0;
    cudaGetDevice(&device);

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, device);
    torch::Tensor torch_gpu_tensor = torch::zeros(c10::IntArrayRef{m, n}, options);
    cudaMemcpy(torch_gpu_tensor.data_ptr(), data, m * n * sizeof(half), cudaMemcpyDeviceToDevice);
    torch::Tensor torch_result = at::sum(torch_gpu_tensor, c10::IntArrayRef{0});
    cudaMemcpy(output, torch_result.data_ptr(), n * sizeof(half), cudaMemcpyDeviceToDevice);
}

void torch_reduce_matrix_rows(const double* data, double* output, int m, int n) {
    int device = 0;
    cudaGetDevice(&device);

    auto options = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA, device);
    torch::Tensor torch_gpu_tensor = torch::zeros(c10::IntArrayRef{m, n}, options);
    cudaMemcpy(torch_gpu_tensor.data_ptr(), data, m * n * sizeof(double), cudaMemcpyDeviceToDevice);
    torch::Tensor torch_result = at::sum(torch_gpu_tensor, c10::IntArrayRef{0});
    cudaMemcpy(output, torch_result.data_ptr(), n * sizeof(double), cudaMemcpyDeviceToDevice);
}

}  // namespace cuda
}  // namespace onnxruntime
