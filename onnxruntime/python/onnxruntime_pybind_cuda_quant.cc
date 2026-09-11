// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Standalone CUDA weight-preprocessing extension module.
//
// This module is intentionally kept SEPARATE from onnxruntime_pybind11_state so
// that `import onnxruntime` never triggers a load-time dependency on the CUDA
// runtime (libcudart). The CUDA weight packing kernels below link CUDA::cudart,
// so this module has a hard libcudart dependency -- but it is imported lazily by
// onnxruntime.python.tools.quantization.cuda_quantizer only when CUDA weight
// prepacking is actually requested.
//
// This approach works for both the legacy in-tree CUDA EP build and the
// CUDA-EP-as-plugin build (onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON), because it
// does not rely on the provider bridge / ProviderInfo_CUDA interface (which is
// not available in plugin builds).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cuda_runtime.h>

#include <memory>
#include <sstream>
#include <stdexcept>

#include "contrib_ops/cuda/llm/fpA_intB_gemm_adaptor.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm_preprocessors.h"

namespace py = pybind11;

namespace {

void ThrowIfCudaError(cudaError_t status, const char* expression) {
  if (status != cudaSuccess) {
    std::ostringstream oss;
    oss << expression << " failed: " << cudaGetErrorString(status);
    throw std::runtime_error(oss.str());
  }
}

struct CudaDeleter {
  void operator()(void* p) const {
    if (p) cudaFree(p);
  }
};

using CudaPtr = std::unique_ptr<void, CudaDeleter>;

// Preprocess quantized weights for CUDA mixed-precision GEMM kernels (FpA_IntB format).
//
// MatMulNBits/QMoE stores quantized weights in (N, K) layout:
//   - N = number of output channels (columns in weight matrix W)
//   - K = number of input features (rows in weight matrix W)
//   - For 4-bit: shape is (N, K/2) bytes where each byte packs 2 elements
//   - For 8-bit: shape is (N, K) bytes
//
// FpA_IntB GEMM kernels expect weights in (K, N) layout (transposed) for efficient
// memory access during matrix multiplication. This function:
//   1. Transposes from (N, K) to (K, N) layout
//   2. Converts unsigned quantized values to signed int8 with zero-point adjustment
//      - 4-bit: uint4 -> int8 with zero_point=8 (range [0,15] -> [-8,7])
//      - 8-bit: uint8 -> int8 with zero_point=128 (range [0,255] -> [-128,127])
//   3. Applies architecture-specific row permutation for optimized tensor core access
//
// Input:  q_weights - Quantized weights from MatMulNBits in (N, K) layout
// Output: Preprocessed weights in (K, N) layout ready for fpA_intB GEMM kernels
py::array_t<int8_t> PackWeightsForMixedGemm(
    py::array_t<uint8_t> q_weights,
    int32_t N,
    int32_t K,
    int32_t bits,
    int32_t force_arch = -1) {
  py::buffer_info q_weights_buf = q_weights.request();

  if (bits != 4 && bits != 8) {
    throw std::invalid_argument("bits must be 4 or 8");
  }
  if (N <= 0 || K <= 0) {
    throw std::invalid_argument("N and K must be positive");
  }
  if (bits == 4 && K % 2 != 0) {
    throw std::invalid_argument("K must be even for 4-bit packed weights");
  }
  if (q_weights_buf.ndim != 2 || q_weights_buf.shape[0] != N || q_weights_buf.shape[1] != K / (8 / bits)) {
    throw std::invalid_argument("q_weights must have shape (N, K / (8 / bits))");
  }

  int n = static_cast<int>(N);
  int k = static_cast<int>(K);

  size_t packed_weight_bytes = static_cast<size_t>(n) * static_cast<size_t>(k) / (8 / bits);
  py::array_t<int8_t> processed_weights({static_cast<pybind11::ssize_t>(packed_weight_bytes)});
  py::buffer_info processed_weights_buf = processed_weights.request();

  auto make_cuda_ptr = [](size_t bytes) -> CudaPtr {
    void* p = nullptr;
    ThrowIfCudaError(cudaMalloc(&p, bytes), "cudaMalloc");
    return CudaPtr(p);
  };

  auto packed_transposed_weight_space = make_cuda_ptr(packed_weight_bytes);
  int8_t* packed_transposed_weight = reinterpret_cast<int8_t*>(packed_transposed_weight_space.get());

  auto fpA_intB_weight_buffer_ = make_cuda_ptr(packed_weight_bytes);
  int8_t* preprocessed_weight = reinterpret_cast<int8_t*>(fpA_intB_weight_buffer_.get());

  const uint8_t* blob_data_cpu = static_cast<const uint8_t*>(q_weights_buf.ptr);

  auto blob_data_gpu_buf = make_cuda_ptr(packed_weight_bytes);
  uint8_t* blob_data_gpu = reinterpret_cast<uint8_t*>(blob_data_gpu_buf.get());

  cudaStream_t stream = cudaStreamLegacy;
  ThrowIfCudaError(cudaMemcpyAsync(blob_data_gpu, blob_data_cpu, packed_weight_bytes, cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync host-to-device");

  if (bits == 4) {
    ::onnxruntime::llm::kernels::fpA_intB_gemv::unpack_uint4_transposed_to_int8_direct_cuda(
        stream, packed_transposed_weight, blob_data_gpu, n, k);
  } else {
    // 8 bits
    ::onnxruntime::llm::kernels::fpA_intB_gemv::transpose_uint8_matrix_and_convert_to_int8(
        stream, packed_transposed_weight, blob_data_gpu, n, k);
  }

  using ::onnxruntime::llm::kernels::weight_only::QuantType;
  QuantType quant_type = bits == 4 ? QuantType::W4_A16 : QuantType::W8_A16;

  int sm = force_arch;
  if (sm < 0) {
    int device_id = 0;
    ThrowIfCudaError(cudaGetDevice(&device_id), "cudaGetDevice");
    cudaDeviceProp device_prop;
    ThrowIfCudaError(cudaGetDeviceProperties(&device_prop, device_id), "cudaGetDeviceProperties");
    sm = device_prop.major * 10 + device_prop.minor;
  }
  sm = ::onnxruntime::llm::kernels::weight_only::get_arch_for_mixed_gemm_weight_preprocess(sm);

  auto permutation_map_buffer = make_cuda_ptr(32 * sizeof(int32_t));

  ::onnxruntime::llm::kernels::weight_only::preprocess_weights_for_mixed_gemm_cuda(
      stream,
      sm,
      preprocessed_weight,
      packed_transposed_weight,
      reinterpret_cast<int32_t*>(permutation_map_buffer.get()),
      {static_cast<size_t>(k), static_cast<size_t>(n)},
      quant_type);

  ThrowIfCudaError(cudaGetLastError(), "preprocess CUDA kernel launch");
  ThrowIfCudaError(cudaMemcpyAsync(processed_weights_buf.ptr, preprocessed_weight, packed_weight_bytes,
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync device-to-host");
  ThrowIfCudaError(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

  return processed_weights;
}

}  // namespace

PYBIND11_MODULE(onnxruntime_cuda_quant_preprocess, m) {
  m.doc() = "CUDA weight-only quantization preprocessing helpers (loaded on demand).";
  m.def("pack_weights_for_cuda_mixed_gemm", &PackWeightsForMixedGemm,
        "Pack quantized weights for CUDA mixed-precision GEMM (FpA_IntB format)",
        py::arg("q_weights"), py::arg("N"), py::arg("K"), py::arg("bits"), py::arg("force_arch") = -1);
}
