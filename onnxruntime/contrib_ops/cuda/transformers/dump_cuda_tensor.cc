// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime_api.h>
#include "core/providers/cuda/cuda_common.h"
#include "core/framework/print_tensor_utils.h"
#include "contrib_ops/cuda/transformers/dump_cuda_tensor.h"
#include "core/platform/env_var_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace transformers {

#if DUMP_TENSOR_LEVEL > 0

// Total number of elements which trigger snippet rather than full dump (default 200). Value 0 disables snippet.
constexpr const char* kTensorSnippetThreshold = "ORT_TENSOR_SNIPPET_THRESHOLD";

// Number of array items in snippet at beginning and end of each dimension (default 3)
constexpr const char* kTensorSnippetEdgeItems = "ORT_TENSOR_SNIPPET_EDGE_ITEMS";

class DumpTensorConfig {
 public:
  static DumpTensorConfig& instance() {
    static DumpTensorConfig instance;
    return instance;
  }

  DumpTensorConfig(const DumpTensorConfig&) = delete;
  DumpTensorConfig& operator=(const DumpTensorConfig&) = delete;

  int get_snippet_threshold() const { return snippet_threshold; }
  int get_snippet_edge_items() const { return snippet_edge_items; }

 private:
  int snippet_threshold;
  int snippet_edge_items;

  DumpTensorConfig() {
    snippet_threshold = ParseEnvironmentVariableWithDefault<int>(kTensorSnippetThreshold,
                                                                 onnxruntime::utils::kDefaultSnippetThreshold);
    snippet_edge_items = ParseEnvironmentVariableWithDefault<int>(kTensorSnippetEdgeItems,
                                                                  onnxruntime::utils::kDefaultSnippetEdgeItems);
  }
  ~DumpTensorConfig() {}
};

template <typename T>
class PinnedHostBuffer {
 public:
  PinnedHostBuffer(size_t length)
      : buffer_(nullptr) {
    CUDA_CALL_THROW(cudaHostAlloc((void**)&buffer_, length * sizeof(T), cudaHostAllocDefault));
  }

  virtual ~PinnedHostBuffer() {
    if (buffer_) {
      CUDA_CALL_THROW(cudaFreeHost(buffer_));
    }
  }

  operator T*() {
    return buffer_;
  }

  operator const T*() const {
    return buffer_;
  }

 protected:
  T* buffer_;
};

template <typename T>
void DumpGpuTensor(const char* name, const T* tensor, int dim0, int dim1, bool is_gpu_tensor) {
  // Occasionally, user will need dump CPU tensor in CUDA EP.
  // In that case, we copy tensor data as well. It is not needed, but it keeps code simple.
  int num_items = dim0 * dim1;
  auto data = std::make_shared<PinnedHostBuffer<T>>(num_items);
  CUDA_CALL_THROW(cudaDeviceSynchronize());
  CUDA_CALL_THROW(cudaMemcpy(*data, tensor, num_items * sizeof(T),
                             is_gpu_tensor ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost));

  if (nullptr != name) {
    std::cout << std::string(name) << std::endl;
  }

  int snippet_threshold = DumpTensorConfig::instance().get_snippet_threshold();
  int snippet_edge_items = DumpTensorConfig::instance().get_snippet_edge_items();
  if (snippet_threshold > 0 && snippet_threshold < num_items) {
    onnxruntime::utils::PrintCpuTensorSnippet<T>(*data, dim0, dim1, snippet_edge_items);
  } else {
    onnxruntime::utils::PrintCpuTensorFull<T>(*data, dim0, dim1);
  }
}

template <typename T>
void DumpGpuTensor(const char* name, const T* tensor, int dim0, int dim1, int dim2, bool is_gpu_tensor) {
  int num_items = dim0 * dim1 * dim2;
  auto data = std::make_shared<PinnedHostBuffer<T>>(num_items);
  CUDA_CALL_THROW(cudaDeviceSynchronize());
  CUDA_CALL_THROW(cudaMemcpy(*data, tensor, num_items * sizeof(T),
                             is_gpu_tensor ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost));

  if (nullptr != name) {
    std::cout << std::string(name) << std::endl;
  }

  int snippet_threshold = DumpTensorConfig::instance().get_snippet_threshold();
  int snippet_edge_items = DumpTensorConfig::instance().get_snippet_edge_items();
  if (snippet_threshold > 0 && snippet_threshold < num_items) {
    onnxruntime::utils::PrintCpuTensorSnippet<T>(*data, dim0, dim1, dim2, snippet_edge_items);
  } else {
    onnxruntime::utils::PrintCpuTensorFull<T>(*data, dim0, dim1, dim2);
  }
}

template <typename T>
void DumpGpuTensor(const char* name, const T* tensor, int dim0, int dim1, int dim2, int dim3, bool is_gpu_tensor) {
  int num_items = dim0 * dim1 * dim2 * dim3;
  auto data = std::make_shared<PinnedHostBuffer<T>>(num_items);
  CUDA_CALL_THROW(cudaDeviceSynchronize());
  CUDA_CALL_THROW(cudaMemcpy(*data, tensor, num_items * sizeof(T),
                             is_gpu_tensor ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost));

  if (nullptr != name) {
    std::cout << std::string(name) << std::endl;
  }

  int snippet_threshold = DumpTensorConfig::instance().get_snippet_threshold();
  int snippet_edge_items = DumpTensorConfig::instance().get_snippet_edge_items();
  if (snippet_threshold > 0 && snippet_threshold < num_items) {
    for (int i = 0; i < dim0; i++) {
      std::cout << "[" << i << "]:" << std::endl;
      onnxruntime::utils::PrintCpuTensorSnippet<T>((*data) + i * dim1 * dim2 * dim3, dim1, dim2, dim3,
                                                   snippet_edge_items);
    }
  } else {
    for (int i = 0; i < dim0; i++) {
      std::cout << "[" << i << "]:" << std::endl;
      onnxruntime::utils::PrintCpuTensorFull<T>((*data) + i * dim1 * dim2 * dim3, dim1, dim2, dim3);
    }
  }
}

void DumpGpuTensor(const char* name, const Tensor& tensor, int dim0, int dim1, int dim2) {
  MLDataType dataType = tensor.DataType();
  bool is_gpu_tensor = (tensor.Location().device.Type() == OrtDevice::GPU);
  if (dataType == DataTypeImpl::GetType<float>()) {
    DumpGpuTensor<float>(name, tensor.Data<float>(), dim0, dim1, dim2, is_gpu_tensor);
  } else if (dataType == DataTypeImpl::GetType<MLFloat16>()) {
    DumpGpuTensor<MLFloat16>(name, tensor.Data<MLFloat16>(), dim0, dim1, dim2, is_gpu_tensor);
  } else if (dataType == DataTypeImpl::GetType<int32_t>()) {
    DumpGpuTensor<int32_t>(name, tensor.Data<int32_t>(), dim0, dim1, dim2, is_gpu_tensor);
  } else if (dataType == DataTypeImpl::GetType<int64_t>()) {
    DumpGpuTensor<int64_t>(name, tensor.Data<int64_t>(), dim0, dim1, dim2, is_gpu_tensor);
  } else {
    assert(0);
  }
}

void DumpGpuTensor(const char* name, const Tensor& tensor, int dim0, int dim1) {
  MLDataType dataType = tensor.DataType();
  bool is_gpu_tensor = (tensor.Location().device.Type() == OrtDevice::GPU);
  if (dataType == DataTypeImpl::GetType<float>()) {
    DumpGpuTensor<float>(name, tensor.Data<float>(), dim0, dim1, is_gpu_tensor);
  } else if (dataType == DataTypeImpl::GetType<MLFloat16>()) {
    DumpGpuTensor<MLFloat16>(name, tensor.Data<MLFloat16>(), dim0, dim1, is_gpu_tensor);
  } else if (dataType == DataTypeImpl::GetType<int32_t>()) {
    DumpGpuTensor<int32_t>(name, tensor.Data<int32_t>(), dim0, dim1, is_gpu_tensor);
  } else if (dataType == DataTypeImpl::GetType<int64_t>()) {
    DumpGpuTensor<int64_t>(name, tensor.Data<int64_t>(), dim0, dim1, is_gpu_tensor);
  } else {
    assert(0);
  }
}

void DumpGpuTensor(const char* name, const Tensor& tensor) {
  const auto& shape = tensor.Shape();

  if (nullptr != name) {
    std::cout << std::string(name) << std::endl;
  }
  std::cout << "Shape:" << shape << std::endl;
  std::cout << tensor.Location().ToString() << std::endl;

  size_t num_dims = shape.NumDimensions();
  if (num_dims >= 3) {
    int dim0 = static_cast<int>(shape.SizeToDimension(num_dims - 2));
    int dim1 = static_cast<int>(shape[num_dims - 2]);
    int dim2 = static_cast<int>(shape[num_dims - 1]);
    DumpGpuTensor(nullptr, tensor, dim0, dim1, dim2);
    return;
  }

  auto num_items = shape.Size();
  size_t num_rows = 1;
  if (num_dims > 1) {
    num_rows = static_cast<size_t>(shape[0]);
  }
  size_t row_size = num_items / num_rows;
  DumpGpuTensor(nullptr, tensor, static_cast<int>(num_rows), static_cast<int>(row_size));
}

void CudaTensorConsoleDumper::Print(const char* name, const float* tensor, int dim0, int dim1) const {
  if (is_enabled_)
    DumpGpuTensor<float>(name, tensor, dim0, dim1, true);
}

void CudaTensorConsoleDumper::Print(const char* name, const MLFloat16* tensor, int dim0, int dim1) const {
  if (is_enabled_)
    DumpGpuTensor<MLFloat16>(name, tensor, dim0, dim1, true);
}

void CudaTensorConsoleDumper::Print(const char* name, const size_t* tensor, int dim0, int dim1) const {
  if (is_enabled_)
    DumpGpuTensor<size_t>(name, tensor, dim0, dim1, true);
}

void CudaTensorConsoleDumper::Print(const char* name, const half* tensor, int dim0, int dim1) const {
  Print(name, reinterpret_cast<const MLFloat16*>(tensor), dim0, dim1);
}

void CudaTensorConsoleDumper::Print(const char* name, const int64_t* tensor, int dim0, int dim1) const {
  if (is_enabled_)
    DumpGpuTensor<int64_t>(name, tensor, dim0, dim1, true);
}

void CudaTensorConsoleDumper::Print(const char* name, const int32_t* tensor, int dim0, int dim1) const {
  if (is_enabled_)
    DumpGpuTensor<int32_t>(name, tensor, dim0, dim1, true);
}

void CudaTensorConsoleDumper::Print(const char* name, const float* tensor, int dim0, int dim1, int dim2) const {
  if (is_enabled_)
    DumpGpuTensor<float>(name, tensor, dim0, dim1, dim2, true);
}

void CudaTensorConsoleDumper::Print(const char* name, const float* tensor, int dim0, int dim1, int dim2, int dim3) const {
  if (is_enabled_)
    DumpGpuTensor<float>(name, tensor, dim0, dim1, dim2, dim3, true);
}

void CudaTensorConsoleDumper::Print(const char* name, const MLFloat16* tensor, int dim0, int dim1, int dim2) const {
  if (is_enabled_)
    DumpGpuTensor<MLFloat16>(name, tensor, dim0, dim1, dim2, true);
}

void CudaTensorConsoleDumper::Print(const char* name, const MLFloat16* tensor, int dim0, int dim1, int dim2, int dim3) const {
  if (is_enabled_)
    DumpGpuTensor<MLFloat16>(name, tensor, dim0, dim1, dim2, dim3, true);
}

void CudaTensorConsoleDumper::Print(const char* name, const half* tensor, int dim0, int dim1, int dim2) const {
  Print(name, reinterpret_cast<const MLFloat16*>(tensor), dim0, dim1, dim2);
}

void CudaTensorConsoleDumper::Print(const char* name, const half* tensor, int dim0, int dim1, int dim2, int dim3) const {
  Print(name, reinterpret_cast<const MLFloat16*>(tensor), dim0, dim1, dim2, dim3);
}

void CudaTensorConsoleDumper::Print(const char* name, const int64_t* tensor, int dim0, int dim1, int dim2) const {
  if (is_enabled_)
    DumpGpuTensor<int64_t>(name, tensor, dim0, dim1, dim2, true);
}

void CudaTensorConsoleDumper::Print(const char* name, const int32_t* tensor, int dim0, int dim1, int dim2) const {
  if (is_enabled_)
    DumpGpuTensor<int32_t>(name, tensor, dim0, dim1, dim2, true);
}

void CudaTensorConsoleDumper::Print(const char* name, const Tensor& tensor) const {
  if (is_enabled_)
    DumpGpuTensor(name, tensor);
}

void CudaTensorConsoleDumper::Print(const char* name, const OrtValue& value) const {
  const Tensor& tensor = value.Get<Tensor>();
  Print(name, tensor);
}

void CudaTensorConsoleDumper::Print(const char* name, int index, bool end_line) const {
  if (!is_enabled_)
    return;

  std::cout << std::string(name) << "[" << index << "]";
  if (end_line) {
    std::cout << std::endl;
  }
}

void CudaTensorConsoleDumper::Print(const char* name, const std::string& value, bool end_line) const {
  if (!is_enabled_)
    return;

  std::cout << std::string(name) << "=" << value;
  if (end_line) {
    std::cout << std::endl;
  }
}

#else
void CudaTensorConsoleDumper::Print(const char*, const float*, int, int) const {
}

void CudaTensorConsoleDumper::Print(const char*, const MLFloat16*, int, int) const {
}

void CudaTensorConsoleDumper::Print(const char*, const size_t*, int, int) const {
}

void CudaTensorConsoleDumper::Print(const char*, const half*, int, int) const {
}

void CudaTensorConsoleDumper::Print(const char*, const int64_t*, int, int) const {
}

void CudaTensorConsoleDumper::Print(const char*, const int32_t*, int, int) const {
}

void CudaTensorConsoleDumper::Print(const char*, const float*, int, int, int) const {
}

void CudaTensorConsoleDumper::Print(const char*, const MLFloat16*, int, int, int) const {
}

void CudaTensorConsoleDumper::Print(const char*, const half*, int, int, int) const {
}

void CudaTensorConsoleDumper::Print(const char*, const int64_t*, int, int, int) const {
}

void CudaTensorConsoleDumper::Print(const char*, const int32_t*, int, int, int) const {
}

void CudaTensorConsoleDumper::Print(const char*, const float*, int, int, int, int) const {
}

void CudaTensorConsoleDumper::Print(const char*, const MLFloat16*, int, int, int, int) const {
}

void CudaTensorConsoleDumper::Print(const char*, const half*, int, int, int, int) const {
}

void CudaTensorConsoleDumper::Print(const char*, const Tensor&) const {
}

void CudaTensorConsoleDumper::Print(const char*, const OrtValue&) const {
}

void CudaTensorConsoleDumper::Print(const char*, int, bool) const {
}

void CudaTensorConsoleDumper::Print(const char*, const std::string&, bool) const {
}
#endif

}  // namespace transformers
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
