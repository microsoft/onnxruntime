// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <cuda_runtime_api.h>
#include "core/providers/cuda/cuda_common.h"
#include "dump_cuda_tensor.h"
#include "core/framework/print_tensor_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace transformers {

#ifdef DEBUG_BEAM_SEARCH
template <typename T>
class PinnedHostBuffer {
 public:
  PinnedHostBuffer(size_t length)
      : buffer_(nullptr) {
    cudaHostAlloc(&buffer_, length * sizeof(T), cudaHostAllocDefault);
  }

  virtual ~PinnedHostBuffer() {
    if (buffer_) {
      cudaFreeHost(buffer_);
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
void DumpGpuTensor(const char* name, const T* tensor, int dim0, int dim1) {
  int num_items = dim0 * dim1;
  auto data = std::make_shared<PinnedHostBuffer<T>>(num_items);
  cudaDeviceSynchronize();
  cudaMemcpy(*data, tensor, num_items * sizeof(T), cudaMemcpyDeviceToHost);

  if (nullptr != name) {
    std::cout << std::string(name) << std::endl;
  }

  if (onnxruntime::utils::kDefaultSnippetThreshold < static_cast<int64_t>(num_items)) {
    onnxruntime::utils::PrintCpuTensorSnippet<T>(*data, dim0, dim1, onnxruntime::utils::kDefaultSnippetEdgeItems);
  } else {
    onnxruntime::utils::PrintCpuTensorFull<T>(*data, dim0, dim1);
  }
}

template <typename T>
void DumpGpuTensor(const char* name, const T* tensor, int dim0, int dim1, int dim2) {
  int num_items = dim0 * dim1 * dim2;
  auto data = std::make_shared<PinnedHostBuffer<T>>(num_items);
  cudaDeviceSynchronize();
  cudaMemcpy(*data, tensor, num_items * sizeof(T), cudaMemcpyDeviceToHost);

  if (nullptr != name) {
    std::cout << std::string(name) << std::endl;
  }

  if (onnxruntime::utils::kDefaultSnippetThreshold < static_cast<int64_t>(num_items)) {
    onnxruntime::utils::PrintCpuTensorSnippet<T>(*data, dim0, dim1, dim2, onnxruntime::utils::kDefaultSnippetEdgeItems);
  } else {
    onnxruntime::utils::PrintCpuTensorFull<T>(*data, dim0, dim1, dim2);
  }
}

void DumpGpuTensor(const char* name, const Tensor& tensor, int dim0, int dim1, int dim2) {
  MLDataType dataType = tensor.DataType();
  if (dataType == DataTypeImpl::GetType<float>()) {
    DumpGpuTensor<float>(name, tensor.Data<float>(), dim0, dim1, dim2);
  } else if (dataType == DataTypeImpl::GetType<MLFloat16>()) {
    DumpGpuTensor<MLFloat16>(name, tensor.Data<MLFloat16>(), dim0, dim1, dim2);
  } else if (dataType == DataTypeImpl::GetType<int32_t>()) {
    DumpGpuTensor<int32_t>(name, tensor.Data<int32_t>(), dim0, dim1, dim2);
  } else if (dataType == DataTypeImpl::GetType<int64_t>()) {
    DumpGpuTensor<int64_t>(name, tensor.Data<int64_t>(), dim0, dim1, dim2);
  } else {
    assert(0);
  }
}

void DumpGpuTensor(const char* name, const Tensor& tensor, int dim0, int dim1) {
  MLDataType dataType = tensor.DataType();
  if (dataType == DataTypeImpl::GetType<float>()) {
    DumpGpuTensor<float>(name, tensor.Data<float>(), dim0, dim1);
  } else if (dataType == DataTypeImpl::GetType<MLFloat16>()) {
    DumpGpuTensor<MLFloat16>(name, tensor.Data<MLFloat16>(), dim0, dim1);
  } else if (dataType == DataTypeImpl::GetType<int32_t>()) {
    DumpGpuTensor<int32_t>(name, tensor.Data<int32_t>(), dim0, dim1);
  } else if (dataType == DataTypeImpl::GetType<int64_t>()) {
    DumpGpuTensor<int64_t>(name, tensor.Data<int64_t>(), dim0, dim1);
  } else {
    assert(0);
  }
}

void DumpGpuTensor(const char* name, const Tensor& tensor) {
  const auto& shape = tensor.Shape();

  size_t num_dims = shape.NumDimensions();
  if (num_dims >= 3) {
    int dim0 = static_cast<int>(shape.SizeToDimension(num_dims - 2));
    int dim1 = static_cast<int>(shape[num_dims - 2]);
    int dim2 = static_cast<int>(shape[num_dims - 1]);
    DumpGpuTensor(name, tensor, dim0, dim1, dim2);
    return;
  }

  auto num_items = shape.Size();
  size_t num_rows = 1;
  if (num_dims > 1) {
    num_rows = static_cast<size_t>(shape[0]);
  }
  size_t row_size = num_items / num_rows;
  DumpGpuTensor(name, tensor, static_cast<int>(num_rows), static_cast<int>(row_size));
}

void CudaTensorConsoleDumper::Print(const char* name, const float* tensor, int dim0, int dim1) const {
  if (is_enabled_)
    DumpGpuTensor<float>(name, tensor, dim0, dim1);
}

void CudaTensorConsoleDumper::Print(const char* name, const MLFloat16* tensor, int dim0, int dim1) const {
  if (is_enabled_)
    DumpGpuTensor<MLFloat16>(name, tensor, dim0, dim1);
}

void CudaTensorConsoleDumper::Print(const char* name, const int64_t* tensor, int dim0, int dim1) const {
  if (is_enabled_)
    DumpGpuTensor<int64_t>(name, tensor, dim0, dim1);
}

void CudaTensorConsoleDumper::Print(const char* name, const int32_t* tensor, int dim0, int dim1) const {
  if (is_enabled_)
    DumpGpuTensor<int32_t>(name, tensor, dim0, dim1);
}

void CudaTensorConsoleDumper::Print(const char* name, const float* tensor, int dim0, int dim1, int dim2) const {
  if (is_enabled_)
    DumpGpuTensor<float>(name, tensor, dim0, dim1, dim2);
}

void CudaTensorConsoleDumper::Print(const char* name, const MLFloat16* tensor, int dim0, int dim1, int dim2) const {
  if (is_enabled_)
    DumpGpuTensor<MLFloat16>(name, tensor, dim0, dim1, dim2);
}

void CudaTensorConsoleDumper::Print(const char* name, const int64_t* tensor, int dim0, int dim1, int dim2) const {
  if (is_enabled_)
    DumpGpuTensor<int64_t>(name, tensor, dim0, dim1, dim2);
}

void CudaTensorConsoleDumper::Print(const char* name, const int32_t* tensor, int dim0, int dim1, int dim2) const {
  if (is_enabled_)
    DumpGpuTensor<int32_t>(name, tensor, dim0, dim1, dim2);
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

void CudaTensorConsoleDumper::Print(const char*, const int64_t*, int, int) const {
}

void CudaTensorConsoleDumper::Print(const char*, const int32_t*, int, int) const {
}

void CudaTensorConsoleDumper::Print(const char*, const float*, int, int, int) const {
}

void CudaTensorConsoleDumper::Print(const char*, const MLFloat16*, int, int, int) const {
}

void CudaTensorConsoleDumper::Print(const char*, const int64_t*, int, int, int) const {
}

void CudaTensorConsoleDumper::Print(const char*, const int32_t*, int, int, int) const {
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