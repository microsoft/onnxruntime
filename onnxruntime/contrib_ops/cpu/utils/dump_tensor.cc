// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/utils/dump_tensor.h"
#include <iomanip>
#include <mutex>
#include <thread>
#include <iostream>
#include "core/framework/print_tensor_utils.h"
#include "contrib_ops/cpu/utils/debug_macros.h"
#include "core/platform/env_var_utils.h"

namespace onnxruntime {
namespace contrib {

#if DUMP_CPU_TENSOR_LEVEL > 0

// Environment variable to enable/disable dumping
constexpr const char* kEnableCpuTensorDumper = "ORT_ENABLE_CPU_DUMP";

// Environment variable to enable/disable dumping thread id
constexpr const char* kDumpThreadId = "ORT_DUMP_THREAD_ID";

// To avoid dumping at the same time from multiple threads
static std::mutex s_mutex;

static bool s_output_thread_id = false;

template <typename T>
void DumpCpuTensor(const char* name, const T* tensor, int dim0, int dim1) {
  std::unique_lock<std::mutex> lock(s_mutex);

  if (s_output_thread_id)
    std::cout << "Thread ID:" << std::this_thread::get_id() << std::endl;

  if (nullptr != name) {
    std::cout << std::string(name) << std::endl;
  }

  if (onnxruntime::utils::kDefaultSnippetThreshold < static_cast<int64_t>(dim0 * dim1)) {
    onnxruntime::utils::PrintCpuTensorSnippet<T>(tensor, dim0, dim1, onnxruntime::utils::kDefaultSnippetEdgeItems);
  } else {
    onnxruntime::utils::PrintCpuTensorFull<T>(tensor, dim0, dim1);
  }
}

template <typename T>
void DumpCpuTensor(const char* name, const T* tensor, int dim0, int dim1, int dim2) {
  std::unique_lock<std::mutex> lock(s_mutex);

  if (s_output_thread_id)
    std::cout << "Thread ID:" << std::this_thread::get_id() << std::endl;

  if (nullptr != name) {
    std::cout << std::string(name) << std::endl;
  }

  if (onnxruntime::utils::kDefaultSnippetThreshold < static_cast<int64_t>(dim0 * dim1 * dim2)) {
    onnxruntime::utils::PrintCpuTensorSnippet<T>(tensor, dim0, dim1, dim2, onnxruntime::utils::kDefaultSnippetEdgeItems);
  } else {
    onnxruntime::utils::PrintCpuTensorFull<T>(tensor, dim0, dim1, dim2);
  }
}

template <typename T>
void DumpCpuTensor(const char* name, const T* tensor, int dim0, int dim1, int dim2, int dim3) {
  std::unique_lock<std::mutex> lock(s_mutex);

  if (s_output_thread_id)
    std::cout << "Thread ID:" << std::this_thread::get_id() << std::endl;

  if (nullptr != name) {
    std::cout << std::string(name) << std::endl;
  }

  if (onnxruntime::utils::kDefaultSnippetThreshold < static_cast<int64_t>(dim0 * dim1 * dim2 * dim3)) {
    for (int i = 0; i < dim0; i++) {
      std::cout << "[" << i << "]:" << std::endl;
      onnxruntime::utils::PrintCpuTensorSnippet<T>(tensor + i * dim1 * dim2 * dim3, dim1, dim2, dim3,
                                                   onnxruntime::utils::kDefaultSnippetEdgeItems);
    }
  } else {
    for (int i = 0; i < dim0; i++) {
      std::cout << "[" << i << "]:" << std::endl;
      onnxruntime::utils::PrintCpuTensorFull<T>(tensor + i * dim1 * dim2 * dim3, dim1, dim2, dim3);
    }
  }
}

void DumpCpuTensor(const char* name, const Tensor& tensor, int dim0, int dim1, int dim2, int dim3) {
  MLDataType dataType = tensor.DataType();
  if (dataType == DataTypeImpl::GetType<float>()) {
    DumpCpuTensor<float>(name, tensor.Data<float>(), dim0, dim1, dim2, dim3);
  } else if (dataType == DataTypeImpl::GetType<MLFloat16>()) {
    DumpCpuTensor<MLFloat16>(name, tensor.Data<MLFloat16>(), dim0, dim1, dim2, dim3);
  } else if (dataType == DataTypeImpl::GetType<int32_t>()) {
    DumpCpuTensor<int32_t>(name, tensor.Data<int32_t>(), dim0, dim1, dim2, dim3);
  } else if (dataType == DataTypeImpl::GetType<int64_t>()) {
    DumpCpuTensor<int64_t>(name, tensor.Data<int64_t>(), dim0, dim1, dim2, dim3);
  } else if (dataType == DataTypeImpl::GetType<int8_t>()) {
    DumpCpuTensor<int8_t>(name, tensor.Data<int8_t>(), dim0, dim1, dim2, dim3);
  } else if (dataType == DataTypeImpl::GetType<uint8_t>()) {
    DumpCpuTensor<uint8_t>(name, tensor.Data<uint8_t>(), dim0, dim1, dim2, dim3);
  } else if (dataType == DataTypeImpl::GetType<UInt4x2>()) {
    DumpCpuTensor<UInt4x2>(name, tensor.Data<UInt4x2>(), dim0, dim1, dim2, dim3);
  } else if (dataType == DataTypeImpl::GetType<Int4x2>()) {
    DumpCpuTensor<Int4x2>(name, tensor.Data<Int4x2>(), dim0, dim1, dim2, dim3);
  } else {
    assert(0);
  }
}

void DumpCpuTensor(const char* name, const Tensor& tensor, int dim0, int dim1, int dim2) {
  MLDataType dataType = tensor.DataType();
  if (dataType == DataTypeImpl::GetType<float>()) {
    DumpCpuTensor<float>(name, tensor.Data<float>(), dim0, dim1, dim2);
  } else if (dataType == DataTypeImpl::GetType<MLFloat16>()) {
    DumpCpuTensor<MLFloat16>(name, tensor.Data<MLFloat16>(), dim0, dim1, dim2);
  } else if (dataType == DataTypeImpl::GetType<int32_t>()) {
    DumpCpuTensor<int32_t>(name, tensor.Data<int32_t>(), dim0, dim1, dim2);
  } else if (dataType == DataTypeImpl::GetType<int64_t>()) {
    DumpCpuTensor<int64_t>(name, tensor.Data<int64_t>(), dim0, dim1, dim2);
  } else if (dataType == DataTypeImpl::GetType<int8_t>()) {
    DumpCpuTensor<int8_t>(name, tensor.Data<int8_t>(), dim0, dim1, dim2);
  } else if (dataType == DataTypeImpl::GetType<uint8_t>()) {
    DumpCpuTensor<uint8_t>(name, tensor.Data<uint8_t>(), dim0, dim1, dim2);
  } else if (dataType == DataTypeImpl::GetType<UInt4x2>()) {
    DumpCpuTensor<UInt4x2>(name, tensor.Data<UInt4x2>(), dim0, dim1, dim2);
  } else if (dataType == DataTypeImpl::GetType<Int4x2>()) {
    DumpCpuTensor<Int4x2>(name, tensor.Data<Int4x2>(), dim0, dim1, dim2);
  } else {
    assert(0);
  }
}

void DumpCpuTensor(const char* name, const Tensor& tensor, int dim0, int dim1) {
  MLDataType dataType = tensor.DataType();
  if (dataType == DataTypeImpl::GetType<float>()) {
    DumpCpuTensor<float>(name, tensor.Data<float>(), dim0, dim1);
  } else if (dataType == DataTypeImpl::GetType<MLFloat16>()) {
    DumpCpuTensor<MLFloat16>(name, tensor.Data<MLFloat16>(), dim0, dim1);
  } else if (dataType == DataTypeImpl::GetType<int32_t>()) {
    DumpCpuTensor<int32_t>(name, tensor.Data<int32_t>(), dim0, dim1);
  } else if (dataType == DataTypeImpl::GetType<int64_t>()) {
    DumpCpuTensor<int64_t>(name, tensor.Data<int64_t>(), dim0, dim1);
  } else if (dataType == DataTypeImpl::GetType<int8_t>()) {
    DumpCpuTensor<int8_t>(name, tensor.Data<int8_t>(), dim0, dim1);
  } else if (dataType == DataTypeImpl::GetType<uint8_t>()) {
    DumpCpuTensor<uint8_t>(name, tensor.Data<uint8_t>(), dim0, dim1);
  } else if (dataType == DataTypeImpl::GetType<UInt4x2>()) {
    DumpCpuTensor<UInt4x2>(name, tensor.Data<UInt4x2>(), dim0, dim1);
  } else if (dataType == DataTypeImpl::GetType<Int4x2>()) {
    DumpCpuTensor<Int4x2>(name, tensor.Data<Int4x2>(), dim0, dim1);
  } else {
    assert(0);
  }
}

void DumpCpuTensor(const char* name, const Tensor& tensor, int dim0) {
  DumpCpuTensor(name, tensor, 1, dim0);
}

void DumpCpuTensor(const char* name, const Tensor& tensor) {
  const auto& shape = tensor.Shape();

  if (nullptr != name) {
    std::cout << std::string(name) << std::endl;
  }
  std::cout << "Shape:" << shape << std::endl;

  size_t num_dims = shape.NumDimensions();
  if (num_dims >= 4) {
    int dim0 = static_cast<int>(shape.SizeToDimension(num_dims - 4));
    int dim1 = static_cast<int>(shape[num_dims - 3]);
    int dim2 = static_cast<int>(shape[num_dims - 2]);
    int dim3 = static_cast<int>(shape[num_dims - 1]);
    DumpCpuTensor(nullptr, tensor, dim0, dim1, dim2, dim3);
    return;
  }

  if (num_dims == 3) {
    int dim0 = static_cast<int>(shape[0]);
    int dim1 = static_cast<int>(shape[1]);
    int dim2 = static_cast<int>(shape[2]);
    DumpCpuTensor(nullptr, tensor, dim0, dim1, dim2);
    return;
  }

  if (num_dims == 2) {
    int dim0 = static_cast<int>(shape[0]);
    int dim1 = static_cast<int>(shape[1]);
    DumpCpuTensor(nullptr, tensor, dim0, dim1);
    return;
  }

  if (num_dims == 1) {
    DumpCpuTensor(nullptr, tensor, static_cast<int>(shape[0]));
  }
}

CpuTensorConsoleDumper::CpuTensorConsoleDumper() {
  is_enabled_ = ParseEnvironmentVariableWithDefault<int>(kEnableCpuTensorDumper, 1) != 0;
  s_output_thread_id = ParseEnvironmentVariableWithDefault<int>(kDumpThreadId, 0) != 0;
}

void CpuTensorConsoleDumper::Print(const std::string& value) const {
  if (!is_enabled_)
    return;

  std::unique_lock<std::mutex> lock(s_mutex);
  if (s_output_thread_id)
    std::cout << "Thread ID:" << std::this_thread::get_id() << std::endl;
  std::cout << value << std::endl;
}

void CpuTensorConsoleDumper::Print(const char* name, const Tensor& tensor) const {
  if (!is_enabled_)
    return;
  DumpCpuTensor(name, tensor);
}

void CpuTensorConsoleDumper::Print(const char* name, const OrtValue& value) const {
  const Tensor& tensor = value.Get<Tensor>();
  Print(name, tensor);
}

#define TENSOR_DUMPER_PRINT_TYPE(dtype)                                                                                     \
  void CpuTensorConsoleDumper::Print(const char* name, const dtype* tensor, int dim0, int dim1) const {                     \
    if (is_enabled_)                                                                                                        \
      DumpCpuTensor<dtype>(name, tensor, dim0, dim1);                                                                       \
  }                                                                                                                         \
  void CpuTensorConsoleDumper::Print(const char* name, const dtype* tensor, int dim0, int dim1, int dim2) const {           \
    if (is_enabled_)                                                                                                        \
      DumpCpuTensor<dtype>(name, tensor, dim0, dim1, dim2);                                                                 \
  }                                                                                                                         \
  void CpuTensorConsoleDumper::Print(const char* name, const dtype* tensor, int dim0, int dim1, int dim2, int dim3) const { \
    if (is_enabled_)                                                                                                        \
      DumpCpuTensor<dtype>(name, tensor, dim0, dim1, dim2, dim3);                                                           \
  }                                                                                                                         \
  void CpuTensorConsoleDumper::Print(const char* name, const dtype* tensor, gsl::span<const int64_t>& dims) const {         \
    PrintTensorByDims<CpuTensorConsoleDumper, dtype>(this, name, tensor, dims);                                             \
  }

TENSOR_DUMPER_PRINT_TYPE(int8_t)
TENSOR_DUMPER_PRINT_TYPE(uint8_t)
TENSOR_DUMPER_PRINT_TYPE(int32_t)
TENSOR_DUMPER_PRINT_TYPE(int64_t)
TENSOR_DUMPER_PRINT_TYPE(float)
TENSOR_DUMPER_PRINT_TYPE(MLFloat16)
TENSOR_DUMPER_PRINT_TYPE(BFloat16)
TENSOR_DUMPER_PRINT_TYPE(UInt4x2)
TENSOR_DUMPER_PRINT_TYPE(Int4x2)
#undef TENSOR_DUMPER_PRINT_TYPE

#else

CpuTensorConsoleDumper::CpuTensorConsoleDumper() {
}

void CpuTensorConsoleDumper::Print(const std::string&) const {
}

void CpuTensorConsoleDumper::Print(const char*, const Tensor&) const {
}

void CpuTensorConsoleDumper::Print(const char*, const OrtValue&) const {
}

#define TENSOR_DUMPER_PRINT_TYPE(dtype)                                                            \
  void CpuTensorConsoleDumper::Print(const char*, const dtype*, int, int) const {                  \
  }                                                                                                \
  void CpuTensorConsoleDumper::Print(const char*, const dtype*, int, int, int) const {             \
  }                                                                                                \
  void CpuTensorConsoleDumper::Print(const char*, const dtype*, int, int, int, int) const {        \
  }                                                                                                \
  void CpuTensorConsoleDumper::Print(const char*, const dtype*, gsl::span<const int64_t>&) const { \
  }

TENSOR_DUMPER_PRINT_TYPE(int8_t)
TENSOR_DUMPER_PRINT_TYPE(uint8_t)
TENSOR_DUMPER_PRINT_TYPE(int32_t)
TENSOR_DUMPER_PRINT_TYPE(int64_t)
TENSOR_DUMPER_PRINT_TYPE(float)
TENSOR_DUMPER_PRINT_TYPE(MLFloat16)
TENSOR_DUMPER_PRINT_TYPE(BFloat16)
TENSOR_DUMPER_PRINT_TYPE(UInt4x2)
TENSOR_DUMPER_PRINT_TYPE(Int4x2)
#undef TENSOR_DUMPER_PRINT_TYPE

#endif

}  // namespace contrib
}  // namespace onnxruntime
