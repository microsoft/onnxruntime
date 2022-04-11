// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dump_tensor.h"
#include <iomanip>
#include "core/framework/print_tensor_utils.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

#ifdef DEBUG_BEAM_SEARCH

template <typename T>
void DumpCpuTensor(const char* name, const T* tensor, int dim0, int dim1) {
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
  if (nullptr != name) {
    std::cout << std::string(name) << std::endl;
  }

  if (onnxruntime::utils::kDefaultSnippetThreshold < static_cast<int64_t>(dim0 * dim1 * dim2)) {
    onnxruntime::utils::PrintCpuTensorSnippet<T>(tensor, dim0, dim1, dim2, onnxruntime::utils::kDefaultSnippetEdgeItems);
  } else {
    onnxruntime::utils::PrintCpuTensorFull<T>(tensor, dim0, dim1, dim2);
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
  } else {
    assert(0);
  }
}

void DumpCpuTensor(const char* name, const Tensor& tensor) {
  const auto& shape = tensor.Shape();

  size_t num_dims = shape.NumDimensions();
  if (num_dims >= 3) {
    int dim0 = static_cast<int>(shape.SizeToDimension(num_dims - 2));
    int dim1 = static_cast<int>(shape[num_dims - 2]);
    int dim2 = static_cast<int>(shape[num_dims - 1]);
    DumpCpuTensor(name, tensor, dim0, dim1, dim2);
    return;
  }

  auto num_items = shape.Size();
  size_t num_rows = 1;
  if (num_dims > 1) {
    num_rows = static_cast<size_t>(shape[0]);
  }
  size_t row_size = num_items / num_rows;
  DumpCpuTensor(name, tensor, static_cast<int>(num_rows), static_cast<int>(row_size));
}

void CpuTensorConsoleDumper::Print(const char* name, const float* tensor, int dim0, int dim1) const {
  if (!is_enabled_)
    return;
  DumpCpuTensor<float>(name, tensor, dim0, dim1);
}

void CpuTensorConsoleDumper::Print(const char* name, const MLFloat16* tensor, int dim0, int dim1) const {
  if (!is_enabled_)
    return;
  DumpCpuTensor<MLFloat16>(name, tensor, dim0, dim1);
}

void CpuTensorConsoleDumper::Print(const char* name, const int64_t* tensor, int dim0, int dim1) const {
  if (!is_enabled_)
    return;
  DumpCpuTensor<int64_t>(name, tensor, dim0, dim1);
}

void CpuTensorConsoleDumper::Print(const char* name, const int32_t* tensor, int dim0, int dim1) const {
  if (!is_enabled_)
    return;
  DumpCpuTensor<int32_t>(name, tensor, dim0, dim1);
}

void CpuTensorConsoleDumper::Print(const char* name, const float* tensor, int dim0, int dim1, int dim2) const {
  if (!is_enabled_)
    return;
  DumpCpuTensor<float>(name, tensor, dim0, dim1, dim2);
}

void CpuTensorConsoleDumper::Print(const char* name, const MLFloat16* tensor, int dim0, int dim1, int dim2) const {
  if (!is_enabled_)
    return;
  DumpCpuTensor<MLFloat16>(name, tensor, dim0, dim1, dim2);
}

void CpuTensorConsoleDumper::Print(const char* name, const int64_t* tensor, int dim0, int dim1, int dim2) const {
  if (!is_enabled_)
    return;
  DumpCpuTensor<int64_t>(name, tensor, dim0, dim1, dim2);
}

void CpuTensorConsoleDumper::Print(const char* name, const int32_t* tensor, int dim0, int dim1, int dim2) const {
  if (!is_enabled_)
    return;
  DumpCpuTensor<int32_t>(name, tensor, dim0, dim1, dim2);
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

void CpuTensorConsoleDumper::Print(const char* name, int index, bool end_line) const {
  if (!is_enabled_)
    return;
  std::cout << std::string(name) << "[" << index << "]";

  if (end_line) {
    std::cout << std::endl;
  }
}

void CpuTensorConsoleDumper::Print(const char* name, const std::string& value, bool end_line) const {
  if (!is_enabled_)
    return;

  std::cout << std::string(name) << "=" << value;

  if (end_line) {
    std::cout << std::endl;
  }
}

#else
void CpuTensorConsoleDumper::Print(const char*, const float*, int, int) const {
}

void CpuTensorConsoleDumper::Print(const char*, const MLFloat16*, int, int) const {
}

void CpuTensorConsoleDumper::Print(const char*, const int64_t*, int, int) const {
}

void CpuTensorConsoleDumper::Print(const char*, const int32_t*, int, int) const {
}

void CpuTensorConsoleDumper::Print(const char*, const float*, int, int, int) const {
}

void CpuTensorConsoleDumper::Print(const char*, const MLFloat16*, int, int, int) const {
}

void CpuTensorConsoleDumper::Print(const char*, const int64_t*, int, int, int) const {
}

void CpuTensorConsoleDumper::Print(const char*, const int32_t*, int, int, int) const {
}

void CpuTensorConsoleDumper::Print(const char*, const Tensor&) const {
}

void CpuTensorConsoleDumper::Print(const char*, const OrtValue&) const {
}

void CpuTensorConsoleDumper::Print(const char*, int, bool) const {
}

void CpuTensorConsoleDumper::Print(const char*, const std::string&, bool) const {
}

#endif

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime