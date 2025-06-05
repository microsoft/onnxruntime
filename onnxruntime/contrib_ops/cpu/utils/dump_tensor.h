// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>
#include <string>
#include "core/framework/ort_value.h"
#include "contrib_ops/cpu/utils/console_dumper.h"

namespace onnxruntime {
namespace contrib {

class CpuTensorConsoleDumper : public IConsoleDumper {
 public:
  CpuTensorConsoleDumper();
  virtual ~CpuTensorConsoleDumper() {}

  void Print(const char* name, const Tensor& value) const override;
  void Print(const char* name, const OrtValue& value) const override;

  void Print(const std::string& value) const override;

  // Output a vector with a threshold for max number of elements to output. Default threshold 0 means no limit.
  template <typename T>
  void Print(const char* name, const std::vector<T>& vec, size_t max_count = 0) const {
    this->Print(name, vec.data(), 1, static_cast<int>(std::min(max_count, vec.size())));
  }

#define TENSOR_DUMPER_PRINT_TYPE(dtype)                                                                     \
  void Print(const char* name, const dtype* tensor, int dim0, int dim1) const override;                     \
  void Print(const char* name, const dtype* tensor, int dim0, int dim1, int dim2) const override;           \
  void Print(const char* name, const dtype* tensor, int dim0, int dim1, int dim2, int dim3) const override; \
  void Print(const char* name, const dtype* tensor, gsl::span<const int64_t>& dims) const override;

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
};

}  // namespace contrib
}  // namespace onnxruntime
