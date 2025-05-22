// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensorprotoutils.h"
#include "core/framework/ort_value.h"
#include "contrib_ops/cpu/utils/console_dumper.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

class CudaTensorConsoleDumper : public onnxruntime::contrib::IConsoleDumper {
 public:
  CudaTensorConsoleDumper();
  virtual ~CudaTensorConsoleDumper() {}

  void Print(const char* name, const Tensor& value) const override;
  void Print(const char* name, const OrtValue& value) const override;
  void Print(const std::string& value) const override;

#define CUDA_DUMPER_PRINT_TYPE(dtype)                                                              \
  void Print(const char* name, const dtype* tensor, int dim0, int dim1) const;                     \
  void Print(const char* name, const dtype* tensor, int dim0, int dim1, int dim2) const;           \
  void Print(const char* name, const dtype* tensor, int dim0, int dim1, int dim2, int dim3) const; \
  void Print(const char* name, const dtype* tensor, gsl::span<const int64_t>& dims) const;

  CUDA_DUMPER_PRINT_TYPE(int8_t)
  CUDA_DUMPER_PRINT_TYPE(uint8_t)
  CUDA_DUMPER_PRINT_TYPE(int32_t)
  CUDA_DUMPER_PRINT_TYPE(int64_t)
  CUDA_DUMPER_PRINT_TYPE(float)
  CUDA_DUMPER_PRINT_TYPE(MLFloat16)
  CUDA_DUMPER_PRINT_TYPE(BFloat16)
  CUDA_DUMPER_PRINT_TYPE(UInt4x2)
  CUDA_DUMPER_PRINT_TYPE(Int4x2)
  CUDA_DUMPER_PRINT_TYPE(half)

#undef CUDA_DUMPER_PRINT_TYPE
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
