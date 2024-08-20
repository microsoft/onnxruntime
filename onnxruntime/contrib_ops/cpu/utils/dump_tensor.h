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
  void Print(const char* name, const float* tensor, int dim0, int dim1) const override;
  void Print(const char* name, const MLFloat16* tensor, int dim0, int dim1) const override;
  void Print(const char* name, const size_t* tensor, int dim0, int dim1) const override;
  void Print(const char* name, const int64_t* tensor, int dim0, int dim1) const override;
  void Print(const char* name, const int32_t* tensor, int dim0, int dim1) const override;

  void Print(const char* name, const float* tensor, int dim0, int dim1, int dim2) const override;
  void Print(const char* name, const MLFloat16* tensor, int dim0, int dim1, int dim2) const override;
  void Print(const char* name, const int64_t* tensor, int dim0, int dim1, int dim2) const override;
  void Print(const char* name, const int32_t* tensor, int dim0, int dim1, int dim2) const override;

  void Print(const char* name, const float* tensor, int dim0, int dim1, int dim2, int dim3) const override;
  void Print(const char* name, const MLFloat16* tensor, int dim0, int dim1, int dim2, int dim3) const override;
  void Print(const char* name, const int64_t* tensor, int dim0, int dim1, int dim2, int dim3) const override;
  void Print(const char* name, const int32_t* tensor, int dim0, int dim1, int dim2, int dim3) const override;

  void Print(const char* name, const int32_t* tensor, gsl::span<const int64_t>& dims) const override;
  void Print(const char* name, const int64_t* tensor, gsl::span<const int64_t>& dims) const override;
  void Print(const char* name, const float* tensor, gsl::span<const int64_t>& dims) const override;
  void Print(const char* name, const MLFloat16* tensor, gsl::span<const int64_t>& dims) const override;

  void Print(const char* name, const Tensor& value) const override;
  void Print(const char* name, const OrtValue& value) const override;
  void Print(const char* name, int index, bool end_line) const override;
  void Print(const char* name, const std::string& value, bool end_line) const override;

  void Print(const std::string& value) const override;

  // Output a vector with a threshold for max number of elements to output. Default threshold 0 means no limit.
  template <typename T>
  void Print(const char* name, const std::vector<T>& vec, size_t max_count = 0) const {
    this->Print(name, vec.data(), 1, static_cast<int>(std::min(max_count, vec.size())));
  }
};

}  // namespace contrib
}  // namespace onnxruntime
