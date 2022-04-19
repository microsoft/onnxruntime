// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include "core/framework/tensorprotoutils.h"
#include "core/framework/ort_value.h"
#include "contrib_ops/cpu/transformers/beam_search_shared.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace transformers {

class CudaTensorConsoleDumper : public onnxruntime::contrib::transformers::IConsoleDumper {
 public:
  CudaTensorConsoleDumper() = default;
  virtual ~CudaTensorConsoleDumper() {}
  void Print(const char* name, const float* tensor, int dim0, int dim1) const override;
  void Print(const char* name, const MLFloat16* tensor, int dim0, int dim1) const override;
  void Print(const char* name, const int64_t* tensor, int dim0, int dim1) const override;
  void Print(const char* name, const int32_t* tensor, int dim0, int dim1) const override;
  void Print(const char* name, const float* tensor, int dim0, int dim1, int dim2) const override;
  void Print(const char* name, const MLFloat16* tensor, int dim0, int dim1, int dim2) const override;
  void Print(const char* name, const int64_t* tensor, int dim0, int dim1, int dim2) const override;
  void Print(const char* name, const int32_t* tensor, int dim0, int dim1, int dim2) const override;
  void Print(const char* name, const Tensor& value) const override;
  void Print(const char* name, const OrtValue& value) const override;
  void Print(const char* name, int index, bool end_line) const override;
  void Print(const char* name, const std::string& value, bool end_line) const override;
};

}  // namespace transformers
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime