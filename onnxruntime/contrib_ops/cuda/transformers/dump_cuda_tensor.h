// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensorprotoutils.h"
#include "core/framework/ort_value.h"
#include "contrib_ops/cpu/utils/console_dumper.h"

#define DUMP_TENSOR_LEVEL 0  // change it to 1 or 2 if want to enable dumping for code not in generation.

#if DUMP_TENSOR_LEVEL > 0
#define DUMP_TENSOR_INIT() onnxruntime::contrib::cuda::transformers::CudaTensorConsoleDumper dumper
#define DUMP_TENSOR(...) dumper.Print(__VA_ARGS__)
#else
#define DUMP_TENSOR_INIT()
#define DUMP_TENSOR(...)
#endif

#if DUMP_TENSOR_LEVEL > 1
#define DUMP_TENSOR_D(...) dumper.Print(__VA_ARGS__)
#else
#define DUMP_TENSOR_D(...)
#endif

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace transformers {

class CudaTensorConsoleDumper : public onnxruntime::contrib::transformers::IConsoleDumper {
 public:
  CudaTensorConsoleDumper() = default;
  virtual ~CudaTensorConsoleDumper() {}

  void Print(const char* name, const size_t* tensor, int dim0, int dim1) const override;

  void Print(const char* name, const int32_t* tensor, int dim0, int dim1) const override;
  void Print(const char* name, const int32_t* tensor, int dim0, int dim1, int dim2) const override;

  void Print(const char* name, const int64_t* tensor, int dim0, int dim1) const override;
  void Print(const char* name, const int64_t* tensor, int dim0, int dim1, int dim2) const override;

  void Print(const char* name, const float* tensor, int dim0, int dim1) const override;
  void Print(const char* name, const float* tensor, int dim0, int dim1, int dim2) const override;
  void Print(const char* name, const float* tensor, int dim0, int dim1, int dim2, int dim3) const;

  void Print(const char* name, const half* tensor, int dim0, int dim1) const;
  void Print(const char* name, const half* tensor, int dim0, int dim1, int dim2) const;
  void Print(const char* name, const half* tensor, int dim0, int dim1, int dim2, int dim3) const;

  void Print(const char* name, const MLFloat16* tensor, int dim0, int dim1) const override;
  void Print(const char* name, const MLFloat16* tensor, int dim0, int dim1, int dim2) const override;
  void Print(const char* name, const MLFloat16* tensor, int dim0, int dim1, int dim2, int dim3) const;

  void Print(const char* name, const BFloat16* tensor, int dim0, int dim1) const;
  void Print(const char* name, const BFloat16* tensor, int dim0, int dim1, int dim2) const;
  void Print(const char* name, const BFloat16* tensor, int dim0, int dim1, int dim2, int dim3) const;

  void Print(const char* name, const Tensor& value) const override;
  void Print(const char* name, const OrtValue& value) const override;
  void Print(const char* name, int index, bool end_line) const override;
  void Print(const char* name, const std::string& value, bool end_line) const override;
};

}  // namespace transformers
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
