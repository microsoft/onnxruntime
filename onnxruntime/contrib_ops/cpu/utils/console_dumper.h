// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include "core/framework/ort_value.h"

// #define DEBUG_GENERATION 1  // uncomment it for debugging generation (like beam search etc)

namespace onnxruntime {
namespace contrib {
namespace transformers {

class IConsoleDumper {
 public:
  IConsoleDumper() : is_enabled_(true) {}
  virtual ~IConsoleDumper() {}
  void Disable() { is_enabled_ = false; }
  bool IsEnabled() const { return is_enabled_; }
  virtual void Print(const char* name, const float* tensor, int dim0, int dim1) const = 0;
  virtual void Print(const char* name, const MLFloat16* tensor, int dim0, int dim1) const = 0;
  virtual void Print(const char* name, const size_t* tensor, int dim0, int dim1) const = 0;
  virtual void Print(const char* name, const int64_t* tensor, int dim0, int dim1) const = 0;
  virtual void Print(const char* name, const int32_t* tensor, int dim0, int dim1) const = 0;
  virtual void Print(const char* name, const float* tensor, int dim0, int dim1, int dim2) const = 0;
  virtual void Print(const char* name, const MLFloat16* tensor, int dim0, int dim1, int dim2) const = 0;
  virtual void Print(const char* name, const int64_t* tensor, int dim0, int dim1, int dim2) const = 0;
  virtual void Print(const char* name, const int32_t* tensor, int dim0, int dim1, int dim2) const = 0;
  virtual void Print(const char* name, const Tensor& value) const = 0;
  virtual void Print(const char* name, const OrtValue& value) const = 0;
  virtual void Print(const char* name, int index, bool end_line) const = 0;
  virtual void Print(const char* name, const std::string& value, bool end_line) const = 0;

 protected:
  bool is_enabled_;
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
