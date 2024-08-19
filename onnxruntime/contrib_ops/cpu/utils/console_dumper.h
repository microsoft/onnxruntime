// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include "core/framework/ort_value.h"
#include "core/framework/float16.h"
#include "contrib_ops/cpu/utils/debug_macros.h"

namespace onnxruntime {
namespace contrib {

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

  virtual void Print(const char* name, const float* tensor, int dim0, int dim1, int dim2, int dim3) const = 0;
  virtual void Print(const char* name, const MLFloat16* tensor, int dim0, int dim1, int dim2, int dim3) const = 0;
  virtual void Print(const char* name, const int64_t* tensor, int dim0, int dim1, int dim2, int dim3) const = 0;
  virtual void Print(const char* name, const int32_t* tensor, int dim0, int dim1, int dim2, int dim3) const = 0;

  virtual void Print(const char* name, const int32_t* tensor, gsl::span<const int64_t>& dims) const = 0;
  virtual void Print(const char* name, const int64_t* tensor, gsl::span<const int64_t>& dims) const = 0;
  virtual void Print(const char* name, const float* tensor, gsl::span<const int64_t>& dims) const = 0;
  virtual void Print(const char* name, const MLFloat16* tensor, gsl::span<const int64_t>& dims) const = 0;

  virtual void Print(const char* name, const Tensor& value) const = 0;
  virtual void Print(const char* name, const OrtValue& value) const = 0;
  virtual void Print(const char* name, int index, bool end_line) const = 0;
  virtual void Print(const char* name, const std::string& value, bool end_line) const = 0;

  virtual void Print(const std::string& value) const = 0;

 protected:
  bool is_enabled_;
};

template <typename TConsoleDumper, typename T>
void PrintTensorByDims(const TConsoleDumper* dumper,
                       const char* name,
                       const T* tensor,
                       gsl::span<const int64_t>& dims) {
  if (dumper->IsEnabled() && (tensor == nullptr || dims.size() == 0)) {
    std::cout << std::string(name) << " is None" << std::endl;
    return;
  }

  auto num_dims = dims.size();
  if (num_dims == 1) {
    dumper->Print(name, tensor, 1, static_cast<int>(dims[0]));
  } else if (num_dims == 2) {
    dumper->Print(name, tensor, static_cast<int>(dims[0]), static_cast<int>(dims[1]));
  } else if (num_dims == 3) {
    dumper->Print(name, tensor, static_cast<int>(dims[0]), static_cast<int>(dims[1]), static_cast<int>(dims[2]));
  } else if (num_dims == 4) {
    dumper->Print(name, tensor,
                  static_cast<int>(dims[0]),
                  static_cast<int>(dims[1]),
                  static_cast<int>(dims[2]),
                  static_cast<int>(dims[3]));
  } else if (num_dims == 5) {
    dumper->Print(name, tensor,
                  static_cast<int>(dims[0]) * static_cast<int>(dims[1]),
                  static_cast<int>(dims[2]),
                  static_cast<int>(dims[3]),
                  static_cast<int>(dims[4]));
  } else {
    ORT_ENFORCE(false, "Unsupported tensor dims");
  }
}
}  // namespace contrib
}  // namespace onnxruntime
