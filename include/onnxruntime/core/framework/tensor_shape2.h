// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>
#include <string>
#include <memory>

namespace onnxruntime {
class TensorShape;
class TensorShape2 {
 public:
  TensorShape2() = default;
  TensorShape2(std::vector<int64_t> dims);
  ~TensorShape2();

  int64_t operator[](size_t idx) const;
  int64_t& operator[](size_t idx);

  size_t NumDimensions() const noexcept;

  /**
     output dimensions nicely formatted
  */
  std::string ToString() const;

  /**
     empty shape or 1D shape (1) is regarded as scalar tensor
  */
  bool IsScalar() const;

 private:
//  std::unique_ptr<TensorShape> tensor_shape_;
  TensorShape* tensor_shape_;
};
}  // namespace onnxruntime
