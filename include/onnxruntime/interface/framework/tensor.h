// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

namespace interface {
struct ITensorShape {
  virtual int64_t NumberOfElements() const = 0;
  virtual const int64_t* GetDimensions(size_t& num_dims) const = 0;
};

}  // namespace interface

}  // namespace onnxruntime