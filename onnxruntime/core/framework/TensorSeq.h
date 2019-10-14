// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensor.h"
#include <vector>

namespace onnxruntime {
// Put this in a separate file to avoid circular dependency between tensor.h and data_types.h
// Data type to represent a sequence of tensors of the same type
struct TensorSeq {
  using value_type = Tensor;  // to satisfy SequenceType template

  // A sequence must be associated with only one data type and all tensors in the seq must be of that type
  // One other alternative of storing the data type of a seq is to templatize the TensorSeq class.
  // The current design follows the Tensor methodology.
  // We also require this because the SequenceEmpty op expects the creation of a seq of a specific type
  // and the SequenceInsert op expects validation of tensors to be added to the seq against this type.
  MLDataType dtype{};

  // TODO: optimization opportunity - if all tensors in the seq are scalars, we can potentially represent them
  // as vector<primitive type>
  std::vector<Tensor> tensors;
};

template <typename TensorElemType>
class SequenceTensorType : public NonTensorType<TensorSeq> {
 public:
  static MLDataType Type();

  bool IsCompatible(const ONNX_NAMESPACE::TypeProto& type_proto) const override {
    return this->IsSequenceCompatible(type_proto);
  }

 private:
  SequenceTensorType() {
    data_types_internal::SetSequenceType<TensorElemType>::Set(this->mutable_type_proto());
  }
};

}  // namespace onnxruntime
