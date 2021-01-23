// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/providers/common.h>

#include "builder_utils.h"
#include "coreml/NeuralNetwork.pb.h"
#include "core/providers/coreml/builders/helper.h"

namespace onnxruntime {
namespace coreml {

common::Status CreateCoreMLWeight(CoreML::Specification::WeightParams& weight,
                                  const ONNX_NAMESPACE::TensorProto& tensor) {
  auto data_type = tensor.data_type();
  if (data_type = ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    const float* data =
        tensor.float_data().empty() ? reinterpret_cast<const float*>(tensor.raw_data().data())
                                    : tensor.float_data().data();

    weight.mutable_floatvalue()->Clear();
    auto num_elements = Product(tensor.dims());
    std::copy(data, data + num_elements,
              google::protobuf::RepeatedFieldBackInserter(weight.mutable_floatvalue()));
  } else {
    // TODO: support other type
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "The initializer of graph doesn't have valid type, name: ",
                           tensor.name(), " type: ", data_type);
  }

  return common::Status::OK();
}

}  // namespace coreml
}  // namespace onnxruntime