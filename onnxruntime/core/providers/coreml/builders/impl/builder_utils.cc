// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef __APPLE__

#include "core/providers/coreml/builders/impl/builder_utils.h"

#include "core/common/narrow.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/shared/utils/utils.h"
#include "core/optimizer/initializer.h"

#include "coreml/NeuralNetwork.pb.h"

namespace onnxruntime {
namespace coreml {

Status ComputeConvPads(const std::vector<int64_t> input_shape,
                       const int64_t weight_size_y,
                       const int64_t weight_size_x,
                       const std::vector<int64_t>& onnx_pads,
                       const std::vector<int64_t>& onnx_strides,
                       const std::vector<int64_t>& onnx_dilations,
                       AutoPadType auto_pad_type,
                       std::vector<int64_t>& pads_out) {
  const int64_t input_size_y = input_shape[2];
  const int64_t input_size_x = input_shape[3];
  const int64_t stride_y = onnx_strides[0];
  const int64_t stride_x = onnx_strides[1];
  const int64_t dilation_y = onnx_dilations[0];
  const int64_t dilation_x = onnx_dilations[1];

  int64_t padding_top = onnx_pads[0];
  int64_t padding_bottom = onnx_pads[2];
  int64_t padding_left = onnx_pads[1];
  int64_t padding_right = onnx_pads[3];

  ORT_RETURN_IF_ERROR(ComputePad(input_size_y,
                                 stride_y, weight_size_y, dilation_y,
                                 auto_pad_type,
                                 padding_top, padding_bottom));
  ORT_RETURN_IF_ERROR(ComputePad(input_size_x,
                                 stride_x, weight_size_x, dilation_x,
                                 auto_pad_type,
                                 padding_left, padding_right));

  pads_out = {padding_top, padding_left, padding_bottom, padding_right};

  return Status::OK();
}

Status HandleAutoPad(const std::vector<int64_t> input_shape,
                     const int64_t weight_size_y,
                     const int64_t weight_size_x,
                     const std::vector<int64_t>& onnx_pads,
                     const std::vector<int64_t>& onnx_strides,
                     const std::vector<int64_t>& onnx_dilations,
                     AutoPadType auto_pad_type,
                     AutoPadType& auto_pad_type_out) {
  auto_pad_type_out = auto_pad_type;
  if (auto_pad_type == AutoPadType::NOTSET && onnx_dilations == std::vector<int64_t>{1, 1} &&
      // ComputeConvPads() only handles known dimensions of input_shape[2] and input_shape[3]
      input_shape[2] != -1 && input_shape[3] != -1) {
    {
      std::vector<int64_t> same_upper_pads;
      ORT_RETURN_IF_ERROR(ComputeConvPads(input_shape, weight_size_y, weight_size_x,
                                          onnx_pads, onnx_strides, onnx_dilations,
                                          AutoPadType::SAME_UPPER, same_upper_pads));
      if (onnx_pads == same_upper_pads) {
        auto_pad_type_out = AutoPadType::SAME_UPPER;
        return Status::OK();
      }
    }

    {
      std::vector<int64_t> same_lower_pads;
      ORT_RETURN_IF_ERROR(ComputeConvPads(input_shape, weight_size_y, weight_size_x,
                                          onnx_pads, onnx_strides, onnx_dilations,
                                          AutoPadType::SAME_LOWER, same_lower_pads));
      if (onnx_pads == same_lower_pads) {
        auto_pad_type_out = AutoPadType::SAME_LOWER;
        return Status::OK();
      }
    }
  }

  return Status::OK();
}

Status CreateCoreMLWeight(CoreML::Specification::WeightParams& weight,
                          const ONNX_NAMESPACE::TensorProto& tensor) {
  const auto data_type = tensor.data_type();
  Initializer unpacked_tensor(tensor);
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      CreateCoreMLWeight(weight, unpacked_tensor.DataAsSpan<float>());
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      CreateCoreMLWeight(weight, unpacked_tensor.DataAsSpan<int32_t>());
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      CreateCoreMLWeight(weight, unpacked_tensor.DataAsSpan<int64_t>());
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The initializer of graph has unsupported type, name: ",
                             tensor.name(), " type: ", data_type);
  }
  return Status::OK();
}

void CreateCoreMLWeight(CoreML::Specification::WeightParams& weight, gsl::span<const float> data) {
  weight.mutable_floatvalue()->Assign(data.begin(), data.end());
}

namespace {
template <typename T>
void CreateCoreMLWeightConvertingDataToFloats(CoreML::Specification::WeightParams& weight, gsl::span<const T> data) {
  google::protobuf::RepeatedField<float> weight_floats{};
  weight_floats.Reserve(narrow<int>(data.size()));
  std::transform(data.begin(), data.end(), google::protobuf::RepeatedFieldBackInserter(&weight_floats),
                 [](T v) { return narrow<float>(v); });
  *weight.mutable_floatvalue() = std::move(weight_floats);
}
}  // namespace

void CreateCoreMLWeight(CoreML::Specification::WeightParams& weight, gsl::span<const int32_t> data) {
  CreateCoreMLWeightConvertingDataToFloats(weight, data);
}

void CreateCoreMLWeight(CoreML::Specification::WeightParams& weight, gsl::span<const int64_t> data) {
  CreateCoreMLWeightConvertingDataToFloats(weight, data);
}

}  // namespace coreml
}  // namespace onnxruntime

#endif
