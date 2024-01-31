// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/impl/builder_utils.h"

#include "core/common/narrow.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/coreml/builders/coreml_spec.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/shared/utils/utils.h"
#include "core/optimizer/initializer.h"

using namespace COREML_SPEC;

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

//
// ML Program Utils
//

namespace {
void SetTensorTypeInfo(MILSpec::TensorType& tensor_type, MILSpec::DataType data_type,
                       const gsl::span<const int32_t> shape) {
  tensor_type.set_datatype(data_type);
  tensor_type.set_rank(shape.size());
  for (const auto& dim : shape) {
    tensor_type.add_dimensions()->mutable_constant()->set_size(dim);
  }
}

void SetTensorTypeInfo(MILSpec::TensorType& tensor_type, MILSpec::DataType data_type,
                       const ONNX_NAMESPACE::TensorShapeProto* shape) {
  tensor_type.set_datatype(data_type);
  if (shape) {
    tensor_type.set_rank(shape->dim_size());
    for (const auto& dim : shape->dim()) {
      if (dim.has_dim_value()) {
        tensor_type.add_dimensions()->mutable_constant()->set_size(dim.dim_value());
      } else {
        tensor_type.add_dimensions()->mutable_unknown()->set_variadic(false);
      }
    }
  }
}

template <typename T1, typename T2 = T1>
void CopyDataToTensorValue(MILSpec::TensorValue& tensor_value, const gsl::span<const T1> data) {
  // need a 'false' that is dependent on the template types to make gcc happy and give a meaningful error message.
  static_assert(false_for_T<T1> && false_for_T<T2>, "Unsupported data type");  // add specializations below as needed
}

template <>
void CopyDataToTensorValue<float>(MILSpec::TensorValue& tensor_value, const gsl::span<const float> data) {
  tensor_value.mutable_floats()->mutable_values()->Add(data.begin(), data.end());
};

template <>
void CopyDataToTensorValue<int32_t>(MILSpec::TensorValue& tensor_value, const gsl::span<const int32_t> data) {
  tensor_value.mutable_ints()->mutable_values()->Add(data.begin(), data.end());
};

template <>
void CopyDataToTensorValue<std::string>(MILSpec::TensorValue& tensor_value, const gsl::span<const std::string> data) {
  tensor_value.mutable_strings()->mutable_values()->Add(data.begin(), data.end());
};

// copy int64_t (used by ONNX for strides/indexes/etc.) to int32_t (used by CoreML)
template <>
void CopyDataToTensorValue<int64_t, int32_t>(MILSpec::TensorValue& tensor_value, const gsl::span<const int64_t> data) {
  auto& int32_out = *tensor_value.mutable_ints()->mutable_values();
  int32_out.Reserve(narrow<int32_t>(data.size()));
  for (const int64_t v : data) {
    int32_out.AddAlreadyReserved(narrow<int32_t>(v));
  }
};

}  // namespace

// convert int64_t ONNX shape to int32_t CoreML shape
std::vector<int32_t> GetCoreMLShape(const gsl::span<const int64_t> dims) {
  std::vector<int32_t> shape;
  shape.reserve(dims.size());
  for (const auto& dim : dims) {
    shape.push_back(narrow<int32_t>(dim));
  }
  return shape;
}

MILSpec::DataType OnnxDataTypeToMILSpec(int onnx_type) {
  switch (static_cast<ONNX_NAMESPACE::TensorProto_DataType>(onnx_type)) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return MILSpec::DataType::FLOAT32;
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return MILSpec::DataType::FLOAT64;
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      return MILSpec::DataType::BFLOAT16;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return MILSpec::DataType::FLOAT16;

    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      return MILSpec::DataType::INT8;
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      return MILSpec::DataType::INT16;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return MILSpec::DataType::INT32;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return MILSpec::DataType::INT64;

    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return MILSpec::DataType::UINT8;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
      return MILSpec::DataType::UINT16;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      return MILSpec::DataType::UINT32;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      return MILSpec::DataType::UINT64;

    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      return MILSpec::DataType::BOOL;
    case ONNX_NAMESPACE::TensorProto_DataType_STRING:
      return MILSpec::DataType::STRING;
    default:
      ORT_THROW("Unsupported data type: ", onnx_type);
  }
}

template <typename T1, typename T2>
MILSpec::Value CreateTensorValue(const gsl::span<const T1> data,
                                 std::optional<const gsl::span<const int32_t>> shape) {
  MILSpec::Value value;
  MILSpec::TensorType& tensor_type = *value.mutable_type()->mutable_tensortype();

  if (shape) {
    SetTensorTypeInfo(tensor_type, DataTypeToMILSpec<T2>(), *shape);
  } else {
    // infer as 1D shape
    std::vector<int32_t> coreml_shape{narrow<int32_t>(data.size())};
    SetTensorTypeInfo(tensor_type, DataTypeToMILSpec<T2>(), coreml_shape);
  }

  MILSpec::TensorValue& tensor_value = *value.mutable_immediatevalue()->mutable_tensor();
  CopyDataToTensorValue<T1, T2>(tensor_value, data);

  return value;
}

template <typename T>
MILSpec::Value CreateScalarTensorValue(const T& data) {
  gsl::span<const T> data_span{&data, 1};
  std::vector<int32_t> shape = {};  // empty for scalar
  return CreateTensorValue<T>(data_span, shape);
}

// explicit specializations for types we handle so the implementation can be in the .cc file
template MILSpec::Value CreateTensorValue<int64_t, int32_t>(const gsl::span<const int64_t> data,
                                                            std::optional<const gsl::span<const int32_t>> shape);

template MILSpec::Value CreateScalarTensorValue(const float& data);
template MILSpec::Value CreateScalarTensorValue(const int32_t& data);
template MILSpec::Value CreateScalarTensorValue(const std::string& data);

COREML_SPEC::MILSpec::NamedValueType CreateNamedTensorValueType(const NodeArg& node_arg) {
  MILSpec::NamedValueType nvt;
  nvt.set_name(node_arg.Name());
  MILSpec::TensorType& tensor_type = *nvt.mutable_type()->mutable_tensortype();

  SetTensorTypeInfo(tensor_type, OnnxDataTypeToMILSpec(node_arg.TypeAsProto()->tensor_type().elem_type()),
                    node_arg.Shape());

  return nvt;
}

void AddOperationInput(MILSpec::Operation& op, std::string_view input_name, std::string_view value_name) {
  MILSpec::Argument arg;
  arg.mutable_arguments()->Add()->set_name(std::string(value_name));

  (*op.mutable_inputs())[input_name] = std::move(arg);
}

void AddOperationOutput(COREML_SPEC::MILSpec::Operation& op, const NodeArg& output) {
  auto& outputs = *op.mutable_outputs();
  auto& output_arg = *outputs.Add();
  output_arg.set_name(output.Name());

  MILSpec::ValueType& value = *output_arg.mutable_type();
  MILSpec::TensorType& tensor_type = *value.mutable_tensortype();

  SetTensorTypeInfo(tensor_type, OnnxDataTypeToMILSpec(output.TypeAsProto()->tensor_type().elem_type()),
                    output.Shape());
}

}  // namespace coreml
}  // namespace onnxruntime
