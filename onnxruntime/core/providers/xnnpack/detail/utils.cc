// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "utils.h"
#include <unordered_map>

#include "core/framework/tensorprotoutils.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/graph/node_attr_utils.h"

#include "core/providers/shared/node_unit/node_unit.h"
#include "onnx/defs/attr_proto_util.h"
#include "core/common/safeint.h"
namespace onnxruntime {
namespace xnnpack {

const char* TensorQtypeToString(enum TensorQuantType type) {
  switch (type) {
    case TensorTypeInvalid:
      return "Invalid";
    case TensorTypeFp32:
      return "FP32";
    case TensorTypeFp16:
      return "FP16";
    case TensorTypeInt8:
      return "QINT8";
    case TensorTypeUint8:
      return "QUINT8";
    case TensorTypeInt32:
      return "QINT32";
    case TensorTypeInt8_Per_Channel:
      return "QCINT8";
    case TensorTypeInt32_Per_Channel:
      return "QCINT32";
  }
  return NULL;
}

bool GetType(const NodeArg& node_arg, int32_t& type) {
  type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  const auto* type_proto = node_arg.TypeAsProto();
  if (!type_proto || !type_proto->has_tensor_type() || !type_proto->tensor_type().has_elem_type()) {
    return false;
  }

  type = type_proto->tensor_type().elem_type();
  return true;
}

bool GetShape(const NodeArg& node_arg, TensorShapeVector& shape) {
  shape.clear();
  const auto* shape_proto = node_arg.Shape();

  if (!shape_proto) {
    return false;
  }

  for (const auto& dim : shape_proto->dim()) {
    shape.push_back(dim.dim_value());
  }

  return true;
}

QuantizedOpType GetQuantizedOpType(const NodeUnit& node_unit) {
  const auto& op_type = node_unit.OpType();
  if (node_unit.UnitType() == NodeUnit::Type::QDQGroup) {
    if (op_type == "Conv")
      return QuantizedOpType::QDQConv;
    else if (op_type == "MaxPool")
      return QuantizedOpType::QDQMaxPool;
    else if (op_type == "AveragePool")
      return QuantizedOpType::QDQAvgPool;
    else if (op_type == "Softmax")
      return QuantizedOpType::QDQSoftmax;
  } else if (node_unit.OpType() == "QLinearConv") {
    return QuantizedOpType::QLinearConv;
  }
  return QuantizedOpType::Unknown;
}

bool IsPaddingTypeSupported(AutoPadType auto_pad) {
  return auto_pad == AutoPadType::NOTSET ||
         auto_pad == AutoPadType::VALID ||
         auto_pad == AutoPadType::SAME_UPPER;
}

typedef std::string ONNXOpType;

static const std::unordered_map<QuantizedOpType, ONNXOpType> qdq_to_onnx_type_map = {
    {QuantizedOpType::QDQConv, "QLinearConv"},
    {QuantizedOpType::QDQAvgPool, "QLinearAveragePool"},
    {QuantizedOpType::QDQSoftmax, "QLinearSoftmax"},
    {QuantizedOpType::QDQMaxPool, "MaxPool"},
};

std::unique_ptr<IndexedSubGraph::MetaDef> FuseQDQGroup(const NodeUnit& node_unit) {
  QuantizedOpType qtype = GetQuantizedOpType(node_unit);
  // create a ComputeCapability for QDQ node.
  std::unique_ptr<IndexedSubGraph::MetaDef> metadef = std::make_unique<IndexedSubGraph::MetaDef>();
  IndexedSubGraph::MetaDef& def = *metadef;
  ORT_ENFORCE(qdq_to_onnx_type_map.count(qtype), "error quantized op to be fused, op name is ", node_unit.Name());
  // inputs
  const auto& inputs = node_unit.Inputs();
  def.name = qdq_to_onnx_type_map.at(qtype);
  // registration
  def.domain = kMSInternalNHWCDomain;  // should always be kMSInternalNHWCDomain
  def.since_version = node_unit.GetNode().SinceVersion();
  // x x-scale x-zp w w-scale w-zp. Some QDQops wouldn't have 9 inputs,
  // but the 5 more unit extra memory is not too expensive
  def.inputs.reserve(9);
  if (qtype == QuantizedOpType::QDQConv) {
    std::for_each(inputs.cbegin(), inputs.cbegin() + 2,
                  [&def](const NodeUnitIODef& arg) {
                    // keep the number of inputs the same by inserting an empty string for a missing optional input
                    def.inputs.push_back(arg.node_arg.Name());
                    const auto& quant_param = arg.quant_param.value();
                    def.inputs.push_back(quant_param.scale.Name());
                    def.inputs.push_back(quant_param.zero_point ? quant_param.zero_point->Name() : "");
                  });
    // y-scale y-zeropoint
    const auto& y_quant_param = node_unit.Outputs()[0].quant_param.value();
    def.inputs.push_back(y_quant_param.scale.Name());
    def.inputs.push_back(y_quant_param.zero_point ? y_quant_param.zero_point->Name() : "");

    // bias
    if (inputs.size() > 2) {
      def.inputs.push_back(inputs[2].node_arg.Name());
    }
  } else if (qtype == QuantizedOpType::QDQAvgPool || qtype == QuantizedOpType::QDQSoftmax) {
    // x x-scale x-zp
    std::for_each(inputs.cbegin(), inputs.cend(),
                  [&def](const NodeUnitIODef& arg) {
                    // keep the number of inputs the same by inserting an empty string for a missing optional input
                    def.inputs.push_back(arg.node_arg.Name());
                    const auto& quant_param = arg.quant_param.value();
                    def.inputs.push_back(quant_param.scale.Name());
                    def.inputs.push_back(quant_param.zero_point ? quant_param.zero_point->Name() : "");
                  });
    // y-scale y-zeropoint
    const auto& y_quant_param = node_unit.Outputs()[0].quant_param.value();
    def.inputs.push_back(y_quant_param.scale.Name());
    def.inputs.push_back(y_quant_param.zero_point ? y_quant_param.zero_point->Name() : "");
  } else if (qtype == QuantizedOpType::QDQMaxPool) {
    // only one input for QDQMaxPool, Tensor:X
    def.inputs.push_back(inputs[0].node_arg.Name());
  } else {
    // all qdq-types are enumerated
    ORT_ENFORCE(0, "unknown QDQ ops", def.name);
  }

  // outputs
  for (const auto& out : node_unit.Outputs()) {
    def.outputs.push_back(out.node_arg.Name());
  }

  // attributes
  // copy existing and add the activation info
  def.attributes.insert(node_unit.GetNode().GetAttributes().begin(), node_unit.GetNode().GetAttributes().end());
  return metadef;
}

// Fuse activation with node_unit.
std::unique_ptr<IndexedSubGraph::MetaDef> FuseActivation(const NodeUnit& node_unit, const Node& activation,
                                                         const GraphViewer& graph) {
  std::unique_ptr<IndexedSubGraph::MetaDef> metadef = std::make_unique<IndexedSubGraph::MetaDef>();
  IndexedSubGraph::MetaDef& def = *metadef;

  // we use the op type/domain to match the static xnnpack Conv or MaxPool kernel
  // registration
  def.name = node_unit.OpType();
  def.domain = node_unit.Domain();  // should always be kMSInternalNHWCDomain
  def.since_version = node_unit.SinceVersion();

  // inputs
  const auto& inputs = node_unit.Inputs();
  def.inputs.reserve(inputs.size());
  std::for_each(inputs.cbegin(), inputs.cend(),
                [&def](const NodeUnitIODef& iodef) {
                  def.inputs.push_back(iodef.node_arg.Name());
                });

  // outputs
  def.outputs.push_back(activation.OutputDefs()[0]->Name());

  // attributes
  // copy existing and add the activation info
  def.attributes = node_unit.GetNode().GetAttributes();

  // use infinity as the default as that's what xnnpack uses if min/max are not set
  float min = -INFINITY;
  float max = INFINITY;

  const auto& activation_type = activation.OpType();
  if (activation_type == "Clip") {
    min = std::numeric_limits<float>::min();
    max = std::numeric_limits<float>::max();
    bool min_max_are_attributes = activation.SinceVersion() == 1 || activation.SinceVersion() == 6;

    if (min_max_are_attributes) {
      ProtoHelperNodeContext nc(activation);
      OpNodeProtoHelper info(&nc);
      min = info.GetAttrOrDefault<float>("min", min);
      max = info.GetAttrOrDefault<float>("max", max);
    } else {
      const auto& clip_inputs = activation.InputDefs();
      const auto num_inputs = clip_inputs.size();

      const auto update_value = [&](size_t idx, float& value_to_set) {
        if (num_inputs > idx) {
          const NodeArg& arg = *clip_inputs[idx];
          if (arg.Exists()) {
            const auto& value = *graph.GetConstantInitializer(arg.Name(), true);
            // these should never be in external data as it makes no sense to put scalars there.
            ORT_ENFORCE(utils::HasExternalData(value) == false,
                        "External data is not supported for the scalar min/max Clip values");

            value_to_set = utils::HasRawData(value)
                               ? *reinterpret_cast<const float*>(value.raw_data().data())
                               : value.float_data()[0];
          }
        }
      };

      update_value(1, min);
      update_value(2, max);
    }
  } else if (activation_type == "Relu") {
    min = 0.f;
  } else {
    ORT_NOT_IMPLEMENTED("No support for fusion of ", node_unit.OpType(), " with ", activation_type);
  }

  InlinedVector<float> activation_params{min, max};
  def.attributes.insert({"activation", utils::MakeAttribute("activation", activation_type)});
  def.attributes.insert({"activation_params", utils::MakeAttribute("activation_params", activation_params)});

  return metadef;
}

const onnx::TensorProto* GetQuantizationScale(const InitializedTensorSet& initializers,
                                              const NodeUnitIODef& io_def) {
  if (io_def.quant_param.has_value() == false) {
    return nullptr;
  }

  onnx::TensorProto tensor_proto_ret;
  const auto scale_name = io_def.quant_param->scale.Name();
  auto it = initializers.find(scale_name);
  if (it == initializers.cend()) {
    return nullptr;
  }
  return it->second;
}

const onnx::TensorProto* GetQuantizationZeroPoint(const InitializedTensorSet& initializers,
                                                  const NodeUnitIODef& io_def) {
  if (!io_def.quant_param.has_value() || !io_def.quant_param->zero_point)
    return nullptr;

  const auto& zero_point_name = io_def.quant_param->zero_point->Name();
  if (!Contains(initializers, zero_point_name)) {
    return nullptr;
  }

  return initializers.at(zero_point_name);
}

// we have uint8,int8 and int8_per-channel
TensorQuantType GetTensorQuantType(const NodeUnit& node_unit, int32_t io_index,
                                   bool is_output, const GraphViewer& graph_viewer) {
  // we do not check the legality of io_index here
  const NodeUnitIODef& iodef = is_output ? node_unit.Outputs()[io_index] : node_unit.Inputs()[io_index];
  TensorQuantType datatype = TensorTypeInvalid;
  int32_t input_type = 0;
  if (!GetType(iodef.node_arg, input_type) || iodef.quant_param.has_value() == false) {
    return TensorTypeInvalid;
  }

  const InitializedTensorSet& initializers = graph_viewer.GetAllInitializedTensors();
  auto* zero_tensor = GetQuantizationZeroPoint(initializers, iodef);
  auto* scale_tensor = GetQuantizationScale(initializers, iodef);

  if (scale_tensor == nullptr || (zero_tensor && zero_tensor->data_type() != input_type)) {
    return TensorTypeInvalid;
  }

  int64_t scales_dim = scale_tensor->dims().empty() ? 1 : scale_tensor->dims()[0];
  int64_t zero_dim = !zero_tensor ? 0 : (zero_tensor->dims().empty() ? 1 : zero_tensor->dims()[0]);
  const auto& quantization_params = iodef.quant_param.value();

  TensorShapeVector tensor_shape;
  if (!GetShape(iodef.node_arg, tensor_shape)) {
    return TensorTypeInvalid;
  }

  std::vector<uint8_t> unpacked_tensor;
  // we have processed float-type in the beginning
  // we do not handle u8s8
  switch (input_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      // zero_point is optional according to https://github.com/onnx/onnx/blob/main/docs/Operators.md#attributes-20
      if (quantization_params.zero_point && (scales_dim != 1 || zero_dim != 1)) {
        LOGS_DEFAULT(VERBOSE) << "unsupported number " << scales_dim
                              << " of scale quantization parameters for UINT8 tensor"
                                 "per-channel uint8 quantization isn't supported";
        break;
      }

      datatype = TensorTypeUint8;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      // symmetry quantization when zero_dim == 0
      if (scales_dim != zero_dim && zero_dim != 0) {
        LOGS_DEFAULT(VERBOSE) << "mismatching number of scale " << scales_dim
                              << " and zero-point " << zero_dim << " quantization parameters for INT8";
        break;
      }

      if (scales_dim == 1) {
        datatype = TensorTypeInt8;
        // layout keeps NCHW, check output channel dim
      } else if (scales_dim == tensor_shape[0]) {
        // default 0 for zero-point if zero_dim == 0
        if (zero_tensor != nullptr) {
          auto status = utils::UnpackInitializerData(*zero_tensor, node_unit.ModelPath(), unpacked_tensor);
          if (!status.IsOK()) {
            LOGS_DEFAULT(ERROR) << "error when unpack zero tensor: "
                                << ", error msg: " << status.ErrorMessage();
            break;
          }
          const int8_t* zero_points = reinterpret_cast<const int8_t*>(unpacked_tensor.data());
          for (size_t i = 0; i < unpacked_tensor.size(); i++) {
            if (zero_points[i] != 0) {
              LOGS_DEFAULT(VERBOSE) << "only support 0 as zero point for per-channel quantization, "
                                    << "zero_points[" << i << "] has value: " << zero_points[i];
              break;
            }
          }
        }

        datatype = TensorTypeInt8_Per_Channel;
      } else {
        LOGS_DEFAULT(VERBOSE) << "mismatching number of quantization parameters  " << scales_dim
                              << " and outer dimension " << tensor_shape[1];
      }
      break;
      // TODO(Jicwen)
    /* case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      break;
      */
    default:
      break;
  }
  return datatype;
}

bool ParseQuantParamFromInfoByOrder(const OpKernelInfo& info,
                                    const InputTensorOrder& scale_zp_indexs,
                                    QuantParam& quant_param) {
  // quant param, which used in create xnnpack_kernel
  // we do not check the error here, as we have done it in op_checker
  // if this input tensor is not exists, its value is -1;
  if (scale_zp_indexs.X_ZERO_POINT >= 0) {
    const Tensor* X_zero_point = nullptr;
    info.TryGetConstantInput(scale_zp_indexs.X_ZERO_POINT, &X_zero_point);

    if (X_zero_point == nullptr) {
      quant_param.X_zero_point_value = 0;
    } else {
      // take all data as uint8, so we can easily parse zero-point and store in out data structure.
      // we will re-cast it to the real datatype (u8 or s8) in the right place
      quant_param.X_zero_point_value = *reinterpret_cast<const uint8_t*>(X_zero_point->DataRaw());
    }
  }

  if (scale_zp_indexs.W_ZERO_POINT >= 0) {
    const Tensor* W_zero_point = nullptr;
    info.TryGetConstantInput(scale_zp_indexs.W_ZERO_POINT, &W_zero_point);

    if (W_zero_point == nullptr) {
      quant_param.W_zero_point_value = 0;
    } else {
      quant_param.W_zero_point_value = *reinterpret_cast<const uint8_t*>(W_zero_point->DataRaw());
    }
  }

  if (scale_zp_indexs.Y_ZERO_POINT >= 0) {
    const Tensor* Y_zero_point = nullptr;
    info.TryGetConstantInput(scale_zp_indexs.Y_ZERO_POINT, &Y_zero_point);

    if (Y_zero_point == nullptr) {
      quant_param.Y_zero_point_value = 0;
    } else {
      quant_param.Y_zero_point_value = *reinterpret_cast<const uint8_t*>(Y_zero_point->DataRaw());
    }
  }

  if (scale_zp_indexs.X_SCALE >= 0) {
    const Tensor* X_scale = nullptr;
    info.TryGetConstantInput(scale_zp_indexs.X_SCALE, &X_scale);
    quant_param.X_scale_value = *(X_scale->template Data<float>());
  }

  if (scale_zp_indexs.W_SCALE >= 0) {
    const Tensor* W_scale = nullptr;
    info.TryGetConstantInput(scale_zp_indexs.W_SCALE, &W_scale);
    quant_param.W_scale_value = *(W_scale->template Data<float>());

    if (!IsScalarOr1ElementVector(W_scale)) {
      quant_param.W_scale_tensor = W_scale;
    }
  }

  if (scale_zp_indexs.Y_SCALE >= 0) {
    const Tensor* Y_scale = nullptr;
    info.TryGetConstantInput(scale_zp_indexs.Y_SCALE, &Y_scale);
    quant_param.Y_scale_value = *(Y_scale->template Data<float>());
  }
  return true;
}

}  // namespace xnnpack
}  // namespace onnxruntime
