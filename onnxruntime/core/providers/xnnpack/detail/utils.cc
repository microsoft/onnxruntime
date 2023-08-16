// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "utils.h"
#include <unordered_map>
#include <vector>

#include "core/common/common.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/graph/node_attr_utils.h"

#include "core/providers/shared/node_unit/node_unit.h"
#include "onnx/defs/attr_proto_util.h"
#include "core/common/safeint.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime {
namespace xnnpack {

const char* OpTypeToString(OpComputeType opCtype) {
  switch (opCtype) {
    case op_compute_type_fp32:
      return "fp32";
    case op_compute_type_fp16:
      return "fp16";
    case op_compute_type_qs8_per_channel:
      return "qc8";
    case op_compute_type_qs8:
      return "qs8";
    case op_compute_type_qu8:
      return "qu8";
    default:
      return "invalid";
  }
}
const char* TensorQtypeToString(enum TensorQuantType type) {
  switch (type) {
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
    default:
      return "invalid";
  }
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
    else if (op_type == "Resize")
      return QuantizedOpType::QDQResize;
    else if (op_type == "ConvTranspose")
      return QuantizedOpType::QDQConvTranspose;

  } else if (node_unit.OpType() == "QLinearConv") {
    return QuantizedOpType::QLinearConv;
  } else if (node_unit.OpType() == "QLinearConvTranspose") {
    return QuantizedOpType::QLinearConvTranspose;
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
    {QuantizedOpType::QDQResize, "Resize"},
    {QuantizedOpType::QDQConvTranspose, "QLinearConvTranspose"},
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
  def.domain = kMSInternalNHWCDomain;                      // should always be kMSInternalNHWCDomain
  def.since_version = node_unit.GetNode().SinceVersion();  // seems we can't change it for registered scheme
  // x x-scale x-zp w w-scale w-zp. Some QDQops wouldn't have 9 inputs,
  // but the 5 more unit extra memory is not too expensive
  def.inputs.reserve(9);
  if (qtype == QuantizedOpType::QDQConv || qtype == QuantizedOpType::QDQConvTranspose) {
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
    if (qtype == QuantizedOpType::QDQConvTranspose) {
      def.since_version = 1;
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
    if (qtype == QuantizedOpType::QDQSoftmax) {
      def.domain = kDynamicDomainByCreate;
      def.since_version = 1;
      def.attributes.emplace("opset", utils::MakeAttribute(std::string("opset"), int64_t(node_unit.SinceVersion())));
    }
  } else if (qtype == QuantizedOpType::QDQMaxPool || qtype == QuantizedOpType::QDQResize) {
    // Don't care about the quantization parameters for MaxPool, Resize
    // where the two ops don't ask for quantization parameters in computation.
    std::for_each(inputs.cbegin(), inputs.cend(),
                  [&def](const NodeUnitIODef& arg) {
                    def.inputs.push_back(arg.node_arg.Name());
                  });
    if (qtype == QuantizedOpType::QDQResize) {
      def.domain = kOnnxDomain;  // QDQResize is not layout sensitive
    }
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
std::unique_ptr<IndexedSubGraph::MetaDef> FuseActivation(const NodeUnit& node_unit, const NodeUnit& activation_unit,
                                                         const GraphViewer& graph) {
  std::unique_ptr<IndexedSubGraph::MetaDef> metadef = std::make_unique<IndexedSubGraph::MetaDef>();
  IndexedSubGraph::MetaDef& def = *metadef;
  const Node& activation = activation_unit.GetNode();

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

std::pair<const onnx::TensorProto*, const onnx::TensorProto*>
GetQuantizationZeroPointAndScale(const GraphViewer& graphview,
                                 const NodeUnitIODef& io_def) {
  std::pair<const onnx::TensorProto*, const onnx::TensorProto*> ret{0, 0};
  if (!io_def.quant_param.has_value()) {
    return ret;
  }

  if (io_def.quant_param.value().zero_point) {
    const auto& zero_point_name = io_def.quant_param->zero_point->Name();
    ret.second = graphview.GetConstantInitializer(zero_point_name, true);
  }

  {
    const auto scale_name = io_def.quant_param->scale.Name();
    ret.first = graphview.GetConstantInitializer(scale_name, true);
  }
  return ret;
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
  auto [scale_tensor, zero_tensor] = GetQuantizationZeroPointAndScale(graph_viewer, iodef);
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
          Initializer zp_val(*zero_tensor, node_unit.ModelPath());
          auto zero_points = zp_val.DataAsSpan<int8_t>();
          for (size_t i = 0; i < zp_val.size(); i++) {
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

template <typename T>
gsl::span<const T> ReadConstantValues(const OpKernelInfo& info, int idx) {
  const onnxruntime::Tensor* tensor = nullptr;

  // this should never happen to throw. op support checker should not choose an op that does not have a constant input
  if (!info.TryGetConstantInput(idx, &tensor)) {
    if constexpr (std::is_same<T, float>::value) {
      ORT_THROW("Could not read constant values from idx ", idx);
    } else {
      // It's legal for zero-point to be null, we just give its default value 0
      static const T default_zp[] = {0};
      return gsl::make_span(default_zp, static_cast<typename gsl::span<T>::size_type>(1));
    }
  }
  return (tensor->DataAsSpan<T>());
}

void GetScaleAndZeroPoint(const OpKernelInfo& info,
                          int scale_idx, std::vector<float>& scale, int zp_idx,
                          uint8_t& zero_point, int32_t x_dtype) {
  auto s_span = ReadConstantValues<float>(info, scale_idx);
  scale.assign(s_span.begin(), s_span.end());

  if (x_dtype == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    zero_point = ReadConstantValues<uint8_t>(info, zp_idx)[0];
  } else if (x_dtype == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
    zero_point = ReadConstantValues<int8_t>(info, zp_idx)[0];
  } else {
    ORT_THROW("invalid dtype of zero point, expected uint8|int8, but got onnx dtype ", x_dtype);
  }
}

// A general function To parse QuantParam for different ops,
// @param info:OpKernelInfo
// @param x_dtype:int32_t|enum ONNX_NAMESPACE::TensorProto_DataType, defined in
// "external/onnx/onnx/onnx-ml.pb.h", to represent the data types of zero_point.
// And scale is always float-type
// @param how_many_input_scale_and_zp:size_t, how many input tensors require quantized params. Typically,
// Conv has three inputs, but bias don't ask for a scale and zero point.
// These definitions are elaborated in onnx schema
// @ret,OpQuantParam, defined in utils.h, to store all scale and zero point.
// All ops have at least one input quant-param(x-scale, x-zero-point)
// and one output quant-param(y-scale, y-zero-point), such as softmax, pool(average-/max-,global-)
// but we might want to adapt irregular ops like, concat/slice, which may have arbitrary inputs.
// This function only works with 8 bits quantization.
OpQuantParam ParseQuantParamForOp(const OpKernelInfo& info, int32_t x_dtype, size_t how_many_input_scale_and_zp) {
  OpQuantParam quant_param;
  int start_idx = 1;
  // take all data as uint8, so we can easily parse zero-point and store in out data structure.
  // we will re-cast it to the real datatype (u8 or s8) in the right place
  // Attention: we are assuming all zero-point being either int8 or uint8

  std::pair<std::vector<float>, uint8_t> param;
  GetScaleAndZeroPoint(info, start_idx, param.first, start_idx + 1, param.second, x_dtype);
  start_idx += 2;
  quant_param.push_back(param);
  for (size_t nThInput = 2; nThInput <= how_many_input_scale_and_zp; ++nThInput) {
    start_idx++;
    // w, w_scale, zero_point
    GetScaleAndZeroPoint(info, start_idx, param.first, start_idx + 1, param.second, x_dtype);
    start_idx += 2;
    quant_param.push_back(param);
  }
  // y_scale, zero_point
  GetScaleAndZeroPoint(info, start_idx, param.first, start_idx + 1, param.second, x_dtype);
  quant_param.push_back(param);
  return quant_param;
}

}  // namespace xnnpack
}  // namespace onnxruntime
