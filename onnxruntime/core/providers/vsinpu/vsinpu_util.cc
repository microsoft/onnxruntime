/****************************************************************************
 *
 *    Copyright (c) 2023 Vivante Corporation
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a
 *    copy of this software and associated documentation files (the "Software"),
 *    to deal in the Software without restriction, including without limitation
 *    the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *    and/or sell copies of the Software, and to permit persons to whom the
 *    Software is furnished to do so, subject to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in
 *    all copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *    DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#include <map>
#include <algorithm>
#include <utility>
#include <unordered_set>
#include "core/providers/vsinpu/vsinpu_util.h"

#include "core/optimizer/initializer.h"
#include "core/providers/shared/utils/utils.h"
namespace onnxruntime {

template <typename T>
struct shared_array_deletor {
  void operator()(T const* ptr) { delete[] ptr; }
};
namespace vsi {
namespace npu {
namespace util {
tim::vx::DataType OnnxDtypeToTIMVXDtype(const int32_t dtype) {
  switch (dtype) {
    case onnx::TensorProto_DataType_FLOAT:
      return tim::vx::DataType::FLOAT32;
    case onnx::TensorProto_DataType_FLOAT16:
      return tim::vx::DataType::FLOAT16;
    case onnx::TensorProto_DataType_INT8:
      return tim::vx::DataType::INT8;
    case onnx::TensorProto_DataType_UINT8:
      return tim::vx::DataType::UINT8;
    case onnx::TensorProto_DataType_INT32:
      return tim::vx::DataType::INT32;
    case onnx::TensorProto_DataType_INT16:
      return tim::vx::DataType::INT16;
    case onnx::TensorProto_DataType_UINT16:
      return tim::vx::DataType::UINT16;
    case onnx::TensorProto_DataType_BOOL:
      return tim::vx::DataType::BOOL8;
    default:
      LOGS_DEFAULT(WARNING) << "Unsupported data type: " << dtype;
      break;
  }
  return tim::vx::DataType::FLOAT32;
}

tim::vx::DataType OnnxDtypeToTIMVXDtype(const ONNX_NAMESPACE::DataType type) {
  static const std::map<std::string, tim::vx::DataType> type_table = {
      {"tensor(float)", tim::vx::DataType::FLOAT32},
      {"tensor(float16)", tim::vx::DataType::FLOAT16},
      {"tensor(int8)", tim::vx::DataType::INT8},
      {"tensor(uint8)", tim::vx::DataType::UINT8},
      {"tensor(int32)", tim::vx::DataType::INT32},
      {"tensor(int16)", tim::vx::DataType::INT16},
      {"tensor(uint16)", tim::vx::DataType::UINT16},
      {"tensor(int64)", tim::vx::DataType::INT64},
      {"tensor(bool)", tim::vx::DataType::BOOL8},
  };
  auto search = type_table.find(*type);
  if (search != type_table.end()) {
    return search->second;
  }
  LOGS_DEFAULT(WARNING) << "Unsupported data type: " << *type;
  return tim::vx::DataType::FLOAT32;
}

tim::vx::ShapeType OnnxShapeToTIMVXShape(const onnxruntime::TensorShape& ts) {
  tim::vx::ShapeType timvx_shape(ts.NumDimensions());
  if (ts.NumDimensions() == 0) {
    timvx_shape.push_back(1);
  } else {
    for (size_t i = 0; i < ts.NumDimensions(); i++) {
      timvx_shape[i] = ts.GetDims()[i];
    }
  }
  return timvx_shape;
}

std::string PrintNode(const onnxruntime::NodeArg& node_arg) {
  auto shape = node_arg.Shape();
  if (shape == nullptr) {
    return "<null>";
  }
  std::string s = node_arg.Name() + ":<";
  if (shape->dim_size() == 0) {
    s += "1>, is a scalar";
    return s;
  }
  for (int i = 0; i < shape->dim_size(); i++) {
    auto dim = shape->dim(i);
    std::string s1;
    std::stringstream ss;
    ss << dim.dim_value();
    ss >> s1;
    s += s1;
    if (i < shape->dim_size() - 1) {
      s += ",";
    } else {
      s += ">";
    }
  }
  return s;
}

std::string PrintNode(const std::vector<int64_t> shape) {
  if (shape.size() == 0) {
    return "<null>";
  }
  std::string s = "<";
  for (std::size_t i = 0; i < shape.size(); i++) {
    auto dim = shape[i];
    std::string s1;
    std::stringstream ss;
    ss << dim;
    ss >> s1;
    s += s1;
    if (i < shape.size() - 1) {
      s += ",";
    } else {
      s += ">";
    }
  }
  return s;
}

size_t GetTensorElementSize(const ONNXTensorElementDataType type) {
  switch (type) {
    case onnx::TensorProto_DataType_INT64:
      return 8;
    case onnx::TensorProto_DataType_FLOAT:
    case onnx::TensorProto_DataType_INT32:
      return 4;
    case onnx::TensorProto_DataType_FLOAT16:
    case onnx::TensorProto_DataType_INT16:
    case onnx::TensorProto_DataType_UINT16:
      return 2;
    case onnx::TensorProto_DataType_INT8:
    case onnx::TensorProto_DataType_UINT8:
    case onnx::TensorProto_DataType_BOOL:
      return 1;
    default:
      break;
  }
  return 0;
}

size_t GetTensorBytes(const Ort::TensorTypeAndShapeInfo& info) {
  return info.GetElementCount() * GetTensorElementSize(info.GetElementType());
}

TensorShape GetTensorShape(const onnxruntime::NodeArg& node_arg) {
  auto shape_proto = node_arg.Shape();
  std::vector<int64_t> dims;
  if (shape_proto != nullptr) {
    for (int i = 0; i < shape_proto->dim_size(); i++) {
      auto dim = shape_proto->dim(i);
      dims.push_back(dim.dim_value());
    }
  }
  if (dims.size() == 0) {
    dims.push_back(1);
  }
  TensorShape ts(dims);
  return ts;
}

std::shared_ptr<uint8_t> UnpackTensor(
    const NodeArg* node_arg, const ONNX_NAMESPACE::TensorProto& initializer) {
  std::shared_ptr<uint8_t> unpackedTensor;
  auto shape = GetTensorShape(*node_arg);
  size_t elementCount = shape.Size();

#define CASE_PROTO(X, Y)                                                      \
  case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_##X: {      \
    size_t tensorByteSize = elementCount * sizeof(Y);                         \
    unpackedTensor.reset(new uint8_t[tensorByteSize],                         \
                         shared_array_deletor<uint8_t>());                    \
    auto status = onnxruntime::utils::UnpackTensor(                           \
        initializer,                                                          \
        initializer.has_raw_data() ? initializer.raw_data().data() : nullptr, \
        initializer.has_raw_data() ? initializer.raw_data().size() : 0,       \
        reinterpret_cast<Y*>(unpackedTensor.get()), elementCount);            \
    if (!status.IsOK()) {                                                     \
      LOGS_DEFAULT(ERROR) << "Unpack tensor data failed.";                    \
    }                                                                         \
    break;                                                                    \
  }
  switch (initializer.data_type()) {
    CASE_PROTO(FLOAT, float);
    CASE_PROTO(DOUBLE, double);
    CASE_PROTO(BOOL, bool);
    CASE_PROTO(INT8, int8_t);
    CASE_PROTO(INT16, int16_t);
    CASE_PROTO(INT32, int32_t);
    CASE_PROTO(INT64, int64_t);
    CASE_PROTO(UINT8, uint8_t);
    CASE_PROTO(UINT16, uint16_t);
    CASE_PROTO(UINT32, uint32_t);
    CASE_PROTO(FLOAT16, onnxruntime::MLFloat16);
    default:
      return nullptr;
  }

  return unpackedTensor;
}

tim::vx::PadType GetPadType(const std::string type) {
  static const std::map<std::string, tim::vx::PadType> type_table = {
      {"NOTSET", tim::vx::PadType::AUTO},
      {"SAME_UPPER", tim::vx::PadType::SAME},
      {"SAME_LOWER", tim::vx::PadType::SAME},
      {"VALID", tim::vx::PadType::VALID},
  };
  auto search = type_table.find(type);
  if (search != type_table.end()) {
    return search->second;
  }
  return tim::vx::PadType::NONE;
}

int32_t ReverseAxis(int32_t origin_axis, int32_t length) {
  int32_t axis = 0;
  if (origin_axis < 0) {
    origin_axis += length;
  }
  axis = length - origin_axis - 1;
  return axis;
}

std::vector<int32_t> ReverseAxis(std::vector<int32_t> origin_axes, int32_t length) {
  std::vector<int32_t> axes;
  for (int32_t& axis : origin_axes) {
    if (axis < 0) {
      axis += length;
    }
    axes.push_back(length - axis - 1);
  }
  std::sort(axes.begin(), axes.end());
  return axes;
}

bool IsTypeSupported(const NodeArg* node_arg) {
  const auto* type_proto = node_arg->TypeAsProto();
  if (!type_proto) {
    return false;
  }

  switch (type_proto->tensor_type().elem_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64:
      return true;
    default:
      return false;
  }
}

QuantizedOpType GetQuantizedOpType(const NodeUnit& node_unit) {
  const auto& op_type = node_unit.OpType();
  if (node_unit.UnitType() == NodeUnit::Type::SingleNode) {
    if (op_type == "DequantizeLinear")
      return QuantizedOpType::DequantizeLinear;
    else if (op_type == "QuantizeLinear")
      return QuantizedOpType::QuantizeLinear;
    else if (op_type == "QLinearConv")
      return QuantizedOpType::QLinearConv;
    else if (op_type == "QLinearMatMul")
      return QuantizedOpType::QLinearMatMul;
    else if (op_type == "QLinearAdd")
      return QuantizedOpType::QLinearAdd;
    else if (op_type == "QLinearMul")
      return QuantizedOpType::QLinearMul;
    else if (op_type == "QLinearSigmoid")
      return QuantizedOpType::QLinearSigmoid;
    else if (op_type == "QLinearAveragePool")
      return QuantizedOpType::QLinearAveragePool;
  } else if (node_unit.UnitType() == NodeUnit::Type::QDQGroup) {
    if (op_type == "Conv")
      return QuantizedOpType::QDQConv;
    else if (op_type == "Resize")
      return QuantizedOpType::QDQResize;
    else if (op_type == "AveragePool")
      return QuantizedOpType::QDQAveragePool;
    else if (op_type == "Add")
      return QuantizedOpType::QDQAdd;
    else if (op_type == "Mul")
      return QuantizedOpType::QDQMul;
    else if (op_type == "Transpose")
      return QuantizedOpType::QDQTranspose;
    else if (op_type == "Reshape")
      return QuantizedOpType::QDQReshape;
    else if (op_type == "Softmax")
      return QuantizedOpType::QDQSoftmax;
    else if (op_type == "Concat")
      return QuantizedOpType::QDQConcat;
    else if (op_type == "Gemm")
      return QuantizedOpType::QDQGemm;
    else if (op_type == "MatMul")
      return QuantizedOpType::QDQMatMul;
  }
  return QuantizedOpType::Unknown;
}

ConvType GetConvType(const NodeUnit& node_unit, const InitializedTensorSet& initializers) {
  NodeAttrHelper helper(node_unit);
  const auto group = helper.Get("group", 1);

  const auto& weight = node_unit.Inputs()[1].node_arg.Name();
  const auto& weight_tensor = *initializers.at(weight);

  // For ONNX we only have 1 conv ops
  // For VSINPU we have 3
  // Input is (W, H, C, N)
  // group == 1,                                   --> regular conv
  // group != 1 && weight is (kW, kH, group, M),       --> depthwise conv
  // group != 1 && weight is (kW, kH, C/group, M), --> grouped conv
  if (group == 1)
    return ConvType::Regular;
  else if ((weight_tensor.dims()[1] == group))
    return ConvType::Depthwise;
  else
    return ConvType::Grouped;
}

bool IsQuantizedConv(QuantizedOpType quant_op_type) {
  return (quant_op_type == QuantizedOpType::QLinearConv) ||
         (quant_op_type == QuantizedOpType::QDQConv);
}

bool IsQuantizedPool(QuantizedOpType quant_op_type) {
  return (quant_op_type == QuantizedOpType::QLinearAveragePool) ||
         (quant_op_type == QuantizedOpType::QDQAveragePool);
}

bool IsQuantizedGemm(QuantizedOpType quant_op_type) {
  return (quant_op_type == QuantizedOpType::QLinearMatMul) ||
         (quant_op_type == QuantizedOpType::QDQGemm) ||
         (quant_op_type == QuantizedOpType::QDQMatMul);
}

bool IsQuantizedBinaryOp(QuantizedOpType quant_op_type) {
  return quant_op_type == QuantizedOpType::QLinearMatMul ||
         quant_op_type == QuantizedOpType::QLinearAdd ||
         quant_op_type == QuantizedOpType::QLinearMul ||
         quant_op_type == QuantizedOpType::QDQAdd ||
         quant_op_type == QuantizedOpType::QDQMul ||
         quant_op_type == QuantizedOpType::QDQGemm ||
         quant_op_type == QuantizedOpType::QDQMatMul ||
         IsQuantizedConv(quant_op_type);
}

bool HasValidBinaryOpQuantizedInputTypes(const NodeUnit& node_unit) {
  auto quant_op_type = GetQuantizedOpType(node_unit);
  int32_t a_input_type, b_input_type;
  if (!IsQuantizedBinaryOp(quant_op_type)) {
    LOGS_DEFAULT(VERBOSE) << "[" << node_unit.OpType() << "] is not a binary qlinear op";
    return false;
  }

  const auto& inputs = node_unit.Inputs();
  if (!GetType(inputs[0].node_arg, a_input_type))
    return false;
  if (!GetType(inputs[1].node_arg, b_input_type))
    return false;

  // QlinearConv/MatMul/QDQGemm/QDQMatMul supports u8u8 or u8s8
  // QLinearAdd/QLinearMul only support u8u8
  bool is_quant_conv_or_gemm = IsQuantizedConv(quant_op_type) || IsQuantizedGemm(quant_op_type);

  bool has_valid_qlinear_conv_weight =
      (b_input_type == ONNX_NAMESPACE::TensorProto_DataType_UINT8 ||
       b_input_type == ONNX_NAMESPACE::TensorProto_DataType_INT8);

  bool has_valid_qlinear_conv_input =
      (a_input_type == ONNX_NAMESPACE::TensorProto_DataType_UINT8 ||
       a_input_type == ONNX_NAMESPACE::TensorProto_DataType_INT8);

  if ((is_quant_conv_or_gemm && !has_valid_qlinear_conv_weight) ||
      (!is_quant_conv_or_gemm && a_input_type != b_input_type)) {
    LOGS_DEFAULT(VERBOSE) << "[" << node_unit.OpType()
                          << "] A Input type: [" << a_input_type
                          << "] B Input type: [" << b_input_type
                          << "] is not supported for now";
    return false;
  }

  return true;
}

void GetQuantizationScaleAndZeroPoint(
    const GraphViewer& graph_viewer, const NodeUnitIODef& io_def, const std::filesystem::path& model_path,
    float& scale, int32_t& zero_point, std::optional<std::vector<float>>& pcq_scales,
    std::optional<std::vector<int32_t>>& pcq_zps) {
  scale = 0.0f;
  zero_point = 0;

  const auto& quant_param = *io_def.quant_param;
  {  // get the scale
    const auto& name = quant_param.scale.Name();
    const auto* s = graph_viewer.GetConstantInitializer(name);
    if (!s) {
      LOGS_DEFAULT(ERROR) << name + " is not a constant initializer";
    };
    Initializer unpacked_tensor(graph_viewer.GetGraph(), *s, model_path);
    scale = unpacked_tensor.DataAsSpan<float>()[0];

    // per channel quantized handling
    if (!unpacked_tensor.dims().empty() && unpacked_tensor.dims()[0] != 0 && unpacked_tensor.dims()[0] != 1) {
      auto scales = unpacked_tensor.DataAsSpan<float>();
      std::vector<float> scales_vec(scales.begin(), scales.end());
      pcq_scales = onnxruntime::make_optional(std::move(scales_vec));
    }
  }

  if (quant_param.zero_point) {  // get the zero point if it exists
    const auto& name = quant_param.zero_point->Name();
    const auto* s = graph_viewer.GetConstantInitializer(name);
    if (!s) {
      LOGS_DEFAULT(ERROR) << name + " is not a constant initializer";
    };
    Initializer unpacked_tensor(graph_viewer.GetGraph(), *s, model_path);
    bool is_i8_zp = unpacked_tensor.data_type() == onnx::TensorProto_DataType_INT8;
    // some qdq conv bias is int32 quantized
    bool is_int32_zp = unpacked_tensor.data_type() == onnx::TensorProto_DataType_INT32;
    zero_point = is_i8_zp
                     ? static_cast<int32_t>(unpacked_tensor.DataAsSpan<int8_t>()[0])
                 : is_int32_zp ? static_cast<int32_t>(unpacked_tensor.DataAsSpan<int32_t>()[0])
                               : static_cast<int32_t>(unpacked_tensor.DataAsByteSpan()[0]);

    // per channel quantized handling
    if (!unpacked_tensor.dims().empty() && unpacked_tensor.dims()[0] != 0 && unpacked_tensor.dims()[0] != 1) {
      auto type = unpacked_tensor.data_type();
      if (is_i8_zp) {
        auto zps = unpacked_tensor.DataAsSpan<int8_t>();
        std::vector<int32_t> zps_vec(zps.begin(), zps.end());
        pcq_zps = onnxruntime::make_optional(std::move(zps_vec));
      } else if (is_int32_zp) {
        auto zps = unpacked_tensor.DataAsByteSpan();
        std::vector<int32_t> zps_vec(zps.begin(), zps.end());
        pcq_zps = onnxruntime::make_optional(std::move(zps_vec));
      } else {
        auto zps = unpacked_tensor.DataAsSpan<int32_t>();
        std::vector<int32_t> zps_vec(zps.begin(), zps.end());
        pcq_zps = onnxruntime::make_optional(std::move(zps_vec));
      }
    }
  }
}

static bool IsInternalQuantizedNodeUnit(const NodeUnit& node_unit) {
  // First, ignore QDQ NodeUnit which is not internal quantized node
  if (node_unit.UnitType() == NodeUnit::Type::QDQGroup)
    return false;

  // These operators can use uint8 input without specific QLinear version of it
  // However, the mode has to be internal to the graph/partition (they cannot consume graph inputs)
  static const std::unordered_set<std::string> internal_quantized_op_types = {
      "Transpose",
      "Resize",
      "Concat",
      "MaxPool",
  };

  const auto& node = node_unit.GetNode();
  if (!Contains(internal_quantized_op_types, node.OpType()))
    return false;

  int32_t input_type;
  ORT_ENFORCE(GetType(*node.InputDefs()[0], input_type));

  return input_type == ONNX_NAMESPACE::TensorProto_DataType_UINT8 ||
         input_type == ONNX_NAMESPACE::TensorProto_DataType_INT8;
}

bool GetType(const NodeArg& node_arg, int32_t& type) {
  type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  const auto* type_proto = node_arg.TypeAsProto();
  if (!type_proto || !type_proto->has_tensor_type() || !type_proto->tensor_type().has_elem_type()) {
    LOGS_DEFAULT(WARNING) << "NodeArg [" << node_arg.Name() << "] has no input type";
    return false;
  }

  type = type_proto->tensor_type().elem_type();
  return true;
}
}  // namespace util
}  // namespace npu
}  // namespace vsi
}  // namespace onnxruntime
