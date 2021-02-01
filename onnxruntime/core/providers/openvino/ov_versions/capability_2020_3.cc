// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#if defined OPENVINO_2020_3

#include "core/providers/shared_library/provider_api.h"

#include "../backend_utils.h"
#include "../backend_manager.h"
#include "capabilities.h"
#include "utils.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4245 5208)
#elif __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <ngraph/ngraph.hpp>
#include <ngraph/frontend/onnx_import/onnx.hpp>
#if defined(_MSC_VER)
#pragma warning(default : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic pop
#endif

namespace onnxruntime {
namespace openvino_ep {

bool IsDimensionSupported(const Node* node) {
  auto node_inputs = node->InputDefs();
  size_t input_dims = 0;
  if (node_inputs[0]->Shape() == nullptr) {
    return true;
  } else {
    input_dims = node_inputs[0]->Shape()->dim_size();
    if (node->OpType().find("Pool") != std::string::npos) {
      if (input_dims != 4 && input_dims != 5)
        return false;
    }

    if (node->OpType() == "Unsqueeze") {
      auto& attributes = node->GetAttributes();
      int64_t axes_size = attributes.count("axes") > 0 ? attributes.at("axes").ints().size() : 0;
      if (input_dims + axes_size > 5)
        return false;
    }

    if (node->OpType() == "Softmax") {
      auto& attributes = node->GetAttributes();
      int64_t axis = attributes.count("axis") > 0 ? attributes.at("axis").i() : 0;
      if (input_dims - axis != 1)
        return false;
    }
  }
  return true;
}

//Ops which are not supported by OpenVINO EP
bool IsUnsupportedOp(std::string name, std::string device) {
  std::set<std::string> unsupported_ops_cpu = {
      "Acosh",
      "And",
      "Asinh",
      "Ceil",
      "ConstantOfShape",
      "CumSum",
      "DequantizeLinear",
      "Equal",
      "Exp",
      "Greater",
      "Hardmax",
      "InstanceNormalization",
      "Less",
      "LogSoftmax",
      "LpNormalization",
      "MatMulInteger",
      "MeanVarianceNormalization",
      "Not",
      "Or",
      "QLinearConv",
      "QuantizeLinear",
      "Reciprocal",
      "ReduceL1",
      "ReduceL2",
      "ReduceLogSumExp",
      "Resize",
      "Round",
      "Scan",
      "Shrink",
      "Softplus",
      "Split",
      "Sqrt",
      "ThresholdedRelu",
      "Upsample",
      "Xor",
  };

  std::set<std::string> unsupported_ops_gpu = {
      "Atanh",
      "Cos",
      "Cosh",
      "Mod",
      "ReduceLogSum",
      "ReduceProd"
      "ReduceSumSquare",
      "Sign",
      "SinFloat",
      "Sinh",
      "Softsign",
  };

  std::set<std::string> unsupported_ops_vpu = {
      "Abs",
      "Acos",
      "Acosh",
      "Asin",
      "Asinh",
      "Atan",
      "Atanh",
      "Cos",
      "Cosh",
      "HardSigmoid",
      "Mod",
      "Sign",
      "Sin",
      "Sinh",
      "Softsign",
      "Tan",
  };

  std::set<std::string> unsupported_ops = {};

  if (device == "CPU") {
    unsupported_ops = unsupported_ops_cpu;
  } else if (device == "GPU") {
    std::merge(unsupported_ops_cpu.begin(), unsupported_ops_cpu.end(),
               unsupported_ops_gpu.begin(), unsupported_ops_gpu.end(),
               std::inserter(unsupported_ops, unsupported_ops.begin()));
  } else if (device == "MYRIAD" || device == "HDDL") {
    std::merge(unsupported_ops_cpu.begin(), unsupported_ops_cpu.end(),
               unsupported_ops_vpu.begin(), unsupported_ops_vpu.end(),
               std::inserter(unsupported_ops, unsupported_ops.begin()));
  }

  return unsupported_ops.find(name) != unsupported_ops.end();
}

// Returns true only if op is in a mode that is not currently supported
static bool IsUnsupportedOpMode(const Node* node, const GraphViewer& graph_viewer) {
  const auto& optype = node->OpType();

  const auto& initializers = graph_viewer.GetAllInitializedTensors();

  if (optype == "Reshape") {
    //nGraph Reshape op currently requires shape info available in advance.
    const auto& shape_arg = node->InputDefs()[1];
    //Empty Initializer check
    if (shape_arg->Shape() == nullptr)
      return false;
    if (shape_arg->Shape()->dim_size() == 1 && shape_arg->Shape()->dim(0).dim_value() == 0)
      return true;
    return initializers.find(shape_arg->Name()) == initializers.end();
  } else if (optype == "MaxPool") {
    //MaxPool "indices" output is not currently supported.
    if (node->OutputDefs().size() > 1) {
      return true;
    }

    // ceil_mode and dilations attrs are not supported in nGraph
    const auto& attributes = node->GetAttributes();
    auto ceil_attr = attributes.find("ceil_mode");
    // default value of ceil_mode (0) is supported.
    if (ceil_attr != attributes.end() && ceil_attr->second().i() != 0) {
      return true;
    }

    if (attributes.find("dilations") != attributes.end()) {
      return true;
    }
    if (!IsDimensionSupported(node))
      return true;
  } else if (optype == "Add" || optype == "Sub" || optype == "Mul") {
    for (size_t i = 0; i < node->InputDefs().size(); i++) {
      if (node->InputDefs()[i]->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64) {
        return true;
      }
    }
  } else if (optype == "Div") {
    for (size_t i = 0; i < node->InputDefs().size(); i++) {
      if (node->InputDefs()[i]->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64 ||
          node->InputDefs()[i]->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32) {
        return true;
      }
    }
  } else if (optype == "Abs") {
    for (size_t i = 0; i < node->InputDefs().size(); i++) {
      if (node->InputDefs()[i]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT)
        return true;
    }
  } else if (optype == "Max" || optype == "Min" || optype == "Mean" || optype == "Sum") {
    if (GetInputCount(node, initializers) == 1)
      return true;
    if (optype == "Max" || optype == "Min") {
      for (size_t i = 0; i < node->InputDefs().size(); i++) {
        auto dtype = node->InputDefs()[i]->TypeAsProto()->tensor_type().elem_type();
        if (dtype == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8 ||
            dtype == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16)
          return true;
      }
    }
  } else if (optype == "Clip") {
    //Only float 16, float and double data types are supported
    const bool data_is_float = node->InputDefs()[0]->Type()->find("float") != std::string::npos;
    const bool data_is_float16 = node->InputDefs()[0]->Type()->find("float16") != std::string::npos;
    const bool data_is_double = node->InputDefs()[0]->Type()->find("double") != std::string::npos;
    return !(data_is_float || data_is_float16 || data_is_double);
  } else if (optype == "OneHot") {
    //nGraph OneHot op currently requires depth info available in advance.
    const auto& depth_arg = node->InputDefs()[1];
    return initializers.find(depth_arg->Name()) == initializers.end();
  } else if (optype == "Conv" || optype == "ConvTranspose") {
    if (GetInputCount(node, initializers) > 1)
      return true;
  } else if (optype == "TopK") {
    //TopK opset 10 is currently not supported.
    //K as input is currently not suppported.
    return node->InputDefs().size() > 1;
  } else if (optype == "ReduceMin") {
    //Only FP32, INT32 and U8 data types are supported
    const bool data_is_float = node->InputDefs()[0]->Type()->find("float") != std::string::npos;
    const bool data_is_int32 = node->InputDefs()[0]->Type()->find("int32") != std::string::npos;
    const bool data_is_u8 = node->InputDefs()[0]->Type()->find("uint8") != std::string::npos;
    return !(data_is_float || data_is_int32 || data_is_u8);
  } else if (optype == "MatMul") {
    //All matmuls except float have computation missmatch
    const bool A_is_float = node->InputDefs()[0]->Type()->find("float") != std::string::npos;
    const bool B_is_float = node->InputDefs()[1]->Type()->find("float") != std::string::npos;
    return (A_is_float && B_is_float) ? false : true;

  } else if (optype == "Pow") {
    //Only supported if the data type of both inputs is same
    auto x_data_type = node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    auto y_data_type = node->InputDefs()[1]->TypeAsProto()->tensor_type().elem_type();
    return x_data_type != y_data_type;
  } else if (optype == "PRelu") {
    auto slope = node->InputDefs()[1];

    //PRelu slope has to be an initializer or needs to come from a constant node
    if (initializers.count(slope->Name()))
      return false;
    else {
      for (auto input_node = node->InputNodesBegin(); input_node != node->InputNodesEnd(); ++input_node) {
        if (GetInputCount(graph_viewer.GetNode((*input_node).Index()), initializers) == 0) {
          return false;
        }
      }
    }
    return true;
  } else if (optype == "Softmax") {
    if (!IsDimensionSupported(node))
      return true;
  } else if (optype == "Unsqueeze") {
    if (!IsDimensionSupported(node))
      return true;
  } else if (optype == "Pad") {
    // Pad is only supported only up to opset 10 (in opset 11 more inputs were added)
    if (node->InputDefs().size() > 1) {
      return true;
    }

    //3D pad with negative padding have computation missmatch
    const auto& attributes = node->GetAttributes();
    auto pad_attr = attributes.find("pads");

    //Negative padding is not supported
    if (pad_attr != attributes.end()) {
      auto& ints = pad_attr->second().ints();
      for (int index = 0; index < ints.size(); index++) {
        if (ints.Get(index) < 0)
          return true;
      }
    }

    auto mode_attr = attributes.find("mode");
    if (mode_attr != attributes.end()) {
      const auto mode = mode_attr->second().s();
      static const std::set<std::string> allowed_modes = {"constant", "reflect"};

      return allowed_modes.count(mode) == 0;
    }
  } else if (optype == "Mod") {
    //Only fmod=1 is supported
    auto& attributes = node->GetAttributes();
    auto fmod = attributes.count("fmod") > 0 ? attributes.at("fmod").i() : 0;
    if (fmod != 1)
      return true;
    //Only FP32 data type is allowed
    for (const auto& input : node->InputDefs()) {
      if (input->Type()->find("float") == std::string::npos)
        return true;
    }
  } else if (optype == "Cast") {
    using onnx_dtype = ONNX_NAMESPACE::TensorProto_DataType;
    const auto supportedCasts = std::set<std::pair<onnx_dtype, onnx_dtype>>{
        {onnx_dtype::TensorProto_DataType_UINT8, onnx_dtype::TensorProto_DataType_FLOAT},
        {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_UINT8},
        {onnx_dtype::TensorProto_DataType_INT16, onnx_dtype::TensorProto_DataType_FLOAT},
        {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_INT16},
        {onnx_dtype::TensorProto_DataType_UINT16, onnx_dtype::TensorProto_DataType_FLOAT},
        {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_UINT16},
        {onnx_dtype::TensorProto_DataType_INT32, onnx_dtype::TensorProto_DataType_FLOAT},
        {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_INT32},
        {onnx_dtype::TensorProto_DataType_UINT8, onnx_dtype::TensorProto_DataType_INT32}};
    auto input_data_type = node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    auto output_data_type = node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

    const auto typePair = std::make_pair(static_cast<onnx_dtype>(input_data_type), static_cast<onnx_dtype>(output_data_type));
    const auto match = supportedCasts.find(typePair);
    if (match == supportedCasts.end()) {
      return true;
    } else
      return false;
  } else if (optype == "Squeeze") {
    //Shape can't have empty axes attribute
    const auto& attributes = node->GetAttributes();
    if (attributes.count("axes") == 0)
      return true;
  } else if (optype == "Slice") {
    //Slice in opset 10 is currently not supported.
    //unsupported inputs: starts, ends, axes, steps
    if (node->InputDefs().size() > 1) {
      return true;
    }
    //nGraph does not properly handle the situation where any value of the "starts" attribute
    //is higher than a corresponding value in the "ends"
    const auto& attributes = node->GetAttributes();
    if (attributes.count("starts") == 0 || attributes.count("ends") == 0) {
      return true;
    }

    const auto& starts = attributes.find("starts")->second().ints();
    const auto& ends = attributes.find("ends")->second().ints();
    for (int i = 0; i < starts.size(); ++i) {
      if (starts.Get(i) > ends.Get(i)) {
        return true;
      }
    }
  } else if (optype == "AveragePool") {
    // ceil_mode attribute is not supported in nGraph
    const auto& attributes = node->GetAttributes();
    auto ceil_attr = attributes.find("ceil_mode");
    // default value of ceil_mode (0) is supported.
    if (ceil_attr != attributes.end() && ceil_attr->second().i() != 0) {
      return true;
    }
    if (!IsDimensionSupported(node))
      return true;
  } else if (optype == "QLinearMatMul") {
    const auto& a_zero_point = node->InputDefs()[2];
    const auto& b_zero_point = node->InputDefs()[5];
    const auto& y_zero_point = node->InputDefs()[7];

    bool non_const_zero_point = false;

    // check if any of the zero points is NOT in the initializers list
    non_const_zero_point |= initializers.find(a_zero_point->Name()) == initializers.end();
    non_const_zero_point |= initializers.find(b_zero_point->Name()) == initializers.end();
    non_const_zero_point |= initializers.find(y_zero_point->Name()) == initializers.end();

    // QLinearMatMul is not supported if any of the zero points is a dynamic input
    return non_const_zero_point;
  } else if (optype == "MatMulInteger") {
    // all MatMulInteger zero points need to be constants
    const auto inputs = node->InputDefs();
    if (inputs.size() == 3) {
      const auto& a_zero_point = node->InputDefs()[2];

      // not found in initializers -> not const
      return initializers.find(a_zero_point->Name()) == initializers.end();
    } else if (inputs.size() == 4) {
      const auto& a_zero_point = node->InputDefs()[2];
      const auto& b_zero_point = node->InputDefs()[3];

      // not found in initializers -> not const
      return initializers.find(a_zero_point->Name()) == initializers.end() ||
             initializers.find(b_zero_point->Name()) == initializers.end();
    }  // else -> azp & bzp are 0 by default according to ONNX spec
  } else if (optype == "ConvInteger") {
    // all ConvInteger zero points need to be constants
    const auto inputs = node->InputDefs();
    if (inputs.size() == 3) {
      const auto& x_zero_point = node->InputDefs()[2];

      // not found in initializers -> not const
      return initializers.find(x_zero_point->Name()) == initializers.end();
    } else if (inputs.size() == 4) {
      const auto& x_zero_point = node->InputDefs()[2];
      const auto& w_zero_point = node->InputDefs()[3];

      // not found in initializers -> not const
      return initializers.find(x_zero_point->Name()) == initializers.end() ||
             initializers.find(w_zero_point->Name()) == initializers.end();
    }  // else -> xzp & wzp are 0 by default according to ONNX spec
  } else if (optype == "Expand") {
    // nGraph only supports constant shape input values
    const auto& shape_input = node->InputDefs()[1];
    return !graph_viewer.IsConstantInitializer(shape_input->Name(), true);
  }

  //Op doesn't fall into known any of unsupported modes.
  return false;
}

static bool IsTypeSupported(const NodeArg* node_arg, bool is_initializer, const std::string& device_type) {
  const auto* type_proto = node_arg->TypeAsProto();
  if (!type_proto) {
    return false;
  }

  if (is_initializer) {
    switch (type_proto->tensor_type().elem_type()) {
      case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL:
      case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT:
      case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32:
      case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64:
        return true;
      default:

#ifndef NDEBUG
        if (openvino_ep::backend_utils::IsDebugEnabled()) {
          std::cout << "Initializer Data Type is not supported" << std::endl;
        }
#endif
        return false;
    }
  } else {
    std::set<int> supported_types_cpu = {
        ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT,
        ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32,
        ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16,
        ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8,
        ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8,
    };

    std::set<int> supported_types_gpu = {
        ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT,
        ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32};
    auto dtype = type_proto->tensor_type().elem_type();

    if (device_type == "CPU" || device_type == "MYRIAD" || device_type == "HDDL") {
      if (supported_types_cpu.find(dtype) != supported_types_cpu.end())
        return true;
      else {
#ifndef NDEBUG
        if (openvino_ep::backend_utils::IsDebugEnabled()) {
          std::cout << "I/O data type is not supported" << std::endl;
        }
#endif
        return false;
      }
    } else if (device_type == "GPU") {
      if (supported_types_gpu.find(dtype) != supported_types_gpu.end())
        return true;
      else {
#ifndef NDEBUG
        if (openvino_ep::backend_utils::IsDebugEnabled()) {
          std::cout << "I/O data type is not supported" << std::endl;
        }
#endif
        return false;
      }
    }
    return true;
  }
}

static bool IsNodeSupported(const std::map<std::string, std::set<std::string>>& op_map,
                            const GraphViewer& graph_viewer,
                            const NodeIndex node_idx, std::string& device_type) {
  const auto& node = graph_viewer.GetNode(node_idx);
  const auto& optype = node->OpType();

#ifndef NDEBUG
  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    std::cout << "Node " << optype << std::endl;
  }
#endif

  const auto& domain = node->Domain();

  /*
  0. Check if node is in the unsupported list
  1. Check input and output data types are supported.
  2. Check if there is unsupported dimension in input and output shapes
  3. Check Op is supported
   3a. Check if Op is of known unsupported modes (edge cases). If yes return false right away.
   3b. If above is not true, check if the op is available in nGraph.
  */

  //Check 0
  if (IsUnsupportedOp(optype, device_type)) {
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "Node is in the unsupported list" << std::endl;
    }
#endif
    return false;
  }

  //Check 1
  bool are_types_supported = true;

  node->ForEachDef([&are_types_supported, &graph_viewer, &device_type](const NodeArg& node_arg, bool is_input) {
    bool is_initializer = false;
    if (is_input) {
      if (graph_viewer.IsConstantInitializer(node_arg.Name(), true))
        is_initializer = true;
    }
    are_types_supported &= IsTypeSupported(&node_arg, is_initializer, device_type);
  });

  if (!are_types_supported) {
    return false;
  }

  //Check 2

  bool has_unsupported_dimension = false;
  node->ForEachDef([&has_unsupported_dimension, &graph_viewer, &device_type](const NodeArg& node_arg, bool is_input) {
    if (is_input) {
      if (graph_viewer.IsConstantInitializer(node_arg.Name(), true))
        return;
    }
    auto shape = node_arg.Shape();
    if (shape != nullptr) {
      //Can't have no dimensions
      if (shape->dim_size() == 0) {
        has_unsupported_dimension = true;
        return;
      } else {
        //Zero dimension check
        for (const auto& dim : shape->dim()) {
          if (utils::HasDimValue(dim) && dim.dim_value() == 0) {
            has_unsupported_dimension = true;
            return;
          }
        }
      }
    }
  });
  if (has_unsupported_dimension) {
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "Dimension check failed" << std::endl;
    }
#endif

    return false;
  }

  //Check 3a
  if (domain == kOnnxDomain && IsUnsupportedOpMode(node, graph_viewer)) {
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "Failed in unsupported op mode" << std::endl;
    }
#endif

    return false;
  }

  //Check 3b
  const auto opset = op_map.find(domain);
  if (opset == op_map.end() || opset->second.find(optype) == opset->second.end()) {
    return false;
  } else {
    return true;
  }
}

static std::vector<NodeIndex>
GetUnsupportedNodeIndices(const GraphViewer& graph_viewer, std::string device, /*out*/ std::unordered_set<std::string>& ng_required_initializers) {
  const auto ng_supported_ops = GetNgSupportedOps(GetOnnxOpSet(graph_viewer));

  std::vector<NodeIndex> unsupported_nodes_idx;

  for (const auto& node_idx : graph_viewer.GetNodesInTopologicalOrder()) {
    if (IsNodeSupported(ng_supported_ops, graph_viewer, node_idx, device)) {
      // Collect inputs that are initializers
      graph_viewer.GetNode(node_idx)->ForEachDef([&ng_required_initializers, &graph_viewer](const NodeArg& node_arg, bool is_input) {
              if(is_input && graph_viewer.GetAllInitializedTensors().count(node_arg.Name())) {
                ng_required_initializers.insert(node_arg.Name());
              } }, true);
    } else {
      unsupported_nodes_idx.push_back(node_idx);
    }
  }

  return unsupported_nodes_idx;
}

std::vector<std::unique_ptr<ComputeCapability>>
GetCapability_2020_3(const GraphViewer& graph_viewer, const std::string device_type) {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  if (graph_viewer.IsSubgraph()) {
    return result;
  }

  // Need access to model_path_
  for (const auto& tensor : graph_viewer.GetAllInitializedTensors()) {
    if (tensor.second->has_data_location() && tensor.second->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
      LOGS_DEFAULT(WARNING) << "[OpenVINO-EP] Initializers with external data location are not currently supported";
      return result;
    }
  }

  // This is a list of initializers that nGraph considers as constants. Example weights, reshape shape etc.
  std::unordered_set<std::string> ng_required_initializers;

  const auto unsupported_nodes = GetUnsupportedNodeIndices(graph_viewer, device_type, ng_required_initializers);

  //If all ops are supported, no partitioning is required. Short-circuit and avoid splitting.
  if (unsupported_nodes.empty()) {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;

    //Fill inputs with names
    std::for_each(graph_viewer.GetInputs().begin(), graph_viewer.GetInputs().end(),
                  [&inputs](const NodeArg* node_arg) { inputs.push_back(node_arg->Name()); });

    /* In scenarios, when there are no inputs or all inputs being initializers,
         ConstantFolding optimization in onnxruntime pre-computes the value.*/
    if (inputs.empty()) {
      return result;
    }

    //If subgraph only has Identity node, EyeLike or Dropout, OpenVINO EP doesn't support it.
    const auto& nodes = graph_viewer.GetNodesInTopologicalOrder();
    if (nodes.size() == 1) {
      const auto& node = graph_viewer.GetNode(nodes[0]);
      if (node->OpType() == "Identity" || node->OpType() == "EyeLike" || node->OpType() == "Dropout")
        return result;
    }

    //Initializers need to be part of meta_def->inputs
    std::for_each(ng_required_initializers.begin(), ng_required_initializers.end(),
                  [&inputs](const std::string& initializer) { inputs.push_back(initializer); });

    //Fill outputs with names
    std::for_each(graph_viewer.GetOutputs().begin(), graph_viewer.GetOutputs().end(),
                  [&outputs](const NodeArg* node_arg) { outputs.push_back(node_arg->Name()); });

    // Create and add this graph to result.
    AppendClusterToSubGraph(graph_viewer.GetNodesInTopologicalOrder(), inputs, outputs, result);

    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Model is fully supported by OpenVINO";
    openvino_ep::BackendManager::GetGlobalContext().is_wholly_supported_graph = true;

  } else {  // unsupported_nodes_idx.empty()
    const auto ng_clusters = GetPartitionedClusters(graph_viewer.GetNodesInTopologicalOrder(), unsupported_nodes);

    auto connected_clusters = GetConnectedClusters(graph_viewer, ng_clusters);

    //Myriad plugin can only load 10 subgraphs
    if (device_type == "MYRIAD" && connected_clusters.size() > 10) {
      std::sort(connected_clusters.begin(), connected_clusters.end(),
                [](const std::vector<NodeIndex>& v1, const std::vector<NodeIndex>& v2) -> bool {
                  return v1.size() > v2.size();
                });
    }
    int no_of_clusters = 0;

    for (const auto& this_cluster : connected_clusters) {
      if (device_type == "MYRIAD" && no_of_clusters == 10) {
        break;
      }
      std::vector<std::string> cluster_graph_inputs, cluster_inputs, const_inputs, cluster_outputs;
      //If subgraph only has Identity node, EyeLike or Dropout, OpenVINO EP doesn't support it.
      if (this_cluster.size() == 1) {
        const auto& node = graph_viewer.GetNode(this_cluster[0]);
        if (node->OpType() == "Identity" || node->OpType() == "EyeLike" || node->OpType() == "Dropout" || node->OpType() == "ReduceMin" || node->OpType() == "Concat" || node->OpType() == "Cast")
          continue;
      }
      GetInputsOutputsOfCluster(graph_viewer, this_cluster, ng_required_initializers, cluster_graph_inputs, cluster_inputs, const_inputs, cluster_outputs);

      bool omit_subgraph = false;
      for (auto index : this_cluster) {
        const auto& node = graph_viewer.GetNode(index);
        if (node->OpType() == "Unsqueeze" || node->OpType() == "Gather" || node->OpType() == "Squeeze") {
          for (const auto& input : node->InputDefs()) {
            auto input_name = input->Name();
            auto it = find(cluster_inputs.begin(), cluster_inputs.end(), input_name);
            if (it != cluster_inputs.end()) {
              omit_subgraph = true;
              break;
            }
          }
        }
      }
      if (omit_subgraph)
        continue;

      /* In scenarios, when there are no inputs or all inputs being initializers,
         ConstantFolding optimization in onnxruntime pre-computes the value.*/
      if (!cluster_inputs.empty() && cluster_inputs.size() > const_inputs.size()) {
        AppendClusterToSubGraph(this_cluster, cluster_inputs, cluster_outputs, result);
        no_of_clusters++;
      }
    }
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Supported subgraphs on OpenVINO: " << no_of_clusters;
  }

  return result;
}

}  // namespace openvino_ep
}  // namespace onnxruntime

#endif  //defined OPENVINO_2020_3
