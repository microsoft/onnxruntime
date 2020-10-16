// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#if defined OPENVINO_2021_1

#include "core/framework/compute_capability.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/graph/graph_utils.h"
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
      auto attributes = node->GetAttributes();
      auto axes = attributes["axes"].ints();
      if (input_dims + axes.size() > 5)
        return false;
    }

  }
  return true;
}

//Ops which are not supported by OpenVINO EP
bool IsOpSupported(std::string name, std::string device) {
  std::set<std::string> common_supported_ops = {
      "Add",
      "And",
      "AveragePool",
      "BatchNormalization",
      "Cast",
      "Clip",
      "Concat",
      "Constant",
      "ConstantOfShape",
      "Conv",
      "ConvTranspose",
      "DepthToSpace",
      "Div",
      "Dropout",
      "Elu",
      "Equal",
      "Erf",
      "Exp",
      "Flatten",
      "Floor",
      "Gather",
      "Gemm",
      "GlobalAveragePool",
      "Greater",
      "Identity",
      "InstanceNormalization",
      "LeakyRelu",
      "Less",
      "Log",
      "LRN",
      "LSTM",
      "MatMul",
      "Max",
      "MaxPool",
      "Mean",
      "Min",
      "Mul",
      "Neg",
      "OneHot",
      "Pad",
      "Pow",
      "PRelu",
      "Reciprocal",
      "ReduceMax",
      "ReduceMean",
      "ReduceMin",
      "ReduceSum",
      "Relu",
      "Reshape",
      "Shape",
      "Sigmoid",
      "Slice",
      "Softmax",
      "SpaceToDepth",
      "Split",
      "Sqrt",
      "Squeeze",
      "Sub",
      "Sum",
      "Tanh",
      "TopK",
      "Transpose",
      "Unsqueeze",
  };

  std::set<std::string> supported_ops_cpu = {
    "Abs",
    "Acos",
    "Acosh",
    "ArgMax",
    "ArgMin",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "Cos",
    "Cosh",
    "GlobalLpPool",
    "HardSigmoid",
    "Not",
    "ReduceLogSum",
    "ReduceProd",
    "ReduceSumSquare",
    "Resize",
    "Selu",
    "Sign",
    "Sinh",
    "Softsign",
    "Tan",
    "NonZero",
    "Upsample"
  };


  std::set<std::string> supported_ops_gpu = {
    "Abs",
    "Asin",
    "Asinh",
    "Atan",
    "Ceil",
    "GlobalLpPool",
    "HardSigmoid",
    "Not",
    "Selu",
    "Tan",
  };
  std::set<std::string> supported_ops_vpu = {
    "Expand",
    "NonMaxSuppression",
    "NonZero",
    "ReduceLogSum",
    "ReduceSumSquare",
    "Resize",
    "RoiAlign",
    "Scatter",
    "SinFloat",
  };

  std::set<std::string> supported_ops = {};

  if (device == "CPU") {
    std::merge(common_supported_ops.begin(), common_supported_ops.end(),
               supported_ops_cpu.begin(), supported_ops_cpu.end(),
               std::inserter(supported_ops,supported_ops.begin()));
  } else if (device == "GPU") {
    std::merge(common_supported_ops.begin(), common_supported_ops.end(),
               supported_ops_gpu.begin(), supported_ops_gpu.end(),
               std::inserter(supported_ops, supported_ops.begin()));
  } else if (device == "MYRIAD" || device == "HDDL") {
    std::merge(common_supported_ops.begin(), common_supported_ops.end(),
               supported_ops_vpu.begin(), supported_ops_vpu.end(),
               std::inserter(supported_ops, supported_ops.begin()));
  }

  return supported_ops.find(name) != supported_ops.end();
}

// Returns true only if op is in a mode that is not currently supported
static bool IsUnsupportedOpMode(const Node* node, const onnxruntime::GraphViewer& graph_viewer, const std::string& device_id) {
  const auto& optype = node->OpType();

  const auto& initializers = graph_viewer.GetAllInitializedTensors();

  if (optype == "MaxPool") {
    //MaxPool "indices" output is not currently supported.
    if (node->OutputDefs().size() > 1) {
      return true;
    }

    const auto& attributes = node->GetAttributes();

    const auto ceil_attr = attributes.find("ceil_mode");
    // default value of ceil_mode (0) is supported.
    if (ceil_attr != attributes.end() && ceil_attr->second.i() != 0) {
      return true;
    }

    //auto pad null value is not supported
    const auto auto_attr = attributes.find("auto_pad");
    if (auto_attr->second.s() == "") {
      return true;
    }
    // dilations attrs are not supported in nGraph
    if (attributes.find("dilations") != attributes.end()) {
      return true;
    }
    if (!IsDimensionSupported(node))
      return true;
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
  } else if (optype == "Conv" || optype == "ConvTranspose") {
    if (GetInputCount(node, initializers) > 1)
      return true;
    auto attributes = node->GetAttributes();
    if (attributes["auto_pad"].s() == "") {
      return true;
    }
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
  } else if (optype == "Identity") {
    const auto& input = node->InputDefs()[0];
    const auto& output = node->OutputDefs()[0];
    auto graph_inputs = graph_viewer.GetInputs();
    auto graph_outputs = graph_viewer.GetOutputs();
    auto input_it = find(graph_inputs.begin(), graph_inputs.end(), input);
    auto output_it = find(graph_outputs.begin(), graph_outputs.end(), output);
    if(input_it != graph_inputs.end() && output_it != graph_outputs.end())
      return true;
  } else if (optype == "Resize") {
    //Resize opset 11 is not supported
    if(node->InputDefs().size() > 2)
      return true;
  } else if (optype == "Unsqueeze") {
    if (!IsDimensionSupported(node))
      return true;
  } else if (optype == "Mod") {
    //Only fmod=1 is supported
    auto attributes = node->GetAttributes();
    auto fmod = attributes["fmod"].i();
    if (fmod != 1)
      return true;
    //Only FP32 data type is allowed
    for (const auto& input : node->InputDefs()) {
      if (input->Type()->find("float") == std::string::npos)
        return true;
    }
  } else if (optype == "Squeeze") {
    //Shape can't have empty axes attribute
    const auto& attributes = node->GetAttributes();
    if (attributes.count("axes") == 0)
      return true;
  } else if (optype == "Slice") {
    //start, end, axes need to be a initializer
    const auto &data_arg = node->InputDefs()[0];
    auto graph_inputs = graph_viewer.GetInputs();
    bool cond_for_slice = false;

    auto it = find(graph_inputs.begin(), graph_inputs.end(), data_arg);
    if(it != graph_inputs.end()){
      if(node->InputDefs().size() > 1){
        const auto &start_arg = node->InputDefs()[1];
        const auto &end_arg = node->InputDefs()[2];
        cond_for_slice |= initializers.find(start_arg->Name()) == initializers.end();
        cond_for_slice |= initializers.find(end_arg->Name()) == initializers.end();
      }
      if (node->InputDefs().size() > 3) {
        const auto &axes_arg = node->InputDefs()[3];
        cond_for_slice |= initializers.find(axes_arg->Name()) == initializers.end();
      }
    }

    return cond_for_slice;
  } else if (optype == "AveragePool") {
    // ceil_mode attribute is not supported in nGraph
    const auto& attributes = node->GetAttributes();
    //auto pad null value is not supported
    const auto auto_attr = attributes.find("auto_pad");
    if (auto_attr->second.s() == "") {
      return true;
    }
    const auto ceil_attr = attributes.find("ceil_mode");
    // default value of ceil_mode (0) is supported.
    if (ceil_attr != attributes.end() && ceil_attr->second.i() != 0) {
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
  } else if (optype == "ArgMax" || optype == "ArgMin") {
    //tensor type does not support select last index
    auto attributes = node->GetAttributes();
    auto last_index_arg = attributes["select_last_index"].i();
    if (last_index_arg != 0)
      return true;
    // tensor type supports float as input for argmax and argmin
    auto dtype = node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    if (dtype != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT) {
      return true;
    }
} else if ((optype == "Equal") || (optype == "And"))  {

    using onnx_dtype = ONNX_NAMESPACE::TensorProto_DataType;
    auto supportedOps = std::set<std::vector<onnx_dtype>>{
        {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_FLOAT },
        {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_INT8, onnx_dtype::TensorProto_DataType_FLOAT },
        {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_INT8 },
        {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_UINT8, onnx_dtype::TensorProto_DataType_FLOAT },
        {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_UINT8 },
        {onnx_dtype::TensorProto_DataType_INT8, onnx_dtype::TensorProto_DataType_INT8, onnx_dtype::TensorProto_DataType_INT8 },
        {onnx_dtype::TensorProto_DataType_INT8, onnx_dtype::TensorProto_DataType_INT8, onnx_dtype::TensorProto_DataType_UINT8 },
        {onnx_dtype::TensorProto_DataType_INT8, onnx_dtype::TensorProto_DataType_UINT8, onnx_dtype::TensorProto_DataType_INT8 },
        {onnx_dtype::TensorProto_DataType_INT32, onnx_dtype::TensorProto_DataType_INT32, onnx_dtype::TensorProto_DataType_INT32 },
        {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_UINT8, onnx_dtype::TensorProto_DataType_FLOAT },
        {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_UINT8 }};

    if (optype == "Equal") {
      supportedOps.insert(std::vector<onnx_dtype>{onnx_dtype::TensorProto_DataType_UINT8, onnx_dtype::TensorProto_DataType_INT32, onnx_dtype::TensorProto_DataType_INT32 }),
      supportedOps.insert(std::vector<onnx_dtype>{onnx_dtype::TensorProto_DataType_UINT8, onnx_dtype::TensorProto_DataType_INT64, onnx_dtype::TensorProto_DataType_INT64 });
      supportedOps.insert(std::vector<onnx_dtype>{onnx_dtype::TensorProto_DataType_BOOL, onnx_dtype::TensorProto_DataType_INT64, onnx_dtype::TensorProto_DataType_INT64 }),
      supportedOps.insert(std::vector<onnx_dtype>{onnx_dtype::TensorProto_DataType_UINT8, onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_FLOAT });
    }

    onnx_dtype input_0_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    onnx_dtype input_1_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->InputDefs()[1]->TypeAsProto()->tensor_type().elem_type();
    onnx_dtype output_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

    const std::vector<onnx_dtype> typePair{output_data_type, input_0_data_type, input_1_data_type};
    const auto match = supportedOps.find(typePair);
    if (match == supportedOps.end()) {
      return true;
    } else
      return false;
  } else if(optype == "Gather") {

    if(device_id == "GPU"){
      const auto& input = node->InputDefs()[0];
      auto graph_inputs = graph_viewer.GetInputs();
      auto it = find(graph_inputs.begin(), graph_inputs.end(), input);
      if(it != graph_inputs.end()){
        const auto &indices_arg = node->InputDefs()[1];
        if (indices_arg->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64)
          return true;
      }
    }
  } else if(optype == "Upsample") {

    //check for attributes
    auto upsample_attr = node->GetAttributes();
    auto upsample_arg = upsample_attr["scales"];
    auto float_size = upsample_arg.floats_size();
    if (float_size > 2 && (upsample_arg.floats(0) != 1.f || upsample_arg.floats(1) != 1.f))
      return true;

    //check for input dimensions
    const auto &x_arg = node->InputDefs()[0];

    auto shape = x_arg->Shape();
    if (shape != nullptr) {
      //input tensor rank cannot be of one dimension
      if (shape->dim_size() == 1) {
         return true;
      }
    }
    // x_arg supports only float, int8 and float16 type
    if ((x_arg->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT) ||
        (x_arg->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) ||
        (x_arg->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16)) {
      return false;
    } else {
      return true;
    }
  }
  //Op doesn't fall into known any of unsupported modes.
  return false;
}

static bool IsTypeSupported(const NodeArg* node_arg, bool is_initializer, const std::string& device_id) {
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
        ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL,
        ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT,
        ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32,
        ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16,
        ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8,
        ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8,
        ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64,
    };

    std::set<int> supported_types_gpu = {
        ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT,
        ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32,
        ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64,
    };
    auto dtype = type_proto->tensor_type().elem_type();

    if (device_id == "CPU" || device_id == "MYRIAD" || device_id == "HDDL") {
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
    } else if (device_id == "GPU") {
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
                            const onnxruntime::GraphViewer& graph_viewer,
                            const NodeIndex node_idx, std::string& device_id) {
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
  if (!IsOpSupported(optype, device_id)) {
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "Node is not in the supported ops list" << std::endl;
    }
#endif
    return false;
  }

  //Check 1
  bool are_types_supported = true;

  node->ForEachDef([&are_types_supported, &graph_viewer, &device_id](const onnxruntime::NodeArg& node_arg, bool is_input) {
    bool is_initializer = false;
    if (is_input) {
      if (graph_viewer.IsConstantInitializer(node_arg.Name(), true))
        is_initializer = true;
    }
    are_types_supported &= IsTypeSupported(&node_arg, is_initializer, device_id);
  });

  if (!are_types_supported) {
    return false;
  }

  //Check 2

  bool has_unsupported_dimension = false;
  node->ForEachDef([&has_unsupported_dimension, &graph_viewer, &device_id, &optype](const onnxruntime::NodeArg& node_arg, bool is_input) {
    if (is_input) {
      if (graph_viewer.IsConstantInitializer(node_arg.Name(), true))
        return;
    }
    auto shape = node_arg.Shape();
    if (shape != nullptr) {
      //Can't have no dimensions
      if (shape->dim_size() == 0) {
        if(optype == "Unsqueeze" || optype == "Squeeze" || optype == "Cast" ||
            optype == "Gather" || optype == "Mul" || optype == "Sub" ||
            optype == "Min" || optype == "Div" || optype == "Floor")
          return;
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
  if (domain == kOnnxDomain && IsUnsupportedOpMode(node, graph_viewer, device_id)) {
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
      graph_viewer.GetNode(node_idx)->ForEachDef([&ng_required_initializers, &graph_viewer](const onnxruntime::NodeArg& node_arg, bool is_input) {
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
GetCapability_2021_1(const onnxruntime::GraphViewer& graph_viewer, std::string device_id) {

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

  const auto unsupported_nodes = GetUnsupportedNodeIndices(graph_viewer, device_id, ng_required_initializers);
  #ifndef NDEBUG
    if(openvino_ep::backend_utils::IsDebugEnabled()){
      std::cout << "No of unsupported nodes " << unsupported_nodes.size() << std::endl;
      for(size_t i = 0; i < unsupported_nodes.size(); i++){
        const auto& node = graph_viewer.GetNode(unsupported_nodes[i]);
        std::cout << "Unsupported node op " << node->OpType() << std::endl;
      }
    }
  #endif

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

    const auto& nodes = graph_viewer.GetNodesInTopologicalOrder();
    //Nodes that work well in models but not as a single node
    if (nodes.size() == 1) {
      const auto& node = graph_viewer.GetNode(nodes[0]);
      if(IsOpSupportedOnlyInModel(node->OpType()))
        return result;
      //If reshape is not an intermediate node, shape needs to be an initializer
      if(node->OpType() == "Reshape"){
        const auto& shape_arg = node->InputDefs()[1];
        if(ng_required_initializers.find(shape_arg->Name()) == ng_required_initializers.end())
        return result;
      } else if (node->OpType() == "Expand") {
        const auto& output = node->OutputDefs()[0];
        if (output->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16)
          return result;
      } else if (node->OpType() == "RoiAlign") {
        using onnx_dtype = ONNX_NAMESPACE::TensorProto_DataType;

        onnx_dtype input_0_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
        onnx_dtype input_1_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->InputDefs()[1]->TypeAsProto()->tensor_type().elem_type();
        onnx_dtype input_2_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->InputDefs()[2]->TypeAsProto()->tensor_type().elem_type();
        onnx_dtype output_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

        if ((input_0_data_type != onnx_dtype::TensorProto_DataType_FLOAT16) ||
            (input_1_data_type != onnx_dtype::TensorProto_DataType_FLOAT16) ||
            (input_2_data_type != onnx_dtype::TensorProto_DataType_FLOAT) ||
            (output_data_type != onnx_dtype::TensorProto_DataType_FLOAT16))
          return result;
      } else if ((node->OpType() == "Greater") || (node->OpType() == "Less")) {

        if (device_id == "MYRIAD") {

          auto input_0_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
          auto input_1_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->InputDefs()[1]->TypeAsProto()->tensor_type().elem_type();
          auto output_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

          if (!((output_data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT) ||
            (output_data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16))) {
            return result;
          }

          if ((input_0_data_type != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16) ||
            (input_1_data_type != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16)) {
            return result;
          }
        }
      }
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

    std::vector<NodeIndex> modified_unsupported_nodes;
    for (const auto& node_idx : graph_viewer.GetNodesInTopologicalOrder()) {
      if(find(unsupported_nodes.begin(), unsupported_nodes.end(), node_idx) != unsupported_nodes.end()){
        modified_unsupported_nodes.push_back(node_idx);
      }
      else{
        const auto& node = graph_viewer.GetNode(node_idx);
        const auto& optype = node->OpType();
        if(optype == "TopK" || optype == "NonZero"){
          modified_unsupported_nodes.push_back(node_idx);
        }
        if(optype == "Gather"){
          if(device_id == "MYRIAD"){
            auto input_data_type = node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
            if(input_data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8){
              modified_unsupported_nodes.push_back(node_idx);
            }
          }
        }
      }
    }
    auto ng_clusters = GetPartitionedClusters(graph_viewer.GetNodesInTopologicalOrder(), modified_unsupported_nodes);

    auto connected_clusters = GetConnectedClusters(graph_viewer, ng_clusters);

    //Myriad plugin can only load 10 subgraphs
    if (device_id == "MYRIAD" && connected_clusters.size() > 10) {
      std::sort(connected_clusters.begin(), connected_clusters.end(),
                [](const std::vector<NodeIndex>& v1, const std::vector<NodeIndex>& v2) -> bool {
                  return v1.size() > v2.size();
                });
    }
    int no_of_clusters = 0;

    for (auto this_cluster : connected_clusters) {
      if (device_id == "MYRIAD" && no_of_clusters == 10) {
        break;
      }
      std::vector<std::string> cluster_graph_inputs, cluster_inputs, const_inputs, cluster_outputs;
      //If subgraph has less then three, graph is considered trivial
      if (this_cluster.size() < 3) {
        continue;
      }
      GetInputsOutputsOfCluster(graph_viewer, this_cluster, ng_required_initializers, cluster_graph_inputs, cluster_inputs, const_inputs, cluster_outputs);

      bool omit_subgraph = false;
      std::map<std::string, int> slice_map;
      //Omitting zero dim subgraphs
      for (auto index : this_cluster) {
        const auto& node = graph_viewer.GetNode(index);
        const auto& optype = node->OpType();
        if (optype == "Mul" || optype == "Transpose" || optype == "Unsqueeze" ||
            optype == "Cast" || optype == "Concat" || optype == "Gather" ||
            optype == "Div" || optype == "Sub" || optype == "Identity") {

            if(optype == "Identity" && device_id != "CPU")
              continue;

            if((optype == "Div" || optype == "Sub") && (device_id != "MYRIAD" &&  device_id != "GPU"))
              continue;
            for (const auto& input : node->InputDefs()) {
              auto input_name = input->Name();
              auto it = find(cluster_graph_inputs.begin(), cluster_graph_inputs.end(), input_name);
              if (it != cluster_graph_inputs.end()) {
                  omit_subgraph = true;
                  break;
              }
            }
        }

        if(optype == "Conv" || optype == "Identity"){
          auto output_name = node->OutputDefs()[0]->Name();
          auto it = find(cluster_outputs.begin(), cluster_outputs.end(), output_name);
          if(it != cluster_outputs.end() && node->GetOutputEdgesCount() != 0){
            omit_subgraph = true;
            break;
          }
        }

        if(optype == "Slice"){
          auto input = node->InputDefs()[0];
          auto input_name = input->Name();
          const bool is_data_int32 = input->Type()->find("int32") != std::string::npos;
          auto it = find(cluster_graph_inputs.begin(), cluster_graph_inputs.end(), input_name);
          if(it != cluster_graph_inputs.end()){
            if(device_id == "MYRIAD" && is_data_int32){
              omit_subgraph = true;
              break;
            }
            if(slice_map.count(input_name) == 0){
              slice_map[input_name] = 1;
            }
            else{
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
      if (!cluster_inputs.empty()){
        AppendClusterToSubGraph(this_cluster, cluster_inputs, cluster_outputs, result);
        no_of_clusters++;
      }
    }
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Supported subgraphs on OpenVINO: " << no_of_clusters;
  }

  return result;
}

} // namespace onnxruntime
} // namespace openvino_ep

#endif //defined OPENVINO_2021_1
