// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include <unordered_set>
#include "core/providers/shared_library/provider_api.h"
#include "../backend_utils.h"
#include "../backend_manager.h"
#include<string>
#include<vector>
#include "data_ops.h"
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

std::set<std::string> ops_supported_only_in_model = {
      "Cast",
      "Concat",
      "ConstantOfShape",
      "Dropout",
      "Expand",
      "EyeLike",
      "Exp",
      "GatherND",
      "Identity",
      "NonMaxSuppression",
      "NonZero",
      "Not",
      "OneHot",
      "Pad",
      "Range",
      "ReduceMin",
      "Resize",
      "Round",
      "Shape",
      "Split",
      "TopK"
}; 

std::vector<supportedop_t> supported_op_mode = {
    {VERSION_2020_4,{"CPU", "GPU"},"Abs"},
    {VERSION_2020_4,{"CPU"},"Acos"},
    {VERSION_2020_4,{"CPU"},"Acosh"},
    {VERSION_2020_4,{"All"},"Add"},
    {VERSION_2020_4,{"All"},"And"},
    {VERSION_2020_4,{"CPU"},"ArgMax"},
    {VERSION_2021_1,{"All"},"ArgMax"},
    {VERSION_2020_4,{"CPU"},"ArgMin"},
    {VERSION_2021_2,{"CPU","MYRIAD"},"ArgMin"},
    {VERSION_2020_4,{"CPU", "GPU"},"Asin"},
    {VERSION_2020_4,{"CPU", "GPU"},"Asinh"},
    {VERSION_2020_4,{"CPU", "GPU"},"Atan"},
    {VERSION_2020_4,{"CPU"},"Atanh"},
    {VERSION_2020_4,{"All"},"AveragePool"},
    {VERSION_2020_4,{"All"},"BatchNormalization"},
    {VERSION_2020_4,{"All"},"Cast"},
    {VERSION_2020_4,{"GPU"},"Ceil"},
    {VERSION_2021_2,{"GPU","MYRIAD"},"Ceil"},
    {VERSION_2020_4,{"All"},"Clip"},
    {VERSION_2020_4,{"All"},"Concat"},
    {VERSION_2020_4,{"All"},"Constant"},
    {VERSION_2020_4,{"All"},"ConstantOfShape"},
    {VERSION_2020_4,{"All"},"Conv"},
    {VERSION_2020_4,{"All"},"ConvTranspose"},
    {VERSION_2020_4,{"CPU"},"Cos"},
    {VERSION_2020_4,{"CPU"},"Cosh"},
    {VERSION_2020_4,{"All"},"DepthToSpace"},
    {VERSION_2020_4,{"All"},"Div"},
    {VERSION_2020_4,{"All"},"Dropout"},
    {VERSION_2020_4,{"All"},"Elu"},
    {VERSION_2020_4,{"All"},"Equal"},
    {VERSION_2020_4,{"All"},"Erf"},
    {VERSION_2020_4,{"All"},"Exp"},
    {VERSION_2021_1,{"MYRIAD"},"Expand"},
    {VERSION_2020_4,{"All"},"Flatten"},
    {VERSION_2020_4,{"All"},"Floor"},
    {VERSION_2020_4,{"All"},"Gather"},
    {VERSION_2021_2,{"MYRIAD"},"GatherND"},
    {VERSION_2020_4,{"All"},"Gemm"},
    {VERSION_2020_4,{"All"},"GlobalAveragePool"},
    {VERSION_2020_4,{"CPU", "GPU"},"GlobalLpPool"},
    {VERSION_2020_4,{"All"},"Greater"},
    {VERSION_2020_4,{"All"},"Identity"},
    {VERSION_2020_4,{"All"},"InstanceNormalization"},
    {VERSION_2020_4,{"CPU", "GPU"},"HardSigmoid"},
    {VERSION_2020_4,{"All"},"LeakyRelu"},
    {VERSION_2020_4,{"All"},"Less"},
    {VERSION_2020_4,{"All"},"Log"},
    {VERSION_2021_2,{"MYRIAD"},"Loop"},
    {VERSION_2020_4,{"All"},"LRN"},
    {VERSION_2020_4,{"All"},"LSTM"},
    {VERSION_2020_4,{"All"},"MatMul"},
    {VERSION_2020_4,{"All"},"Max"},
    {VERSION_2020_4,{"All"},"MaxPool"},
    {VERSION_2020_4,{"All"},"Mean"},
    {VERSION_2020_4,{"All"},"Min"},
    {VERSION_2020_4,{"All"},"Mul"},
    {VERSION_2020_4,{"All"},"Neg"},
    {VERSION_2021_1,{"All"},"NonMaxSuppression"},
    {VERSION_2021_1,{"CPU", "MYRIAD"},"NonZero"},
    {VERSION_2021_1,{"All"},"Not"},
    {VERSION_2020_4,{"CPU", "GPU"},"Not"},
    {VERSION_2020_4,{"All"},"OneHot"},
    {VERSION_2020_4,{"All"},"Pad"},
    {VERSION_2020_4,{"All"},"Pow"},
    {VERSION_2020_4,{"All"},"PRelu"},
    {VERSION_2021_2,{"MYRIAD"},"Range"},
    {VERSION_2020_4,{"All"},"Reciprocal"},
    {VERSION_2020_4,{"CPU", "MYRIAD"},"ReduceLogSum"},
    {VERSION_2020_4,{"All"},"ReduceMax"},
    {VERSION_2020_4,{"All"},"ReduceMean"},
    {VERSION_2020_4,{"All"},"ReduceMin"},
    {VERSION_2020_4,{"CPU"},"ReduceProd"},
    {VERSION_2020_4,{"All"},"ReduceSum"},
    {VERSION_2020_4,{"CPU", "MYRIAD"},"ReduceSumSquare"},
    {VERSION_2020_4,{"All"},"Relu"},
    {VERSION_2020_4,{"CPU"},"Resize"},
    {VERSION_2020_4,{"All"},"Reshape"},
    {VERSION_2021_1,{"All"},"RoiAlign"},
    {VERSION_2021_2,{"MYRIAD"},"Round"},
    {VERSION_2021_1,{"MYRIAD"},"Scatter"},
    {VERSION_2021_2,{"MYRIAD"},"ScatterElements"},
    {VERSION_2020_4,{"CPU", "GPU"},"Selu"},
    {VERSION_2020_4,{"All"},"Shape"},
    {VERSION_2020_4,{"All"},"Sigmoid"},
    {VERSION_2020_4,{"CPU"},"Sign"},
    {VERSION_2020_4,{"CPU"},"Sign"},
    {VERSION_2020_4,{"CPU"},"Sinh"},
    {VERSION_2020_4,{"MYRIAD"},"SinFloat"},
    {VERSION_2020_4,{"All"},"Slice"},
    {VERSION_2020_4,{"All"},"Softmax"},
    {VERSION_2020_4,{"All"},"SpaceToDepth"},
    {VERSION_2020_4,{"All"},"Split"},
    {VERSION_2020_4,{"All"},"Sqrt"},
    {VERSION_2020_4,{"All"},"Squeeze"},
    {VERSION_2020_4,{"CPU"},"Softsign"},
    {VERSION_2020_4,{"All"},"Sub"},
    {VERSION_2020_4,{"All"},"Sum"},
    {VERSION_2020_4,{"CPU", "GPU"},"Tan"},
    {VERSION_2020_4,{"All"},"Tanh"},
    {VERSION_2021_2,{"MYRIAD"},"Tile"},
    {VERSION_2020_4,{"All"},"Transpose"},
    {VERSION_2020_4,{"All"},"TopK"},
    {VERSION_2020_4,{"All"},"Unsqueeze"},
    {VERSION_2021_1,{"CPU", "GPU"},"Upsample"},
    {VERSION_2021_2,{"MYRIAD"},"Where"},
    
};

void Capability::populate_types_supported() {

    std::set<int> supported_types_initializer = {
      ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL,
      ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT,
      ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32,
      ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64,
    };
    
    std::set<int> supported_types_vpu = {
      ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL,
      ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT,
      ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32,
      ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16,
      ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8,
      ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8,
      ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64,
    };

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
}

void Capability::populate_op_mode_supported() {

  no_dimension_supported.push_back({VERSION_2020_4,{"All"},"Unsqueeze"});
  no_dimension_supported.push_back({VERSION_2020_4,{"All"},"Squeeze"});
  no_dimension_supported.push_back({VERSION_2020_4,{"All"},"Cast"});
  no_dimension_supported.push_back({VERSION_2020_4,{"All"},"Gather"});
  no_dimension_supported.push_back({VERSION_2020_4,{"All"},"Mul"});
  no_dimension_supported.push_back({VERSION_2020_4,{"All"},"Sub"});
  no_dimension_supported.push_back({VERSION_2020_4,{"All"},"Min"});
  no_dimension_supported.push_back({VERSION_2020_4,{"All"},"Div"});
  no_dimension_supported.push_back({VERSION_2020_4,{"All"},"Floor"});
  no_dimension_supported.push_back({VERSION_2021_2,{"All"},"Where"});
  no_dimension_supported.push_back({VERSION_2021_2,{"All"},"Range"});
  no_dimension_supported.push_back({VERSION_2021_2,{"Myriad"},"ArgMin"});
  no_dimension_supported.push_back({VERSION_2021_2,{"Myriad"},"Max"});
  no_dimension_supported.push_back({VERSION_2021_2,{"Myriad"},"Add"});
  no_dimension_supported.push_back({VERSION_2021_2,{"Myriad"},"Less"});
  no_dimension_supported.push_back({VERSION_2021_2,{"Myriad"},"Greater"});
  no_dimension_supported.push_back({VERSION_2021_2,{"Myriad"},"Clip"});
  no_dimension_supported.push_back({VERSION_2021_2,{"Myriad"},"Resize"});
  no_dimension_supported.push_back({VERSION_2021_2,{"Myriad"},"Equal"});


  subgraph_supported.push_back({VERSION_2020_4,{"All"},"Mul"});
  subgraph_supported.push_back({VERSION_2020_4,{"All"},"Transpose"});
  subgraph_supported.push_back({VERSION_2020_4,{"All"},"Unsqueeze"});
  subgraph_supported.push_back({VERSION_2020_4,{"All"},"Cast"});
  subgraph_supported.push_back({VERSION_2020_4,{"All"},"Concat"});
  subgraph_supported.push_back({VERSION_2020_4,{"All"},"Gather"});
  subgraph_supported.push_back({VERSION_2020_4,{"Myriad"},"Div"});
  subgraph_supported.push_back({VERSION_2020_4,{"Myriad"},"Sub"});
  subgraph_supported.push_back({VERSION_2021_1,{"CPU"},"Identity"});
  subgraph_supported.push_back({VERSION_2021_1,{"CPU"},"Div"});
  subgraph_supported.push_back({VERSION_2021_1,{"CPU"},"Sub"});
    
  
  //populate unsupportedmode_t
  {
    unsupportedopmode_t obj = {VERSION_2020_4, {"All"}, 
     [this](const Node* node, const Provider_InitializedTensorSet&) {
      if (node->OutputDefs().size() > 1)
        return true;
      const auto& attributes = node->GetAttributes();
      auto ceil_attr = attributes.find("ceil_mode");
      if (ceil_attr != attributes.end() && ceil_attr->second().i() != 0)
        return true;
      auto auto_attr = attributes.find("auto_pad");
      if (auto_attr->second().s() == "") 
        return true;
      if (attributes.find("dilations") != attributes.end()) 
        return true;
      return(this->check_if_dimension_unsupported(node));
    }
  };
    _confirmation_map.insert({"MaxPool", obj});
  } 
  {
    unsupportedopmode_t obj = {VERSION_2020_4, {"All"}, 
     [this](const Node* node, const Provider_InitializedTensorSet&) {
      for (size_t i = 0; i < node->InputDefs().size(); i++) {
        if (node->InputDefs()[i]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT)
          return true;
      }
      return false;
    }
    };
    _confirmation_map.insert({"Abs", obj});
  } 
  {
    unsupportedopmode_t obj = {VERSION_2020_4, {"All"}, 
     [this](const Node* node, const Provider_InitializedTensorSet& initializers) {
        if (GetInputCount(node, initializers) == 1)
          return true;
        return false;  
     }
    };
    _confirmation_map.insert({"Max", obj});
    _confirmation_map.insert({"Min", obj});
    _confirmation_map.insert({"Mean", obj});
    _confirmation_map.insert({"Sum", obj});
  }
  {
    unsupportedopmode_t obj = {VERSION_2020_4, {"All"}, 
     [this](const Node* node, const Provider_InitializedTensorSet&) {
        for (size_t i = 0; i < node->InputDefs().size(); i++) {
        auto dtype = node->InputDefs()[i]->TypeAsProto()->tensor_type().elem_type();
        if (dtype == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8 ||
            dtype == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16)
          return true;
      }
      return false;
     }
    };
    _confirmation_map.insert({"Max", obj});
    _confirmation_map.insert({"Min", obj});
  } 
  {
    unsupportedopmode_t obj = {VERSION_2020_4, {"All"}, 
     [this](const Node* node, const Provider_InitializedTensorSet&) {
      const bool data_is_float = node->InputDefs()[0]->Type()->find("float") != std::string::npos;
      const bool data_is_float16 = node->InputDefs()[0]->Type()->find("float16") != std::string::npos;
      const bool data_is_double = node->InputDefs()[0]->Type()->find("double") != std::string::npos;
      return !(data_is_float || data_is_float16 || data_is_double);
     }
    };
    _confirmation_map.insert({"Clip", obj});
  }
  {  
    unsupportedopmode_t obj = {VERSION_2020_4, {"All"},
     [this](const Node* node, const Provider_InitializedTensorSet& initializers) {
      if (GetInputCount(node, initializers) > 1)
        return true;
      return false;
     } };
    _confirmation_map.insert({"Conv", obj});
    _confirmation_map.insert({"ConvTranspose", obj});
  }
  {
     unsupportedopmode_t obj = {VERSION_2020_4, {"All"}, 
     [this](const Node* node, const Provider_InitializedTensorSet&) {
        return node->InputDefs().size() > 1;
     }
    };
    _confirmation_map.insert({"TopK", obj});
  }
  {
      unsupportedopmode_t obj = {VERSION_2020_4, {"All"}, 
      [this](const Node* node, const Provider_InitializedTensorSet&) {
        const bool data_is_float = node->InputDefs()[0]->Type()->find("float") != std::string::npos;
        const bool data_is_int32 = node->InputDefs()[0]->Type()->find("int32") != std::string::npos;
        const bool data_is_u8 = node->InputDefs()[0]->Type()->find("uint8") != std::string::npos;
        return !(data_is_float || data_is_int32 || data_is_u8);
      }
      };
      _confirmation_map.insert({"ReduceMin", obj});
  }
  {
      unsupportedopmode_t obj = {VERSION_2020_4, {"All"},
      [this](const Node* node, const Provider_InitializedTensorSet&) {
        const bool A_is_float = node->InputDefs()[0]->Type()->find("float") != std::string::npos;
        const bool B_is_float = node->InputDefs()[1]->Type()->find("float") != std::string::npos;
        return (A_is_float && B_is_float) ? false : true;
      }
      };
      _confirmation_map.insert({"MatMul", obj});
  }
  {
    unsupportedopmode_t obj = {VERSION_2020_4, {"All"},
      [this](const Node* node, const Provider_InitializedTensorSet&) {
        auto x_data_type = node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
        auto y_data_type = node->InputDefs()[1]->TypeAsProto()->tensor_type().elem_type();
        return x_data_type != y_data_type;
      }
      };
    _confirmation_map.insert({"Pow", obj});
  }
  {
    unsupportedopmode_t obj = {VERSION_2020_4, {"All"}, 
      [this](const Node* node, const Provider_InitializedTensorSet& initializers) {
        auto slope = node->InputDefs()[1];
        //PRelu slope has to be an initializer or needs to come from a constant node
        if (initializers.count(slope->Name()))
          return false;
        else {
          for (auto input_node = node->InputNodesBegin(); input_node != node->InputNodesEnd(); ++input_node) {
            if (GetInputCount(this->graph_viewer.GetNode((*input_node).Index()), initializers) == 0) 
             return false;
          } }
        return true;
      }
    };
    _confirmation_map.insert({"PRelu", obj});
  }
  {
    unsupportedopmode_t obj = {VERSION_2020_4, {"All"},
      [this](const Node* node, const Provider_InitializedTensorSet&) {
        const auto& input = node->InputDefs()[0];
        const auto& output = node->OutputDefs()[0];
        auto graph_inputs = this->graph_viewer.GetInputs();
        auto graph_outputs = this->graph_viewer.GetOutputs();
        auto input_it = find(graph_inputs.begin(), graph_inputs.end(), input);
        auto output_it = find(graph_outputs.begin(), graph_outputs.end(), output);
        if (input_it != graph_inputs.end() && output_it != graph_outputs.end())
          return true;
        return false;  
      }
    };
    _confirmation_map.insert({"Identity", obj});
  }
  {
    unsupportedopmode_t obj = {VERSION_2020_4, {"All"}, 
      [this](const Node* node, const Provider_InitializedTensorSet&) {
        if (node->InputDefs().size() > 2)
          return true;
        return false;
      }  
    };
    _confirmation_map.insert({"Resize", obj});
  }
  {
    unsupportedopmode_t obj = {VERSION_2020_4, {"All"}, 
      [this](const Node* node, const Provider_InitializedTensorSet&) {
        return this->check_if_dimension_unsupported(node);
      }
    };
    _confirmation_map.insert({"Unsqueeze", obj});
  }
  {
    unsupportedopmode_t obj = {VERSION_2020_4, {"All"}, 
      [this](const Node* node, const Provider_InitializedTensorSet&) {
        auto& attributes = node->GetAttributes();
        auto fmod = attributes.count("fmod") > 0 ? attributes.at("fmod").i() : 0;
        if (fmod != 1) return true;
        for (const auto& input : node->InputDefs()) {
          if (input->Type()->find("float") == std::string::npos)
            return true;
        }
        return false;
      }
    };
    _confirmation_map.insert({"Mod", obj});
  }
  {
    unsupportedopmode_t obj = {VERSION_2020_4, {"All"}, 
      [this](const Node* node, const Provider_InitializedTensorSet&) {
        const auto& attributes = node->GetAttributes();
        if (attributes.count("axes") == 0)
          return true;
        return false;
      }
    };
    _confirmation_map.insert({"Squeeze", obj});
  }
  {
    unsupportedopmode_t obj = {VERSION_2020_4, {"All"}, 
      [this](const Node* node, const Provider_InitializedTensorSet& initializers) {
        bool cond_for_slice = false;
        if (node->InputDefs().size() > 1) {
          const auto& start_arg = node->InputDefs()[1];
          const auto& end_arg = node->InputDefs()[2];
          cond_for_slice |= initializers.find(start_arg->Name()) == initializers.end();
          cond_for_slice |= initializers.find(end_arg->Name()) == initializers.end();
        }
        if (node->InputDefs().size() > 3) {
          const auto& axes_arg = node->InputDefs()[3];
          cond_for_slice |= initializers.find(axes_arg->Name()) == initializers.end();
        }
        return cond_for_slice;
      }
    };
    _confirmation_map.insert({"Slice", obj});
  }
  {
    unsupportedopmode_t obj = {VERSION_2020_4, {"All"}, 
      [this](const Node* node, const Provider_InitializedTensorSet&) {
        const auto& attributes = node->GetAttributes();
        auto ceil_attr = attributes.find("ceil_mode");
        if (ceil_attr != attributes.end() && ceil_attr->second().i() != 0) return true;
        return check_if_dimension_unsupported(node);
      }
    };
    _confirmation_map.insert({"AveragePool", obj});
  }   
  {
     unsupportedopmode_t obj = {VERSION_2020_4, {"All"},
      [this](const Node* node, const Provider_InitializedTensorSet& initializers) {
        bool non_const_zero_point = false;
        // check if any of the zero points is NOT in the initializers list
        non_const_zero_point |= initializers.find(node->InputDefs()[2]->Name()) == initializers.end();
        non_const_zero_point |= initializers.find(node->InputDefs()[5]->Name()) == initializers.end();
        non_const_zero_point |= initializers.find(node->InputDefs()[7]->Name()) == initializers.end();
        // QLinearMatMul is not supported if any of the zero points is a dynamic input
        return non_const_zero_point;
      }
    };
    _confirmation_map.insert({"QLinearMatMul", obj});
  }
  {
    unsupportedopmode_t obj = {VERSION_2020_4, {"All"},
      [this](const Node* node, const Provider_InitializedTensorSet& initializers) {
        if (node->InputDefs().size() == 3) {
          return initializers.find(node->InputDefs()[2]->Name()) == initializers.end();
        } else if (node->InputDefs().size() == 4) {
          return ((initializers.find(node->InputDefs()[2]->Name()) == initializers.end()) ||
                  (initializers.find(node->InputDefs()[2]->Name()) == initializers.end())) ;
        }
        return false;
      }
    };
    _confirmation_map.insert({"MatMulInteger", obj});
  }
  {
     unsupportedopmode_t obj = {VERSION_2020_4, {"All"},
      [this](const Node* node, const Provider_InitializedTensorSet& initializers) {
        if (node->InputDefs().size() == 3) {
          return (initializers.find(node->InputDefs()[2]->Name()) == initializers.end());
        } else if (node->InputDefs().size() == 4) {
          return initializers.find(node->InputDefs()[2]->Name()) == initializers.end() ||
             initializers.find(node->InputDefs()[3]->Name()) == initializers.end();
        } 
        return false;
      }
    };
  _confirmation_map.insert({"ConvInteger", obj});
  }
  {
     unsupportedopmode_t obj = {VERSION_2020_4, {"All"},
      [this](const Node* node, const Provider_InitializedTensorSet&) {
        auto& attributes = node->GetAttributes();
        auto last_index_arg = attributes.count("select_last_index") > 0 ? attributes.at("select_last_index").i() : 0;
        if (last_index_arg != 0)
          return true;
        if (node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT)
          return true;
        return false;  
      }
    };
  _confirmation_map.insert({"ArgMax", obj});
  _confirmation_map.insert({"ArgMin", obj});
  }
  {
      unsupportedopmode_t obj = {VERSION_2020_4, {"All"}, 
      [this](const Node* node, const Provider_InitializedTensorSet&) {
        using onnx_dtype = ONNX_NAMESPACE::TensorProto_DataType;
      auto supportedOps = std::set<std::vector<onnx_dtype>>{
        {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_FLOAT},
        {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_INT8, onnx_dtype::TensorProto_DataType_FLOAT},
        {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_INT8},
        {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_UINT8, onnx_dtype::TensorProto_DataType_FLOAT},
        {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_UINT8},
        {onnx_dtype::TensorProto_DataType_INT8, onnx_dtype::TensorProto_DataType_INT8, onnx_dtype::TensorProto_DataType_INT8},
        {onnx_dtype::TensorProto_DataType_INT8, onnx_dtype::TensorProto_DataType_INT8, onnx_dtype::TensorProto_DataType_UINT8},
        {onnx_dtype::TensorProto_DataType_INT8, onnx_dtype::TensorProto_DataType_UINT8, onnx_dtype::TensorProto_DataType_INT8},
        {onnx_dtype::TensorProto_DataType_INT32, onnx_dtype::TensorProto_DataType_INT32, onnx_dtype::TensorProto_DataType_INT32},
        {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_UINT8, onnx_dtype::TensorProto_DataType_FLOAT},
        {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_UINT8}};

        if (node->OpType() == "Equal") {
          supportedOps.insert(std::vector<onnx_dtype>{onnx_dtype::TensorProto_DataType_UINT8, onnx_dtype::TensorProto_DataType_INT32, onnx_dtype::TensorProto_DataType_INT32}),
          supportedOps.insert(std::vector<onnx_dtype>{onnx_dtype::TensorProto_DataType_UINT8, onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_FLOAT});
        }

        onnx_dtype input_0_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
        onnx_dtype input_1_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->InputDefs()[1]->TypeAsProto()->tensor_type().elem_type();
        onnx_dtype output_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
        const std::vector<onnx_dtype> typePair{output_data_type, input_0_data_type, input_1_data_type};
        const auto match = supportedOps.find(typePair); 
        if (match == supportedOps.end()) 
          return true;
        else
        return false;
      }
    };
  _confirmation_map.insert({"Equal", obj});
  _confirmation_map.insert({"And", obj});
  }

}

bool Capability::check_if_op_is_supported(std::string name, std::vector<supportedop_t>& op_list) {
  for (size_t i=0; i < sizeof(op_list); i++) {
    if (op_list[i].optype == name) {
      if (op_list[i].device_type[0] == "All") {
        if (version_id <= op_list[i].version )
          return true;           
      } else {
        auto it = std::find (op_list[i].device_type.begin(), op_list[i].device_type.end(), device_id); 
        if (it != op_list[i].device_type.end()) {
           if (version_id <= op_list[i].version )
          return true;  
        }
      }
    }
  }
  return false;
}

bool Capability::check_if_type_is_supported(const NodeArg* node_arg, bool is_initializer) {
  
  const auto* type_proto = node_arg->TypeAsProto();
  if (!type_proto) {
    return false;
  }

  if (is_initializer) {
    auto dtype = type_proto->tensor_type().elem_type();
    if (supported_types_initializer.find(dtype) != supported_types_initializer.end())
      return true;
    else {
    
#ifndef NDEBUG
        if (openvino_ep::backend_utils::IsDebugEnabled()) {
          std::cout << "Initializer Data Type is not supported" << std::endl;
        }
#endif
        return false;
    }
  } else {
    
    auto dtype = type_proto->tensor_type().elem_type();

    if (device_id == "MYRIAD" || device_id == "HDDL" || device_id.find("HETERO") != std::string::npos || device_id.find("MULTI") != std::string::npos) {
      if (supported_types_vpu.find(dtype) != supported_types_vpu.end())
        return true;
      else {
#ifndef NDEBUG
        if (openvino_ep::backend_utils::IsDebugEnabled()) {
          std::cout << "I/O data type is not supported" << std::endl;
        }
#endif
        return false;
      }
    } else if (device_id == "CPU") {
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
      auto prec_str = openvino_ep::BackendManager::GetGlobalContext().precision_str;
      if (prec_str == "FP32" && dtype == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16)
        return false;
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

bool Capability::check_if_unsupported_op_mode(const Node* node) {
  bool result = false;
  const auto& optype = node->OpType();
  const auto& initializers = graph_viewer.GetAllInitializedTensors();

  auto iter = _confirmation_map.equal_range(optype);
  for (auto it=iter.first; it!=iter.second; ++it) {
    auto ob = it->second;
    if ((ob.dev == "All")||(ob.dev == device_id)) {
      if (ob.ver <= version_id) {
          ob.func(node, initializers);
          return result;
      }
    }
  }
  return result;
}

bool Capability::check_if_dimension_unsupported(const Node* node) {
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
  }
  return true;
}

bool Capability::check_if_node_is_supported(const std::map<std::string, std::set<std::string>>& op_map,
                                            const NodeIndex node_idx) {
  
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
  if (!check_if_op_is_supported(optype, supported_op_mode)) {
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "Node is not in the supported ops list" << std::endl;
    }
#endif
    return false;
  }

  //Check 1
  bool are_types_supported = true;

  node->ForEachDef([this, &are_types_supported](const NodeArg& node_arg, bool is_input) {
    bool is_initializer = false;
    if (is_input) {
      if (this->graph_viewer.IsConstantInitializer(node_arg.Name(), true))
        is_initializer = true;
    }
    bool type_is_supported = check_if_type_is_supported(&node_arg, is_initializer);
    are_types_supported &= type_is_supported;
  });

  if (!are_types_supported) {
    return false;
  }

  //Check 2

  bool has_unsupported_dimension = false;
  node->ForEachDef([&has_unsupported_dimension, this, &optype](const NodeArg& node_arg, bool is_input) {
    if (is_input) {
      if (this->graph_viewer.IsConstantInitializer(node_arg.Name(), true))
        return;
    }
    auto shape = node_arg.Shape();
    if (shape != nullptr) {
      //Can't have no dimensions
      if ((shape->dim_size() == 0) && (check_if_op_is_supported(optype, no_dimension_supported))) {
        has_unsupported_dimension = true;
        return;
      } else {
        //Zero dimension check
        for (const auto& dim : shape->dim()) {
          if (utils::HasDimValue(dim) && dim.dim_value() == 0) {
            if ((device_id.find("MYRIAD") != std::string::npos) && (optype == "Resize"))
              return;
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
  if (domain == kOnnxDomain && check_if_unsupported_op_mode(node)) {
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

std::vector<NodeIndex> Capability::GetUnsupportedNodeIndices(std::unordered_set<std::string>& ng_required_initializers) {  

    const auto ng_supported_ops = GetNgSupportedOps(GetOnnxOpSet(graph_viewer));

    std::vector<NodeIndex> unsupported_nodes_idx;

    for (const auto& node_idx : graph_viewer.GetNodesInTopologicalOrder()) {
      if (check_if_node_is_supported(ng_supported_ops, node_idx)) {
        // Collect inputs that are initializers
        graph_viewer.GetNode(node_idx)->ForEachDef([&ng_required_initializers, this](const NodeArg& node_arg, bool is_input) {
            if(is_input && this->graph_viewer.GetAllInitializedTensors().count(node_arg.Name())) {
                ng_required_initializers.insert(node_arg.Name());
              } }, true);
      } else {
        unsupported_nodes_idx.push_back(node_idx);
      }
    }
    return unsupported_nodes_idx;        
}

bool Capability::IsOpSupportedOnlyInModel(std::string name) {
  return ops_supported_only_in_model.find(name) != ops_supported_only_in_model.end();
}

bool Capability::CheckSpecialConditionForClusterSizeOne(std::unordered_set<std::string>& ng_required_initializers, const Node* node) {
    if (node->OpType() == "Reshape") {
          const auto& shape_arg = node->InputDefs()[1];
          if (ng_required_initializers.find(shape_arg->Name()) == ng_required_initializers.end())
            return true;
    } else if (node->OpType() == "Expand") {
        const auto& output = node->OutputDefs()[0];
        if (output->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16)
          return true;
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
          return true;
    } else if ((node->OpType() == "Greater") || (node->OpType() == "Less")) {
        if (device_id.find("MYRIAD") != std::string::npos) {
          auto input_0_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
          auto input_1_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->InputDefs()[1]->TypeAsProto()->tensor_type().elem_type();
          auto output_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

          if (!((output_data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT) ||
                (output_data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16))) {
            return true;
          }

          if ((input_0_data_type != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16) ||
              (input_1_data_type != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16)) {
            return true;
          }
        }
    }
    return false;
}

bool Capability::DoNotOmitSubGraph(const std::string& name) {
  return check_if_op_is_supported(name, subgraph_supported);
}

bool Capability::NodePushBack(const Node *node, const std::string& optype) {
  if (optype == "TopK" || optype == "NonZero") {
    return true;
  }
  if (optype == "Gather") {
    if (device_id.find("MYRIAD") != std::string::npos) {
      auto input_data_type = node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
      if (input_data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8) {
        return true;
      }
    }
  }
  return false;
}



}  // namespace openvino_ep
}  // namespace onnxruntime