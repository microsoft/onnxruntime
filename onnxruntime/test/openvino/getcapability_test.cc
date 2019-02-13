// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session.h"
#include "gtest/gtest.h"
#include "test/framework/test_utils.h"
#include "test/test_environment.h"
#include "core/graph/model.h"
#include "core/framework/compute_capability.h"
// #include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

void AddAttribute_Helper(onnxruntime::Node& node, const std::string& attr_name, int64_t attr_value){
    ONNX_NAMESPACE::AttributeProto attr;
    attr.set_name(attr_name);
    attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
    attr.set_i(attr_value);
    node.AddAttribute(attr_name,attr);
}
void AddAttribute_Helper(onnxruntime::Node& node, const std::string& attr_name, std::initializer_list<int64_t> attr_value) {
  ONNX_NAMESPACE::AttributeProto attr;
  attr.set_name(attr_name);
  attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
  for (auto v : attr_value) {
    attr.add_ints(v);
  }
  node.AddAttribute(attr_name, attr);
}

TEST(OpenVINOExecutionProviderTest, GetCapability_Sanity) {

    static const std::string MODEL_URI = "testdata/transform/abs-2id-max.onnx";

    OpenVINOExecutionProviderInfo info;
    std::shared_ptr<Model> model;
    ASSERT_TRUE(Model::Load(MODEL_URI,model).IsOK());
    Graph& graph = model->MainGraph();
    auto opv_provider = std::make_unique<OpenVINOExecutionProvider>(info);
    GraphViewer graph_viewer(graph);
    std::vector<const KernelRegistry*> kernel_reg;
    std::vector<std::unique_ptr<ComputeCapability>> compute_cap = opv_provider->GetCapability(graph_viewer,kernel_reg);
    for(auto& cc : compute_cap){

        for(auto nodeIndex : cc->sub_graph->nodes){

            auto node = graph_viewer.GetNode(nodeIndex);
            ASSERT_TRUE(node->OpType() == "Identity");
            //TODO: Uncomment these
            // ASSERT_FALSE(node->OpType() == "Max");
            // ASSERT_FALSE(node->OpType() == "Abs");
        }
    }
}

TEST(OpenVINOExecutionProviderTest, GetCapability_Group) {

    std::unordered_map<std::string, int> domain_map;
    domain_map[kOnnxDomain] = 9;
    onnxruntime::Model model("graph_1", false, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(),domain_map);
    auto& graph = model.MainGraph();
    std::vector<onnxruntime::NodeArg*> inputs;
    std::vector<onnxruntime::NodeArg*> outputs;

    ONNX_NAMESPACE::TypeProto float_tensor;

    float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

    ONNX_NAMESPACE::TypeProto float_tensor1;

    float_tensor1.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    float_tensor1.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    float_tensor1.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
    float_tensor1.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);

    ONNX_NAMESPACE::TypeProto int_tensor;

    int_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    int_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    int_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
    int_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

    auto& input_arg1 = graph.GetOrCreateNodeArg("X", &float_tensor);
    auto& input_arg2 = graph.GetOrCreateNodeArg("Y", &float_tensor);
    inputs.push_back(&input_arg1);
    inputs.push_back(&input_arg2);
    auto& output_arg = graph.GetOrCreateNodeArg("add_out", &float_tensor);
    outputs.push_back(&output_arg);

    graph.AddNode("add_1", "Add", "Adding", inputs, outputs);

    inputs.clear();
    inputs.push_back(&output_arg);
    auto& output_arg1 = graph.GetOrCreateNodeArg("transpose_out", &float_tensor1);
    outputs.clear();
    outputs.push_back(&output_arg1);
    auto& transpose_node = graph.AddNode("tranpose_1", "Transpose", "Transposing", inputs, outputs);
    ONNX_NAMESPACE::AttributeProto perm_attr;
    perm_attr.set_name("perm");
    perm_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
    perm_attr.add_ints(0);
    perm_attr.add_ints(2);
    perm_attr.add_ints(1);
    transpose_node.AddAttribute("perm", perm_attr);

    auto& input_arg3 = graph.GetOrCreateNodeArg("Max_input2", &float_tensor1);
    inputs.clear();
    inputs.push_back(&output_arg1);
    inputs.push_back(&input_arg3);
    auto& output_arg2 = graph.GetOrCreateNodeArg("Max_Out", &float_tensor1);
    outputs.clear();
    outputs.push_back(&output_arg2);

    graph.AddNode("Max1", "Max", "Max operator", inputs, outputs);

    inputs.clear();
    inputs.push_back(&output_arg2);
    auto& output_arg3 = graph.GetOrCreateNodeArg("Identity_out", &float_tensor1);
    outputs.clear();
    outputs.push_back(&output_arg3);
    graph.AddNode("Identity1", "Identity", "Identity operator", inputs, outputs);

    inputs.clear();
    inputs.push_back(&output_arg3);
    auto& output_arg4 = graph.GetOrCreateNodeArg("Identity2_out", &float_tensor1);
    outputs.clear();
    outputs.push_back(&output_arg4);
    graph.AddNode("Identity2", "Identity", "Identity operator", inputs, outputs);

    inputs.clear();
    auto& input_arg4 = graph.GetOrCreateNodeArg("Reshape_input", &int_tensor);
    inputs.push_back(&output_arg4);
    inputs.push_back(&input_arg4);
    auto& output_arg5 = graph.GetOrCreateNodeArg("out", &float_tensor);
    outputs.clear();
    outputs.push_back(&output_arg5);
    graph.AddNode("Reshape1", "Reshape", "Reshape operator", inputs,outputs);

    auto status = graph.Resolve();
    ASSERT_TRUE(status.IsOK());
    std::string model_file_name = "my_model.onnx";
    status = onnxruntime::Model::Save(model, model_file_name);

    OpenVINOExecutionProviderInfo info;
    std::shared_ptr<Model> model1;

    ASSERT_TRUE(Model::Load(model_file_name,model1).IsOK());
    Graph& graph1 = model1->MainGraph();
    auto opv_provider = std::make_unique<OpenVINOExecutionProvider>(info);
    GraphViewer graph_viewer(graph1);
    std::vector<const KernelRegistry*> kernel_reg;
    std::vector<std::unique_ptr<ComputeCapability>> compute_cap = opv_provider->GetCapability(graph_viewer,kernel_reg);
    //TODO: Change this to 2
    ASSERT_EQ(compute_cap.size(),1);
    for(auto& cc : compute_cap) {

        auto nodes_list = cc->sub_graph->nodes;

        for(auto nodeIndex : nodes_list){

            auto node = graph_viewer.GetNode(nodeIndex);
            //TODO: Change this to ASSERT_FALSE after GetCapability code is merged
            ASSERT_TRUE(node->OpType() == "Max");
        }
    }
}


TEST(OpenVINOExecutionProviderTest, GetCapability_Conv) {


    std::unordered_map<std::string, int> domain_map;
    domain_map[kOnnxDomain] = 9;
    onnxruntime::Model model("graph_1", false, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(),domain_map);
    auto& graph = model.MainGraph();
    std::vector<onnxruntime::NodeArg*> inputs;
    std::vector<onnxruntime::NodeArg*> outputs;

    ONNX_NAMESPACE::TypeProto float_tensor;
    float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(7);

    ONNX_NAMESPACE::TypeProto float_tensor1;
    float_tensor1.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    float_tensor1.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    float_tensor1.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    float_tensor1.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &float_tensor);
    auto& input_arg_2 = graph.GetOrCreateNodeArg("W", &float_tensor1);
    inputs.push_back(&input_arg_1);
    inputs.push_back(&input_arg_2);
    auto& output_arg = graph.GetOrCreateNodeArg("node_out", &float_tensor);
    outputs.push_back(&output_arg);
    auto& conv_node = graph.AddNode("node_1", "Conv", "ConvolutionNode",inputs,outputs);


    AddAttribute_Helper(conv_node,"dilations", {1});
    AddAttribute_Helper(conv_node,"pads", {0,0});
    AddAttribute_Helper(conv_node,"group", 1);
    AddAttribute_Helper(conv_node,"strides", {1});
    AddAttribute_Helper(conv_node,"kernel_shape", {1});

    auto status = graph.Resolve();
    ASSERT_TRUE(status.IsOK());
    std::string model_file_name = "conv.onnx";
    status = onnxruntime::Model::Save(model,model_file_name);

}
}  // namespace test
}  // namespace onnxruntime
