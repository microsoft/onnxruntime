// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include <stddef.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/compute_capability.h"
#include "core/framework/tensorprotoutils.h"

#include "openvino_execution_provider.h"
#include "openvino_layer.h"
#include "core/graph/model.h"
#include "openvino_graph.h"

#include "core/util/protobuf_parsing_utils.h"
namespace onnxruntime {

//using namespace InferenceEngine;

constexpr const char* OpenVINO = "OpenVINO";


OpenVINOExecutionProvider::OpenVINOExecutionProvider(OpenVINOExecutionProviderInfo& info)
: IExecutionProvider {onnxruntime::kOpenVINOExecutionProvider} {
  (void)info;

  DeviceAllocatorRegistrationInfo device_info( {OrtMemTypeDefault, [](int) {return std::make_unique<CPUAllocator>(std::make_unique<OrtAllocatorInfo>(OPENVINO,OrtDeviceAllocator,0,OrtMemTypeDefault));}, std::numeric_limits<size_t>::max()});
  InsertAllocator(CreateAllocator(device_info));
}

static ONNX_NAMESPACE::ModelProto GetModelProtoFromFusedNode(const onnxruntime::GraphViewer& graph_viewer) {
//   const auto& attributes = fused_node->GetAttributes();
//   const auto& initializers = attributes.at("initializers").tensors();

  ONNX_NAMESPACE::ModelProto model_proto;
  auto graph_proto = model_proto.mutable_graph();
//   const auto& fused_graph = fused_node->GetFunctionBody()->Body();

  for (const auto& node : graph_viewer.Nodes()) {
    node.ToProto(*(graph_proto->add_node()));
  }

  for (const auto& input : graph_viewer.GetInputs()) {
    auto valueInfoProto = graph_proto->add_input();
    *valueInfoProto = input->ToProto();
  }

  for (const auto& output : graph_viewer.GetOutputs()) {
    auto valueInfoProto = graph_proto->add_output();
    *valueInfoProto = output->ToProto();
  }

  for (const auto& initializer : graph_viewer.GetAllInitializedTensors()) {
    graph_proto->add_initializer()->CopyFrom(*initializer.second);
  }

  auto opset = model_proto.add_opset_import();
  opset->set_domain(kOnnxDomain);
  opset->set_version(graph_viewer.DomainToVersionMap().at(kOnnxDomain));
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  return model_proto;
}

static common::Status SaveModel(ONNX_NAMESPACE::ModelProto& model_proto, const std::string& file_path){
    int fd;
    Status status = Env::Default().FileOpenWr(file_path,fd);

    google::protobuf::io::FileOutputStream output(fd);
    const bool result = model_proto.SerializeToZeroCopyStream(&output) && output.Flush();
    if(result)
        return Status::OK();
    else
        return Status::OK();

}
std::vector<std::unique_ptr<ComputeCapability>> OpenVINOExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph_viewer,
    const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {

    std::vector<std::unique_ptr<ComputeCapability>> result;

    auto domain_map = graph_viewer.DomainToVersionMap();
    int opset_version = 0;
    auto it = domain_map.find(kOnnxDomain);
    if(it != domain_map.end())
        opset_version = it->second;

    // std::vector<std::vector<onnxruntime::NodeIndex>> groups;
    std::vector<onnxruntime::NodeIndex> group;
    int counter = 0;

    auto node_indexes = graph_viewer.GetNodesInTopologicalOrder();
    std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();

    // bool newSubGraph(true);
    bool isGraphSupported = true;
    for (auto index : node_indexes) {

        auto node = graph_viewer.GetNode(index);

        auto layer = openvino_ep::OpenVINOLayer(node->OpType());

        if (layer.getName() == "NotSupported" && layer.getOpsetVersion() < opset_version) {

            isGraphSupported = false;
            return result;
        }
    }
    std::set<const onnxruntime::NodeArg*> fused_inputs, fused_outputs;


    if(isGraphSupported){
        for(auto index : node_indexes){
            sub_graph->nodes.push_back(index);
        }
        for(auto input : graph_viewer.GetInputs()){
            fused_inputs.insert(input);
        }
        for(auto output : graph_viewer.GetOutputs()){
            fused_outputs.insert(output);
        }
        auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
        meta_def->name = "OpenVINOKernel_" + std::to_string(counter++);
        meta_def->domain = "OpenVINO";
        meta_def->since_version = 1;
        // meta_def->attributes["initializers"] = izers_attr;

        for(auto input : fused_inputs){
            meta_def->inputs.push_back(input->Name());
        }

        for (auto output : fused_outputs) {
            meta_def->outputs.push_back(output->Name());
        }

        sub_graph->SetMetaDef(meta_def);
        result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));

        auto model_proto = GetModelProtoFromFusedNode(graph_viewer);
        SaveModel(model_proto,"ov_model.onnx");
    }

    return result;
}



common::Status OpenVINOExecutionProvider::Compile(
    const std::vector<onnxruntime::Node*>& fused_nodes,
    std::vector<NodeComputeInfo>& node_compute_funcs) {

    // for (const auto& fused_node : fused_nodes) {

    //     auto fused_graph = fused_node->GetFunctionBody()->Body();

        // auto opset = model_proto.add_opset_import();
        // opset->set_domain(kOnnxDomain);
        // opset->set_version(fused_graph.DomainToVersionMap().at(kOnnxDomain));
        // model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
        // SaveModel(model_proto,"mo_model.onnx");
    // }

  for (auto fused_node : fused_nodes) {
    std::shared_ptr<openvino_ep::OpenVINOGraph> openvino_graph;
    try {
      openvino_graph = std::make_shared<openvino_ep::OpenVINOGraph>(fused_node,
          std::string(info_.device));

    } catch (const char* msg) {
      std::cerr << "Caught Compiler exception: " << msg << std::endl;
      return Status(common::StatusCategory::ONNXRUNTIME,
          common::StatusCode::NOT_IMPLEMENTED, msg);
    }

    NodeComputeInfo compute_info;

    compute_info.create_state_func =
        [openvino_graph] (ComputeContext* context, FunctionState* state) {

          OpenVINOEPFunctionState* p = new OpenVINOEPFunctionState();
          p->allocate_func = context->allocate_func;
          p->destroy_func = context->release_func;
          p->allocator_handle = context->allocator_handle;
          p->openvino_graph = openvino_graph;
          *state = static_cast<FunctionState>(p);
          return 0;
        };

    compute_info.compute_func =
        [] (FunctionState state, ONNXRunTimeTensor* input_tensors, size_t num_inputs,
            ONNXRunTimeTensor* output_tensors, size_t num_outputs) {

          (void)num_outputs;
          auto function_state = static_cast<OpenVINOEPFunctionState*>(state);

          try {
            function_state->openvino_graph->Infer(input_tensors, num_inputs, output_tensors,
                num_outputs, function_state->allocate_func, function_state->allocator_handle);
          } catch (const char* msg) {
            std::cerr << "Caught Runtime exception: " << msg << std::endl;
            return common::StatusCode::RUNTIME_EXCEPTION;
          }

          return common::StatusCode::OK;
        };

    compute_info.release_state_func =
        [](FunctionState state) {
          if (state) {
            OpenVINOEPFunctionState* function_state = static_cast<OpenVINOEPFunctionState*>(state);
            delete function_state;
          }
        };

    node_compute_funcs.push_back(compute_info);
  }

  return Status::OK();
}

} // namespace onnxruntime
