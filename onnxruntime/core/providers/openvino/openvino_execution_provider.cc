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

namespace onnxruntime {

//using namespace InferenceEngine;

constexpr const char* OpenVINO = "OpenVINO";


OpenVINOExecutionProvider::OpenVINOExecutionProvider(OpenVINOExecutionProviderInfo& info)
: IExecutionProvider {onnxruntime::kOpenVINOExecutionProvider} {
  (void)info;
#ifdef onnxruntime_USE_OPENVINO_CPU_FP32
  info_.device = "CPU_FP32";
#elif onnxruntime_USE_OPENVINO_GPU_FP32
  info_.device = "GPU_FP32";
#elif openvino_USE_OPENVINO_GPU_FP16
  info_.device = "GPU_FP16";
#elif onnxruntime_USE_OPENVINO_MYRIAD
  info_.device = "MYRIAD_FP16";
#elif onnxruntime_USE_OPENVINO_VAD_R
  info_.device = "HDDL_FP16";
#else
  info_.device = "CPU_FP32";
#endif

  DeviceAllocatorRegistrationInfo device_info( {OrtMemTypeDefault, [](int) {return std::make_unique<CPUAllocator>(std::make_unique<OrtAllocatorInfo>(OPENVINO,OrtDeviceAllocator,0,OrtMemTypeDefault));}, std::numeric_limits<size_t>::max()});
  InsertAllocator(CreateAllocator(device_info));
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

    std::vector<std::vector<onnxruntime::NodeIndex>> groups;
    int counter = 0;

    bool newSubGraph(true);
    for (auto& node : graph_viewer.Nodes()) {

        auto layer = OpenVINOLayer(node.OpType());

        if (layer.getName() != "NotSupported" && layer.getOpsetVersion() >= opset_version) {


            if(layer.getName() == "FullyConnectedGemm"){

                auto attributes = node.GetAttributes();
                auto broadcast = attributes["broadcast"].i();
                auto transB = attributes["transB"].i();
                auto formal_params = node.Op()->inputs();
                size_t output_size = 1;
                size_t dims_c = 1;

                for(size_t i = 0; i < formal_params.size(); i++){
                    auto formal_name = formal_params[i].GetName();

                    if(formal_name == "A"){

                        auto shape_vector = onnxruntime::utils::GetTensorShapeFromTensorShapeProto(*(node.InputDefs()[i]->Shape()));
                        output_size *= shape_vector[0];
                    }else if (formal_name == "B"){
                        std::string W_name = node.InputDefs()[i]->Name();

                        const ONNX_NAMESPACE::TensorProto* tensor_proto;
                        graph_viewer.GetInitializedTensor(W_name,tensor_proto);
                        InferenceEngine::SizeVector dims;
                        for(int i = 0; i < tensor_proto->dims_size(); i++){
                            dims.push_back(size_t(tensor_proto->dims(i)));
                        }
                        if(transB){
                            output_size *= dims[0];
                        }else
                        {
                            output_size *= dims[1];
                        }

                    }else if (formal_name == "C"){
                        std::string name = node.InputDefs()[i]->Name();

                        const ONNX_NAMESPACE::TensorProto* tensor_proto;
                        graph_viewer.GetInitializedTensor(name,tensor_proto);
                        InferenceEngine::SizeVector dims;
                        for(int i = 0; i < tensor_proto->dims_size(); i++){
                            dims.push_back(size_t(tensor_proto->dims(i)));
                        }
                        dims_c *= dims[1];
                        std::cout << "dims c " << dims_c << std::endl;
                    }
                }

                if(broadcast == 1 && output_size != dims_c ){
                    std::cout << "Gemm op not supported " << std::endl;
                    break;
                }

            }

            if(layer.getName() == "Norm"){

                auto attributes = node.GetAttributes();
                auto bias = attributes["bias"].f();

                if(bias != 1){
                    std::cout << "LRN bias not equal to 1 is not supported " << std::endl;
                    break;
                }
            }


            if(newSubGraph){

                groups.emplace_back();
                newSubGraph = false;
            }
            groups.back().emplace_back(node.Index());
        }
        else{
            std::cout << "Op not supported " << node.OpType() << std::endl;
            //This node is not supported
            newSubGraph = true;
        }
    }

    for(const auto& group : groups){
        if(!group.empty()) {
            std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
            std::set<const onnxruntime::NodeArg*> fused_inputs, fused_outputs;
            for (auto index : group) {
                sub_graph->nodes.push_back(index);
                auto node = graph_viewer.GetNode(index);
                for (auto input : node->InputDefs()) {
                    auto it = fused_outputs.find(input);
                    if (it != fused_outputs.end()) {
                        fused_outputs.erase(it);
                    } else {
                        fused_inputs.insert(input);
                    }
                }
                for (auto output : node->OutputDefs()) {
                    auto it = fused_inputs.find(output);
                    if (it != fused_inputs.end()) {
                        fused_inputs.erase(it);
                    } else {
                        fused_outputs.insert(output);
                    }
                }
            }

            auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
            meta_def->name = "OpenVINOKernel_" + std::to_string(counter++);
            meta_def->domain = "OpenVINO";
            meta_def->since_version = 1;

            for(auto input : fused_inputs){
                meta_def->inputs.push_back(input->Name());
            }

            for (auto output : fused_outputs) {
                meta_def->outputs.push_back(output->Name());
            }

            sub_graph->SetMetaDef(meta_def);
            result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
        }
    }
    return result;
}

common::Status OpenVINOExecutionProvider::Compile(
    const std::vector<onnxruntime::Node*>& fused_nodes,
    std::vector<NodeComputeInfo>& node_compute_funcs) {

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
