// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include <stddef.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <Python.h>

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

//static common::Status SaveModel(ONNX_NAMESPACE::ModelProto& model_proto, const std::string& file_path){
//    int fd;
//    Status status = Env::Default().FileOpenWr(file_path,fd);
//
//    google::protobuf::io::FileOutputStream output(fd);
//    const bool result = model_proto.SerializeToZeroCopyStream(&output) && output.Flush();
//    if(result)
//        return Status::OK();
//    else
//        return Status::OK();
//
//}

std::vector<std::unique_ptr<ComputeCapability>> OpenVINOExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph_viewer,
    const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {


    std::vector<std::unique_ptr<ComputeCapability>> result;
    bool precision_fp32 = true;
    std::string device_id = "CPU";

    #ifdef OPENVINO_CONFIG_GPU_FP32
        device_id = "GPU";
    #endif

    #ifdef OPENVINO_CONFIG_GPU_FP16
        precision_fp32 = false;
        device_id = "GPU";
    #endif

    #ifdef OPENVINO_CONFIG_MYRIAD
        precision_fp32 = false;
    #endif

    #ifdef OPENVINO_CONFIG_VAD_R
        precision_fp32 = false;
    #endif


    auto domain_map = graph_viewer.DomainToVersionMap();
    int opset_version = 0;
    auto it = domain_map.find(kOnnxDomain);
    if(it != domain_map.end())
        opset_version = it->second;

    // std::vector<std::vector<onnxruntime::NodeIndex>> groups;
    std::vector<onnxruntime::NodeIndex> group;
    int counter = 0;
    auto initializers = graph_viewer.GetAllInitializedTensors();

    auto node_indexes = graph_viewer.GetNodesInTopologicalOrder();
    std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();


    auto model_proto = GetModelProtoFromFusedNode(graph_viewer);

    auto graph_proto = model_proto.mutable_graph();
    int input_dims = 0;
    int output_dims = 0;

    if(graph_viewer.GetInputs().size() !=0)
        input_dims = graph_proto->input(0).type().tensor_type().shape().dim_size();

    if(graph_viewer.GetOutputs().size() !=0)
        output_dims = graph_proto->output(0).type().tensor_type().shape().dim_size();



    //GPU Plugin does not support single dimensional input
    if(device_id == "GPU"){
        if(input_dims == 1 || input_dims == 5 || output_dims == 5)
            return result;
    }





    // bool newSubGraph(true);
    bool isGraphSupported = true;
    for (auto index : node_indexes) {

        auto node = graph_viewer.GetNode(index);


        auto layer = openvino_ep::OpenVINOLayer(node->OpType());


        if (layer.getName() == "NotSupported" && layer.getOpsetVersion() < opset_version) {

            isGraphSupported = false;
            return result;
        }
        //Gemm, BatchNorm, Conv and Reshape cant take more than 1 input
        if(node->OpType() == "Gemm" || node->OpType() == "BatchNormalization" || node->OpType() == "Conv" || node->OpType() == "Reshape" || node->OpType() == "MatMul"){


            int count = 0;
            for(auto input : node->InputDefs()){
                auto name = input->Name();
                auto it = initializers.find(name);
                if(it == initializers.end()){
                    count++;
                }
            }
            if(count > 1){
                return result;
            }
        }
        //Dropout or Identity can't have graph inputs
        if(node->OpType() == "Dropout" || node->OpType() == "Identity" || node->OpType() == "Concat") {

            auto graph_inputs = graph_viewer.GetInputs();
            for(auto input : node->InputDefs()){
                auto it = find(graph_inputs.begin(), graph_inputs.end(), input);
                if(it != graph_inputs.end()){
                    return result;
                }
            }
        }

        if(node->OpType() == "MaxPool" || node->OpType() == "AveragePool"){
            auto attributes = node->GetAttributes();
            auto auto_pad = attributes["auto_pad"].s();
            if(auto_pad == ""){
                return result;
            }
            if(node->OutputDefs().size() > 1){
                return result;
            }
        }
        //Transpose with no attr is not supported
        if(node->OpType() == "Transpose"){
            auto attributes = node->GetAttributes();
            auto perm = attributes["perm"].ints();
            if(perm.size() == 0){
                return result;
            }


            const auto* type_proto = node->InputDefs()[0]->TypeAsProto();
            if(type_proto->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_STRING){
                return result;
            }

            if(device_id == "GPU"){

                auto graph_inputs = graph_viewer.GetInputs();
                auto it = find(graph_inputs.begin(), graph_inputs.end(), node->InputDefs()[0]);
                if(it != graph_inputs.end()){
                    if(input_dims == 3){
                        return result;
                    }
                }
                // else{
                //     const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
                //     graph_viewer.GetInitializedTensor(node->InputDefs()[0]->Name(), tensor_proto);
                //     if(tensor_proto->dims_size() == 3){
                //         return result;
                //     }
                // }
            }
        }

        if(node->OpType() == "Softmax"){
            auto attributes = node->GetAttributes();
            auto axis = attributes["axis"].i();
            if(axis != 1)
                return result;
        }

    }
    std::set<const onnxruntime::NodeArg*> fused_inputs, fused_outputs;



    if(isGraphSupported){

        std::string model_proto_strbuf;
        model_proto.SerializeToString(&model_proto_strbuf);


        PyObject *pModule, *pOutput,*pFunc;

        Py_Initialize();
        pModule = PyImport_ImportModule("openvino_mo");

        if(pModule != NULL){

            if(precision_fp32){
                pFunc = PyObject_GetAttrString(pModule,"convert_fp32");
            }
            else{
                pFunc = PyObject_GetAttrString(pModule,"convert_fp16");
            }

            // Prepare ModelProto Input to Python
            PyObject* pFileName = PyByteArray_FromStringAndSize(model_proto_strbuf.c_str(), model_proto_strbuf.size());
            PyObject* pArgs = PyTuple_New(1);
            PyTuple_SetItem(pArgs, 0, pFileName);


            if(pFunc && PyCallable_Check(pFunc)){

                // Call the Python function
                pOutput = PyObject_CallObject(pFunc, pArgs);

                if(!PyTuple_CheckExact(pOutput)){
                    return result;
                }
                else{
                    Py_DECREF(pOutput);
                }
            }
        }


        for(auto index : node_indexes){
            sub_graph->nodes.push_back(index);
        }
        for(auto input : graph_viewer.GetInputs()){
            fused_inputs.insert(input);
        }
        for(auto output : graph_viewer.GetOutputs()){
            fused_outputs.insert(output);
        }

        ONNX_NAMESPACE::AttributeProto model_proto_str_attr;
        model_proto_str_attr.set_name("model_proto_str");
        model_proto_str_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING);
        model_proto_str_attr.set_s(model_proto_strbuf);

        auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
        meta_def->attributes["model_proto_str"] = model_proto_str_attr;
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
