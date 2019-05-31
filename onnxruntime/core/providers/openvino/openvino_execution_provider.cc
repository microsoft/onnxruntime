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
#include "core/session/onnxruntime_cxx_api.h"

#include "openvino_execution_provider.h"
#include "openvino_layer.h"
#include "core/graph/model.h"
#include "openvino_graph.h"

#include "core/util/protobuf_parsing_utils.h"
namespace onnxruntime {

constexpr const char* OpenVINO = "OpenVINO";

OpenVINOExecutionProvider::OpenVINOExecutionProvider(OpenVINOExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kOpenVINOExecutionProvider} {
  (void)info;

  DeviceAllocatorRegistrationInfo device_info({OrtMemTypeDefault, [](int) { return std::make_unique<CPUAllocator>(std::make_unique<OrtAllocatorInfo>(OPENVINO, OrtDeviceAllocator, 0, OrtMemTypeDefault)); }, std::numeric_limits<size_t>::max()});
  InsertAllocator(CreateAllocator(device_info));
}

static ONNX_NAMESPACE::ModelProto GetModelProtoFromFusedNode(const onnxruntime::GraphViewer& graph_viewer) {
  ONNX_NAMESPACE::ModelProto model_proto;
  auto graph_proto = model_proto.mutable_graph();

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
  device_id = "MYRIAD";
#endif

#ifdef OPENVINO_CONFIG_VAD_R
  precision_fp32 = false;
  device_id = "HDDL";
#endif

    auto domain_map = graph_viewer.DomainToVersionMap();
    int opset_version = 0;
    auto it = domain_map.find(kOnnxDomain);
    if(it != domain_map.end())
        opset_version = it->second;

    std::vector<onnxruntime::NodeIndex> group;
    int counter = 0;
    auto initializers = graph_viewer.GetAllInitializedTensors();

    auto node_indexes = graph_viewer.GetNodesInTopologicalOrder();
    std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();


    auto model_proto = GetModelProtoFromFusedNode(graph_viewer);

    auto graph_proto = model_proto.mutable_graph();
    int input_dims = 0;
    int output_dims = 0;
    int num_inputs = graph_viewer.GetInputs().size();
    int num_outputs = graph_viewer.GetOutputs().size();

    std::vector<std::vector<int>> input_arrays;


    if(num_inputs !=0 )
        input_dims = graph_proto->input(0).type().tensor_type().shape().dim_size();

    if(num_outputs !=0)
        output_dims = graph_proto->output(0).type().tensor_type().shape().dim_size();

    if(num_inputs != 0){

        for(int i = 0; i < num_inputs; i++){

            int input_dims_size = graph_proto->input(i).type().tensor_type().shape().dim_size();
            std::vector<int> temp_arr;

            for(int j = 0; j < input_dims_size; j++){
                temp_arr.push_back(graph_proto->input(i).type().tensor_type().shape().dim(j).dim_value());
            }
            input_arrays.push_back(temp_arr);
        }
    }


    //GPU Plugin does not support single dimensional input
    if(device_id == "GPU"){
        if(input_dims == 1 || input_dims == 5 || output_dims == 5)
            return result;
    }

    bool isGraphSupported = true;
    bool OpSum = false, OpGemm = false;
    for (auto index : node_indexes) {

        auto node = graph_viewer.GetNode(index);

        //Use ForEachDef

        auto layer = openvino_ep::OpenVINOLayer(node->OpType());


        if (layer.getName() == "NotSupported" && layer.getOpsetVersion() < opset_version) {

            isGraphSupported = false;
            return result;
        }

        if(node->OpType() == "Sum"){

            OpSum = true;
        }

        if(node->OpType() == "Gemm"){

            OpGemm = true;

        }


        //Gemm, BatchNorm, Conv and Reshape cant take more than 1 input
        if(node->OpType() == "BatchNormalization" || node->OpType() == "Conv" || node->OpType() == "Reshape"){


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

        if(device_id == "MYRIAD" || device_id == "HDDL"){

            if(node->OpType() == "Reshape" || node->OpType() == "Unsqueeze" || node->OpType() == "Flatten"){

                if(input_arrays[0][0] != 1) {
                    return result;
                }
            }
            if(node->OpType() == "Gemm"){
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
        }


        if(node->OpType() == "MatMul"){


            for(size_t i = 0; i < node->InputDefs().size(); i++){

                if(node->InputDefs()[i]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT){
                    return result;
                }
            }


            auto iter = node->OutputNodesBegin();

            if(iter == node->OutputNodesEnd()){
                return result;
            }

            for(auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it){
                auto out_node = graph_viewer.GetNode((*it).Index());

                if(out_node->OpType() != "Add"){
                    return result;
                }
            }

            bool isGraphInput = false;

            auto graph_inputs = graph_viewer.GetInputs();
            for(auto input : node->InputDefs()) {
                auto it = find(graph_inputs.begin(), graph_inputs.end(), input);
                if(it != graph_inputs.end()){
                    isGraphInput = true;
                }
            }
            if(isGraphInput){

                size_t input_dims_size = input_arrays[0].size();


                for(int i = 0; i < num_inputs; i++){

                    if(input_arrays[i].size() > 2)
                        return result;

                    if(input_dims_size != input_arrays[i].size())
                        return result;
                }
            }

            //Disable MNIST for MYRIAD and HDDL
            if(device_id == "MYRIAD" || device_id == "HDDL")
                return result;
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
            if(auto_pad == "" || auto_pad == "SAME_LOWER"){ // || auto_pad == "SAME_UPPER"){
                return result;
            }

            auto dilations_ints = attributes["dilations"].ints();
            if(dilations_ints.size() != 0){
                if(dilations_ints[0] > 1)
                    return result;
            }

            auto ceil_mode = attributes["ceil_mode"].i();
            if(ceil_mode != 0)
                return result;
            if(node->OutputDefs().size() > 1){
                return result;
            }
        }

        //We only support 4D and 5D blobs for pooling
        if(node->OpType() == "GlobalMaxPool" || node->OpType() == "MaxPool" || node->OpType() == "AveragePool" || node->OpType() == "GlobalAveragePool")
        {
                auto graph_inputs = graph_viewer.GetInputs();
                auto it = find(graph_inputs.begin(), graph_inputs.end(), node->InputDefs()[0]);
                if(it != graph_inputs.end()){
                    if(device_id == "MYRIAD" || device_id == "HDDL"){
                        if(input_dims != 3 || input_dims != 4)
                            return result;
                    }
                    else if(input_dims < 4 || input_dims > 5){
                        return result;
                    }
                }
        }

        //Transpose with no attr is not supported
        if(node->OpType() == "Transpose"){
            auto attributes = node->GetAttributes();
            auto perm = attributes["perm"].ints();
            if(perm.size() == 0 || perm.size() > 5){
                return result;
            }


            const auto* type_proto = node->InputDefs()[0]->TypeAsProto();
            if(type_proto->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_STRING){
                return result;
            }


                auto graph_inputs = graph_viewer.GetInputs();
                auto it = find(graph_inputs.begin(), graph_inputs.end(), node->InputDefs()[0]);
                if(it != graph_inputs.end()){
                    if(input_dims == 3 || input_dims == 2 || input_dims > 5){
                        return result;
                    }
                }
        }

        if(node->OpType() == "Unsqueeze"){
            auto graph_inputs = graph_viewer.GetInputs();
            auto attributes = node->GetAttributes();
            auto axes = attributes["axes"].ints();
            auto it = find(graph_inputs.begin(), graph_inputs.end(), node->InputDefs()[0]);
            if(it != graph_inputs.end()){
                if(input_dims + axes.size() > 5){
                    return result;
                }
            }
            const auto* type_proto = node->InputDefs()[0]->TypeAsProto();
            if(type_proto->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT)
                return result;
        }

        if(node->OpType() == "Reshape"){
            if(output_dims == 0){
                return result;
            }
        }

        if(node->OpType() == "Mul" || node->OpType() == "Add" || node->OpType() == "Sum"){


            auto graph_inputs = graph_viewer.GetInputs();
            auto it = find(graph_inputs.begin(), graph_inputs.end(), node->InputDefs()[0]);

            if(it != graph_inputs.end()){

                size_t dims = input_arrays[0].size();
                if(dims == 0)
                    return result;

                for(int i = 0; i < num_inputs; i++){

                    if(dims != input_arrays[i].size())
                        return result;
                }
                for(int i = 0; i < num_inputs; i++){
                    for(size_t j = 0; j < dims; j++){

                        if(input_arrays[0][j] != input_arrays[i][j]){
                            isGraphSupported = false;
                            break;
                        }
                    }
                }
            }
        }

        if(node->OpType() == "Softmax"){
            auto graph_inputs = graph_viewer.GetInputs();
            auto it = find(graph_inputs.begin(), graph_inputs.end(), node->InputDefs()[0]);
            if(it != graph_inputs.end()){
                if(input_dims != 2){
                    return result;
                }
            }
            auto attributes = node->GetAttributes();
            auto axis = attributes["axis"].i();
            if(axis != 1)
                return result;
        }


        if(node->OpType() == "Flatten"){
            auto attributes = node->GetAttributes();
            auto axis = attributes["axis"].i();
            if(device_id == "MYRIAD" || device_id == "HDDL")
            {
                if(axis != 1)
                    return result;
            }
        }

    }
    //Disable Resnet DUC for MYRIAD and HDDL
    if(OpSum == true && OpGemm == false){
        if(device_id == "MYRIAD" || device_id == "HDDL"){
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
        else{
            return result;
        }

        for(auto index : node_indexes){
            sub_graph->nodes.push_back(index);
            auto* node = graph_viewer.GetNode(index);

            // Track graph inputs and initializers
            for(auto input_def : node->InputDefs()) {
              if(fused_outputs.find(input_def) == fused_outputs.end()) {
                fused_inputs.insert(input_def);
              } else {
                fused_outputs.erase(input_def);
              }
            }

            // Track graph outputs
            for(auto output_def : node->OutputDefs()) {
              if(fused_inputs.find(output_def) == fused_inputs.end()) {
                fused_outputs.insert(output_def);
              } else {
                fused_inputs.erase(output_def);
              }
            }
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

        for (auto input : fused_inputs) {
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
        [openvino_graph](ComputeContext* context, FunctionState* state) {
          OpenVINOEPFunctionState* p = new OpenVINOEPFunctionState();
          p->allocate_func = context->allocate_func;
          p->destroy_func = context->release_func;
          p->allocator_handle = context->allocator_handle;
          p->openvino_graph = openvino_graph;
          *state = static_cast<FunctionState>(p);
          return 0;
        };

    compute_info.compute_func =
        [](FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
          Ort::CustomOpApi ort{*api};

          auto function_state = static_cast<OpenVINOEPFunctionState*>(state);

          try {
            function_state->openvino_graph->Infer(ort, context);
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

}  // namespace onnxruntime
