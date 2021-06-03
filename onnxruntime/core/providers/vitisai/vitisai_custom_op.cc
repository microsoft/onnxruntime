// Copyright(C) Xilinx Inc.
// Licensed under the MIT License

#include <fstream>
#include <iostream>
#include <string>

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#if defined(_MSC_VER)
#pragma warning(default : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic pop
#endif

#include <chrono>

#include <pyxir/common/xbuffer.hpp>
#include <pyxir/frontend/onnx.hpp>
#include <pyxir/runtime/run_options.hpp>

#include "vitisai_custom_op.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/common/logging/logging.h"
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace vitisai_ep {

static ONNX_NAMESPACE::ModelProto GetModelProtoFromFusedNode(const onnxruntime::Node* fused_node,
                                                             const logging::Logger& logger) {
  const auto* node_function = fused_node->GetFunctionBody();

  ORT_ENFORCE(node_function != nullptr, "Could not extract function body for node: ",
              fused_node->Name());

  const Graph& node_subgraph = node_function->Body();
  onnxruntime::Model model{node_subgraph.Name(), true, ModelMetaData{}, PathString{},
                           IOnnxRuntimeOpSchemaRegistryList{}, node_subgraph.DomainToVersionMap(),
                           std::vector<ONNX_NAMESPACE::FunctionProto>(), logger};

  ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  *(model_proto.mutable_graph()) = node_subgraph.ToGraphProto();

  return model_proto;
}

VitisAICustomOp::VitisAICustomOp(const ComputeContext* context,
                                 const onnxruntime::Node* fused_node,
                                 const std::string& backend_type,
                                 const std::string& export_runtime_module,
                                 const std::string& load_runtime_module,
                                 const logging::Logger* logger)
  : backend_type_(backend_type), export_runtime_module_(export_runtime_module),
    load_runtime_module_(load_runtime_module)
{
  SetLogger(logger);

  allocate_func_ = context->allocate_func;
  release_func_ = context->release_func;
  allocator_ = context->allocator_handle;
  name_ = context->node_name;

  model_proto_ = GetModelProtoFromFusedNode(fused_node, *GetLogger());
  std::istringstream model_stream{model_proto_.SerializeAsString()};
  xg_ = pyxir::onnx::import_onnx_model(model_stream);
  
  // If the `load_runtime_module` provider option is empty we  build a PyXIR
  // runtime module from scratch. Otherwise, we load the runtime module from
  // the provided file.   
  if (load_runtime_module_.empty()) {
    pyxir::partition(xg_, std::vector<std::string>{backend_type_}, "");

    auto input_defs = fused_node->InputDefs();
    for (auto idef : input_defs) {
      in_tensor_names_.push_back(idef->Name());
    }

    auto output_defs = fused_node->OutputDefs();
    for (auto odef : output_defs) {
      out_tensor_names_.push_back(odef->Name());
    }
    
    pyxir::RunOptionsHolder run_options(new pyxir::runtime::RunOptions());
    run_options->on_the_fly_quantization = true;
    run_options->export_runtime_module_path = export_runtime_module_;
    rt_mod_ = pyxir::build_rt(xg_, backend_type_, in_tensor_names_,
                              out_tensor_names_, "vai", run_options);
  } else {
    std::ifstream in_file(load_runtime_module_);
    std::stringstream buffer;
    buffer << in_file.rdbuf();
    std::string serialized_rt_mod = buffer.str();
    in_file.close();

    std::istringstream sstream(serialized_rt_mod);
    rt_mod_.reset(new pyxir::runtime::RuntimeModule());
    rt_mod_->deserialize(sstream);
    in_tensor_names_ = rt_mod_->get_in_tensor_names();
    out_tensor_names_ = rt_mod_->get_out_tensor_names();
  }
}

VitisAICustomOp::~VitisAICustomOp() {}


Status VitisAICustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const { 
  Ort::CustomOpApi ort{*api};
  const unsigned num_inputs = (unsigned) xg_->get_nb_inputs();

  ssize_t batch_size = 1;
  std::vector<pyxir::XBufferHolder> in_tensors;
  std::vector<pyxir::XBufferHolder> out_tensors;

  // Initialize input tensors.
  try {
    for (unsigned i = 0; i < num_inputs; ++i) {
      const OrtValue* input_tensor = ort.KernelContext_GetInput(context, i);
      auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
      auto tensor_type = ort.GetTensorElementType(tensor_info);
      auto ort_shape = ort.GetTensorShape(tensor_info);
      std::vector<ssize_t> tensor_shape{ort_shape.begin(), ort_shape.end()};
      batch_size = tensor_shape[0];
      
      if (tensor_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
          "VITIS-AI EP input onnx tensor data type: " + std::to_string(tensor_type) + " not supported.");
      
      void* input_data = const_cast<void*>(ort.GetTensorData<void>(input_tensor));
      in_tensors.push_back(std::shared_ptr<pyxir::XBuffer>(
        new pyxir::XBuffer(input_data, 4, "f", tensor_shape.size(), tensor_shape,
                            false, false)));
    }
  } catch (const std::exception& exp) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, name_ + ": Exception while copying input data to Pyxir: " + std::string(exp.what()));
  } catch (...) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, name_ + ": Unknown exception while copying input data to Pyxir");
  }

  // Initialize output tensors
  try {
    for (unsigned i = 0; i < out_tensor_names_.size(); ++i) {
      auto shape = xg_->get(out_tensor_names_[i])->shapes[0];
      std::vector<ssize_t> out_shape{shape.begin(), shape.end()};
      out_shape[0] = batch_size;
      std::vector<int64_t> ort_shape{out_shape.begin(), out_shape.end()};

      OrtValue* output_tensor = ort.KernelContext_GetOutput(context, i, ort_shape.data(), ort_shape.size());
      auto tensor_info = ort.GetTensorTypeAndShape(output_tensor);
      auto tensor_type = ort.GetTensorElementType(tensor_info);
      if (tensor_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
          "VITIS-AI EP input onnx tensor data type: " + std::to_string(tensor_type) + " not supported.");

      void* output_data = ort.GetTensorMutableData<void>(output_tensor);
      out_tensors.push_back(std::shared_ptr<pyxir::XBuffer>(
        new pyxir::XBuffer(output_data, 4, "f", out_shape.size(), out_shape,
                            false, false)));
    }
  } catch (const std::exception& exp) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, name_ + ": Exception while creating Pyxir output Tensor: " + std::string(exp.what()));
  } catch (...) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, name_ + ": Unknown exception while creating Pyxir output Tensor");
  }

  // Run the graph through Vitis-AI Pyxir
  try {
    // std::lock_guard<std::mutex> lock(compute_lock_);
    rt_mod_->execute(in_tensors, out_tensors);
  } catch (const std::exception& exp) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, name_ + ": Exception while executing Pyxir computation: " + std::string(exp.what()));
  } catch (...) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, name_ + ": Unknown exception while executing Pyxir computation");
  }
  
  return Status::OK();
}

}  // namespace vitisai_ep
}  // namespace onnxruntime
