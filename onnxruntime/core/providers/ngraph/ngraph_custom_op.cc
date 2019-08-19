// Copyright(C) 2019 Intel Corporation
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
#include <ngraph/frontend/onnx_import/onnx.hpp>
#if defined(_MSC_VER)
#pragma warning(default : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic pop
#endif

#include "ngraph_custom_op.h"
#include "core/common/logging/logging.h"
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace ngraph_ep {

static bool check_ngraph_dump_ops() {
#ifdef _WIN32
  size_t env_name_len = 0;
  char* env_name = nullptr;
  return (_dupenv_s(&env_name, &env_name_len, "ONNXRUNTIME_NGRAPH_DUMP_OPS") == 0 && env_name != nullptr);
#else
  return (std::getenv("ONNXRUNTIME_NGRAPH_DUMP_OPS") != nullptr);
#endif
}

NGRAPHCustomOp::NGRAPHCustomOp(const ComputeContext* context,
                               const ONNX_NAMESPACE::ModelProto& model_proto,
                               const std::shared_ptr<ngraph::runtime::Backend>& ng_backend) :
  ng_backend_{ng_backend}, model_proto_{model_proto}
{
  allocate_func_ = context->allocate_func;
  release_func_ = context->release_func;
  allocator_ = context->allocator_handle;
  name_ = context->node_name;

  if (check_ngraph_dump_ops()) {
    std::fstream dump(name_ + ".onnx", std::ios::out | std::ios::trunc | std::ios::binary);
    model_proto_.SerializeToOstream(&dump);
  }
}

NGRAPHCustomOp::~NGRAPHCustomOp() {
  for (const auto& compiled_exe : ng_exe_map_) {
    ng_backend_->remove_compiled_function(compiled_exe.second);
  }
}

//This method gets called in critical path of execution: Optimize
void NGRAPHCustomOp::Initialize(const OrtCustomOpApi* api, OrtKernelContext* context) const {
  Ort::CustomOpApi ort{*api};

  size_t num_inputs = ort.KernelContext_GetInputCount(context);

  //Key for ng_exe_map
  std::string uniq_input_shape;

  //Optimizing for general case of 4D tensors
  uniq_input_shape.reserve(4 * sizeof(int64_t) * num_inputs + num_inputs);

  for (size_t i = 0; i < num_inputs; i++) {
    const OrtValue* input_tensor = ort.KernelContext_GetInput(context, i);
    auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
    auto tensor_shape = ort.GetTensorShape(tensor_info);
    ort.ReleaseTensorTypeAndShapeInfo(tensor_info);

    const auto ndim = tensor_shape.size();
    uniq_input_shape.append(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    uniq_input_shape.append(reinterpret_cast<const char*>(tensor_shape.data()), ndim * sizeof(int64_t));
  }

  auto it = ng_exe_map_.insert({uniq_input_shape, nullptr});  //TODO: Limit the size of map with configurable size.

  //ng_exe with current shape already exists
  if (!it.second) {
    ng_curr_exe_ = it.first->second;
    return;
  } else {
    auto graph_proto = model_proto_.mutable_graph();

    LOGS_DEFAULT(INFO) << "[NGRAPHCustomOp] Compiling customOp: " << name_;

    // Clear previous shapes if any and set new input shapes
    for (size_t i = 0; i < num_inputs; i++) {
      auto g_in_shape = graph_proto->mutable_input((int)i)->mutable_type()->mutable_tensor_type()->mutable_shape();
      g_in_shape->clear_dim();

      const OrtValue* input_tensor = ort.KernelContext_GetInput(context, i);
      auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
      auto tensor_shape = ort.GetTensorShape(tensor_info);
      ort.ReleaseTensorTypeAndShapeInfo(tensor_info);

      for (size_t dim = 0; dim < tensor_shape.size(); dim++) {
        g_in_shape->add_dim()->set_dim_value(tensor_shape[dim]);
      }
    }

    std::istringstream model_stream{model_proto_.SerializeAsString()};
    std::shared_ptr<ngraph::Function> ng_function;
    try {
      ng_function = ngraph::onnx_import::import_onnx_model(model_stream);
    } catch (const std::exception& exp) {
      LOGS_DEFAULT(FATAL) << "[NGRAPHCustomOp] " << " - " << name_ << " - "
                          << "Exception while importing model to nGraph: " << std::string(exp.what());
      throw;
    } catch (...) {
      LOGS_DEFAULT(FATAL) << "[NGRAPHCustomOp] " << " - " << name_ << " - "
                          << "Unknown exception while importing model to nGraph";
      throw;
    }

    for (auto& result : ng_function->get_results()) {
      result->set_needs_default_layout(true);
    }

    // Finally compile nGraph with backend.
    try {
      ng_curr_exe_ = ng_backend_->compile(ng_function);
    } catch (const std::exception& exp) {
      LOGS_DEFAULT(FATAL) << "[NGRAPHCustomOp] " << " - " << name_ << " - "
                          << "Exception while compiling ngraph::Function: " << std::string(exp.what());
    } catch (...) {
      LOGS_DEFAULT(FATAL) << "[NGRAPHCustomOp] " << " - " << name_ << " - " << "Unknown exception while compiling ngraph::Function";
    }
    it.first->second = ng_curr_exe_;
  }
}  // namespace ngraph_ep

//This method gets called in critical path of execution: Optimize
Status NGRAPHCustomOp::Compute(const OrtCustomOpApi* api, OrtKernelContext* context) const {
  Ort::CustomOpApi ort{*api};

  //TODO: Minimize locked region
  std::lock_guard<std::mutex> lock(compute_lock_);

  // Initialize nGraph function if it is not already initialized.
  Initialize(api, context);

  ORT_ENFORCE(ng_curr_exe_ != nullptr);

  std::vector<std::shared_ptr<ngraph::runtime::Tensor>> ng_inputs;
  std::vector<std::shared_ptr<ngraph::runtime::Tensor>> ng_outputs;

  // Write ONNXR input data to nGraph input tensors.
  try {
    unsigned input_index = 0;
    for (const auto& ng_param : ng_curr_exe_->get_parameters()) {
      const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index++);
      void* input_data = const_cast<void*>(ort.GetTensorData<void>(input_tensor));
      ng_inputs.emplace_back(ng_backend_->create_tensor(ng_param->get_output_element_type(0), ng_param->get_output_shape(0), input_data));
    }
  } catch (const std::exception& exp) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, name_ + ": Exception while copying input data to nGraph: " + std::string(exp.what()));
  } catch (...) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, name_ + ": Unknown exception while copying input data to nGraph");
  }

  // Initialize output tensors
  try {
    //TODO: Optimize
    unsigned output_index = 0;
    for (auto& ng_result : ng_curr_exe_->get_results()) {
      const auto& dtype = ng_result->get_element_type();
      const auto& shape = ng_result->get_shape();

      std::vector<int64_t> ort_shape{shape.begin(), shape.end()};
      OrtValue* output_tensor = ort.KernelContext_GetOutput(context, output_index++, ort_shape.data(), ort_shape.size());
      void* output_data = ort.GetTensorMutableData<void>(output_tensor);
      ng_outputs.emplace_back(ng_backend_->create_tensor(dtype, shape, output_data));
    }
  } catch (const std::exception& exp) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, name_ + ": Exception while creating nGraph output Tensor: " + std::string(exp.what()));
  } catch (...) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, name_ + ": Unknown exception while creating nGraph output Tensor");
  }

  // Run the graph through nGraph.
  try {
    if (!ng_curr_exe_->call(ng_outputs, ng_inputs))
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, name_ + ": Error while executing nGraph computation");
  } catch (const std::exception& exp) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, name_ + ": Exception while executing nGraph computation: " + std::string(exp.what()));
  } catch (...) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, name_ + ": Unknown exception while executing nGraph computation");
  }

  return Status::OK();
}

}  // namespace ngraph_ep
}  // namespace onnxruntime
