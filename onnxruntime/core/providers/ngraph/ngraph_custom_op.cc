// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include <fstream>
#include <iostream>
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <ngraph/frontend/onnx_import/onnx.hpp>
#pragma GCC diagnostic pop

#include "ngraph_custom_op.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {
namespace ngraph_ep {

static DType GetDataType(const ngraph::element::Type& ng_type) {
  switch (ng_type.get_type_enum()) {
    case ngraph::element::Type_t::f32:
      return DType::TFloat32;
    case ngraph::element::Type_t::f64:
      return DType::TDouble;
    case ngraph::element::Type_t::boolean:
      return DType::TBool;
    case ngraph::element::Type_t::u8:
      return DType::TUint8;
    case ngraph::element::Type_t::i8:
      return DType::TInt8;
    case ngraph::element::Type_t::u16:
      return DType::TUint16;
    case ngraph::element::Type_t::i16:
      return DType::TInt16;
    case ngraph::element::Type_t::u32:
      return DType::TUint32;
    case ngraph::element::Type_t::i32:
      return DType::TInt32;
    case ngraph::element::Type_t::u64:
      return DType::TUint64;
    case ngraph::element::Type_t::i64:
      return DType::TInt64;
    default:
      throw "Unsupported DataType";
  }
}

NGRAPHCustomOp::NGRAPHCustomOp(const ComputeContext* context, const ONNX_NAMESPACE::ModelProto& model_proto,
                               const std::shared_ptr<ngraph::runtime::Backend>& ng_backend)
    : ng_backend_{ng_backend},
      model_proto_{model_proto} {
  allocate_func_ = context->allocate_func;
  release_func_ = context->release_func;
  allocator_ = context->allocator_handle;
  name_ = context->node_name;

  if (std::getenv("ONNXRUNTIME_NGRAPH_DUMP_OPS") != nullptr) {
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
void NGRAPHCustomOp::Initialize(const ONNXRunTimeTensor* input_tensors, const size_t& num_inputs) const {
  LOGS_DEFAULT(INFO) << "nGraph compiling customOp: " << name_;

  //Key for ng_exe_map
  std::string uniq_input_shape;

  //Optimizing for general case of 4D tensors
  uniq_input_shape.reserve(4 * sizeof(int64_t) * num_inputs + num_inputs);

  for (size_t i = 0; i < num_inputs; i++) {
    const auto& ndim = input_tensors[i].ndim;
    uniq_input_shape.append(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    uniq_input_shape.append(reinterpret_cast<const char*>(input_tensors[i].shape), ndim * sizeof(int64_t));
  }

  auto it = ng_exe_map_.insert({uniq_input_shape, nullptr});  //TODO: Limit the size of map with configurable size.

  //ng_exe with current shape already exists
  if (!it.second) {
    ng_curr_exe_ = it.first->second;
    return;
  } else {
    auto graph_proto = model_proto_.mutable_graph();
    // Clear previous shapes if any and set new input shapes
    for (size_t i = 0; i < num_inputs; i++) {
      auto g_in_shape = graph_proto->mutable_input(i)->mutable_type()->mutable_tensor_type()->mutable_shape();
      g_in_shape->clear_dim();

      for (size_t dim = 0; dim < input_tensors[i].ndim; dim++) {
        g_in_shape->add_dim()->set_dim_value(input_tensors[i].shape[dim]);
      }
    }

    std::istringstream model_stream{model_proto_.SerializeAsString()};
    std::shared_ptr<ngraph::Function> ng_function;
    try {
      ng_function = ngraph::onnx_import::import_onnx_model(model_stream);
    } catch (const std::exception& exp) {
      LOGS_DEFAULT(FATAL) << "[" << name_ << "] "
                          << "Exception while converting onnx to nGraph: " << std::string(exp.what());
      throw;
    } catch (...) {
      LOGS_DEFAULT(FATAL) << "[" << name_ << "] "
                          << "Unknown exception while converting onnx to nGraph";
      throw;
    }

    for (auto& result : ng_function->get_results()) {
      result->set_needs_default_layout(true);
    }

    // Finally compile nGraph with backend.
    try {
      ng_curr_exe_ = ng_backend_->compile(ng_function);
    } catch (const std::exception& exp) {
      LOGS_DEFAULT(FATAL) << "Exception while compiling nGraph Op: " << name_ << std::string(exp.what());
    } catch (...) {
      LOGS_DEFAULT(FATAL) << "Unknown exception while compiling nGraph Op: " << name_;
    }
    it.first->second = ng_curr_exe_;
  }
}  // namespace ngraph_ep

//This method gets called in critical path of execution: Optimize
Status NGRAPHCustomOp::Compute(const ONNXRunTimeTensor* input_tensors, const size_t num_inputs, ONNXRunTimeTensor* const output_tensors, const size_t num_outputs) const {
  ORT_UNUSED_PARAMETER(num_outputs);

  //TODO: Minimize locked region
  std::lock_guard<std::mutex> lock(compute_lock_);

  // Initialize nGraph function if it is not already initialized.
  Initialize(input_tensors, num_inputs);

  ORT_ENFORCE(ng_curr_exe_ != nullptr);

  std::vector<std::shared_ptr<ngraph::runtime::Tensor>> ng_inputs;
  std::vector<std::shared_ptr<ngraph::runtime::Tensor>> ng_outputs;

  // Write ONNXR input data to nGraph input tensors.
  try {
    auto& in_tensor = input_tensors;
    for (const auto& ng_param : ng_curr_exe_->get_parameters()) {
      ng_inputs.emplace_back(ng_backend_->create_tensor(ng_param->get_output_element_type(0), ng_param->get_output_shape(0), (in_tensor++)->data));
    }
  } catch (const std::exception& exp) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Exception while copying input data to nGraph: " + std::string(exp.what()));
  } catch (...) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unknown exception while copying input data to nGraph");
  }

  // Initialize output tensors
  try {
    //TODO: Optimize
    auto onxr_output = output_tensors;
    for (auto& ng_result : ng_curr_exe_->get_results()) {
      const auto& dtype = ng_result->get_element_type();
      const auto& shape = ng_result->get_shape();

      onxr_output->dtype = GetDataType(dtype);
      onxr_output->ndim = shape.size();
      onxr_output->shape = new int64_t[onxr_output->ndim];

      size_t num_elements = 1;
      for (size_t dim = 0; dim < shape.size(); dim++) {
        num_elements *= shape[dim];
        onxr_output->shape[dim] = shape[dim];
      }

      onxr_output->data = (*(allocate_func_))(allocator_, 64, num_elements * sizeof(onxr_output->dtype));

      ng_outputs.emplace_back(ng_backend_->create_tensor(dtype, shape, onxr_output->data));
      ++onxr_output;
    }
  } catch (const std::exception& exp) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Exception while creating nGraph output Tensor: " + std::string(exp.what()));
  } catch (...) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unknown exception while creating nGraph output Tensor");
  }

  // Run the graph through nGraph.
  try {
    if (!ng_curr_exe_->call(ng_outputs, ng_inputs))
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Error while executing nGraph computation");
  } catch (const std::exception& exp) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Exception while executing nGraph computation: " + std::string(exp.what()));
  } catch (...) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unknown exception while executing nGraph computation");
  }

  return Status::OK();
}

}  // namespace ngraph_ep
}  // namespace onnxruntime
