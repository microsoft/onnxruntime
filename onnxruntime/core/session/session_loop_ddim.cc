// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/session_loop.h"

#include "core/common/safeint.h"
#include "core/framework/data_transfer.h"
#include "core/framework/run_options.h"
#include "core/framework/tensor.h"
#include "core/framework/stateful_execution_utils.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"

namespace onnxruntime {
using namespace stateful_execution;

namespace {
std::unique_ptr<NodeArg> Argument(std::string_view name, int type, gsl::span<const int64_t> shape) {
  ONNX_NAMESPACE::TypeProto proto;
  auto* tensor = proto.mutable_tensor_type();
  tensor->set_elem_type(type);
  auto* dimensions = tensor->mutable_shape();
  for (const auto dimension : shape) dimensions->add_dim()->set_dim_value(dimension);
  return std::make_unique<NodeArg>(std::string(name), &proto);
}
}  // namespace

Status SessionLoop::ConfigureDdim() {
  const auto& options = *options_.ddim;
  ORT_RETURN_IF_ERROR(DdimScheduler::Create(options, ddim_));
  const auto& inputs = *session_.GetModelInputs().second;
  const auto& outputs = *session_.GetModelOutputs().second;
  for (const auto& name : {options.initialization_input, options.positive_conditioning_input,
                           options.negative_conditioning_input, options.latent_output}) {
    ORT_RETURN_IF(FindArgument(inputs, name) || FindArgument(outputs, name), "DDIM logical and model ports collide");
  }
  auto shape = options.latent_shape;
  shape[0] *= options.guidance_mode == DdimGuidanceMode::Batched ? 2 : 1;
  auto check_model = [&](const std::string& name, const NodeArg* actual, int type,
                         gsl::span<const int64_t> dimensions) -> Status {
    ORT_RETURN_IF_NOT(actual, "Unknown DDIM model port: ", name);
    const auto expected = Argument(name, type, dimensions);
    return CheckCompatibleTypes(*expected, *actual);
  };
  ORT_RETURN_IF_ERROR(check_model(options.latent_input, FindArgument(inputs, options.latent_input),
                                  ONNX_NAMESPACE::TensorProto_DataType_FLOAT, shape));
  ORT_RETURN_IF_ERROR(check_model(options.prediction_output, FindArgument(outputs, options.prediction_output),
                                  ONNX_NAMESPACE::TensorProto_DataType_FLOAT, shape));
  const bool integer = options.timestep_format == DdimTimestepFormat::Int64Scalar ||
                       options.timestep_format == DdimTimestepFormat::Int64Vector;
  const bool vector = options.timestep_format == DdimTimestepFormat::Int64Vector ||
                      options.timestep_format == DdimTimestepFormat::Float32Vector;
  const std::array<int64_t, 1> one{1};
  ORT_RETURN_IF_ERROR(check_model(options.timestep_input, FindArgument(inputs, options.timestep_input),
                                  integer ? ONNX_NAMESPACE::TensorProto_DataType_INT64
                                          : ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                                  vector ? gsl::span<const int64_t>(one) : gsl::span<const int64_t>{}));
  const auto* conditioning = FindArgument(inputs, options.conditioning_input);
  ORT_RETURN_IF_NOT(conditioning, "Unknown DDIM conditioning model port");
  ORT_RETURN_IF_ERROR(CheckTensorType(*conditioning));
  auto type = *conditioning->TypeAsProto();
  auto* tensor = type.mutable_tensor_type();
  ORT_RETURN_IF(tensor->elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                "DDIM conditioning must be FP32");
  if (tensor->has_shape()) {
    auto* dimensions = tensor->mutable_shape();
    ORT_RETURN_IF(dimensions->dim_size() == 0, "DDIM conditioning must have a batch dimension");
    const auto& batch = dimensions->dim(0);
    ORT_RETURN_IF(batch.has_dim_value() && batch.dim_value() != shape[0], "DDIM model conditioning batch mismatch");
    dimensions->mutable_dim(0)->set_dim_value(options.latent_shape[0]);
  }
  ddim_inputs_[0] = Argument(options.initialization_input, ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                             options.latent_shape);
  ddim_inputs_[1] = std::make_unique<NodeArg>(options.positive_conditioning_input, &type);
  ddim_inputs_[2] = std::make_unique<NodeArg>(options.negative_conditioning_input, &type);
  ddim_output_ = Argument(options.latent_output, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, options.latent_shape);
  return Status::OK();
}

const NodeArg* SessionLoop::InputArgument(std::string_view name) const {
  if (ddim_) {
    for (const auto& input : ddim_inputs_) {
      if (input->Name() == name) return input.get();
    }
    const auto& options = *options_.ddim;
    if (name == options.latent_input || name == options.timestep_input || name == options.conditioning_input) {
      return nullptr;
    }
  }
  return FindArgument(*session_.GetModelInputs().second, name);
}

const NodeArg* SessionLoop::OutputArgument(std::string_view name) const {
  if (ddim_) return name == ddim_output_->Name() ? ddim_output_.get() : nullptr;
  return FindArgument(*session_.GetModelOutputs().second, name);
}

Status SessionLoop::CheckValue(std::string_view name, const OrtValue& value) const {
  const auto* argument = InputArgument(name);
  ORT_RETURN_IF_NOT(argument, "Unknown SessionLoop input: ", name);
  if (ddim_) {
    const auto& options = *options_.ddim;
    if (name == options.initialization_input) return ddim_->CheckInitialization(value);
    if (name == options.positive_conditioning_input || name == options.negative_conditioning_input) {
      ORT_RETURN_IF_ERROR(ddim_->CheckConditioning(value));
    }
  }
  return CheckInput(*argument, value);
}

Status SessionLoop::CheckSource(std::string_view name, const NodeArg& source) const {
  const auto* target = InputArgument(name);
  ORT_RETURN_IF_NOT(target, "Unknown SessionLoop input: ", name);
  ORT_RETURN_IF_ERROR(CheckTensorType(source));
  if (ddim_ && name == options_.ddim->initialization_input &&
      source.TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_UINT64) {
    ORT_RETURN_IF(source.Shape() && source.Shape()->dim_size() != 0, "DDIM seed source must be scalar");
    return Status::OK();
  }
  return CheckCompatibleTypes(source, *target);
}

Status SessionLoopExecution::InitializeDdim() {
  if (!loop_->ddim_) return Status::OK();
  std::unique_ptr<IOBinding> binding;
  ORT_RETURN_IF_ERROR(loop_->session_.NewIOBinding(&binding));
  ORT_RETURN_IF_ERROR(loop_->ddim_->Initialize(initial_inputs_.inputs.at(loop_->options_.ddim->initialization_input),
                                               binding->GetCPUAllocator(kCpuExecutionProvider), ddim_initial_));
  ddim_latents_ = ddim_initial_;
  return Status::OK();
}

Status SessionLoopExecution::StepDdim(SessionLoopStepResult& out) {
  const auto& options = *loop_->options_.ddim;
  const int calls = options.guidance_mode == DdimGuidanceMode::Batched ? 1 : 2;
  InlinedVector<OrtValue> predictions;
  for (int call = 0; call < calls; ++call) {
    std::unique_ptr<IOBinding> binding;
    ORT_RETURN_IF_ERROR(loop_->session_.NewIOBinding(&binding));
    const auto allocator = binding->GetCPUAllocator(kCpuExecutionProvider);
    DdimModelInputs inputs;
    ORT_RETURN_IF_ERROR(loop_->ddim_->Prepare(ddim_latents_,
                                              initial_inputs_.inputs.at(options.positive_conditioning_input),
                                              initial_inputs_.inputs.at(options.negative_conditioning_input),
                                              progress_.iterations, call != 0, allocator, inputs));
    auto bind = [&](const std::string& name, const OrtValue& value) -> Status {
      ORT_RETURN_IF_ERROR(binding->BindInput(name, value));
      if (value.Get<Tensor>().Location().device.UsesCpuMemory()) {
        staging_.cpu_input_request_bytes =
            SafeInt<size_t>(staging_.cpu_input_request_bytes) + value.Get<Tensor>().SizeInBytes();
      }
      return Status::OK();
    };
    for (const auto& [name, value] : initial_inputs_.inputs) {
      if (name != options.initialization_input && name != options.positive_conditioning_input &&
          name != options.negative_conditioning_input) {
        ORT_RETURN_IF_ERROR(bind(name, value));
      }
    }
    ORT_RETURN_IF_ERROR(bind(options.latent_input, inputs.latents));
    ORT_RETURN_IF_ERROR(bind(options.conditioning_input, inputs.conditioning));
    ORT_RETURN_IF_ERROR(bind(options.timestep_input, inputs.timestep));
    ORT_RETURN_IF_ERROR(binding->BindOutput(options.prediction_output, OrtDevice{}));
    ORT_RETURN_IF_ERROR(binding->SynchronizeInputs());
    RunOptions run_options;
    ORT_RETURN_IF_ERROR(loop_->session_.Run(run_options, *binding));
    progress_.invocations = SafeInt<int64_t>(progress_.invocations) + 1;
    ORT_RETURN_IF_ERROR(binding->SynchronizeOutputs());
    ORT_RETURN_IF(binding->GetOutputs().size() != 1, "DDIM requires one prediction result");
    ORT_RETURN_IF_ERROR(loop_->ddim_->CheckPrediction(binding->GetOutputs()[0]));
    const auto& source = binding->GetOutputs()[0].Get<Tensor>();
    ORT_RETURN_IF_NOT(source.Location().device.UsesCpuMemory(), "DDIM prediction must be CPU accessible");
    staging_.cpu_output_bytes = SafeInt<size_t>(staging_.cpu_output_bytes) + source.SizeInBytes();
    OrtValue snapshot;
    Tensor::InitOrtValue(source.DataType(), source.Shape(), allocator, snapshot);
    ORT_RETURN_IF_ERROR(CPUDataTransfer{}.CopyTensor(source, *snapshot.GetMutable<Tensor>()));
    staging_.cpu_snapshot_copy_bytes = SafeInt<size_t>(staging_.cpu_snapshot_copy_bytes) + source.SizeInBytes();
    predictions.push_back(std::move(snapshot));
  }
  std::unique_ptr<IOBinding> binding;
  ORT_RETURN_IF_ERROR(loop_->session_.NewIOBinding(&binding));
  OrtValue next;
  ORT_RETURN_IF_ERROR(loop_->ddim_->Update(ddim_latents_, predictions, progress_.iterations,
                                           binding->GetCPUAllocator(kCpuExecutionProvider), next));
  ORT_RETURN_IF(cancelled_.load(), "DDIM cancelled at update boundary; call Reset before reuse");
  ddim_latents_ = std::move(next);
  progress_.iterations = SafeInt<int64_t>(progress_.iterations) + 1;
  EvaluateExits({});
  progress_.outputs = {ddim_latents_};
  auto result = progress_;
  result.executed = true;
  requires_reset_ = false;
  out = std::move(result);
  return Status::OK();
}

}  // namespace onnxruntime
