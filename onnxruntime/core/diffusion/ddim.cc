// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/diffusion/ddim.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string_view>

#include "core/common/inlined_containers.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace {

Status HostTensor(const OrtValue& value) {
  ORT_RETURN_IF_NOT(value.IsTensor() && value.Get<Tensor>().Location().device.UsesCpuMemory(),
                    "DDIM requires CPU tensors");
  return Status::OK();
}

Status FiniteFloat(const OrtValue& value) {
  ORT_RETURN_IF_ERROR(HostTensor(value));
  const auto& tensor = value.Get<Tensor>();
  ORT_RETURN_IF_NOT(tensor.IsDataType<float>() && tensor.Shape().Size() > 0,
                    "DDIM requires nonempty FP32 tensors");
  const auto values = tensor.DataAsSpan<float>();
  ORT_RETURN_IF_NOT(std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); }),
                    "DDIM tensor contains NaN or infinity");
  return Status::OK();
}

OrtValue AllocateFloat(const TensorShape& shape, const AllocatorPtr& allocator) {
  OrtValue value;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), shape, allocator, value);
  return value;
}

OrtValue Concatenate(const Tensor& first, const Tensor& second, const AllocatorPtr& allocator) {
  auto shape = first.Shape().AsShapeVector();
  shape[0] *= 2;
  auto result = AllocateFloat(TensorShape(shape), allocator);
  auto* destination = result.GetMutable<Tensor>()->MutableData<float>();
  const auto a = first.DataAsSpan<float>();
  const auto b = second.DataAsSpan<float>();
  std::copy(b.begin(), b.end(), std::copy(a.begin(), a.end(), destination));
  return result;
}

}  // namespace

Status DdimScheduler::Validate(const DdimOptions& options) {
  ORT_RETURN_IF(options.profile != "sd15_ddim_v1", "Unsupported DDIM profile: ", options.profile);
  ORT_RETURN_IF(options.num_inference_steps < 1 || options.num_inference_steps > 999,
                "SD 1.5 DDIM requires 1 through 999 inference steps");
  ORT_RETURN_IF(!std::isfinite(options.guidance_scale) || options.guidance_scale < 0,
                "DDIM guidance scale must be finite and nonnegative");
  ORT_RETURN_IF(options.guidance_mode != DdimGuidanceMode::Batched &&
                    options.guidance_mode != DdimGuidanceMode::Sequential,
                "Unknown DDIM guidance mode");
  ORT_RETURN_IF(options.timestep_format != DdimTimestepFormat::Int64Scalar &&
                    options.timestep_format != DdimTimestepFormat::Int64Vector &&
                    options.timestep_format != DdimTimestepFormat::Float32Scalar &&
                    options.timestep_format != DdimTimestepFormat::Float32Vector,
                "Unknown DDIM timestep format");
  size_t elements = 1;
  const auto maximum = std::min(static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                                static_cast<uint64_t>(std::numeric_limits<size_t>::max() / (2 * sizeof(float))));
  for (const auto dimension : options.latent_shape) {
    ORT_RETURN_IF(dimension <= 0 || static_cast<uint64_t>(dimension) > maximum / elements,
                  "Invalid or overflowing DDIM latent dimensions");
    elements *= static_cast<size_t>(dimension);
  }
  InlinedHashSet<std::string_view> names;
  for (const auto* name : {&options.latent_input, &options.timestep_input, &options.conditioning_input,
                           &options.prediction_output, &options.initialization_input,
                           &options.positive_conditioning_input, &options.negative_conditioning_input,
                           &options.latent_output}) {
    ORT_RETURN_IF(name->empty() || name->find('\0') != std::string::npos || !names.insert(*name).second,
                  "DDIM port names must be nonempty and distinct");
  }
  return Status::OK();
}

DdimScheduler::DdimScheduler(DdimOptions options) : options_(std::move(options)) {
  const float first = std::sqrt(0.00085f);
  const float last = std::sqrt(0.012f);
  float product = 1.f;
  for (size_t i = 0; i < alpha_products_.size(); ++i) {
    const float root = first + (last - first) * (static_cast<float>(i) / 999.f);
    product *= 1.f - root * root;
    alpha_products_[i] = product;
  }
}

Status DdimScheduler::Create(DdimOptions options, std::shared_ptr<const DdimScheduler>& out) {
  ORT_RETURN_IF_ERROR(Validate(options));
  out = std::shared_ptr<const DdimScheduler>(new DdimScheduler(std::move(options)));
  return Status::OK();
}

Status DdimScheduler::Timestep(int64_t index, int64_t& out) const {
  ORT_RETURN_IF(index < 0 || index >= options_.num_inference_steps, "DDIM timestep index out of range");
  out = (options_.num_inference_steps - 1 - index) * (1000 / options_.num_inference_steps) + 1;
  return Status::OK();
}

Status DdimScheduler::CheckLatents(const OrtValue& value) const {
  ORT_RETURN_IF_ERROR(HostTensor(value));
  ORT_RETURN_IF(value.Get<Tensor>().Shape() != TensorShape(gsl::make_span(options_.latent_shape)),
                "DDIM latent shape mismatch");
  return FiniteFloat(value);
}

Status DdimScheduler::CheckConditioning(const OrtValue& value) const {
  ORT_RETURN_IF_ERROR(HostTensor(value));
  const auto& shape = value.Get<Tensor>().Shape();
  ORT_RETURN_IF(shape.NumDimensions() == 0 || shape[0] != options_.latent_shape[0],
                "DDIM conditioning batch mismatch");
  ORT_RETURN_IF(shape.Size() > std::numeric_limits<int64_t>::max() / 2 ||
                    static_cast<uint64_t>(shape.Size()) > std::numeric_limits<size_t>::max() / (2 * sizeof(float)),
                "DDIM conditioning size overflows");
  return FiniteFloat(value);
}

Status DdimScheduler::CheckInitialization(const OrtValue& input) const {
  ORT_RETURN_IF_ERROR(HostTensor(input));
  const auto& tensor = input.Get<Tensor>();
  if (tensor.IsDataType<uint64_t>()) {
    ORT_RETURN_IF(tensor.Shape().NumDimensions() != 0, "DDIM seed must be a uint64 scalar");
    return Status::OK();
  }
  return CheckLatents(input);
}

Status DdimScheduler::CheckPrediction(const OrtValue& value) const {
  ORT_RETURN_IF_ERROR(HostTensor(value));
  auto shape = options_.latent_shape;
  if (options_.guidance_mode == DdimGuidanceMode::Batched) shape[0] *= 2;
  ORT_RETURN_IF(value.Get<Tensor>().Shape() != TensorShape(gsl::make_span(shape)), "DDIM prediction shape mismatch");
  return FiniteFloat(value);
}

Status DdimScheduler::Initialize(const OrtValue& input, const AllocatorPtr& allocator, OrtValue& out) const {
  ORT_RETURN_IF_ERROR(CheckInitialization(input));
  const auto& tensor = input.Get<Tensor>();
  if (!tensor.IsDataType<uint64_t>()) {
    auto value = AllocateFloat(tensor.Shape(), allocator);
    const auto source = tensor.DataAsSpan<float>();
    std::copy(source.begin(), source.end(), value.GetMutable<Tensor>()->MutableData<float>());
    out = std::move(value);
    return Status::OK();
  }
  auto value = AllocateFloat(TensorShape(gsl::make_span(options_.latent_shape)), allocator);
  auto samples = value.GetMutable<Tensor>()->MutableDataAsSpan<float>();
  std::mt19937_64 engine(*tensor.Data<uint64_t>());
  for (size_t i = 0; i < samples.size(); i += 2) {
    const double u1 = static_cast<double>((engine() >> 11) + 1) * 0x1.0p-53;
    const double u2 = static_cast<double>(engine() >> 11) * 0x1.0p-53;
    const double radius = std::sqrt(-2. * std::log(u1));
    const double angle = 6.2831853071795864769 * u2;
    samples[i] = static_cast<float>(radius * std::cos(angle));
    if (i + 1 < samples.size()) samples[i + 1] = static_cast<float>(radius * std::sin(angle));
  }
  out = std::move(value);
  return Status::OK();
}

Status DdimScheduler::Prepare(const OrtValue& latents, const OrtValue& positive, const OrtValue& negative,
                              int64_t index, bool conditional, const AllocatorPtr& allocator,
                              DdimModelInputs& out) const {
  ORT_RETURN_IF_ERROR(CheckLatents(latents));
  ORT_RETURN_IF_ERROR(CheckConditioning(positive));
  ORT_RETURN_IF_ERROR(CheckConditioning(negative));
  ORT_RETURN_IF(positive.Get<Tensor>().Shape() != negative.Get<Tensor>().Shape(),
                "DDIM conditioning shapes differ");
  int64_t timestep = 0;
  ORT_RETURN_IF_ERROR(Timestep(index, timestep));
  DdimModelInputs result;
  if (options_.guidance_mode == DdimGuidanceMode::Batched) {
    result.latents = Concatenate(latents.Get<Tensor>(), latents.Get<Tensor>(), allocator);
    result.conditioning = Concatenate(negative.Get<Tensor>(), positive.Get<Tensor>(), allocator);
  } else {
    result.latents = latents;
    result.conditioning = conditional ? positive : negative;
  }
  const bool vector = options_.timestep_format == DdimTimestepFormat::Int64Vector ||
                      options_.timestep_format == DdimTimestepFormat::Float32Vector;
  const TensorShape shape = vector ? TensorShape{1} : TensorShape{};
  if (options_.timestep_format == DdimTimestepFormat::Int64Scalar ||
      options_.timestep_format == DdimTimestepFormat::Int64Vector) {
    Tensor::InitOrtValue(DataTypeImpl::GetType<int64_t>(), shape, allocator, result.timestep);
    *result.timestep.GetMutable<Tensor>()->MutableData<int64_t>() = timestep;
  } else {
    result.timestep = AllocateFloat(shape, allocator);
    *result.timestep.GetMutable<Tensor>()->MutableData<float>() = static_cast<float>(timestep);
  }
  out = std::move(result);
  return Status::OK();
}

Status DdimScheduler::Update(const OrtValue& latents, gsl::span<const OrtValue> predictions, int64_t index,
                             const AllocatorPtr& allocator, OrtValue& out) const {
  ORT_RETURN_IF_ERROR(CheckLatents(latents));
  const bool batched = options_.guidance_mode == DdimGuidanceMode::Batched;
  ORT_RETURN_IF(predictions.size() != (batched ? 1u : 2u), "DDIM prediction count mismatch");
  for (const auto& value : predictions) {
    ORT_RETURN_IF_ERROR(CheckPrediction(value));
  }
  int64_t timestep = 0;
  ORT_RETURN_IF_ERROR(Timestep(index, timestep));
  const int64_t previous = timestep - 1000 / options_.num_inference_steps;
  const float alpha = alpha_products_[static_cast<size_t>(timestep)];
  const float alpha_previous = alpha_products_[static_cast<size_t>(std::max(int64_t{0}, previous))];
  const float noise = std::sqrt(1.f - alpha);
  const float original = std::sqrt(alpha);
  const float next_original = std::sqrt(alpha_previous);
  const float next_noise = std::sqrt(1.f - alpha_previous);
  const auto input = latents.Get<Tensor>().DataAsSpan<float>();
  const float* unconditional = predictions[0].Get<Tensor>().Data<float>();
  const float* conditional = batched ? unconditional + input.size() : predictions[1].Get<Tensor>().Data<float>();
  auto value = AllocateFloat(latents.Get<Tensor>().Shape(), allocator);
  auto result = value.GetMutable<Tensor>()->MutableDataAsSpan<float>();
  for (size_t i = 0; i < input.size(); ++i) {
    const float epsilon = unconditional[i] + options_.guidance_scale * (conditional[i] - unconditional[i]);
    result[i] = next_original * ((input[i] - noise * epsilon) / original) + next_noise * epsilon;
    ORT_RETURN_IF(!std::isfinite(result[i]), "DDIM update produced NaN or infinity");
  }
  out = std::move(value);
  return Status::OK();
}

}  // namespace onnxruntime
