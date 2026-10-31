// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <memory>
#include <string>

#include "core/common/common.h"
#include "core/framework/allocator.h"
#include "core/framework/ort_value.h"

namespace onnxruntime {

enum class DdimGuidanceMode { Batched,
                              Sequential };
enum class DdimTimestepFormat { Int64Scalar,
                                Int64Vector,
                                Float32Scalar,
                                Float32Vector };

struct DdimOptions {
  std::string profile = "sd15_ddim_v1";
  std::array<int64_t, 4> latent_shape{};
  int64_t num_inference_steps = 0;
  float guidance_scale = 0;
  DdimGuidanceMode guidance_mode = DdimGuidanceMode::Batched;
  DdimTimestepFormat timestep_format = DdimTimestepFormat::Int64Scalar;
  std::string latent_input;
  std::string timestep_input;
  std::string conditioning_input;
  std::string prediction_output;
  std::string initialization_input;
  std::string positive_conditioning_input;
  std::string negative_conditioning_input;
  std::string latent_output;
};

struct DdimModelInputs {
  OrtValue latents;
  OrtValue conditioning;
  OrtValue timestep;
};

class DdimScheduler {
 public:
  static Status Validate(const DdimOptions& options);
  static Status Create(DdimOptions options, std::shared_ptr<const DdimScheduler>& out);
  const DdimOptions& Options() const { return options_; }
  const std::array<float, 1000>& AlphaProducts() const { return alpha_products_; }
  Status Timestep(int64_t index, int64_t& out) const;
  Status CheckInitialization(const OrtValue& input) const;
  Status Initialize(const OrtValue& input, const AllocatorPtr& allocator, OrtValue& out) const;
  Status CheckLatents(const OrtValue& value) const;
  Status CheckConditioning(const OrtValue& value) const;
  Status CheckPrediction(const OrtValue& value) const;
  Status Prepare(const OrtValue& latents, const OrtValue& positive, const OrtValue& negative,
                 int64_t index, bool conditional, const AllocatorPtr& allocator, DdimModelInputs& out) const;
  Status Update(const OrtValue& latents, gsl::span<const OrtValue> predictions, int64_t index,
                const AllocatorPtr& allocator, OrtValue& out) const;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DdimScheduler);

 private:
  explicit DdimScheduler(DdimOptions options);
  const DdimOptions options_;
  std::array<float, 1000> alpha_products_;
};

}  // namespace onnxruntime
