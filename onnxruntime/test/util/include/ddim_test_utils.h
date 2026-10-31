// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>

#include "nlohmann/json.hpp"
#include "core/session/onnxruntime_c_api.h"
#include "test/util/include/stateful_execution_test_utils.h"

namespace onnxruntime::test::stateful {

// The returned record borrows names/dimensions until the copying setter is called.
inline OrtStatefulDdimConfig PublicDdimConfig(const DdimOptions& options) {
  return {sizeof(OrtStatefulDdimConfig),
          ORT_STATEFUL_DDIM_SD15_V1,
          options.latent_shape.data(),
          options.latent_shape.size(),
          options.num_inference_steps,
          options.guidance_scale,
          static_cast<OrtStatefulDdimGuidanceMode>(options.guidance_mode),
          static_cast<OrtStatefulDdimTimestepFormat>(options.timestep_format),
          options.latent_input.c_str(),
          options.timestep_input.c_str(),
          options.conditioning_input.c_str(),
          options.prediction_output.c_str(),
          options.initialization_input.c_str(),
          options.positive_conditioning_input.c_str(),
          options.negative_conditioning_input.c_str(),
          options.latent_output.c_str()};
}

inline nlohmann::ordered_json DdimPolicyJson(const DdimOptions& options) {
  const std::array<const char*, 2> modes{"batched", "sequential"};
  const std::array<const char*, 4> formats{"int64_scalar", "int64_vector", "float32_scalar", "float32_vector"};
  return {{"profile", options.profile},
          {"latent_shape", options.latent_shape},
          {"num_inference_steps", options.num_inference_steps},
          {"guidance_scale", options.guidance_scale},
          {"guidance_mode", modes.at(static_cast<size_t>(options.guidance_mode))},
          {"timestep_format", formats.at(static_cast<size_t>(options.timestep_format))},
          {"latent_input", options.latent_input},
          {"timestep_input", options.timestep_input},
          {"conditioning_input", options.conditioning_input},
          {"prediction_output", options.prediction_output},
          {"initialization_input", options.initialization_input},
          {"positive_conditioning_input", options.positive_conditioning_input},
          {"negative_conditioning_input", options.negative_conditioning_input},
          {"latent_output", options.latent_output}};
}

inline nlohmann::ordered_json DdimChainJson(const DdimOptions& options) {
  using Json = nlohmann::ordered_json;
  Json execution{{"kind", "session_loop"}};
  for (const auto& [logical, external] : {
           std::pair{options.initialization_input, "init"},
           std::pair{options.positive_conditioning_input, "positive"},
           std::pair{options.negative_conditioning_input, "negative"}}) {
    execution["inputs"][logical] = {{"kind", "external_input"}, {"name", external}};
  }
  execution["loop"] = {{"ddim", DdimPolicyJson(options)},
                       {"exit_conditions", Json::array({{{"kind", "max_iterations"},
                                                         {"limit", options.num_inference_steps}}})}};
  Json chain;
  chain["steps"] = Json::array({{{"id", "denoise"}, {"component", "denoiser"}, {"executor_info", {{"ort", std::move(execution)}}}}});
  chain["executor_info"]["ort"]["outputs"]["latents"] =
      {{"kind", "model_output"}, {"step", "denoise"}, {"output", options.latent_output}};
  return chain;
}

}  // namespace onnxruntime::test::stateful
