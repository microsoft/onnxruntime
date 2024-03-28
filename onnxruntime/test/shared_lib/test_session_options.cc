// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/abi_session_options_impl.h"
#include "gmock/gmock.h"

using namespace onnxruntime;

TEST(CApiTest, session_options_graph_optimization_level) {
  // Test set optimization level succeeds when valid level is provided.
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
}

TEST(CApiTest, session_options_deterministic_compute) {
  // Manual validation currently. Check that SetDeterministicCompute in abi_session_options.cc is hit.
  Ort::SessionOptions options;
  options.SetDeterministicCompute(true);
}

#if !defined(ORT_MINIMAL_BUILD) && !defined(ORT_EXTENDED_MINIMAL_BUILD) && !defined(ORT_NO_EXCEPTIONS)

TEST(CApiTest, session_options_oversized_affinity_string) {
  Ort::SessionOptions options;
  std::string long_affinity_str(onnxruntime::kMaxStrLen + 1, '0');
  try {
    options.AddConfigEntry(kOrtSessionOptionsConfigIntraOpThreadAffinities, long_affinity_str.c_str());
    ASSERT_TRUE(false) << "Creation of config should have thrown exception";
  } catch (const std::exception& ex) {
    ASSERT_THAT(ex.what(), testing::HasSubstr("Config value is longer than maximum length: "));
  }
}

#define TRANSFORMER_NAMES                                         \
  {                                                               \
    "ConstantFolding", "FuseConvBN", "FuseMatMulAdd",             \
        "FuseMatMulAddFusion", "FuseTransposes", "NopTransformer" \
  }

TEST(CApiTest, session_options_disable_transformers) {
  Ort::SessionOptions options;

  constexpr const char* disable_transformers[] = TRANSFORMER_NAMES;
  options.DisableRulesAndOptimizers(std::begin(disable_transformers), std::end(disable_transformers));

  const std::string disable_transformers_str[] = TRANSFORMER_NAMES;
  options.DisableRulesAndOptimizers(std::begin(disable_transformers_str), std::end(disable_transformers_str));

  constexpr std::string_view disable_transformers_sv[] = TRANSFORMER_NAMES;
  options.DisableRulesAndOptimizers(std::begin(disable_transformers_sv), std::end(disable_transformers_sv));

  const std::vector<std::string> disable_transformers_vec = {TRANSFORMER_NAMES};
  options.DisableRulesAndOptimizers(std::begin(disable_transformers_vec), std::end(disable_transformers_vec));

  constexpr std::array<std::string_view, std::size(disable_transformers)>
      disable_transformers_array = TRANSFORMER_NAMES;
  options.DisableRulesAndOptimizers(std::begin(disable_transformers_array), std::end(disable_transformers_array));

  const OrtSessionOptions* raw_options = options;
  ASSERT_EQ(std::size(disable_transformers), raw_options->value.disabled_rules_and_transformers.size());
}

#endif
