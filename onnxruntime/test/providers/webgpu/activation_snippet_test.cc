// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cctype>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace webgpu {
namespace test {

namespace {

// Extract numeric literals that are not part of identifiers.
std::vector<std::string> ExtractNumericLiterals(const std::string& wgsl) {
  std::vector<std::string> literals;
  for (size_t i = 0; i < wgsl.size();) {
    const bool starts_number =
        (std::isdigit(static_cast<unsigned char>(wgsl[i])) != 0) ||
        (wgsl[i] == '-' && i + 1 < wgsl.size() && std::isdigit(static_cast<unsigned char>(wgsl[i + 1])) != 0);
    const bool part_of_identifier =
        i > 0 && (std::isalnum(static_cast<unsigned char>(wgsl[i - 1])) != 0 || wgsl[i - 1] == '_');
    if (!starts_number || part_of_identifier) {
      i++;
      continue;
    }
    size_t end = i + 1;
    while (end < wgsl.size() && (std::isdigit(static_cast<unsigned char>(wgsl[end])) != 0 || wgsl[end] == '.' ||
                                 wgsl[end] == 'e' || wgsl[end] == 'E' ||
                                 ((wgsl[end] == '+' || wgsl[end] == '-') &&
                                  (wgsl[end - 1] == 'e' || wgsl[end - 1] == 'E')))) {
      end++;
    }
    literals.push_back(wgsl.substr(i, end - i));
    i = end;
  }
  return literals;
}

// Count mantissa digits, excluding sign, exponent, decimal point, and edge zeros.
size_t CountSignificantDigits(const std::string& literal) {
  const std::string mantissa = literal.substr(0, literal.find_first_of("eE"));
  std::string digits;
  for (char c : mantissa) {
    if (std::isdigit(static_cast<unsigned char>(c)) != 0) {
      digits.push_back(c);
    }
  }
  const size_t first = digits.find_first_not_of('0');
  if (first == std::string::npos) {
    return 0;
  }
  return digits.find_last_not_of('0') - first + 1;
}

// Render with round-trip precision.
std::string RenderFullPrecision(float value) {
  std::ostringstream oss;
  oss << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
  return oss.str();
}

Activation MakeActivation(ActivationKind kind, float param_0 = 0.0f, float param_1 = 0.0f) {
  Activation activation;
  activation.activation_kind_ = kind;
  activation.activation_params_.values_[0] = param_0;
  activation.activation_params_.values_[1] = param_1;
  return activation;
}

const std::vector<ActivationKind>& AllActivationKinds() {
  static const std::vector<ActivationKind> kinds{
      ActivationKind::Relu, ActivationKind::Sigmoid, ActivationKind::Clip,
      ActivationKind::HardSigmoid, ActivationKind::LeakyRelu, ActivationKind::Tanh,
      ActivationKind::QuickGelu, ActivationKind::HardSwish, ActivationKind::Elu,
      ActivationKind::Gelu, ActivationKind::GeluTanh, ActivationKind::Softplus,
      ActivationKind::ThresholdedRelu, ActivationKind::Erf};
  return kinds;
}

}  // namespace

// Check emitted text because the truncation is below reliable inference tolerances.
TEST(WebGpuActivationSnippetTest, ConstantsAreEmittedAtFullPrecision) {
  for (ActivationKind kind : AllActivationKinds()) {
    const Activation activation = MakeActivation(kind, 0.125f, 0.375f);
    for (const std::string& wgsl : {GetActivationSnippet(activation, "vec4<f32>", "f32"),
                                    GetActivationDeclaration(activation, "vec4<f32>", "f32")}) {
      for (const std::string& literal : ExtractNumericLiterals(wgsl)) {
        // A round-trippable literal needs enough digits to represent its parsed f32.
        const float parsed = std::strtof(literal.c_str(), nullptr);
        const std::string full_precision = RenderFullPrecision(parsed);
        EXPECT_GE(CountSignificantDigits(literal), CountSignificantDigits(full_precision))
            << "literal '" << literal << "' looks truncated (the value it parses to needs '"
            << full_precision << "') for activation kind " << static_cast<int>(kind);
      }
    }
  }
}

TEST(WebGpuActivationSnippetTest, KnownConstantsAreNotTruncated) {
  struct Expectation {
    ActivationKind kind;
    bool declaration;
    float value;
    const char* description;
  };
  const std::vector<Expectation> expectations{
      {ActivationKind::Gelu, false, 0.70710678118654752f, "Gelu 1/sqrt(2)"},
      {ActivationKind::Gelu, true, 0.3275911f, "erf r0"},
      {ActivationKind::Gelu, true, 1.061405429f, "erf r5"},
      {ActivationKind::Gelu, true, -1.453152027f, "erf r4"},
      {ActivationKind::GeluTanh, false, 0.035677408136300125f, "Gelu-tanh cubic coefficient"},
      {ActivationKind::GeluTanh, false, 0.79788456080286535f, "Gelu-tanh sqrt(2/pi)"},
      {ActivationKind::HardSwish, false, 1.0f / 6.0f, "HardSwish 1/6"},
  };

  for (const Expectation& expectation : expectations) {
    const Activation activation = MakeActivation(expectation.kind);
    const std::string wgsl = expectation.declaration
                                 ? GetActivationDeclaration(activation, "vec4<f32>", "f32")
                                 : GetActivationSnippet(activation, "vec4<f32>", "f32");

    bool found = false;
    for (const std::string& literal : ExtractNumericLiterals(wgsl)) {
      if (std::strtof(literal.c_str(), nullptr) == expectation.value) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << expectation.description << " was not emitted at full precision. WGSL was:\n"
                       << wgsl;
  }
}

TEST(WebGpuActivationSnippetTest, ParametersAreReadFromUniformsNotBaked) {
  const std::vector<ActivationKind> parameterized{
      ActivationKind::Clip, ActivationKind::HardSigmoid, ActivationKind::LeakyRelu,
      ActivationKind::Elu, ActivationKind::ThresholdedRelu};

  for (ActivationKind kind : parameterized) {
    const Activation activation = MakeActivation(kind, 0.13579f, 0.24680f);
    const std::string wgsl = GetActivationSnippet(activation, "vec4<f32>", "f32");

    EXPECT_NE(wgsl.find("uniforms.activation_param_0"), std::string::npos)
        << "kind " << static_cast<int>(kind) << " does not read its parameter from a uniform: " << wgsl;
    EXPECT_EQ(wgsl.find("0.13579"), std::string::npos) << "parameter value baked into shader: " << wgsl;
    EXPECT_EQ(wgsl.find("0.2468"), std::string::npos) << "parameter value baked into shader: " << wgsl;
  }
}

TEST(WebGpuActivationSnippetTest, QuickGeluUnitAlphaIsASeparateShaderAndCacheKey) {
  const Activation silu = MakeActivation(ActivationKind::QuickGelu, 1.0f);
  const Activation general = MakeActivation(ActivationKind::QuickGelu, 1.702f);

  EXPECT_TRUE(silu.HasUnitQuickGeluAlpha());
  EXPECT_FALSE(general.HasUnitQuickGeluAlpha());

  const std::string silu_wgsl = GetActivationSnippet(silu, "vec4<f32>", "f32");
  const std::string general_wgsl = GetActivationSnippet(general, "vec4<f32>", "f32");

  EXPECT_EQ(silu_wgsl.find("activation_param_0"), std::string::npos)
      << "SiLU must not read an alpha uniform: " << silu_wgsl;
  EXPECT_NE(general_wgsl.find("activation_param_0"), std::string::npos)
      << "non-unit alpha must read its alpha from a uniform: " << general_wgsl;
  EXPECT_NE(silu_wgsl, general_wgsl);

  EXPECT_NE(silu.ToString(), general.ToString());
}

TEST(WebGpuActivationSnippetTest, ParameterValuesDoNotAffectTheCacheKey) {
  const Activation leaky_a = MakeActivation(ActivationKind::LeakyRelu, 0.01f);
  const Activation leaky_b = MakeActivation(ActivationKind::LeakyRelu, 0.25f);
  EXPECT_EQ(leaky_a.ToString(), leaky_b.ToString());
  EXPECT_EQ(GetActivationSnippet(leaky_a, "vec4<f32>", "f32"),
            GetActivationSnippet(leaky_b, "vec4<f32>", "f32"));

  const Activation clip_a = MakeActivation(ActivationKind::Clip, 0.0f, 6.0f);
  const Activation clip_b = MakeActivation(ActivationKind::Clip, -1.0f, 1.0f);
  EXPECT_EQ(clip_a.ToString(), clip_b.ToString());
  EXPECT_EQ(GetActivationSnippet(clip_a, "vec4<f32>", "f32"),
            GetActivationSnippet(clip_b, "vec4<f32>", "f32"));

  EXPECT_NE(MakeActivation(ActivationKind::Relu).ToString(),
            MakeActivation(ActivationKind::Sigmoid).ToString());
}

TEST(WebGpuActivationSnippetTest, ParameterlessActivationsReadNoUniforms) {
  const std::vector<ActivationKind> parameterless{
      ActivationKind::None, ActivationKind::Relu, ActivationKind::Sigmoid, ActivationKind::Tanh,
      ActivationKind::HardSwish, ActivationKind::Gelu, ActivationKind::GeluTanh,
      ActivationKind::Softplus, ActivationKind::Erf};

  for (ActivationKind kind : parameterless) {
    const Activation activation = MakeActivation(kind, 7.25f, 9.5f);
    const std::string snippet = GetActivationSnippet(activation, "vec4<f32>", "f32");
    const std::string declaration = GetActivationDeclaration(activation, "vec4<f32>", "f32");
    EXPECT_EQ(snippet.find("uniforms.activation_param"), std::string::npos)
        << "kind " << static_cast<int>(kind) << " unexpectedly reads a parameter uniform: " << snippet;
    EXPECT_EQ(declaration.find("uniforms.activation_param"), std::string::npos)
        << "kind " << static_cast<int>(kind) << " helper unexpectedly reads a parameter uniform: " << declaration;
  }
}

// WGSL has no forward declarations; helper definitions must accompany their uses.
TEST(WebGpuActivationSnippetTest, HelperIsEmittedWheneverItIsCalled) {
  for (ActivationKind kind : AllActivationKinds()) {
    const Activation activation = MakeActivation(kind, 0.5f, 1.5f);
    const std::string snippet = GetActivationSnippet(activation, "vec4<f32>", "f32");
    const std::string declaration = GetActivationDeclaration(activation, "vec4<f32>", "f32");

    if (snippet.find("fused_act_erf") != std::string::npos) {
      EXPECT_NE(declaration.find("fn fused_act_erf"), std::string::npos)
          << "kind " << static_cast<int>(kind) << " calls fused_act_erf but declares no helper";
    }
    if (snippet.find("fused_act_tanh") != std::string::npos) {
      EXPECT_NE(declaration.find("fn fused_act_tanh"), std::string::npos)
          << "kind " << static_cast<int>(kind) << " calls fused_act_tanh but declares no helper";
    }
    if (!declaration.empty()) {
      const bool referenced = snippet.find("fused_act_erf") != std::string::npos ||
                              snippet.find("fused_act_tanh") != std::string::npos;
      EXPECT_TRUE(referenced) << "kind " << static_cast<int>(kind) << " emits an unused helper";
    }
  }
}

TEST(WebGpuActivationSnippetTest, EveryKindProducesASnippet) {
  for (ActivationKind kind : AllActivationKinds()) {
    const Activation activation = MakeActivation(kind, 0.5f, 1.5f);
    EXPECT_FALSE(GetActivationSnippet(activation, "vec4<f32>", "f32").empty())
        << "activation kind " << static_cast<int>(kind) << " produced no snippet";
  }
  EXPECT_TRUE(GetActivationSnippet(MakeActivation(ActivationKind::None), "vec4<f32>", "f32").empty());
}

}  // namespace test
}  // namespace webgpu
}  // namespace onnxruntime
