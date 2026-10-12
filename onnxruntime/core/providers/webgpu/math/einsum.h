// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {
struct SymbolInfo {
  size_t count{0};
  int64_t dim_value{0};
};

using EinsumSymbolIndices = std::map<std::string, std::vector<size_t>>;
#define WEBGPU_EINSUM_TERM_CONFIG(F) F(EinsumSymbolIndices, symbol_to_indices)
WEBGPU_DECLARE_SPECIALIZATION(EinsumTerm, WEBGPU_EINSUM_TERM_CONFIG);
#undef WEBGPU_EINSUM_TERM_CONFIG

class EinsumEquation {
 public:
  EinsumEquation(const std::vector<const Tensor*>& inputs, const std::string& equation);
  std::vector<int64_t> output_dims;
  std::map<std::string, SymbolInfo> symbol_to_info_;
  std::vector<EinsumTerm> lhs_;
  EinsumTerm rhs_;

 private:
  bool has_ellipsis_{false};
  std::vector<int64_t> ellipsis_dims_;
  void AddSymbol(const std::string& symbol, int64_t dim_value);
  EinsumTerm ProcessTerm(const std::string& term,
                         bool is_input,
                         gsl::span<const int64_t> dims);
};

#define WEBGPU_EINSUM_PROGRAM_CONFIG(F)         \
  F(gsl::span<const EinsumTerm>, input_terms_)  \
  F(gsl::span<const EinsumTerm>, output_terms_) \
  F(bool, is_scalar_)

struct EinsumProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_EINSUM_PROGRAM_CONFIG);
  static Config MakeConfig(const EinsumEquation& equation) {
    Config result{};
    result.input_terms_ = gsl::span<const EinsumTerm>{equation.lhs_};
    result.output_terms_ = gsl::span<const EinsumTerm>{&equation.rhs_, 1};
    result.is_scalar_ = bool{equation.output_dims.empty()};

    return result;
  }
  static constexpr std::string_view name = "Einsum";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_EINSUM_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_EINSUM_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_EINSUM_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_EINSUM_PROGRAM_CONFIG

using EinsumProgram = ConfiguredProgram<EinsumProgramShader>;

class Einsum final : public WebGpuKernel {
 public:
  Einsum(const OpKernelInfo& info) : WebGpuKernel(info) {
    std::string equation;
    ORT_ENFORCE(info.GetAttr("equation", &equation).IsOK());
    equation_ = equation;
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  std::string equation_;
};

}  // namespace webgpu
}  // namespace onnxruntime
