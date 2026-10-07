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
  std::vector<int> input_indices;
  int64_t dim_value{0};
};

struct EinsumTerm {
  // Indices of the symbols in the term which cannot be negative.
  std::map<std::string, std::vector<size_t>> symbol_to_indices;
  // The index of the input tensor in the Einsum equation.
  // This is -1 for the output term.
  int input_index{-1};
};

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
  void AddSymbol(const std::string& symbol, int64_t dim_value, int input_index);
  EinsumTerm ProcessTerm(const std::string& term,
                         bool is_input,
                         gsl::span<const int64_t> dims,
                         int index = -1);
};

using EinsumInputSymbols = std::map<std::string, std::vector<int>>;
using EinsumSymbolIndices = std::map<std::string, std::vector<size_t>>;
#define WEBGPU_EINSUM_PROGRAM_CONFIG(F)               \
  F(size_t, input_count_)                             \
  F(EinsumInputSymbols, symbol_to_inputs_)            \
  F(std::vector<EinsumSymbolIndices>, input_indices_) \
  F(EinsumSymbolIndices, output_indices_)             \
  F(bool, is_scalar_)

struct EinsumProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_EINSUM_PROGRAM_CONFIG);
    Config(size_t input_count, EinsumEquation&& equation)
        : input_count_{input_count},
          output_indices_{std::move(equation.rhs_.symbol_to_indices)},
          is_scalar_{equation.output_dims.empty()} {
      for (auto& [symbol, info] : equation.symbol_to_info_) {
        symbol_to_inputs_.emplace(symbol, std::move(info.input_indices));
      }
      input_indices_.reserve(equation.lhs_.size());
      for (auto& term : equation.lhs_) input_indices_.push_back(std::move(term.symbol_to_indices));
    }
  };
  static constexpr std::string_view name = "Einsum";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32});
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
