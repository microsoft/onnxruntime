// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {
struct SymbolInfo {
  int count{0};
  std::vector<int> input_indices;
  int dim_value{0};
};

struct EinsumTerm {
  std::map<std::string, std::vector<int>> symbol_to_indices;
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
  void AddSymbol(const std::string& symbol, int dim_value, int input_index);
  EinsumTerm ProcessTerm(const std::string& term, bool is_input, gsl::span<const int64_t> dims, int index = -1);
};

class EinsumProgram final : public Program<EinsumProgram> {
 public:
  EinsumProgram(int input_count, const EinsumEquation& parsed_equation)
      : Program{"Einsum"},
        input_count_(input_count),
        parsed_equation_{parsed_equation} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  // TODO: add uniform variables dynamically.
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"j_max", ProgramUniformVariableDataType::Uint32},
      {"output_size", ProgramUniformVariableDataType::Uint32});

 private:
  int input_count_;
  const EinsumEquation& parsed_equation_;
};

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
