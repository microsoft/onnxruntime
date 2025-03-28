// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/einsum.h"
#include <regex>
#include <vector>
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_EINSUM_VERSIONED_KERNEL(start, end)           \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                         \
      Einsum,                                                \
      kOnnxDomain,                                          \
      start,                                                \
      end,                                                  \
      kWebGpuExecutionProvider,                             \
      (*KernelDefBuilder::Create())                         \
          .TypeConstraint("T", WebGpuSupportedNumberTypes()),\
      Einsum);

#define WEBGPU_EINSUM_KERNEL(version)                        \
  ONNX_OPERATOR_KERNEL_EX(                                   \
      Einsum,                                               \
      kOnnxDomain,                                          \
      version,                                              \
      kWebGpuExecutionProvider,                             \
      (*KernelDefBuilder::Create())                         \
          .TypeConstraint("T", WebGpuSupportedNumberTypes()),\
      Einsum);

WEBGPU_EINSUM_VERSIONED_KERNEL(12, 12)
WEBGPU_EINSUM_KERNEL(13)

namespace {
const std::regex symbol_pattern("[a-zA-Z]|\\.\\.\\.");
const std::regex term_pattern("([a-zA-Z]|\\.\\.\\.)+");
const std::regex lhs_pattern("(([a-zA-Z]|\\.\\.\\.)+,)*([a-zA-Z]|\\.\\.\\.)+");
}  // namespace

Einsum::EinsumEquation::EinsumEquation(const std::vector<const Tensor*>& inputs, const std::string& equation) {
  std::string lhs, rhs;
  size_t arrow_pos = equation.find("->");
  if (arrow_pos != std::string::npos) {
    lhs = equation.substr(0, arrow_pos);
    rhs = equation.substr(arrow_pos + 2);
  } else {
    lhs = equation;
    rhs = "";
  }

  if (!std::regex_match(lhs, lhs_pattern)) {
    ORT_THROW("Invalid LHS term");
  }

  // Parse LHS terms
  std::string::size_type pos = 0;
  std::string::size_type find;
  int input_idx = 0;
  while ((find = lhs.find(',', pos)) != std::string::npos) {
    auto term = lhs.substr(pos, find - pos);
    if (!std::regex_match(term, term_pattern)) {
      ORT_THROW("Invalid LHS term");
    }
    auto dims = inputs[input_idx]->Shape().GetDims();
    lhs_.push_back(ProcessTerm(term, true, dims, input_idx));
    pos = find + 1;
    input_idx++;
  }
  auto last_term = lhs.substr(pos);
  if (!std::regex_match(last_term, term_pattern)) {
    ORT_THROW("Invalid LHS term");
  }
  auto dims = inputs[input_idx]->Shape().GetDims();
  lhs_.push_back(ProcessTerm(last_term, true, dims, input_idx));

  // Initialize RHS if not specified
  if (rhs.empty()) {
    for (const auto& pair : symbol_to_info_) {
      if (pair.second.count == 1 || pair.first == "...") {
        rhs += pair.first;
      }
    }
  } else {
    if (!std::regex_match(rhs, term_pattern)) {
      ORT_THROW("Invalid RHS");
    }
  }

  // Compute output dims
  std::sregex_iterator it(rhs.begin(), rhs.end(), symbol_pattern);
  std::sregex_iterator end;
  for (; it != end; ++it) {
    std::string symbol = it->str();
    if (symbol == "...") {
      output_dims.insert(output_dims.end(), ellipsis_dims_.begin(), ellipsis_dims_.end());
    } else {
      auto info_it = symbol_to_info_.find(symbol);
      if (info_it == symbol_to_info_.end()) {
        ORT_THROW("Invalid RHS symbol");
      }
      output_dims.push_back(info_it->second.dim_value);
    }
  }

  rhs_ = ProcessTerm(rhs, false, output_dims);
}

void Einsum::EinsumEquation::AddSymbol(const std::string& symbol, int dim_value, int input_index) {
  auto it = symbol_to_info_.find(symbol);
  if (it != symbol_to_info_.end()) {
    if (it->second.dim_value != dim_value && it->second.count != 1) {
      ORT_THROW("Dimension mismatch");
    }
    it->second.count++;
    it->second.input_indices.push_back(input_index);
  } else {
    SymbolInfo info;
    info.count = 1;
    info.dim_value = dim_value;
    info.input_indices.push_back(input_index);
    symbol_to_info_[symbol] = info;
  }
}

Einsum::EinsumTerm Einsum::EinsumEquation::ProcessTerm(
    const std::string& term,
    bool is_input,
    const std::vector<int64_t>& dims,
    int index) {
  EinsumTerm einsum_term;
  einsum_term.input_index = index;

  const int64_t rank = static_cast<int64_t>(dims.size());
  bool ellipsis = false;
  std::vector<int64_t> ellipsis_dims;
  int64_t next_dim = 0;

  std::sregex_iterator it(term.begin(), term.end(), symbol_pattern);
  std::sregex_iterator end;
  int i = 0;
  for (; it != end; ++it, ++i) {
    std::string symbol = it->str();
    if (symbol == "...") {
      if (ellipsis) {
        ORT_THROW("Only one ellipsis is allowed per input term");
      }
      ellipsis = true;
      int64_t ellipsis_dim_length = rank - std::distance(std::sregex_iterator(term.begin(), term.end(), symbol_pattern)) + 1;
      if (ellipsis_dim_length < 0) {
        ORT_THROW("Ellipsis out of bounds");
      }
      ellipsis_dims.assign(dims.begin() + next_dim, dims.begin() + next_dim + ellipsis_dim_length);
      if (has_ellipsis_) {
        if (ellipsis_dims_ != ellipsis_dims) {
          ORT_THROW("Ellipsis dimensions mismatch");
        }
      } else if (is_input) {
        has_ellipsis_ = true;
        ellipsis_dims_ = ellipsis_dims;
      } else {
        ORT_THROW("Ellipsis must be specified in the LHS");
      }
      // Add '0', '1', '2', '3', '4', etc to represent ellipsis dimensions
      for (size_t j = 0; j < ellipsis_dims.size(); ++j) {
        std::string symbol_j(1, static_cast<char>('0' + j));
        einsum_term.symbol_to_indices[symbol_j].push_back(i + j);
        AddSymbol(symbol_j, static_cast<int>(dims[next_dim++]), index);
      }
    } else {
      einsum_term.symbol_to_indices[symbol].push_back(i + (has_ellipsis_ ? ellipsis_dims_.size() - 1 : 0));
      AddSymbol(symbol, static_cast<int>(dims[next_dim++]), index);
    }
  }
  return einsum_term;
}

Status EinsumProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Parse equation and prepare equation parser
  EinsumEquation equation(shader.GetInputs(), equation_);

  // Add all input tensors
  std::vector<ShaderVariableHelper> inputs;
  for (size_t i = 0; i < shader.GetInputs().size(); ++i) {
    inputs.push_back(shader.AddInput("input" + std::to_string(i), ShaderUsage::UseUniform));
  }

  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  // Initialize output indices
  shader.MainFunctionBody() << "  var outputIndices = " << output.offsetToIndices("global_idx") << ";\n";

  // Initialize variables for input indices
  for (size_t i = 0; i < inputs.size(); ++i) {
    shader.MainFunctionBody() << "  var input" << i << "Indices: " << inputs[i].type.indices << ";\n";
  }

  // Copy output indices to input indices for direct mapped dimensions
  for (const auto& symbol : equation.rhs_.symbol_to_indices) {
    const auto& outputIndex = symbol.second[0];
    for (const auto& term : equation.lhs_) {
      if (auto it = term.symbol_to_indices.find(symbol.first); it != term.symbol_to_indices.end()) {
        for (const auto& inputIndex : it->second) {
          shader.MainFunctionBody() << "  input" << term.input_index << "Indices[" << inputIndex << "] = outputIndices[" << outputIndex << "];\n";
        }
      }
    }
  }

  // Initialize reduction operations
  shader.MainFunctionBody() << "  var sum = output_value_t(0);\n";

  // Generate loops for reduced dimensions
  for (const auto& symbol_info : equation.symbol_to_info_) {
    if (!equation.rhs_.symbol_to_indices.contains(symbol_info.first)) {
      shader.MainFunctionBody() << "  for (var " << symbol_info.first << ": u32 = 0; "
                               << symbol_info.first << " < " << symbol_info.second.dim_value << "; "
                               << symbol_info.first << "++) {\n";
      // Set indices for reduced dimensions
      for (const auto& input_idx : symbol_info.second.input_indices) {
        const auto& term = equation.lhs_[input_idx];
        if (auto it = term.symbol_to_indices.find(symbol_info.first); it != term.symbol_to_indices.end()) {
          for (const auto& idx : it->second) {
            shader.MainFunctionBody() << "    input" << input_idx << "Indices[" << idx << "] = " << symbol_info.first << ";\n";
          }
        }
      }
    }
  }

  // Compute product of inputs
  shader.MainFunctionBody() << "  var prod = output_value_t(1);\n";
  for (size_t i = 0; i < inputs.size(); ++i) {
    shader.MainFunctionBody() << "  prod *= " << inputs[i].getByIndices("input" + std::to_string(i) + "Indices") << ";\n";
  }
  shader.MainFunctionBody() << "  sum += prod;\n";

  // Close reduction loops
  for (const auto& symbol_info : equation.symbol_to_info_) {
    if (!equation.rhs_.symbol_to_indices.contains(symbol_info.first)) {
      shader.MainFunctionBody() << "  }\n";
    }
  }

  // Write output
  shader.MainFunctionBody() << "  " << output.setByOffset("global_idx", "sum") << ";\n";

  return Status::OK();
}

Status Einsum::ComputeInternal(ComputeContext& context) const {
  if (context.InputCount() < 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Einsum requires at least one input tensor.");
  }

  std::vector<const Tensor*> input_tensors;
  for (int i = 0; i < context.InputCount(); ++i) {
    input_tensors.push_back(context.Input<Tensor>(i));
  }

  EinsumEquation equation(input_tensors, equation_);
  const std::vector<int64_t>& output_dims = equation.output_dims;
  int64_t output_size = std::accumulate(output_dims.begin(), output_dims.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());

  Tensor* Y = context.Output(0, output_dims);
  if (output_size == 0) {
    return Status::OK();
  }

  EinsumProgram program{equation_};

  for (size_t i = 0; i < input_tensors.size(); ++i) {
    program.AddInput({input_tensors[i], ProgramTensorMetadataDependency::Type});
  }

  program.SetDispatchGroupSize(static_cast<uint32_t>((output_size + 63) / 64))
      .AddOutput({Y, ProgramTensorMetadataDependency::Type})
      .AddUniformVariables({{static_cast<uint32_t>(output_size)}});  // output_size

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
