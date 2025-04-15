// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/einsum.h"

#include <algorithm>
#include <regex>
#include <vector>

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_EINSUM_TYPED_KERNEL_DECL(version)                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      Einsum,                                                         \
      kOnnxDomain,                                                    \
      version,                                                        \
      float,                                                          \
      kWebGpuExecutionProvider,                                       \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), \
      Einsum);

WEBGPU_EINSUM_TYPED_KERNEL_DECL(12);

// Regular expressions for equation parsing.
static const std::regex symbol_pattern("[a-zA-Z]|\\.\\.\\.");
static const std::regex term_pattern("([a-zA-Z]|\\.\\.\\.)+");
static const std::regex lhs_pattern("(([a-zA-Z]|\\.\\.\\.)+,)*([a-zA-Z]|\\.\\.\\.)+");

// Helper function to remove all whitespaces in a given string.
std::string RemoveAllWhitespace(const std::string& str) {
  std::string result = str;
  result.erase(std::remove_if(result.begin(), result.end(), ::isspace), result.end());
  return result;
}

bool IsInteger(const std::string& s) {
  static const std::regex pattern(R"(^\d+$)");
  return std::regex_match(s, pattern);
}

EinsumEquation::EinsumEquation(const std::vector<const Tensor*>& inputs,
                               const std::string& raw_equation) {
  std::string lhs, rhs, equation = RemoveAllWhitespace(raw_equation);
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

  // Parse LHS terms.
  size_t pos = 0;
  size_t find;
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

  // Initialize RHS if not specified.
  if (rhs.empty()) {
    bool ellipsis_dim_calculated = false;
    for (const auto& pair : symbol_to_info_) {
      // Skip symbols that appear more than once (except underscore symbols)
      // or if we've already handled underscore symbols
      bool is_ellipsis_dim_symbol = IsInteger(pair.first);
      bool should_skip = ((!is_ellipsis_dim_symbol && pair.second.count != 1) ||
                          (is_ellipsis_dim_symbol && ellipsis_dim_calculated));

      if (should_skip) {
        continue;
      }

      if (IsInteger(pair.first)) {
        rhs += "...";
        ellipsis_dim_calculated = true;
      } else {
        rhs += pair.first;
      }
    }
  } else {
    if (!std::regex_match(rhs, term_pattern)) {
      ORT_THROW("Invalid RHS");
    }
  }

  // Compute output dims.
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

void EinsumEquation::AddSymbol(const std::string& symbol, int dim_value, int input_index) {
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

EinsumTerm EinsumEquation::ProcessTerm(const std::string& term,
                                       bool is_input,
                                       gsl::span<const int64_t> dims,
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
      std::sregex_iterator symbol_it(term.begin(), term.end(), symbol_pattern);
      std::sregex_iterator symbol_end;
      int64_t ellipsis_dim_length = rank - std::distance(symbol_it, symbol_end) + 1;
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
      // Add '0', '1', '2', '3', '4', etc to represent ellipsis dimensions.
      for (int j = 0; j < ellipsis_dims.size(); ++j) {
        std::string symbol_j = std::to_string(j);
        einsum_term.symbol_to_indices[symbol_j].push_back(i + j);
        AddSymbol(symbol_j, static_cast<int>(dims[next_dim++]), index);
      }
    } else {
      einsum_term.symbol_to_indices[symbol].push_back(
          i + (has_ellipsis_ ? static_cast<int>(ellipsis_dims_.size()) - 1 : 0));
      AddSymbol(symbol, static_cast<int>(dims[next_dim++]), index);
    }
  }
  return einsum_term;
}

Status EinsumProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Add inputs and output.
  const ShaderVariableHelper& input0 = shader.AddInput("input0", ShaderUsage::UseUniform);

  std::vector<std::reference_wrapper<const ShaderVariableHelper>> inputs;
  inputs.push_back(input0);

  for (int i = 1; i < input_count_; ++i) {
    inputs.push_back(shader.AddInput("input" + std::to_string(i), ShaderUsage::UseUniform));
  }

  const ShaderVariableHelper& output =
      shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  // Helper variables for shader generation.
  std::vector<std::string> idx_copy;
  std::string init_prod = "var prod = 1.0;";
  std::string init_sum = "var sum = 0.0;";
  std::string update_sum = "sum += prod;";
  std::vector<std::string> reduce_ops_set_indices;
  std::vector<std::string> reduce_ops_loop_headers;
  std::vector<std::string> reduce_ops_loop_footers;
  std::vector<std::string> reduce_op_compute;
  bool is_reduce_ops_without_loop =
      parsed_equation_.symbol_to_info_.size() == parsed_equation_.rhs_.symbol_to_indices.size();

  for (const auto& pair : parsed_equation_.symbol_to_info_) {
    const std::string& symbol = pair.first;
    const SymbolInfo& info = pair.second;

    if (parsed_equation_.rhs_.symbol_to_indices.find(symbol) !=
        parsed_equation_.rhs_.symbol_to_indices.end()) {
      // Process output dimensions.
      auto rhs_indices = parsed_equation_.rhs_.symbol_to_indices.find(symbol);
      if (rhs_indices != parsed_equation_.rhs_.symbol_to_indices.end() &&
          !rhs_indices->second.empty()) {
        int output_index = rhs_indices->second[0];
        int lhs_term_index = 0;
        for (const auto& term : parsed_equation_.lhs_) {
          if (std::find(info.input_indices.begin(), info.input_indices.end(), lhs_term_index) ==
              info.input_indices.end()) {
            lhs_term_index++;
            continue;
          }

          auto it = term.symbol_to_indices.find(symbol);
          if (it == term.symbol_to_indices.end()) {
            ORT_THROW("Invalid symbol error");
          }

          for (auto input_index : it->second) {
            idx_copy.push_back(inputs[lhs_term_index].get().IndicesSet(
                "input" + std::to_string(lhs_term_index) + "Indices", std::to_string(input_index),
                output.IndicesGet("outputIndices", std::to_string(output_index))));
          }

          lhs_term_index++;
        }
      }
    } else {
      // Process reduction dimensions.
      int rhs_term_index = 0;
      for (const auto& term : parsed_equation_.lhs_) {
        if (std::find(info.input_indices.begin(), info.input_indices.end(), rhs_term_index) ==
            info.input_indices.end()) {
          rhs_term_index++;
          continue;
        }

        auto it = term.symbol_to_indices.find(symbol);
        if (it == term.symbol_to_indices.end()) {
          ORT_THROW("Invalid symbol error");
        }

        for (auto input_index : it->second) {
          reduce_ops_set_indices.push_back(inputs[rhs_term_index].get().IndicesSet(
              "input" + std::to_string(rhs_term_index) + "Indices", std::to_string(input_index),
              symbol));
        }

        std::string get_indices_str = "prod *= " +
                                      inputs[rhs_term_index].get().GetByIndices(
                                          "input" + std::to_string(rhs_term_index) + "Indices") +
                                      ";";
        if (std::find(reduce_op_compute.begin(), reduce_op_compute.end(), get_indices_str) ==
            reduce_op_compute.end()) {
          reduce_op_compute.push_back(get_indices_str);
        }

        rhs_term_index++;
      }

      reduce_ops_loop_headers.push_back("for(var " + symbol + ": u32 = 0; " + symbol + " < " +
                                        std::to_string(info.dim_value) + "; " + symbol + "++) {");
      reduce_ops_loop_footers.push_back("}");
    }
  }

  std::vector<std::string> reduce_ops = idx_copy;

  // Generate shader code based on reduction type.
  if (is_reduce_ops_without_loop) {
    // Direct multiplication without reduction loops.
    std::string sum_statement = "let sum = " + inputs[0].get().GetByIndices("input0Indices");
    for (int i = 1; i < inputs.size(); ++i) {
      sum_statement +=
          " * " + inputs[i].get().GetByIndices("input" + std::to_string(i) + "Indices");
    }

    sum_statement += ";";

    reduce_ops.push_back(sum_statement);
  } else {
    // Reduction operation with loops.
    reduce_ops.push_back(init_sum);
    for (const auto& header : reduce_ops_loop_headers) {
      reduce_ops.push_back(header);
    }
    for (const auto& set_idx : reduce_ops_set_indices) {
      reduce_ops.push_back(set_idx);
    }
    reduce_ops.push_back(init_prod);
    for (const auto& compute : reduce_op_compute) {
      reduce_ops.push_back(compute);
    }
    reduce_ops.push_back(update_sum);
    for (const auto& footer : reduce_ops_loop_footers) {
      reduce_ops.push_back(footer);
    }
  }

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size");
  shader.MainFunctionBody() << "var outputIndices = " << output.OffsetToIndices("global_idx")
                            << ";\n";

  // Define input indices with appropriate types.
  for (int i = 0; i < input_count_; i++) {
    const auto& input = inputs[i].get();
    int rank = input.Rank();

    // Construct WGSL type string based on rank.
    std::string indices_type;
    if (rank < 2) {
      indices_type = "u32";
    } else if (rank <= 4) {
      indices_type = "vec" + std::to_string(rank) + "<u32>";
    } else {
      indices_type = "array<u32, " + std::to_string(rank) + ">";
    }

    shader.MainFunctionBody() << "var input" << i << "Indices: " << indices_type << ";\n";
  }

  // Add reduce operations.
  for (const auto& op : reduce_ops) {
    shader.MainFunctionBody() << op << "\n";
  }

  // Set output value.
  shader.MainFunctionBody() << output.SetByOffset("global_idx", "sum") << "\n";

  return Status::OK();
}

Status Einsum::ComputeInternal(ComputeContext& context) const {
  if (context.InputCount() < 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Einsum requires at least one input tensor.");
  }

  std::vector<const Tensor*> input_tensors;
  for (int i = 0; i < context.InputCount(); ++i) {
    input_tensors.push_back(context.Input<Tensor>(i));
  }

  EinsumEquation equation(input_tensors, equation_);
  const std::vector<int64_t>& output_dims = equation.output_dims;
  Tensor* Y = context.Output(0, output_dims);
  int64_t output_size = Y->Shape().Size();
  if (output_size == 0) {
    return Status::OK();
  }

  // Create program with input count and the parsed equation.
  EinsumProgram program{context.InputCount(), equation};

  for (int i = 0; i < input_tensors.size(); ++i) {
    program.AddInput({input_tensors[i], ProgramTensorMetadataDependency::TypeAndRank});
  }

  // Add output and base uniforms.
  program.CacheHint(equation_)
      .SetDispatchGroupSize(static_cast<uint32_t>((output_size + 63) / 64))
      .AddOutput({Y, ProgramTensorMetadataDependency::TypeAndRank})
      .AddUniformVariables({static_cast<uint32_t>(output_size)});

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
