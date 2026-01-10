// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/einsum.h"

#include <algorithm>
#include <regex>
#include <set>
#include <vector>

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

namespace {
// Regular expressions for equation parsing.
static const std::regex symbol_pattern("[a-zA-Z]|\\.\\.\\.");
static const std::regex term_pattern("([a-zA-Z]|\\.\\.\\.)+");
// Term can be empty in some cases like ,...i->...i, so allow empty term here.
static const std::regex lhs_pattern("(([a-zA-Z]|\\.\\.\\.)*,)*([a-zA-Z]|\\.\\.\\.)*");

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
}  // namespace

#define WEBGPU_EINSUM_KERNEL_DECL(version)                                            \
  ONNX_OPERATOR_KERNEL_EX(                                                            \
      Einsum, kOnnxDomain, version, kWebGpuExecutionProvider,                         \
      (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()), \
      Einsum);

WEBGPU_EINSUM_KERNEL_DECL(12);

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
    if (!term.empty() && !std::regex_match(term, term_pattern)) {
      ORT_THROW("Invalid LHS term");
    }
    auto dims = inputs[input_idx]->Shape().GetDims();
    lhs_.push_back(ProcessTerm(term, true, dims, input_idx));
    pos = find + 1;
    input_idx++;
  }
  auto last_term = lhs.substr(pos);
  if (!last_term.empty() && !std::regex_match(last_term, term_pattern)) {
    ORT_THROW("Invalid LHS term");
  }
  auto dims = inputs[input_idx]->Shape().GetDims();
  lhs_.push_back(ProcessTerm(last_term, true, dims, input_idx));

  if (!rhs.empty() && !std::regex_match(rhs, term_pattern)) {
    ORT_THROW("Invalid RHS term");
  }

  // Handle empty RHS differently for implicit vs explicit modes.
  // Implicit mode - arrow is not in the equation where the equation "ij,jk" equals to "ij,jk->ik"
  // which is actually a matrix multiplication.
  // Explicit mode - arrow is in the equation where the equation "ij,jk->" contains two steps, first
  // step is a matrix multiplication just like the implicit mode, and the second step is to sum up
  // the matrix produced by the first step to a scalar.
  bool is_implicit_mode = arrow_pos == std::string::npos;
  if (rhs.empty() && is_implicit_mode) {
    // Implicit mode without RHS specified - construct output with repeated symbols
    bool ellipsis_dim_calculated = false;
    for (const auto& pair : symbol_to_info_) {
      // Skip when symbol appears multiple times (except ellipsis dimensions)
      // or when ellipsis dimensions have already been processed.
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

void EinsumEquation::AddSymbol(const std::string& symbol, int64_t dim_value, int input_index) {
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

EinsumTerm EinsumEquation::ProcessTerm(const std::string& term, bool is_input,
                                       gsl::span<const int64_t> dims, int index) {
  EinsumTerm einsum_term;
  einsum_term.input_index = index;

  // If the term is empty, return the einsum_term with empty symbol_to_indices.
  // This is important for the case where the equation contains scalar like ",i...,->i...", in which
  // case the term is empty. We need the term to generate the correct shader code.
  if (term.empty()) {
    return einsum_term;
  }

  const size_t rank = dims.size();
  bool ellipsis = false;
  std::vector<int64_t> ellipsis_dims;
  size_t next_dim = 0;

  std::sregex_iterator it(term.begin(), term.end(), symbol_pattern);
  std::sregex_iterator end;
  for (size_t i = 0; it != end; ++it, ++i) {
    std::string symbol = it->str();
    if (symbol == "...") {
      if (ellipsis) {
        ORT_THROW("Only one ellipsis is allowed per input term");
      }
      ellipsis = true;
      std::sregex_iterator symbol_it(term.begin(), term.end(), symbol_pattern);
      std::sregex_iterator symbol_end;
      size_t symbol_distance = std::distance(symbol_it, symbol_end) - 1;
      if (rank < symbol_distance) {
        ORT_THROW("Ellipsis out of bounds");
      }
      ellipsis_dims.assign(dims.begin() + next_dim,
                           dims.begin() + (next_dim + rank - symbol_distance));
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
      for (size_t j = 0; j < ellipsis_dims.size(); ++j) {
        std::string symbol_j = std::to_string(j);
        einsum_term.symbol_to_indices[symbol_j].push_back(i + j);
        AddSymbol(symbol_j, dims[next_dim++], index);
      }
    } else {
      einsum_term.symbol_to_indices[symbol].push_back(
          i + (has_ellipsis_ ? ellipsis_dims_.size() - 1 : 0));
      AddSymbol(symbol, dims[next_dim++], index);
    }
  }
  return einsum_term;
}

Status EinsumProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Add inputs and output.
  const ShaderVariableHelper& input0 =
      shader.AddInput("input0", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);

  std::vector<std::reference_wrapper<const ShaderVariableHelper>> inputs;
  inputs.push_back(input0);

  for (size_t i = 1; i < input_count_; ++i) {
    inputs.push_back(shader.AddInput("input" + std::to_string(i),
                                     ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias));
  }

  const ShaderVariableHelper& output =
      shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  // Helper variables for shader generation.
  std::string init_prod = "var prod = output_element_t(1);";
  std::string init_sum = "var sum = output_element_t(0);";
  std::string update_sum = "sum += prod;";
  std::vector<std::string> idx_copy;
  std::vector<std::string> reduce_ops;
  std::vector<std::string> reduce_ops_set_indices;
  std::vector<std::string> reduce_ops_loop_headers;
  std::vector<std::string> reduce_ops_loop_footers;
  std::vector<std::string> reduce_op_compute;
  bool is_reduce_ops_without_loop =
      parsed_equation_.symbol_to_info_.size() == parsed_equation_.rhs_.symbol_to_indices.size();
  std::set<std::string> uniform_symbol_set;
  for (const auto& pair : parsed_equation_.symbol_to_info_) {
    const std::string& symbol = pair.first;
    const SymbolInfo& info = pair.second;
    if (parsed_equation_.rhs_.symbol_to_indices.find(symbol) !=
        parsed_equation_.rhs_.symbol_to_indices.end()) {
      // Find the indices in the right-hand side (output) term for the current symbol
      auto rhs_indices = parsed_equation_.rhs_.symbol_to_indices.find(symbol);
      // Skip if symbol doesn't appear in output or has no indices
      // This means this symbol is not needed for output calculation
      if (rhs_indices == parsed_equation_.rhs_.symbol_to_indices.end() ||
          rhs_indices->second.empty()) {
        continue;
      }

      int lhs_term_index = 0;
      for (const auto& term : parsed_equation_.lhs_) {
        // Skip if the current input tensor index is not associated with this symbol
        // This check ensures we only process input indices that actually have this symbol.
        if (std::find(info.input_indices.begin(), info.input_indices.end(), lhs_term_index) ==
            info.input_indices.end()) {
          lhs_term_index++;
          continue;
        }

        auto it = term.symbol_to_indices.find(symbol);
        if (it == term.symbol_to_indices.end()) {
          ORT_THROW("Invalid symbol error");
        }

        // For each input index associated with the current symbol in this term
        for (auto input_index : it->second) {
          // Copy output indices to input indices for dimensions that appear in both input and
          // output Example: For equation "ij,jk->ik", when symbol='i', this copies the 'i' index
          // from output to input0 Format like: input0Indices[0] = outputIndices[0], for the 'i'
          // symbol
          idx_copy.push_back(inputs[lhs_term_index].get().IndicesSet(
              "input" + std::to_string(lhs_term_index) + "Indices", std::to_string(input_index),
              output.IndicesGet("outputIndices", std::to_string(rhs_indices->second[0]))));
        }

        lhs_term_index++;
      }
    } else {
      int lhs_term_index = 0;
      for (const auto& term : parsed_equation_.lhs_) {
        // Always construct the string for multiplying the input value to the product accumulator
        // Format like: prod *= get_input0_by_indices(input0Indices);
        std::string get_indices_str = "prod *= " +
                                      inputs[lhs_term_index].get().GetByIndices(
                                          "input" + std::to_string(lhs_term_index) + "Indices") +
                                      ";";

        // Only add this computation to reduce_op_compute if it hasn't been added before
        // This prevents duplicate multiplications for the same input term since the same symbol
        // can appear in multiple terms.
        if (std::find(reduce_op_compute.begin(), reduce_op_compute.end(), get_indices_str) ==
            reduce_op_compute.end()) {
          reduce_op_compute.push_back(get_indices_str);
        }

        // Skip if the current input tensor index is not associated with this symbol
        // This check ensures we only process input indices that actually have this symbol.
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
          // Set the input indices for the current input tensor at the given input_index position
          // Format like: input0Indices[1] = j, given equation "ij,jk->ik".
          reduce_ops_set_indices.push_back(inputs[lhs_term_index].get().IndicesSet(
              "input" + std::to_string(lhs_term_index) + "Indices", std::to_string(input_index),
              symbol));

          // Check if we've already processed this symbol to avoid duplicate loop generation
          if (uniform_symbol_set.find(symbol) == uniform_symbol_set.end()) {
            // Add symbol to tracked set to prevent duplicate processing
            uniform_symbol_set.insert(symbol);

            // Generate a WGSL loop header for reduction over this dimension
            // Format like: for(var j: u32 = 0; j < uniforms.input0_shape[1]; j++) {, given equation
            // "ij,jk->ik".
            reduce_ops_loop_headers.push_back("for(var " + symbol + ": u32 = 0; " + symbol + " < " +
                                              "uniforms.input" + std::to_string(lhs_term_index) +
                                              "_shape[" + std::to_string(input_index) + "]; " +
                                              symbol + "++) {");

            // Add corresponding loop closing brace
            reduce_ops_loop_footers.push_back("}");
          }
        }

        lhs_term_index++;
      }
    }
  }

  // Generate shader code based on reduction type.
  if (is_reduce_ops_without_loop) {
    // Direct multiplication without reduction loops.
    std::string sum_statement = "let sum = " + inputs[0].get().GetByIndices("input0Indices");
    for (size_t i = 1; i < inputs.size(); ++i) {
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

  // Add safety check to ensure workgroup sizes don't exceed output tensor dimensions
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size");

  // Special handling for scalar output
  bool is_scalar_output = parsed_equation_.output_dims.empty();
  if (is_scalar_output) {
    // For scalar output, only process the first workgroup thread. This is a special case where the
    // output is a single scalar value. The global index is set to 0, and the rest of the threads
    // are ignored. This is important for the case where the equation is finally reduced to a
    // scalar. For example, the equation "ij->" is a matrix summation and the output is a scalar,
    // the shader code will only execute for the first workgroup thread. There may be some space for
    // optimization here.
    shader.MainFunctionBody() << "if (global_idx != 0u) { return; }\n";
  } else {
    // Convert global linear index to N-dimensional indices for the output tensor
    // This maps a 1D global thread ID to the corresponding N-D output tensor coordinates
    shader.MainFunctionBody() << "var outputIndices = " << output.OffsetToIndices("global_idx")
                              << ";\n";
  }

  // Define input indices with appropriate types.
  for (size_t i = 0; i < input_count_; i++) {
    shader.MainFunctionBody() << "var input" << i << "Indices: input" << std::to_string(i)
                              << "_indices_t;\n";
  }

  // Copy output indices to input indices.
  for (const auto& idx : idx_copy) {
    shader.MainFunctionBody() << idx << "\n";
  }

  // Add reduce operations.
  for (const auto& op : reduce_ops) {
    shader.MainFunctionBody() << op << "\n";
  }

  // Handle output value assignment based on the output type (scalar or tensor)
  if (is_scalar_output) {
    // For scalar output, write the sum to the first (and only) output element at offset 0
    shader.MainFunctionBody() << output.SetByOffset("0", "sum") << "\n";
  } else {
    // For tensor output, write the sum to the output element at the current global thread index
    // This maps each thread's result to the corresponding position in the output tensor
    shader.MainFunctionBody() << output.SetByOffset("global_idx", "sum") << "\n";
  }

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

  // TODO: The EinsumEquation initialization could potentially be done during model loading
  // based on input/output shape inference results. This would improve runtime performance
  // by avoiding redundant initialization on every compute call.
  EinsumEquation equation(input_tensors, equation_);
  const std::vector<int64_t>& output_dims = equation.output_dims;
  Tensor* Y = context.Output(0, output_dims);
  int64_t output_size = Y->Shape().Size();
  if (output_size == 0) {
    return Status::OK();
  }

  // Create program with input count and the parsed equation.
  EinsumProgram program{input_tensors.size(), equation};

  for (size_t i = 0; i < input_tensors.size(); ++i) {
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
