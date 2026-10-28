// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/einsum.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <set>
#include <vector>

#include "core/common/inlined_containers.h"
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
  std::erase_if(result, [](unsigned char c) { return std::isspace(c); });
  return result;
}

bool IsInteger(const std::string& s) {
  static const std::regex pattern(R"(^\d+$)");
  return std::regex_match(s, pattern);
}
}  // namespace

#define WEBGPU_EINSUM_KERNEL_DECL(start_version, end_version)                         \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                  \
      Einsum, kOnnxDomain, start_version, end_version, kWebGpuExecutionProvider,      \
      (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()), \
      Einsum);

WEBGPU_EINSUM_KERNEL_DECL(12, 27);

ONNX_OPERATOR_KERNEL_EX(
    Einsum,
    kOnnxDomain,
    28,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()),
    Einsum);

EinsumEquation::EinsumEquation(const std::vector<const Tensor*>& inputs,
                               const std::string& raw_equation) {
  const std::string equation = RemoveAllWhitespace(raw_equation);
  const size_t arrow = equation.find("->");
  const std::string lhs = equation.substr(0, arrow);
  std::string rhs = arrow == std::string::npos ? "" : equation.substr(arrow + 2);
  ORT_ENFORCE(std::regex_match(lhs, lhs_pattern), "Invalid LHS term");
  ORT_ENFORCE(rhs.empty() || std::regex_match(rhs, term_pattern), "Invalid RHS term");

  InlinedVector<std::string> terms;
  size_t start = 0;
  do {
    const size_t comma = lhs.find(',', start);
    terms.push_back(lhs.substr(start, comma == std::string::npos ? comma : comma - start));
    if (comma == std::string::npos) break;
    start = comma + 1;
  } while (true);
  ORT_ENFORCE(terms.size() == inputs.size(), "Einsum equation and input count mismatch");

  // Determine the common ellipsis rank before assigning right-aligned batch labels.
  size_t ellipsis_rank = 0;
  for (size_t i = 0; i < terms.size(); ++i) {
    size_t labels = 0;
    size_t ellipses = 0;
    for (std::sregex_iterator it(terms[i].begin(), terms[i].end(), symbol_pattern), end; it != end; ++it) {
      if (it->str() == "...")
        ++ellipses;
      else
        ++labels;
    }
    const size_t rank = inputs[i]->Shape().NumDimensions();
    ORT_ENFORCE(ellipses <= 1 && (ellipses ? rank >= labels : rank == labels),
                "Einsum term does not match input rank");
    if (ellipses != 0) {
      has_ellipsis_ = true;
      ellipsis_rank = std::max(ellipsis_rank, rank - labels);
    }
  }
  ellipsis_dims_.assign(ellipsis_rank, 1);
  for (size_t i = 0; i < terms.size(); ++i) {
    lhs_.push_back(ProcessTerm(terms[i], true, inputs[i]->Shape().GetDims(), narrow<int>(i)));
  }
  for (size_t i = 0; i < ellipsis_rank; ++i) {
    ellipsis_dims_[i] = symbol_to_info_.at(std::to_string(i)).dim_value;
  }

  if (arrow == std::string::npos) {
    if (has_ellipsis_) rhs = "...";
    for (const auto& [symbol, info] : symbol_to_info_) {
      if (!IsInteger(symbol) && info.count == 1) rhs += symbol;
    }
  }
  InlinedHashSet<std::string> output_labels;
  for (std::sregex_iterator it(rhs.begin(), rhs.end(), symbol_pattern), end; it != end; ++it) {
    const std::string symbol = it->str();
    ORT_ENFORCE(output_labels.insert(symbol).second, "Repeated RHS symbol");
    if (symbol == "...") {
      ORT_ENFORCE(has_ellipsis_, "Ellipsis must be specified in the LHS");
      output_dims.insert(output_dims.end(), ellipsis_dims_.begin(), ellipsis_dims_.end());
    } else {
      const auto info = symbol_to_info_.find(symbol);
      ORT_ENFORCE(info != symbol_to_info_.end(), "Invalid RHS symbol");
      output_dims.push_back(info->second.dim_value);
    }
  }
  rhs_ = ProcessTerm(rhs, false, output_dims);
}

void EinsumEquation::AddSymbol(const std::string& symbol, int64_t dim_value, int input_index) {
  auto [it, inserted] = symbol_to_info_.try_emplace(symbol);
  auto& info = it->second;
  if (inserted) {
    info.dim_value = dim_value;
  } else {
    ORT_ENFORCE(info.dim_value == dim_value || info.dim_value == 1 || dim_value == 1,
                "Einsum dimensions cannot broadcast");
    if (info.dim_value == 1) info.dim_value = dim_value;
  }
  ++info.count;
  info.input_indices.push_back(input_index);
}

EinsumTerm EinsumEquation::ProcessTerm(const std::string& term, bool is_input,
                                       gsl::span<const int64_t> dims, int index) {
  EinsumTerm result;
  result.input_index = index;
  size_t named_labels = 0;
  for (std::sregex_iterator it(term.begin(), term.end(), symbol_pattern), end; it != end; ++it) {
    if (it->str() != "...") ++named_labels;
  }
  ORT_ENFORCE(dims.size() >= named_labels, "Einsum term exceeds tensor rank");
  const size_t local_ellipsis_rank = dims.size() - named_labels;
  size_t dimension = 0;
  auto append = [&](const std::string& symbol) {
    auto& indices = result.symbol_to_indices[symbol];
    if (!indices.empty()) {
      ORT_ENFORCE(dims[indices.front()] == dims[dimension], "Einsum diagonal dimensions must match");
    }
    indices.push_back(dimension);
    if (is_input) AddSymbol(symbol, dims[dimension], index);
    ++dimension;
  };
  for (std::sregex_iterator it(term.begin(), term.end(), symbol_pattern), end; it != end; ++it) {
    const std::string symbol = it->str();
    if (symbol == "...") {
      for (size_t j = ellipsis_dims_.size() - local_ellipsis_rank; j < ellipsis_dims_.size(); ++j) {
        append(std::to_string(j));
      }
    } else {
      append(symbol);
    }
  }
  ORT_ENFORCE(dimension == dims.size(), "Einsum term does not match tensor rank");
  return result;
}

Status EinsumProgram::GenerateShaderCode(ShaderHelper& shader) const {
  InlinedVector<const ShaderVariableHelper*> inputs;
  inputs.reserve(input_count_);
  for (size_t i = 0; i < input_count_; ++i) {
    inputs.push_back(&shader.AddInput("input" + std::to_string(i),
                                      ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias));
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  auto& body = shader.MainFunctionBody();
  body << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size");
  if (!parsed_equation_.output_dims.empty()) {
    body << "let output_indices = " << output.OffsetToIndices("global_idx") << ";\n";
  }
  for (size_t i = 0; i < input_count_; ++i) {
    body << "var input" << i << "_indices: input" << i << "_indices_t;\n";
  }

  // Runtime extents keep the shader valid when a cached program's singleton dimensions change.
  auto input_extent = [&](size_t input, size_t dimension) {
    return inputs[input]->IndicesGet("uniforms.input" + std::to_string(input) + "_shape", narrow<int>(dimension));
  };
  auto assign_symbol = [&](const std::string& symbol, const std::string& value) {
    for (size_t i = 0; i < input_count_; ++i) {
      const auto found = parsed_equation_.lhs_[i].symbol_to_indices.find(symbol);
      if (found == parsed_equation_.lhs_[i].symbol_to_indices.end()) continue;
      for (size_t dimension : found->second) {
        body << inputs[i]->IndicesSet("input" + std::to_string(i) + "_indices", narrow<int>(dimension),
                                      "select(0u, " + value + ", " + input_extent(i, dimension) + " != 1u)")
             << "\n";
      }
    }
  };
  for (const auto& [symbol, indices] : parsed_equation_.rhs_.symbol_to_indices) {
    assign_symbol(symbol, output.IndicesGet("output_indices", narrow<int>(indices.front())));
  }

  body << "var sum = output_element_t(0);\n";
  size_t reduction_count = 0;
  for (const auto& [symbol, info] : parsed_equation_.symbol_to_info_) {
    ORT_UNUSED_PARAMETER(info);
    if (parsed_equation_.rhs_.symbol_to_indices.contains(symbol)) continue;
    std::string extent;
    for (size_t i = 0; i < input_count_; ++i) {
      const auto found = parsed_equation_.lhs_[i].symbol_to_indices.find(symbol);
      if (found == parsed_equation_.lhs_[i].symbol_to_indices.end()) continue;
      const auto next = input_extent(i, found->second.front());
      extent = extent.empty() ? next : "select(" + extent + ", " + next + ", " + extent + " == 1u)";
    }
    const std::string counter = "reduce_" + symbol;
    body << "for (var " << counter << " = 0u; " << counter << " < " << extent << "; " << counter << "++) {\n";
    assign_symbol(symbol, counter);
    ++reduction_count;
  }
  body << "var product = output_element_t(1);\n";
  for (size_t i = 0; i < input_count_; ++i) {
    body << "product *= " << inputs[i]->GetByIndices("input" + std::to_string(i) + "_indices") << ";\n";
  }
  body << "sum += product;\n";
  for (size_t i = 0; i < reduction_count; ++i) body << "}\n";
  body << output.SetByOffset("global_idx", "sum") << "\n";
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
