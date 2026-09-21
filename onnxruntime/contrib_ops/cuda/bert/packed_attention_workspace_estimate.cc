// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_CONTRIB_OPS) && !defined(BUILD_CUDA_EP_AS_PLUGIN)

// Must precede the dual-world adapter header so this translation unit gets the
// shared-provider bridge's Node declaration.
#include "core/providers/shared_library/provider_api.h"

#include "contrib_ops/cuda/bert/packed_attention_workspace_estimate.h"

#include <algorithm>
#include <limits>
#include <string>

#include "core/common/float16.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/mha_runner.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

bool HasExactQkvHiddenSizes(const PackedAttentionWorkspaceEstimateConfig& config) {
  return config.qkv_hidden_sizes_count == config.qkv_hidden_sizes.size();
}

bool IsValidConfig(const PackedAttentionWorkspaceEstimateConfig& config) {
  if ((config.element_size != 2 && config.element_size != 4) ||
      config.num_heads <= 0 ||
      config.num_heads > std::numeric_limits<int32_t>::max()) {
    return false;
  }

  if (config.op != PackedAttentionWorkspaceOperator::PackedAttention) {
    return config.qkv_hidden_sizes_count == 0;
  }

  if (config.qkv_hidden_sizes_count != 0 &&
      config.qkv_hidden_sizes_count != config.qkv_hidden_sizes.size()) {
    return false;
  }

  for (size_t i = 0; i < config.qkv_hidden_sizes_count; ++i) {
    if (config.qkv_hidden_sizes[i] < 0 ||
        config.qkv_hidden_sizes[i] > std::numeric_limits<int32_t>::max()) {
      return false;
    }
  }
  if (HasExactQkvHiddenSizes(config)) {
    const int64_t q = config.qkv_hidden_sizes[0];
    const int64_t k = config.qkv_hidden_sizes[1];
    const int64_t v = config.qkv_hidden_sizes[2];
    if (q != k || q % config.num_heads != 0 || v % config.num_heads != 0 ||
        q + k + v > std::numeric_limits<int32_t>::max()) {
      return false;
    }
  }
  return true;
}

bool HasZeroDimension(const PackedAttentionWorkspaceEstimateConfig& config) {
  return config.op == PackedAttentionWorkspaceOperator::PackedAttention &&
         HasExactQkvHiddenSizes(config) &&
         std::find(config.qkv_hidden_sizes.begin(), config.qkv_hidden_sizes.end(), int64_t{0}) !=
             config.qkv_hidden_sizes.end();
}

bool IsRequiredInputPresent(gsl::span<const WorkspaceInputShape> input_shapes,
                            size_t input_index) {
  return GetWorkspaceInputShape(input_shapes, input_index).GetState() !=
         WorkspaceInputShapeState::Missing;
}

const TensorShape* GetKnownShape(gsl::span<const WorkspaceInputShape> input_shapes,
                                 size_t input_index) {
  return GetWorkspaceInputShape(input_shapes, input_index).GetShape();
}

PackedAttentionShape ToPackedAttentionShape(const TensorShape& shape) {
  PackedAttentionShape result;
  const auto& dimensions = shape.GetDims();
  result.rank = dimensions.size();
  const size_t count = std::min(dimensions.size(), result.dimensions.size());
  for (size_t i = 0; i < count; ++i) {
    result.dimensions[i] = dimensions[i];
  }
  return result;
}

bool HasZeroDimension(gsl::span<const WorkspaceInputShape> input_shapes) {
  for (const auto& input_shape : input_shapes) {
    const TensorShape* shape = input_shape.GetShape();
    if (shape != nullptr &&
        std::find(shape->GetDims().begin(), shape->GetDims().end(), int64_t{0}) !=
            shape->GetDims().end()) {
      return true;
    }
  }

  return false;
}

bool HasValidPmhaLayoutPresence(gsl::span<const WorkspaceInputShape> input_shapes) {
  const TensorShape* query = GetKnownShape(input_shapes, 0);
  if (query == nullptr) {
    return false;
  }

  const bool has_key = GetWorkspaceInputShape(input_shapes, 1).GetState() !=
                       WorkspaceInputShapeState::Missing;
  const bool has_value = GetWorkspaceInputShape(input_shapes, 2).GetState() !=
                         WorkspaceInputShapeState::Missing;
  if (query->NumDimensions() == 4) {
    return !has_key && !has_value;
  }

  if (query->NumDimensions() == 2) {
    return has_key && has_value;
  }

  return false;
}

bool IsTrtRouteReachable(int32_t qk_head_size,
                         int32_t v_head_size,
                         PackedAttentionHeadSizeDomain head_size_domain,
                         int32_t sequence_length_bound,
                         int sm,
                         const AttentionKernelOptions& kernel_options) {
  if (sequence_length_bound <= 0) {
    return false;
  }

  const bool enable_flash_attention = kernel_options.UseTrtFlashAttention();
  if (head_size_domain == PackedAttentionHeadSizeDomain::Exact) {
    return qk_head_size == v_head_size &&
           (FusedMHARunnerFP16v2::IsSupported(
                sm, qk_head_size, 1, enable_flash_attention) ||
            FusedMHARunnerFP16v2::IsSupported(
                sm, qk_head_size, sequence_length_bound, enable_flash_attention));
  }

  const int32_t common_head_size_bound = std::min(qk_head_size, v_head_size);
  return FusedMHARunnerFP16v2::IsAnySupportedHeadSize(
             sm, common_head_size_bound, 1, enable_flash_attention) ||
         FusedMHARunnerFP16v2::IsAnySupportedHeadSize(
             sm, common_head_size_bound, sequence_length_bound, enable_flash_attention);
}

#if USE_MEMORY_EFFICIENT_ATTENTION
bool IsMemoryEfficientRouteReachable(
    int32_t qk_head_size,
    int32_t v_head_size,
    PackedAttentionHeadSizeDomain head_size_domain,
    int sm,
    bool is_half) {
  if (head_size_domain == PackedAttentionHeadSizeDomain::Exact) {
    return has_memory_efficient_attention(
        sm, is_half, false, qk_head_size, v_head_size);
  }

  return has_memory_efficient_attention_for_head_size_bounds(
      sm, is_half, false, qk_head_size, v_head_size);
}
#endif

#if USE_FLASH_ATTENTION
bool IsFlashRouteReachableForBounds(const PackedMultiHeadAttentionProblem& problem,
                                    const cudaDeviceProp& device_prop) {
  const int32_t common_head_size_bound =
      std::min(problem.qk_head_size, problem.v_head_size);
  if (common_head_size_bound <= 0) {
    return false;
  }

  return onnxruntime::flash::is_any_supported_head_size<MLFloat16>(
      device_prop, static_cast<size_t>(common_head_size_bound),
      static_cast<size_t>(problem.num_heads),
      static_cast<size_t>(problem.num_heads));
}
#endif

std::optional<PackedAttentionWorkspaceAggregate> EstimatePa(
    const PackedAttentionWorkspaceEstimateConfig& config,
    gsl::span<const WorkspaceInputShape> input_shapes,
    const cudaDeviceProp& device_prop,
    const AttentionKernelOptions& kernel_options) {
  for (size_t input_index = 0; input_index < 5; ++input_index) {
    if (!IsRequiredInputPresent(input_shapes, input_index)) {
      return std::nullopt;
    }
  }

  PackedAttentionInputShapes inputs;
  const TensorShape* input = GetKnownShape(input_shapes, 0);
  const TensorShape* weights = GetKnownShape(input_shapes, 1);
  const TensorShape* bias = GetKnownShape(input_shapes, 2);
  const TensorShape* token_offset = GetKnownShape(input_shapes, 3);
  const TensorShape* cumulative_sequence_length = GetKnownShape(input_shapes, 4);
  if (input == nullptr || weights == nullptr || bias == nullptr ||
      token_offset == nullptr || cumulative_sequence_length == nullptr) {
    return std::nullopt;
  }

  inputs.input = ToPackedAttentionShape(*input);
  inputs.weights = ToPackedAttentionShape(*weights);
  inputs.bias = ToPackedAttentionShape(*bias);
  inputs.token_offset = ToPackedAttentionShape(*token_offset);
  inputs.cumulative_sequence_length = ToPackedAttentionShape(*cumulative_sequence_length);
  inputs.element_size = config.element_size;
  inputs.num_heads = config.num_heads;
  inputs.qkv_hidden_sizes_count = config.qkv_hidden_sizes_count;
  inputs.qkv_hidden_sizes = config.qkv_hidden_sizes;

  const auto& attention_bias_shape = GetWorkspaceInputShape(input_shapes, 5);
  inputs.has_attention_bias =
      attention_bias_shape.GetState() != WorkspaceInputShapeState::Missing;
  if (inputs.has_attention_bias) {
    const TensorShape* shape = attention_bias_shape.GetShape();
    if (shape == nullptr) {
      return std::nullopt;
    }
    inputs.attention_bias = ToPackedAttentionShape(*shape);
  }

  const auto problem_result = BuildPackedAttentionProblem(inputs);
  if (!problem_result.status.IsOK()) {
    return std::nullopt;
  }

  const auto routes =
      GetPackedAttentionReachableBackendsForBounds(
          problem_result.problem,
          HasExactQkvHiddenSizes(config)
              ? PackedAttentionHeadSizeDomain::Exact
              : PackedAttentionHeadSizeDomain::UpperBound,
          device_prop, kernel_options);
  // A reachable route is intentionally evaluated at the original maximum
  // geometry. Current QKV, projection, and MEA accumulator formulas are
  // componentwise monotonic even when route eligibility is not.
  const auto aggregate =
      GetPackedAttentionWorkspaceAggregateForBounds(problem_result.problem, routes);
  return aggregate.status.IsOK()
             ? std::optional<PackedAttentionWorkspaceAggregate>{aggregate}
             : std::nullopt;
}

std::optional<PackedAttentionWorkspaceAggregate> EstimatePmha(
    const PackedAttentionWorkspaceEstimateConfig& config,
    gsl::span<const WorkspaceInputShape> input_shapes,
    const cudaDeviceProp& device_prop,
    const AttentionKernelOptions& kernel_options) {
  for (size_t input_index : {size_t{0}, size_t{4}, size_t{5}}) {
    if (!IsRequiredInputPresent(input_shapes, input_index)) {
      return std::nullopt;
    }
  }

  if (!HasValidPmhaLayoutPresence(input_shapes)) {
    return std::nullopt;
  }

  PackedMultiHeadAttentionInputShapes inputs;
  const TensorShape* query = GetKnownShape(input_shapes, 0);
  const TensorShape* token_offset = GetKnownShape(input_shapes, 4);
  const TensorShape* cumulative_sequence_length = GetKnownShape(input_shapes, 5);
  if (query == nullptr || token_offset == nullptr || cumulative_sequence_length == nullptr) {
    return std::nullopt;
  }

  inputs.query = ToPackedAttentionShape(*query);
  inputs.token_offset = ToPackedAttentionShape(*token_offset);
  inputs.cumulative_sequence_length = ToPackedAttentionShape(*cumulative_sequence_length);
  inputs.element_size = config.element_size;
  inputs.num_heads = config.num_heads;

  const auto copy_optional_shape =
      [&input_shapes](size_t input_index, bool& present, PackedAttentionShape& destination) {
        const auto& input_shape = GetWorkspaceInputShape(input_shapes, input_index);
        present = input_shape.GetState() != WorkspaceInputShapeState::Missing;
        if (!present) {
          return true;
        }

        const TensorShape* shape = input_shape.GetShape();
        if (shape == nullptr) {
          return false;
        }
        destination = ToPackedAttentionShape(*shape);
        return true;
      };

  if (!copy_optional_shape(1, inputs.has_key, inputs.key) ||
      !copy_optional_shape(2, inputs.has_value, inputs.value) ||
      !copy_optional_shape(3, inputs.has_bias, inputs.bias) ||
      !copy_optional_shape(6, inputs.has_attention_bias, inputs.attention_bias)) {
    return std::nullopt;
  }

  const auto problem_result = BuildPackedMultiHeadAttentionProblem(inputs);
  if (!problem_result.status.IsOK()) {
    return std::nullopt;
  }

  const auto routes =
      GetPackedMultiHeadAttentionReachableBackendsForBounds(
          problem_result.problem, device_prop, kernel_options);
  // Keep route recipes at the original maximum geometry; only the
  // reachability probes use smaller supported head witnesses.
  const auto aggregate =
      GetPackedMultiHeadAttentionWorkspaceAggregateForBounds(problem_result.problem, routes);
  return aggregate.status.IsOK()
             ? std::optional<PackedAttentionWorkspaceAggregate>{aggregate}
             : std::nullopt;
}

std::optional<PackedAttentionWorkspaceEstimateConfig> ConfigFromNode(const Node& node) {
  PackedAttentionWorkspaceEstimateConfig config;
  if (node.OpType() == "PackedAttention") {
    config.op = PackedAttentionWorkspaceOperator::PackedAttention;
  } else if (node.OpType() == "PackedMultiHeadAttention") {
    config.op = PackedAttentionWorkspaceOperator::PackedMultiHeadAttention;
  } else {
    return std::nullopt;
  }

  const auto& input_defs = node.InputDefs();
  if (input_defs.empty() || input_defs[0] == nullptr || !input_defs[0]->Exists()) {
    return std::nullopt;
  }

  const auto* type_proto = input_defs[0]->TypeAsProto();
  if (type_proto == nullptr || !type_proto->has_tensor_type()) {
    return std::nullopt;
  }

  switch (type_proto->tensor_type().elem_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      config.element_size = 2;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      config.element_size = 4;
      break;
    default:
      return std::nullopt;
  }

  bool found_num_heads = false;
  for (const auto& attr : node.GetAttributes()) {
    if (attr.first == "num_heads") {
      config.num_heads = attr.second.i();
      found_num_heads = true;
    } else if (attr.first == "qkv_hidden_sizes") {
      config.qkv_hidden_sizes_count = static_cast<size_t>(attr.second.ints_size());
      const size_t count = std::min(config.qkv_hidden_sizes_count,
                                    config.qkv_hidden_sizes.size());
      for (size_t i = 0; i < count; ++i) {
        config.qkv_hidden_sizes[i] = attr.second.ints(static_cast<int>(i));
      }
    }
  }

  return found_num_heads ? std::optional<PackedAttentionWorkspaceEstimateConfig>{config}
                         : std::nullopt;
}

}  // namespace

PackedAttentionBackendMask GetPackedAttentionReachableBackendsForBounds(
    const PackedAttentionProblem& problem,
    PackedAttentionHeadSizeDomain head_size_domain,
    const cudaDeviceProp& device_prop,
    const AttentionKernelOptions& kernel_options) {
  const int sm = device_prop.major * 10 + device_prop.minor;
  PackedAttentionBackendMask routes = PackedAttentionBackendMask::Unfused;
  const bool trt_candidate =
      problem.element_size == 2 &&
      kernel_options.UseTrtFusedAttention() &&
      !problem.has_attention_bias &&
      IsTrtRouteReachable(
          problem.qk_head_size, problem.v_head_size,
          head_size_domain, problem.sequence_length, sm, kernel_options);
  if (trt_candidate) {
    routes = routes | PackedAttentionBackendMask::Trt;
  }

  bool mea_feasible = false;
#if USE_MEMORY_EFFICIENT_ATTENTION
  // PackedAttention currently does not consult UseEfficientAttention().
  mea_feasible =
      (problem.has_attention_bias
           ? problem.sequence_length >= static_cast<int32_t>(4 * problem.element_size)
           : true) &&
      problem.element_size == 2 &&
      IsMemoryEfficientRouteReachable(
          problem.qk_head_size, problem.v_head_size,
          head_size_domain, sm, true);
#endif

  return mea_feasible
             ? routes | PackedAttentionBackendMask::MemoryEfficient
             : routes;
}

PackedAttentionBackendMask GetPackedMultiHeadAttentionReachableBackendsForBounds(
    const PackedMultiHeadAttentionProblem& problem,
    const cudaDeviceProp& device_prop,
    const AttentionKernelOptions& kernel_options) {
  const int sm = device_prop.major * 10 + device_prop.minor;

  PackedAttentionBackendMask routes = PackedAttentionBackendMask::Unfused;

  bool flash_feasible = false;
#if USE_FLASH_ATTENTION
  flash_feasible =
      problem.element_size == 2 &&
      kernel_options.UseFlashAttention() &&
      !problem.has_attention_bias &&
      IsFlashRouteReachableForBounds(problem, device_prop) &&
      (problem.qkv_format != PackedMultiHeadAttentionQkvFormat::Packed ||
       problem.sequence_length >= kernel_options.MinSeqLenForFlashAttentionPackedQkv());
#endif
  if (flash_feasible) {
    routes = routes | PackedAttentionBackendMask::Flash;
  }

  const bool trt_candidate =
      problem.element_size == 2 &&
      kernel_options.UseTrtFusedAttention() &&
      !problem.has_attention_bias &&
      IsTrtRouteReachable(
          problem.qk_head_size, problem.v_head_size,
          PackedAttentionHeadSizeDomain::UpperBound,
          problem.sequence_length, sm, kernel_options);
  if (trt_candidate) {
    routes = routes | PackedAttentionBackendMask::Trt;
  }

  bool mea_feasible = false;
#if USE_MEMORY_EFFICIENT_ATTENTION
  mea_feasible =
      kernel_options.UseEfficientAttention() &&
      (problem.has_attention_bias
           ? problem.sequence_length >= static_cast<int32_t>(4 * problem.element_size)
           : true) &&
      (problem.element_size == 2 ||
       problem.sequence_length >= kernel_options.MinSeqLenForEfficientAttentionFp32()) &&
      IsMemoryEfficientRouteReachable(
          problem.qk_head_size, problem.v_head_size,
          PackedAttentionHeadSizeDomain::UpperBound, sm,
          problem.element_size == 2);
#endif

  return mea_feasible
             ? routes | PackedAttentionBackendMask::MemoryEfficient
             : routes;
}

std::optional<PackedAttentionWorkspaceAggregate> EstimatePackedAttentionWorkspace(
    const PackedAttentionWorkspaceEstimateConfig& config,
    gsl::span<const WorkspaceInputShape> input_shapes,
    const cudaDeviceProp& device_prop,
    const AttentionKernelOptions& kernel_options) {
  if (!IsValidConfig(config)) {
    return std::nullopt;
  }

  // WorkspaceInputShape has no provenance: a zero may be an estimation hint,
  // not proof that runtime output is empty.
  if (HasZeroDimension(config) || HasZeroDimension(input_shapes)) {
    return std::nullopt;
  }

  if (config.op == PackedAttentionWorkspaceOperator::PackedAttention) {
    return EstimatePa(config, input_shapes, device_prop, kernel_options);
  }

  if (config.op == PackedAttentionWorkspaceOperator::PackedMultiHeadAttention) {
    return EstimatePmha(config, input_shapes, device_prop, kernel_options);
  }

  return std::nullopt;
}

std::optional<PackedAttentionWorkspaceAggregate> EstimatePackedAttentionWorkspace(
    const Node& node,
    gsl::span<const WorkspaceInputShape> input_shapes,
    const cudaDeviceProp& device_prop,
    const AttentionKernelOptions& kernel_options) {
  const auto config = ConfigFromNode(node);
  if (!config.has_value()) {
    return std::nullopt;
  }

  return EstimatePackedAttentionWorkspace(
      *config, input_shapes, device_prop, kernel_options);
}

void SetPackedAttentionWorkspaceRequirements(
    const PackedAttentionWorkspaceAggregate& estimate,
    InlinedVector<WorkspaceRequirement>& requirements) {
  requirements.clear();
  if (!estimate.status.IsOK() || estimate.total_workspace_bytes == 0) {
    return;
  }

  requirements.push_back(
      WorkspaceRequirement{estimate.total_workspace_bytes, /*slot_id=*/0,
                           /*alignment_bytes=*/kPackedAttentionWorkspaceAlignment});
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#endif  // !defined(DISABLE_CONTRIB_OPS) && !defined(BUILD_CUDA_EP_AS_PLUGIN)
