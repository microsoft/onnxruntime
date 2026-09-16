// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS) && !defined(BUILD_CUDA_EP_AS_PLUGIN)

// Must be first so this translation unit uses the shared-provider Node world.
#include "core/providers/shared_library/provider_api.h"

#include "contrib_ops/cuda/bert/group_query_attention_workspace_estimate.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>

#include "core/common/float16.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace {

// Positional indices from the Microsoft-domain GroupQueryAttention v1 schema.
constexpr size_t kQuery = 0;
constexpr size_t kKey = 1;
constexpr size_t kValue = 2;
constexpr size_t kPastKey = 3;
constexpr size_t kPastValue = 4;
constexpr size_t kSequenceLengths = 5;
constexpr size_t kTotalSequenceLength = 6;
constexpr size_t kCosCache = 7;
constexpr size_t kSinCache = 8;
constexpr size_t kPositionIds = 9;
constexpr size_t kAttentionBias = 10;
constexpr size_t kHeadSink = 11;
constexpr size_t kKScale = 12;
constexpr size_t kVScale = 13;
constexpr size_t kQNorm = 14;
constexpr size_t kKNorm = 15;

bool Present(gsl::span<const WorkspaceInputShape> shapes, size_t index) {
  return GetWorkspaceInputShape(shapes, index).GetState() != WorkspaceInputShapeState::Missing;
}

const TensorShape* Shape(gsl::span<const WorkspaceInputShape> shapes, size_t index) {
  return GetWorkspaceInputShape(shapes, index).GetShape();
}

bool PositiveDims(const TensorShape& shape) {
  return std::all_of(shape.GetDims().begin(), shape.GetDims().end(),
                     [](int64_t dim) { return dim > 0; });
}

bool PairPresent(gsl::span<const WorkspaceInputShape> shapes, size_t left, size_t right) {
  return Present(shapes, left) == Present(shapes, right);
}

std::optional<GQAKvQuantizationType> ParseQuantization(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  if (value == "NONE") return GQAKvQuantizationType::None;
  if (value == "PER_TENSOR") return GQAKvQuantizationType::PerTensor;
  if (value == "PER_CHANNEL") return GQAKvQuantizationType::PerChannel;
  return std::nullopt;
}

bool HasZeroDimension(gsl::span<const WorkspaceInputShape> shapes) {
  for (const auto& input : shapes) {
    const auto* shape = input.GetShape();
    if (shape != nullptr &&
        std::find(shape->GetDims().begin(), shape->GetDims().end(), int64_t{0}) !=
            shape->GetDims().end()) {
      return true;
    }
  }
  return false;
}

bool ValidateAuxiliaryShapes(const GQAWorkspaceEstimateConfig& config,
                             gsl::span<const WorkspaceInputShape> shapes,
                             int64_t batch_bound,
                             int64_t sequence_bound,
                             int64_t head_bound,
                             int64_t total_kv_bound) {
  const TensorShape* sequence_lengths = Shape(shapes, kSequenceLengths);
  const TensorShape* total = Shape(shapes, kTotalSequenceLength);
  if (sequence_lengths == nullptr || total == nullptr ||
      sequence_lengths->NumDimensions() == 0 ||
      !(total->NumDimensions() == 0 ||
        (total->NumDimensions() == 1 && (*total)[0] == 1))) {
    return false;
  }
  size_t batch_dimension_count = 0;
  for (int64_t dim : sequence_lengths->GetDims()) {
    if (dim != 1 && dim != batch_bound) return false;
    if (dim == batch_bound && batch_bound != 1) ++batch_dimension_count;
  }
  if (batch_bound != 1 && batch_dimension_count != 1) return false;

  if (!PairPresent(shapes, kCosCache, kSinCache) ||
      !PairPresent(shapes, kQNorm, kKNorm)) {
    return false;
  }
  if (config.do_rotary && !Present(shapes, kCosCache)) return false;
  if (Present(shapes, kCosCache)) {
    const auto* cos = Shape(shapes, kCosCache);
    const auto* sin = Shape(shapes, kSinCache);
    if (cos == nullptr || sin == nullptr || cos->NumDimensions() != 2 ||
        sin->NumDimensions() != 2 || !PositiveDims(*cos) || !PositiveDims(*sin)) {
      return false;
    }
    const int64_t rotary_width = (*cos)[1];
    if (rotary_width != (*sin)[1] || rotary_width % 8 != 0 ||
        rotary_width > head_bound / 2) {
      return false;
    }
  }
  if (Present(shapes, kQNorm)) {
    const auto* q = Shape(shapes, kQNorm);
    const auto* k = Shape(shapes, kKNorm);
    if (q == nullptr || k == nullptr || q->NumDimensions() != 1 ||
        k->NumDimensions() != 1 || (*q)[0] <= 0 || (*k)[0] <= 0 ||
        std::min((*q)[0], (*k)[0]) < head_bound) {
      return false;
    }
  }
  if (Present(shapes, kHeadSink)) {
    const auto* sink = Shape(shapes, kHeadSink);
    if (sink == nullptr || sink->NumDimensions() != 1 ||
        (*sink)[0] != config.num_heads) {
      return false;
    }
  }
  if (Present(shapes, kAttentionBias)) {
    const auto* bias = Shape(shapes, kAttentionBias);
    if (bias == nullptr || bias->NumDimensions() != 4 || !PositiveDims(*bias) ||
        ((*bias)[0] != 1 && (*bias)[0] != batch_bound) ||
        ((*bias)[1] != 1 && (*bias)[1] != config.num_heads) ||
        (*bias)[2] < sequence_bound || (*bias)[3] < total_kv_bound) {
      return false;
    }
  }
  if (Present(shapes, kPositionIds)) {
    const auto* positions = Shape(shapes, kPositionIds);
    if (positions == nullptr || positions->NumDimensions() != 2 ||
        !PositiveDims(*positions) || (*positions)[0] != batch_bound ||
        (*positions)[1] < sequence_bound) {
      return false;
    }
  }
  return true;
}

bool ValidateScaleShape(GQAKvQuantizationType type, const TensorShape* shape,
                        int64_t kv_heads, int64_t head_bound) {
  if (type == GQAKvQuantizationType::None) return true;
  if (shape == nullptr || !PositiveDims(*shape)) return false;
  if (type == GQAKvQuantizationType::PerTensor) {
    return std::all_of(shape->GetDims().begin(), shape->GetDims().end(),
                       [](int64_t dim) { return dim == 1; });
  }
  if (shape->NumDimensions() > 4) return false;
  const auto& dims = shape->GetDims();
  if (dims.size() >= 2 && dims[dims.size() - 2] != 1 &&
      dims[dims.size() - 2] > kv_heads) {
    return false;
  }
  return dims.back() == 1 || dims.back() <= head_bound;
}

std::optional<GQAWorkspaceBounds> BuildBounds(
    const GQAWorkspaceEstimateConfig& config,
    gsl::span<const WorkspaceInputShape> shapes,
    const cudaDeviceProp& device_prop,
    const AttentionKernelOptions& kernel_options) {
  if (config.qkv_element_size != 2 ||
      (config.cache_element_size != 1 && config.cache_element_size != 2) ||
      config.num_heads <= 0 || config.kv_num_heads <= 0 ||
      config.num_heads % config.kv_num_heads != 0 ||
      config.num_heads > std::numeric_limits<int32_t>::max() ||
      config.kv_num_heads > std::numeric_limits<int32_t>::max() ||
      (config.causal != 0 && config.causal != 1) ||
      (config.local_window_size != -1 && config.local_window_size <= 0) ||
      (config.sliding_window_cache && config.local_window_size <= 0) ||
      (config.causal == 0 && config.local_window_size != -1) ||
      !std::isfinite(config.softcap) ||
      !Present(shapes, kQuery) || !Present(shapes, kSequenceLengths) ||
      !Present(shapes, kTotalSequenceLength) ||
      !PairPresent(shapes, kKey, kValue) ||
      !PairPresent(shapes, kPastKey, kPastValue) ||
      !PairPresent(shapes, kKScale, kVScale) ||
      HasZeroDimension(shapes)) {
    return std::nullopt;
  }

  // WorkspaceInputShape exposes the total_sequence_length scalar's shape, not
  // its value. For a non-windowed cache that value can exceed past dim 2 and
  // directly scales backend workspace. Non-windowed execution can also copy a
  // full past tensor when only one past/present pair aliases, but alias state is
  // unavailable here. Windowed execution is bounded by the cache shape and the
  // runtime requires both past/present pairs to alias before allocating scratch.
  if (!config.sliding_window_cache) return std::nullopt;

  const bool packed = !Present(shapes, kKey);
  const TensorShape* query = Shape(shapes, kQuery);
  if (query == nullptr || query->NumDimensions() != 3 || !PositiveDims(*query)) {
    return std::nullopt;
  }
  int64_t batch_bound = (*query)[0];
  int64_t sequence_bound = (*query)[1];
  int64_t head_bound = 0;
  if (packed) {
    const int64_t head_factor = config.num_heads + 2 * config.kv_num_heads;
    if ((*query)[2] < head_factor) return std::nullopt;
    head_bound = (*query)[2] / head_factor;
  } else {
    const auto* key = Shape(shapes, kKey);
    const auto* value = Shape(shapes, kValue);
    if (key == nullptr || value == nullptr || key->NumDimensions() != 3 ||
        value->NumDimensions() != 3 || !PositiveDims(*key) || !PositiveDims(*value)) {
      return std::nullopt;
    }
    batch_bound = std::min({batch_bound, (*key)[0], (*value)[0]});
    sequence_bound = std::min({sequence_bound, (*key)[1], (*value)[1]});
    head_bound = std::min({(*query)[2] / config.num_heads,
                           (*key)[2] / config.kv_num_heads,
                           (*value)[2] / config.kv_num_heads});
  }
  if (head_bound < 8 || batch_bound <= 0 || sequence_bound <= 0) return std::nullopt;

  // Cache capacity is the only sound bound for total KV length available at
  // this API. Without past tensors, the CPU scalar value is unavailable and a
  // single-token invocation can name an arbitrarily larger total length.
  if (!Present(shapes, kPastKey)) return std::nullopt;
  const auto* past_key = Shape(shapes, kPastKey);
  const auto* past_value = Shape(shapes, kPastValue);
  if (past_key == nullptr || past_value == nullptr ||
      past_key->NumDimensions() != 4 || past_value->NumDimensions() != 4 ||
      !PositiveDims(*past_key) || !PositiveDims(*past_value)) {
    return std::nullopt;
  }
  batch_bound = std::min({batch_bound, (*past_key)[0], (*past_value)[0]});
  if ((*past_key)[1] < config.kv_num_heads || (*past_value)[1] < config.kv_num_heads) {
    return std::nullopt;
  }
  const int64_t capacity_bound = std::min((*past_key)[2], (*past_value)[2]);
  int64_t cache_head_bound = std::min((*past_key)[3], (*past_value)[3]);
  if (config.kv_cache_bit_width == 4) {
    if (cache_head_bound > std::numeric_limits<int64_t>::max() / 2) return std::nullopt;
    cache_head_bound *= 2;
  }
  head_bound = std::min(head_bound, cache_head_bound);
  if (batch_bound <= 0 || capacity_bound <= 0 || head_bound < 8 ||
      capacity_bound < config.local_window_size ||
      !ValidateAuxiliaryShapes(config, shapes, batch_bound, sequence_bound,
                               head_bound, capacity_bound)) {
    return std::nullopt;
  }

  // Runtime rejects attention bias for sliding-window GQA.
  if (Present(shapes, kAttentionBias)) return std::nullopt;

  const bool k_quantized = config.k_quantization != GQAKvQuantizationType::None;
  const bool v_quantized = config.v_quantization != GQAKvQuantizationType::None;
  if (k_quantized != v_quantized ||
      (k_quantized && (!Present(shapes, kKScale) || config.cache_element_size != 1 ||
                       (config.kv_cache_bit_width != 8 && config.kv_cache_bit_width != 4))) ||
      (!k_quantized && (config.cache_element_size != 2 ||
                        config.kv_cache_bit_width != 0))) {
    return std::nullopt;
  }
  if (Present(shapes, kKScale) &&
      (!ValidateScaleShape(config.k_quantization, Shape(shapes, kKScale),
                           config.kv_num_heads, head_bound) ||
       !ValidateScaleShape(config.v_quantization, Shape(shapes, kVScale),
                           config.kv_num_heads, head_bound))) {
    return std::nullopt;
  }

  GQAWorkspaceBounds bounds;
  bounds.qkv_element_size = config.qkv_element_size;
  bounds.cache_element_size = config.cache_element_size;
  bounds.batch_size_bound = batch_bound;
  bounds.sequence_length_bound = sequence_bound;
  bounds.num_heads = config.num_heads;
  bounds.kv_num_heads = config.kv_num_heads;
  bounds.head_size_bound = head_bound;
  bounds.present_kv_cache_capacity_bound = capacity_bound;
  bounds.kv_cache_bit_width = config.kv_cache_bit_width;
  bounds.k_quantization = config.k_quantization;
  bounds.v_quantization = config.v_quantization;
  bounds.is_windowed_kv_cache = config.sliding_window_cache;
  bounds.do_rotary = config.do_rotary;
  bounds.is_packed_qkv = packed;
  bounds.use_qk_norm = Present(shapes, kQNorm);
  bounds.prompt_reachable = true;
  bounds.decode_reachable = true;
  bounds.device_major = device_prop.major;
  bounds.device_minor = device_prop.minor;
  bounds.multi_processor_count = device_prop.multiProcessorCount;
  bounds.is_bf16 = config.is_bf16;
  bounds.local_window_size = config.local_window_size;
  bounds.xqa_head_sink_storage =
      Present(shapes, kHeadSink)
          ? (config.head_sink_is_prepacked ? GQAXqaHeadSinkStorage::PrepackedFp32
                                           : GQAXqaHeadSinkStorage::DynamicConversion)
          : GQAXqaHeadSinkStorage::None;

  const bool has_bias = Present(shapes, kAttentionBias);
  const bool has_sink = Present(shapes, kHeadSink);
  const bool smooth_supported_by_xqa = !config.smooth_softmax || has_sink;
  const bool quantized_xqa = k_quantized && config.kv_cache_bit_width == 8;
  const int64_t group_size = config.num_heads / config.kv_num_heads;
  const bool xqa_head_reachable =
      (head_bound >= 64) &&
      ((head_bound >= 64 && IsSupportedGQAXqaHeadSize(64)) ||
       (head_bound >= 128 && IsSupportedGQAXqaHeadSize(128)) ||
       (head_bound >= 256 && IsSupportedGQAXqaHeadSize(256)));
  const bool fp8_cache = k_quantized && config.cache_is_fp8;
  const bool xqa_reachable =
      config.enable_xqa && config.causal == 1 && !has_bias &&
      device_prop.major >= 8 && config.softcap == 0.0f &&
      smooth_supported_by_xqa && (!bounds.use_qk_norm || !k_quantized) &&
      (!k_quantized || config.kv_cache_bit_width == 8) &&
      xqa_head_reachable &&
      IsSupportedGQAXqaGroupSize(group_size, k_quantized) &&
      (!fp8_cache || device_prop.major >= 9 ||
       (device_prop.major == 8 && device_prop.minor == 9));
  if (xqa_reachable) {
    bounds.reachable_backends =
        bounds.reachable_backends | GQAReachableBackend::Xqa;
    bounds.xqa_kv_type = quantized_xqa
                             ? (config.cache_is_fp8 ? GQAXqaKvType::Fp8
                                                    : GQAXqaKvType::Int8)
                             : GQAXqaKvType::None;
  }

  bool flash_reachable = false;
#if USE_FLASH_ATTENTION
  flash_reachable =
      !has_bias && kernel_options.UseFlashAttention() &&
      onnxruntime::flash::is_any_supported_head_size<MLFloat16>(
          device_prop, static_cast<size_t>(head_bound),
          static_cast<size_t>(config.num_heads),
          static_cast<size_t>(config.kv_num_heads));
#endif
  if (flash_reachable) {
    bounds.reachable_backends =
        bounds.reachable_backends | GQAReachableBackend::Flash;
    if (!config.disable_flash_decode && !k_quantized && !bounds.use_qk_norm &&
        !config.sliding_window_cache) {
      bounds.reachable_backends =
          bounds.reachable_backends | GQAReachableBackend::FlashFastDecode;
    }
  }

  bool mea_reachable = false;
#if USE_MEMORY_EFFICIENT_ATTENTION
  mea_reachable =
      !k_quantized && !has_bias && kernel_options.UseEfficientAttention() &&
      has_memory_efficient_attention_for_head_size_bounds(
          device_prop.major * 10 + device_prop.minor,
          /*is_half=*/!config.is_bf16, config.is_bf16,
          static_cast<int>(std::min<int64_t>(head_bound, std::numeric_limits<int>::max())),
          static_cast<int>(std::min<int64_t>(head_bound, std::numeric_limits<int>::max())));
#endif
  if (mea_reachable) {
    bounds.reachable_backends =
        bounds.reachable_backends | GQAReachableBackend::MemoryEfficient;
  }
  if (!k_quantized && !config.smooth_softmax && !has_sink) {
    bounds.reachable_backends =
        bounds.reachable_backends | GQAReachableBackend::Unfused;
  }

  const bool cudnn_reachable =
      !has_bias && !k_quantized && config.softcap == 0.0f &&
      !config.smooth_softmax && !has_sink && config.local_window_size == -1 &&
      (kernel_options.UseCudnnFlashAttention() ||
       (kernel_options.AllowCudnnFlashAttentionAuto() && device_prop.major >= 9));
  if (cudnn_reachable) {
    bounds.reachable_backends =
        bounds.reachable_backends | GQAReachableBackend::Cudnn;
  }
  return bounds;
}

std::optional<GQAWorkspaceEstimateConfig> ConfigFromNode(const Node& node) {
  if (node.OpType() != "GroupQueryAttention") return std::nullopt;
  GQAWorkspaceEstimateConfig config;
  bool found_heads = false;
  bool found_kv_heads = false;
  for (const auto& [name, attr] : node.GetAttributes()) {
    if (name == "num_heads") {
      config.num_heads = attr.i();
      found_heads = true;
    } else if (name == "kv_num_heads") {
      config.kv_num_heads = attr.i();
      found_kv_heads = true;
    } else if (name == "causal") {
      config.causal = attr.i();
    } else if (name == "local_window_size") {
      config.local_window_size = attr.i();
    } else if (name == "sliding_window_cache") {
      config.sliding_window_cache = attr.i() == 1;
    } else if (name == "do_rotary") {
      config.do_rotary = attr.i() == 1;
    } else if (name == "smooth_softmax") {
      config.smooth_softmax = attr.i() == 1;
    } else if (name == "softcap") {
      config.softcap = attr.f();
    } else if (name == "kv_cache_bit_width") {
      config.kv_cache_bit_width = attr.i();
    } else if (name == "k_quant_type") {
      const auto type = ParseQuantization(attr.s());
      if (!type.has_value()) return std::nullopt;
      config.k_quantization = *type;
    } else if (name == "v_quant_type") {
      const auto type = ParseQuantization(attr.s());
      if (!type.has_value()) return std::nullopt;
      config.v_quantization = *type;
    }
  }
  if (!found_heads || !found_kv_heads) return std::nullopt;
  const auto& inputs = node.InputDefs();
  if (inputs.empty() || inputs[0] == nullptr) return std::nullopt;
  const auto* query_type = inputs[0]->TypeAsProto();
  if (query_type == nullptr || !query_type->has_tensor_type()) return std::nullopt;
  const int query_elem = query_type->tensor_type().elem_type();
  if (query_elem != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 &&
      query_elem != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) {
    return std::nullopt;
  }
  config.qkv_element_size = 2;
  config.is_bf16 = query_elem == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;

  const NodeArg* cache_arg =
      inputs.size() > kPastKey && inputs[kPastKey] != nullptr && inputs[kPastKey]->Exists()
          ? inputs[kPastKey]
          : nullptr;
  if (cache_arg == nullptr || cache_arg->TypeAsProto() == nullptr ||
      !cache_arg->TypeAsProto()->has_tensor_type()) {
    return std::nullopt;
  }
  const int cache_elem = cache_arg->TypeAsProto()->tensor_type().elem_type();
  config.cache_is_fp8 =
      cache_elem == ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN;
  config.cache_element_size =
      cache_elem == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 ||
              cache_elem == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16
          ? 2
          : 1;
  config.enable_xqa =
      ParseEnvironmentVariableWithDefault<int>("ORT_ENABLE_XQA", 1) != 0;
  config.disable_flash_decode =
      ParseEnvironmentVariableWithDefault<bool>("ORT_DISABLE_FLASH_DECODE", false);
  return config;
}

}  // namespace

std::optional<GQAWorkspaceAggregate> EstimateGroupQueryAttentionWorkspace(
    const GQAWorkspaceEstimateConfig& config,
    gsl::span<const WorkspaceInputShape> input_shapes,
    const cudaDeviceProp& device_prop,
    const AttentionKernelOptions& kernel_options) {
  const auto bounds = BuildBounds(config, input_shapes, device_prop, kernel_options);
  if (!bounds.has_value()) return std::nullopt;
  const auto aggregate = GetGQAWorkspaceAggregateForBounds(*bounds);
  return aggregate.status.IsOK()
             ? std::optional<GQAWorkspaceAggregate>{aggregate}
             : std::nullopt;
}

std::optional<GQAWorkspaceAggregate> EstimateGroupQueryAttentionWorkspace(
    const Node& node,
    gsl::span<const WorkspaceInputShape> input_shapes,
    const cudaDeviceProp& device_prop,
    const AttentionKernelOptions& kernel_options) {
  const auto config = ConfigFromNode(node);
  return config.has_value()
             ? EstimateGroupQueryAttentionWorkspace(
                   *config, input_shapes, device_prop, kernel_options)
             : std::nullopt;
}

void SetGroupQueryAttentionWorkspaceRequirements(
    const GQAWorkspaceAggregate& estimate,
    InlinedVector<WorkspaceRequirement>& requirements) {
  requirements.clear();
  if (!estimate.status.IsOK() || estimate.total_workspace_bytes == 0) return;
  requirements.push_back(
      {estimate.total_workspace_bytes, 0, kGQAWorkspaceAlignment});
}

void SetGroupQueryAttentionLevel1MemoryEstimate(
    const GQAWorkspaceAggregate& workspace,
    Level1MemoryEstimate& estimate) {
  if (workspace.status.IsOK() && workspace.total_workspace_bytes != 0) {
    estimate.runtime_workspace_bytes = workspace.total_workspace_bytes;
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#endif
