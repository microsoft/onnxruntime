// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/moe/qmoe_shared_cpu.h"
#include "contrib_ops/cpu/moe/qmoe_shared_layout.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>

#include "core/common/float16.h"
#include "core/common/inlined_containers.h"
#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/framework/allocator.h"
#include "core/platform/threadpool.h"
#if !defined(ORT_MINIMAL_BUILD)
#include "core/framework/kernel_pilot.h"
#endif

namespace onnxruntime::contrib {
namespace {

enum class ElementType : uint8_t {
  kFloat32 = 0,
  kFloat16 = 1,
  kBFloat16 = 2
};

ElementType GetElementType(const Tensor& tensor) {
  if (tensor.IsDataType<MLFloat16>()) return ElementType::kFloat16;
  if (tensor.IsDataType<BFloat16>()) return ElementType::kBFloat16;
  ORT_ENFORCE(tensor.IsDataType<float>(), "Shared QMoE requires float, float16, or bfloat16 values.");
  return ElementType::kFloat32;
}

float ReadElement(const void* data, ElementType type, size_t index) {
  switch (type) {
    case ElementType::kFloat16:
      return static_cast<float>(static_cast<const MLFloat16*>(data)[index]);
    case ElementType::kBFloat16:
      return static_cast<float>(static_cast<const BFloat16*>(data)[index]);
    default:
      return static_cast<const float*>(data)[index];
  }
}

void ReadActivationRowToFloat(const void* data, ElementType type, size_t offset, int64_t count, float* dst) {
  switch (type) {
    case ElementType::kFloat16:
      qmoe_shared::HalfBitsToFloatRow(reinterpret_cast<const uint16_t*>(data) + offset, dst, count);
      break;
    case ElementType::kBFloat16:
      qmoe_shared::BFloat16BitsToFloatRow(reinterpret_cast<const uint16_t*>(data) + offset, dst, count);
      break;
    default:
      std::memcpy(dst, static_cast<const float*>(data) + offset, static_cast<size_t>(count) * sizeof(float));
      break;
  }
}

float RoundElement(float value, ElementType type) {
  switch (type) {
    case ElementType::kFloat16:
      return qmoe_shared::HalfToFloat(qmoe_shared::FloatToHalf(value));
    case ElementType::kBFloat16:
      return qmoe_shared::BFloat16ToFloat(qmoe_shared::FloatToBFloat16(value));
    default:
      return value;
  }
}

void WriteElement(void* data, ElementType type, size_t index, float value) {
  switch (type) {
    case ElementType::kFloat16:
      static_cast<MLFloat16*>(data)[index] = MLFloat16::FromBits(qmoe_shared::FloatToHalf(value));
      break;
    case ElementType::kBFloat16:
      static_cast<BFloat16*>(data)[index] = BFloat16::FromBits(qmoe_shared::FloatToBFloat16(value));
      break;
    default:
      static_cast<float*>(data)[index] = value;
  }
}

Status ValidateWeightShape(const Tensor& tensor, int64_t experts, int64_t n, int64_t k,
                           int64_t bits, const std::string& quant_type) {
  ORT_RETURN_IF_NOT(tensor.IsDataType<uint8_t>() && tensor.Shape().NumDimensions() == 3,
                    "Shared QMoE weights must be rank-3 uint8 tensors.");
  const auto& shape = tensor.Shape();
  const int64_t row_bytes = narrow<int64_t>((SafeInt<size_t>(k) * bits + 7) / 8);
  const bool canonical = shape[0] == experts && shape[1] == n && shape[2] == row_bytes;
  const bool sm80_or_fp4_alias = bits == 4 && shape[0] == experts && shape[1] == k && shape[2] == n / 2;
  ORT_RETURN_IF_NOT(canonical || sm80_or_fp4_alias,
                    "Shared QMoE weight shape does not match its logical projection dimensions.");
  if (quant_type == "int" && bits == 4) {
    ORT_RETURN_IF_NOT(n % qmoe_shared::kColumnsInterleaved == 0 && k % qmoe_shared::kRowsPerTile == 0,
                      "Shared SM80 INT4 QMoE requires N divisible by 4 and K divisible by 64.");
  }
  const size_t expected_bytes = SafeInt<size_t>(experts) * n * row_bytes;
  ORT_RETURN_IF_NOT(tensor.SizeInBytes() == expected_bytes, "Shared QMoE weight byte extent mismatch.");
  return Status::OK();
}

}  // namespace

Status ComputeSharedQMoE(OpKernelContext* context, const SharedQMoEAttributes& attributes) {
  const auto* input = context->Input<Tensor>(0);
  const auto* router = context->Input<Tensor>(1);
  const auto* weights1 = context->Input<Tensor>(2);
  const auto* scales1 = context->Input<Tensor>(3);
  const auto* bias1 = context->Input<Tensor>(4);
  const auto* weights2 = context->Input<Tensor>(5);
  const auto* scales2 = context->Input<Tensor>(6);
  const auto* bias2 = context->Input<Tensor>(7);
  const auto* router_weights = context->Input<Tensor>(14);
  const auto* global_scale1 = context->Input<Tensor>(15);
  const auto* global_scale2 = context->Input<Tensor>(16);
  ORT_RETURN_IF_NOT(input && router && weights1 && scales1 && weights2 && scales2,
                    "Shared QMoE requires input, routing, weights, and scales.");
  const bool is_nvfp4 = (attributes.quant_type == "nvfp4");
  const bool is_mxfp4 = (attributes.quant_type == "fp4");
  const bool is_fp4 = is_nvfp4 || is_mxfp4;
  ORT_RETURN_IF_NOT((attributes.quant_type == "int" && (attributes.weight_bits == 4 || attributes.weight_bits == 3)) ||
                        (is_fp4 && attributes.weight_bits == 4),
                    "Shared QMoE requires prepacked INT4, raw INT3, or row-major NVFP4/MXFP4 configuration.");
  if (!is_fp4) {
    ORT_RETURN_IF_NOT(global_scale1 == nullptr && global_scale2 == nullptr,
                      "Shared integer QMoE does not accept global scales.");
  } else if (is_nvfp4) {
    ORT_RETURN_IF_NOT(global_scale1 != nullptr && global_scale2 != nullptr,
                      "Shared NVFP4 QMoE requires fc1_global_scale and fc2_global_scale.");
  }
  for (int index = 8; index < context->InputCount(); ++index) {
    if (index == 14 || (is_fp4 && (index == 15 || index == 16))) continue;
    ORT_RETURN_IF_NOT(context->Input<Tensor>(index) == nullptr,
                      "Shared QMoE does not support FC3, explicit zero points, or auxiliary scales.");
  }

  const auto& input_shape = input->Shape();
  const auto& s1 = scales1->Shape();
  const auto& s2 = scales2->Shape();
  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 2 || input_shape.NumDimensions() == 3,
                    "Shared QMoE input must have rank 2 or 3.");
  ORT_RETURN_IF_NOT(s1.NumDimensions() == 3 && s2.NumDimensions() == 3,
                    "Shared QMoE scales must have shape [E,N,K/block_size].");
  const int64_t hidden = input_shape[input_shape.NumDimensions() - 1];
  const int64_t experts = s1[0];
  const int64_t block = is_nvfp4 ? 16 : (is_mxfp4 ? 32 : attributes.block_size);
  ORT_RETURN_IF_NOT(hidden > 0 && experts > 0 && s2[2] > 0 &&
                        (is_nvfp4 ? (block == 16) : (block == 32 || block == 64 || block == 128)),
                    "Shared QMoE projection dimensions and block size must be valid.");
  const int64_t intermediate = narrow<int64_t>(SafeInt<size_t>(s2[2]) * block);
  const int64_t fc1_n = narrow<int64_t>(SafeInt<size_t>(intermediate) * 2);
  const int64_t rows = input_shape.Size() / hidden;
  ORT_RETURN_IF_NOT(s1[1] == fc1_n && s1[2] == hidden / block && hidden % block == 0 &&
                        s2[0] == experts && s2[1] == hidden,
                    "Shared QMoE scale grids disagree with input or projection dimensions.");
  ORT_RETURN_IF_NOT(attributes.top_k > 0 && attributes.top_k <= experts,
                    "Shared QMoE top_k must be within the expert count.");
  ORT_RETURN_IF_NOT(router->Shape() == TensorShape({rows, experts}),
                    "Shared QMoE router_probs must have shape [rows,experts].");
  if (router_weights != nullptr) {
    ORT_RETURN_IF_NOT(router_weights->Shape() == router->Shape(),
                      "Shared QMoE router_weights must match router_probs.");
  }
  ORT_RETURN_IF_ERROR(ValidateWeightShape(*weights1, experts, fc1_n, hidden, attributes.weight_bits, attributes.quant_type));
  ORT_RETURN_IF_ERROR(ValidateWeightShape(*weights2, experts, hidden, intermediate, attributes.weight_bits, attributes.quant_type));
  ORT_RETURN_IF_NOT(!bias1 || bias1->Shape() == TensorShape({experts, fc1_n}),
                    "Shared QMoE FC1 bias must have shape [E,2*I].");
  ORT_RETURN_IF_NOT(!bias2 || bias2->Shape() == TensorShape({experts, hidden}),
                    "Shared QMoE FC2 bias must have shape [E,H].");

  const ElementType boundary_type = GetElementType(*input);
  ElementType scale_type = boundary_type;
  if (is_nvfp4) {
    ORT_RETURN_IF_NOT(scales1->IsDataType<Float8E4M3FN>() && scales2->IsDataType<Float8E4M3FN>(),
                      "Shared NVFP4 QMoE requires float8e4m3fn block scales.");
  } else if (is_mxfp4) {
    ORT_RETURN_IF_NOT(scales1->IsDataType<Float8E8M0>() && scales2->IsDataType<Float8E8M0>(),
                      "Shared MXFP4 QMoE requires float8e8m0 block scales.");
  } else {
    scale_type = GetElementType(*scales1);
    ORT_RETURN_IF_NOT(GetElementType(*scales2) == scale_type,
                      "Shared QMoE FC1 and FC2 scale types must match.");
  }
  if (global_scale1 != nullptr) {
    ORT_RETURN_IF_NOT(global_scale1->IsDataType<float>() && global_scale1->Shape() == TensorShape({experts}),
                      "Shared FP4 QMoE fc1_global_scale must be float [E].");
  }
  if (global_scale2 != nullptr) {
    ORT_RETURN_IF_NOT(global_scale2->IsDataType<float>() && global_scale2->Shape() == TensorShape({experts}),
                      "Shared FP4 QMoE fc2_global_scale must be float [E].");
  }
  const ElementType bias1_type = bias1 ? GetElementType(*bias1) : boundary_type;
  const ElementType bias2_type = bias2 ? GetElementType(*bias2) : boundary_type;
  ORT_RETURN_IF_NOT(!bias1 || !bias2 || bias1_type == bias2_type,
                    "Shared QMoE FC1 and FC2 bias types must match.");

  auto* output = context->Output(0, input_shape);
  if (rows == 0) return Status::OK();

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
  const size_t route_count = SafeInt<size_t>(rows) * attributes.top_k;
  auto route_experts = IAllocator::MakeUniquePtr<int>(allocator, route_count);
  auto route_scales = IAllocator::MakeUniquePtr<float>(allocator, route_count);
  auto route_outputs = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(route_count) * hidden);
  InlinedVector<InlinedVector<size_t>> expert_routes;
  expert_routes.reserve(narrow<size_t>(experts));
  for (int64_t e = 0; e < experts; ++e) expert_routes.emplace_back();
  struct RankedExpert {
    float logit;
    int expert;
  };
  InlinedVector<RankedExpert> ranking;
  ranking.reserve(narrow<size_t>(experts));
  for (int64_t e = 0; e < experts; ++e) ranking.push_back({0.0f, narrow<int>(e)});
  const ElementType router_type = GetElementType(*router);
  const ElementType router_weight_type = router_weights ? GetElementType(*router_weights) : router_type;
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t e = 0; e < experts; ++e) {
      ranking[narrow<size_t>(e)] = {ReadElement(router->DataRaw(), router_type, row * experts + e), narrow<int>(e)};
    }
    std::partial_sort(ranking.begin(), ranking.begin() + attributes.top_k, ranking.end(),
                      [](const RankedExpert& a, const RankedExpert& b) {
                        return a.logit > b.logit || (a.logit == b.logit && a.expert < b.expert);
                      });
    const float max_logit = ranking[0].logit;
    float denominator = 0.0f;
    const int64_t probability_count = attributes.normalize_routing_weights ? attributes.top_k : experts;
    if (router_weights == nullptr) {
      for (int64_t j = 0; j < probability_count; ++j) {
        denominator += std::exp(ranking[narrow<size_t>(j)].logit - max_logit);
      }
    }
    for (int64_t j = 0; j < attributes.top_k; ++j) {
      const int expert = ranking[narrow<size_t>(j)].expert;
      const size_t route = narrow<size_t>(row * attributes.top_k + j);
      route_experts.get()[route] = expert;
      float scale = router_weights ? ReadElement(router_weights->DataRaw(), router_weight_type,
                                                 row * experts + expert)
                                   : std::exp(ranking[narrow<size_t>(j)].logit - max_logit);
      if (router_weights == nullptr) {
        scale = denominator == 0.0f ? 0.0f : scale / denominator;
      }
      route_scales.get()[route] = scale;
      expert_routes[narrow<size_t>(expert)].push_back(route);
    }
  }

  InlinedVector<int64_t> active_experts;
  active_experts.reserve(narrow<size_t>(experts));
  for (int64_t expert = 0; expert < experts; ++expert) {
    if (!expert_routes[narrow<size_t>(expert)].empty()) active_experts.push_back(expert);
  }
  auto* thread_pool = context->GetOperatorThreadPool();
  const int workers = std::min(narrow<int>(active_experts.size()),
                               thread_pool ? concurrency::ThreadPool::DegreeOfParallelism(thread_pool) : 1);
  constexpr int64_t kRowsPerTile = qmoe_shared::kMaxRowsPerCall;
  const size_t worker_elements = SafeInt<size_t>(kRowsPerTile) * (SafeInt<size_t>(hidden) * 3 + intermediate);
  auto scratch = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(workers) * worker_elements);

  const size_t w1_stride = weights1->SizeInBytes() / narrow<size_t>(experts);
  const size_t s1_stride = scales1->SizeInBytes() / narrow<size_t>(experts);
  const size_t b1_stride = bias1 ? (bias1->SizeInBytes() / narrow<size_t>(experts)) : 0;
  const size_t w2_stride = weights2->SizeInBytes() / narrow<size_t>(experts);
  const size_t s2_stride = scales2->SizeInBytes() / narrow<size_t>(experts);
  const size_t b2_stride = bias2 ? (bias2->SizeInBytes() / narrow<size_t>(experts)) : 0;
  const auto* w1_base = static_cast<const uint8_t*>(weights1->DataRaw());
  const auto* s1_base = static_cast<const uint8_t*>(scales1->DataRaw());
  const auto* b1_base = bias1 ? static_cast<const uint8_t*>(bias1->DataRaw()) : nullptr;
  const auto* w2_base = static_cast<const uint8_t*>(weights2->DataRaw());
  const auto* s2_base = static_cast<const uint8_t*>(scales2->DataRaw());
  const auto* b2_base = bias2 ? static_cast<const uint8_t*>(bias2->DataRaw()) : nullptr;
  const float* g1_base = global_scale1 ? global_scale1->Data<float>() : nullptr;
  const float* g2_base = global_scale2 ? global_scale2->Data<float>() : nullptr;
  const qmoe_shared::SwiGluParams swiglu{
      attributes.activation_alpha, attributes.activation_beta, attributes.swiglu_limit};
  const bool bf16_boundaries = (boundary_type == ElementType::kBFloat16);
  const bool fp16_boundaries = (boundary_type == ElementType::kFloat16);

  std::atomic<size_t> next_expert{0};
  concurrency::ThreadPool::TrySimpleParallelFor(thread_pool, workers, [&](std::ptrdiff_t worker) {
    float* gathered = scratch.get() + worker * worker_elements;
    float* reordered = gathered + kRowsPerTile * hidden;
    float* activated = reordered + kRowsPerTile * hidden;
    float* fc2 = activated + kRowsPerTile * intermediate;
    for (size_t task = next_expert.fetch_add(1); task < active_experts.size(); task = next_expert.fetch_add(1)) {
      const int64_t expert = active_experts[task];
      const auto& routes = expert_routes[narrow<size_t>(expert)];
      if (is_fp4) {
        qmoe_shared::Nvfp4ExpertView view{};
        view.fc1_weights = w1_base + static_cast<size_t>(expert) * w1_stride;
        view.fc1_scales = s1_base + static_cast<size_t>(expert) * s1_stride;
        view.fc1_global_scale = g1_base ? g1_base[expert] : 1.0f;
        view.fc1_bias = b1_base ? reinterpret_cast<const uint16_t*>(b1_base + static_cast<size_t>(expert) * b1_stride)
                                : nullptr;
        view.fc2_weights = w2_base + static_cast<size_t>(expert) * w2_stride;
        view.fc2_scales = s2_base + static_cast<size_t>(expert) * s2_stride;
        view.fc2_global_scale = g2_base ? g2_base[expert] : 1.0f;
        view.fc2_bias = b2_base ? reinterpret_cast<const uint16_t*>(b2_base + static_cast<size_t>(expert) * b2_stride)
                                : nullptr;
        view.bias_is_bf16 = (bias1_type == ElementType::kBFloat16);
        view.is_mxfp4 = is_mxfp4;
        for (size_t start = 0; start < routes.size(); start += kRowsPerTile) {
          const int64_t count = narrow<int64_t>(std::min(routes.size() - start, narrow<size_t>(kRowsPerTile)));
          for (int64_t r = 0; r < count; ++r) {
            const size_t token = routes[start + r] / narrow<size_t>(attributes.top_k);
            ReadActivationRowToFloat(input->DataRaw(), boundary_type, token * hidden, hidden, gathered + r * hidden);
            qmoe_shared::ReorderNvfp4ActivationRow(gathered + r * hidden, reordered + r * hidden, hidden);
          }
          qmoe_shared::Nvfp4Fc1SwiGluRange(hidden, intermediate, view, reordered, count, swiglu,
                                           activated, 0, fc1_n, /*force_scalar=*/false,
                                           bf16_boundaries, fp16_boundaries);
          qmoe_shared::Nvfp4Fc2Range(hidden, intermediate, view, activated, count, fc2,
                                     0, hidden, /*force_scalar=*/false, /*apply_bias=*/true,
                                     /*round_scale_bf16=*/!fp16_boundaries);
          for (int64_t r = 0; r < count; ++r) {
            float* destination = route_outputs.get() + routes[start + r] * hidden;
            for (int64_t n = 0; n < hidden; ++n) destination[n] = RoundElement(fc2[r * hidden + n], boundary_type);
          }
        }
        continue;
      }
      if (attributes.weight_bits == 3) {
        qmoe_shared::Int3ExpertView view{};
        view.fc1_weights = w1_base + static_cast<size_t>(expert) * w1_stride;
        view.fc1_scales = reinterpret_cast<const uint16_t*>(s1_base + static_cast<size_t>(expert) * s1_stride);
        view.fc1_bias = b1_base ? reinterpret_cast<const uint16_t*>(b1_base + static_cast<size_t>(expert) * b1_stride)
                                : nullptr;
        view.fc2_weights = w2_base + static_cast<size_t>(expert) * w2_stride;
        view.fc2_scales = reinterpret_cast<const uint16_t*>(s2_base + static_cast<size_t>(expert) * s2_stride);
        view.fc2_bias = b2_base ? reinterpret_cast<const uint16_t*>(b2_base + static_cast<size_t>(expert) * b2_stride)
                                : nullptr;
        view.block_size = block;
        view.scale_is_bf16 = (scale_type == ElementType::kBFloat16);
        view.bias_is_bf16 = (bias1_type == ElementType::kBFloat16);
        view.scale_is_fp32 = (scale_type == ElementType::kFloat32);
        view.bias_is_fp32 = (bias1_type == ElementType::kFloat32);
        for (size_t start = 0; start < routes.size(); start += kRowsPerTile) {
          const int64_t count = narrow<int64_t>(std::min(routes.size() - start, narrow<size_t>(kRowsPerTile)));
          for (int64_t r = 0; r < count; ++r) {
            const size_t token = routes[start + r] / narrow<size_t>(attributes.top_k);
            ReadActivationRowToFloat(input->DataRaw(), boundary_type, token * hidden, hidden, gathered + r * hidden);
          }
          qmoe_shared::Int3Fc1SwiGluRange(hidden, intermediate, view, gathered, count, swiglu,
                                          activated, 0, fc1_n, /*force_scalar=*/false,
                                          bf16_boundaries, fp16_boundaries);
          qmoe_shared::Int3Fc2Range(hidden, intermediate, view, activated, count, fc2,
                                    0, hidden, /*force_scalar=*/false, /*apply_bias=*/true);
          for (int64_t r = 0; r < count; ++r) {
            float* destination = route_outputs.get() + routes[start + r] * hidden;
            for (int64_t n = 0; n < hidden; ++n) destination[n] = RoundElement(fc2[r * hidden + n], boundary_type);
          }
        }
        continue;
      }
      qmoe_shared::Int4ExpertView view{};
      view.fc1_weights = w1_base + static_cast<size_t>(expert) * w1_stride;
      view.fc1_scales = reinterpret_cast<const uint16_t*>(s1_base + static_cast<size_t>(expert) * s1_stride);
      view.fc1_bias = b1_base ? reinterpret_cast<const uint16_t*>(b1_base + static_cast<size_t>(expert) * b1_stride)
                              : nullptr;
      view.fc2_weights = w2_base + static_cast<size_t>(expert) * w2_stride;
      view.fc2_scales = reinterpret_cast<const uint16_t*>(s2_base + static_cast<size_t>(expert) * s2_stride);
      view.fc2_bias = b2_base ? reinterpret_cast<const uint16_t*>(b2_base + static_cast<size_t>(expert) * b2_stride)
                              : nullptr;
      view.block_size = block;
      view.scales_are_block_major = false;
      view.weights_are_prepacked = true;
      view.scale_is_bf16 = (scale_type == ElementType::kBFloat16);
      view.bias_is_bf16 = (bias1_type == ElementType::kBFloat16);
      view.scale_is_fp32 = (scale_type == ElementType::kFloat32);
      view.bias_is_fp32 = (bias1_type == ElementType::kFloat32);

      for (size_t start = 0; start < routes.size(); start += kRowsPerTile) {
        const int64_t count = narrow<int64_t>(std::min(routes.size() - start, narrow<size_t>(kRowsPerTile)));
        for (int64_t r = 0; r < count; ++r) {
          const size_t token = routes[start + r] / narrow<size_t>(attributes.top_k);
          ReadActivationRowToFloat(input->DataRaw(), boundary_type, token * hidden, hidden, gathered + r * hidden);
          qmoe_shared::ReorderActivationRow(gathered + r * hidden, reordered + r * hidden, hidden);
        }
        qmoe_shared::Int4Fc1SwiGluRange(hidden, intermediate, view, reordered, count, swiglu,
                                        activated, 0, fc1_n, /*force_scalar=*/false,
                                        bf16_boundaries, fp16_boundaries);
        qmoe_shared::Int4Fc2Range(hidden, intermediate, view, activated, count, fc2,
                                  0, hidden, /*force_scalar=*/false, /*apply_bias=*/true);
        for (int64_t r = 0; r < count; ++r) {
          float* destination = route_outputs.get() + routes[start + r] * hidden;
          for (int64_t n = 0; n < hidden; ++n) destination[n] = RoundElement(fc2[r * hidden + n], boundary_type);
        }
      }
    }
  });

  void* output_data = output->MutableDataRaw();
  concurrency::ThreadPool::TrySimpleParallelFor(thread_pool, narrow<int>(rows), [&](std::ptrdiff_t row) {
    for (int64_t n = 0; n < hidden; ++n) {
      float sum = 0.0f;
      for (int64_t j = 0; j < attributes.top_k; ++j) {
        const size_t route = narrow<size_t>(row * attributes.top_k + j);
        sum += route_scales.get()[route] * route_outputs.get()[route * hidden + n];
      }
      WriteElement(output_data, boundary_type, row * hidden + n, sum);
    }
  });
#if !defined(ORT_MINIMAL_BUILD)
  if (attributes.track_experts) {
    auto* pilot = context->GetKernelPilot();
    ORT_RETURN_IF_NOT(pilot, "MoE expert tracking is enabled but its collector is unavailable.");
    auto& usage = pilot->Moe();
    ORT_RETURN_IF_ERROR(usage.BeginInvocation(narrow<size_t>(experts)));
    ORT_RETURN_IF_ERROR(usage.Collect(gsl::make_span(route_experts.get(), route_count)));
  }
#endif
  return Status::OK();
}

}  // namespace onnxruntime::contrib
