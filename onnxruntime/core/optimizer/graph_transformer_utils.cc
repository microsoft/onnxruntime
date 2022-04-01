// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/graph_transformer_utils.h"

#include <algorithm>
#include <variant>

#include "core/optimizer/conv_activation_fusion.h"
#include "core/optimizer/nhwc_transformer.h"
#include "core/optimizer/qdq_transformer/qdq_final_cleanup.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.h"
#include "core/optimizer/selectors_actions/selector_action_transformer_apply_contexts.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/optimizer/conv_add_act_fusion.h"

#if !defined(ORT_MINIMAL_BUILD)

#include "core/mlas/inc/mlas.h"
#include "core/optimizer/attention_fusion.h"
#include "core/optimizer/bias_dropout_fusion.h"
#include "core/optimizer/bias_gelu_fusion.h"
#include "core/optimizer/bias_softmax_fusion.h"
#include "core/optimizer/cast_elimination.h"
#include "core/optimizer/common_subexpression_elimination.h"
#include "core/optimizer/constant_folding.h"
#include "core/optimizer/conv_add_fusion.h"
#include "core/optimizer/conv_bn_fusion.h"
#include "core/optimizer/conv_mul_fusion.h"
#include "core/optimizer/div_mul_fusion.h"
#include "core/optimizer/dropout_elimination.h"
#include "core/optimizer/dynamic_quantize_matmul_fusion.h"
#include "core/optimizer/embed_layer_norm_fusion.h"
#include "core/optimizer/expand_elimination.h"
#include "core/optimizer/fast_gelu_fusion.h"
#include "core/optimizer/free_dim_override_transformer.h"
#include "core/optimizer/gelu_approximation.h"
#include "core/optimizer/gelu_fusion.h"
#include "core/optimizer/gemm_activation_fusion.h"
#include "core/optimizer/gemm_sum_fusion.h"
#include "core/optimizer/gemm_transpose_fusion.h"
#include "core/optimizer/identity_elimination.h"
#include "core/optimizer/layer_norm_fusion.h"
#include "core/optimizer/matmul_add_fusion.h"
#include "core/optimizer/matmul_integer_to_float.h"
#include "core/optimizer/matmul_scale_fusion.h"
#include "core/optimizer/matmul_transpose_fusion.h"
#include "core/optimizer/nchwc_transformer.h"
#include "core/optimizer/noop_elimination.h"
#include "core/optimizer/not_where_fusion.h"
#include "core/optimizer/qdq_transformer/clip_quantizelinear.h"
#include "core/optimizer/qdq_transformer/qdq_propagation.h"
#include "core/optimizer/qdq_transformer/qdq_s8_to_u8.h"
#include "core/optimizer/qdq_transformer/relu_quantizelinear.h"
#include "core/optimizer/relu_clip_fusion.h"
#include "core/optimizer/reshape_fusion.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/skip_layer_norm_fusion.h"
#include "core/optimizer/slice_elimination.h"
#include "core/optimizer/transpose_optimizer/ort_transpose_optimizer.h"
#include "core/optimizer/unsqueeze_elimination.h"

#endif  // !defined(ORT_MINIMAL_BUILD)

namespace onnxruntime::optimizer_utils {

static void FilterTransformers(InlinedVector<std::unique_ptr<GraphTransformer>>& transformers,
                               const InlinedHashSet<std::string>& transformers_to_disable) {
  if (transformers_to_disable.empty()) return;

  transformers.erase(
      std::remove_if(transformers.begin(), transformers.end(),
                     [&, transformers_to_disable_end = transformers_to_disable.end()](const std::unique_ptr<GraphTransformer>& transformer) {
                       return !transformer ||
                              transformers_to_disable.find(transformer->Name()) != transformers_to_disable_end;
                     }),
      transformers.end());
}

#if !defined(ORT_MINIMAL_BUILD)

std::string GenerateRuleBasedTransformerName(TransformerLevel level) {
  return "Level" + std::to_string(static_cast<uint32_t>(level)) + "_RuleBasedTransformer";
}

InlinedVector<std::unique_ptr<RewriteRule>> GenerateRewriteRules(
    TransformerLevel level,
    const InlinedHashSet<std::string>& rules_to_disable) {
  InlinedVector<std::unique_ptr<RewriteRule>> rules;
  switch (level) {
    case TransformerLevel::Level1:
      rules.push_back(std::make_unique<EliminateIdentity>());
      rules.push_back(std::make_unique<EliminateSlice>());
      rules.push_back(std::make_unique<UnsqueezeElimination>());
      rules.push_back(std::make_unique<EliminateDropout>());
      rules.push_back(std::make_unique<ExpandElimination>());
      rules.push_back(std::make_unique<CastElimination>());
      rules.push_back(std::make_unique<NoopElimination>());
      rules.push_back(std::make_unique<DivMulFusion>());
      rules.push_back(std::make_unique<FuseReluClip>());
      rules.push_back(std::make_unique<GemmSumFusion>());
      rules.push_back(std::make_unique<GemmTransposeFusion>());
      rules.push_back(std::make_unique<NotWhereFusion>());
      rules.push_back(std::make_unique<ConvAddFusion>());
      rules.push_back(std::make_unique<ConvMulFusion>());
      rules.push_back(std::make_unique<ConvBNFusion>());
      rules.push_back(std::make_unique<ClipQuantFusion>());
      rules.push_back(std::make_unique<ReluQuantFusion>());
      break;

    case TransformerLevel::Level2:
      // No level2 rules available today
      break;

    case TransformerLevel::Level3:
      break;

    default:
      ORT_THROW("Unsupported optimization level: ", static_cast<int>(level));
  }

  if (rules_to_disable.empty()) {
    return rules;
  } else {
    InlinedVector<std::unique_ptr<RewriteRule>> filtered_list;
    const auto end = rules_to_disable.cend();
    std::for_each(rules.begin(), rules.end(),
                  [&](std::unique_ptr<RewriteRule>& item) {
                    if ((item != nullptr) && (rules_to_disable.find(item->Name()) == end)) {
                      filtered_list.push_back(std::move(item));
                    }
                  });

    return filtered_list;
  }
}

std::unique_ptr<RuleBasedGraphTransformer> GenerateRuleBasedGraphTransformer(
    TransformerLevel level,
    const InlinedHashSet<std::string>& rules_to_disable,
    const InlinedHashSet<std::string_view>& compatible_execution_providers) {
  auto rewrite_rules_to_register = GenerateRewriteRules(level, rules_to_disable);
  if (rewrite_rules_to_register.empty()) {
    return nullptr;
  }

  std::unique_ptr<RuleBasedGraphTransformer> rule_transformer =
      std::make_unique<RuleBasedGraphTransformer>(GenerateRuleBasedTransformerName(level),
                                                  compatible_execution_providers);
  for (auto& entry : rewrite_rules_to_register) {
    ORT_THROW_IF_ERROR(rule_transformer->Register(std::move(entry)));
  }

  return rule_transformer;
}

InlinedVector<std::unique_ptr<GraphTransformer>> GenerateTransformers(
    TransformerLevel level,
    const SessionOptions& session_options,
    const IExecutionProvider& cpu_execution_provider, /*required by constant folding*/
    const InlinedHashSet<std::string>& rules_and_transformers_to_disable) {
  InlinedVector<std::unique_ptr<GraphTransformer>> transformers;
  const bool disable_quant_qdq =
      session_options.config_options.GetConfigOrDefault(kOrtSessionOptionsDisableQuantQDQ, "0") == "1";
#ifndef DISABLE_CONTRIB_OPS
  const InlinedHashSet<std::string_view> cpu_ep = {onnxruntime::kCpuExecutionProvider};
#endif
  switch (level) {
    case TransformerLevel::Level1: {
      // RewriteRule optimizations are the simplest (they generally remove unnecessary nodes and are cheap to run)
      // so run them first so there is potentially less for the more intensive optimizations like ConstantFolding,
      // CommonSubexpressionElimination and TransposeOptimizer to do.
      auto rule_transformer = GenerateRuleBasedGraphTransformer(level, rules_and_transformers_to_disable, {});
      if (rule_transformer != nullptr) {
        transformers.emplace_back(std::move(rule_transformer));
      }

      // no filtering on execution provider for L1 optimizations as they only use official ONNX operators
      transformers.emplace_back(std::make_unique<CommonSubexpressionElimination>());
      transformers.emplace_back(std::make_unique<ConstantFolding>(cpu_execution_provider, !disable_quant_qdq));
      transformers.emplace_back(std::make_unique<MatMulAddFusion>());
      transformers.emplace_back(std::make_unique<ReshapeFusion>());
      transformers.emplace_back(std::make_unique<FreeDimensionOverrideTransformer>(
          session_options.free_dimension_overrides));
      auto cpu_allocator = cpu_execution_provider.GetAllocator(0, OrtMemTypeDefault);
      transformers.emplace_back(std::make_unique<TransposeOptimizer>(std::move(cpu_allocator)));

      if (!disable_quant_qdq) {
        transformers.emplace_back(std::make_unique<QDQPropagationTransformer>());
      }

    } break;

    case TransformerLevel::Level2: {
      const bool enable_quant_qdq_cleanup =
          session_options.config_options.GetConfigOrDefault(kOrtSessionOptionsEnableQuantQDQCleanup, "0") == "1";
#ifndef DISABLE_CONTRIB_OPS
      const bool qdq_is_int8_allowed =
          session_options.config_options.GetConfigOrDefault(kOrtSessionOptionsQDQIsInt8Allowed,
                                                            QDQIsInt8Allowed() ? "1" : "0") == "1";
      const bool enable_gelu_approximation =
          session_options.config_options.GetConfigOrDefault(kOrtSessionOptionsEnableGeluApproximation, "0") == "1";

      const InlinedHashSet<std::string_view> cuda_rocm_eps = {onnxruntime::kCudaExecutionProvider,
                                                              onnxruntime::kRocmExecutionProvider};
      const InlinedHashSet<std::string_view> cpu_cuda_rocm_eps = {onnxruntime::kCpuExecutionProvider,
                                                                  onnxruntime::kCudaExecutionProvider,
                                                                  onnxruntime::kRocmExecutionProvider};
      const InlinedHashSet<std::string_view> cpu_cuda_rocm_acl_armnn_eps = {onnxruntime::kCpuExecutionProvider,
                                                                            onnxruntime::kCudaExecutionProvider,
                                                                            onnxruntime::kRocmExecutionProvider,
                                                                            onnxruntime::kAclExecutionProvider,
                                                                            onnxruntime::kArmNNExecutionProvider};

      if (!disable_quant_qdq) {
        // currently we don't support QDQS8ToU8Transformer in a minimal build and if supported, this needs to run in
        // Level 1 during export and not Level 2 at runtime as it would result in overlapping optimizations which
        // runtime optimization does not support, so add session config value here to force qdqisint8allowed to be true.
        if (!qdq_is_int8_allowed) {
          transformers.emplace_back(std::make_unique<QDQS8ToU8Transformer>(cpu_ep));
        }
        transformers.emplace_back(std::make_unique<QDQSelectorActionTransformer>(qdq_is_int8_allowed));
      }

      transformers.emplace_back(std::make_unique<GemmActivationFusion>(cpu_ep));
      transformers.emplace_back(std::make_unique<MatMulIntegerToFloatFusion>(cpu_ep));
      transformers.emplace_back(std::make_unique<DynamicQuantizeMatMulFusion>(cpu_ep));

      transformers.emplace_back(std::make_unique<ConvActivationFusion>(cpu_cuda_rocm_acl_armnn_eps));

      transformers.emplace_back(std::make_unique<GeluFusion>(cpu_cuda_rocm_eps));
      transformers.emplace_back(std::make_unique<LayerNormFusion>(cpu_cuda_rocm_eps));
      transformers.emplace_back(std::make_unique<SimplifiedLayerNormFusion>(cpu_cuda_rocm_eps));
      transformers.emplace_back(std::make_unique<AttentionFusion>(cpu_cuda_rocm_eps));
      transformers.emplace_back(std::make_unique<EmbedLayerNormFusion>(cpu_cuda_rocm_eps));

      transformers.emplace_back(std::make_unique<BiasDropoutFusion>(cuda_rocm_eps));
      transformers.emplace_back(std::make_unique<MatmulTransposeFusion>(cpu_cuda_rocm_eps));
      transformers.emplace_back(std::make_unique<BiasGeluFusion>(cpu_cuda_rocm_eps));
      transformers.emplace_back(std::make_unique<BiasSoftmaxFusion>(cpu_cuda_rocm_eps));
      transformers.emplace_back(std::make_unique<SkipLayerNormFusion>(cpu_cuda_rocm_eps));

      transformers.emplace_back(std::make_unique<FastGeluFusion>(cpu_cuda_rocm_eps));

      transformers.emplace_back(std::make_unique<MatMulScaleFusion>(cpu_cuda_rocm_eps));

      // GeluApproximation has side effects which may change results. It needs to be manually enabled,
      // or alternatively the model can be updated offline using a model conversion script
      //   e.g. fusion_gelu_approximation function used by onnxruntime/python/tools/transformers/onnx_model_bert.py
      if (enable_gelu_approximation) {
        transformers.emplace_back(std::make_unique<GeluApproximation>(cpu_cuda_rocm_eps));
      }

#endif
      if (enable_quant_qdq_cleanup) {
        transformers.emplace_back(std::make_unique<QDQFinalCleanupTransformer>());
      }
    } break;

    case TransformerLevel::Level3: {
#ifndef DISABLE_CONTRIB_OPS
      // Register the NCHWc layout transformer if supported by the platform.
      if (MlasNchwcGetBlockSize() > 1) {
        transformers.emplace_back(std::make_unique<NchwcTransformer>());
      }
      auto cpu_allocator = cpu_execution_provider.GetAllocator(0, OrtMemTypeDefault);
      transformers.emplace_back(std::make_unique<NhwcTransformer>(std::move(cpu_allocator)));
      // NCHWCtransformer should have a higher priority versus this. Because NCHWCtransformer also do the similiar things
      // of fusion patterns and target on CPU. However, NCHWCtransformer will reorder the layout to nchwc which is only available for
      // x86-64 cpu, not edge cpu like arm. But This tranformer could be used by opencl-ep/cpu-ep. So
      // we will prefer NhwcTransformer once ort runs on x86-64 CPU, otherwise ConvAddActivationFusion is enabled.
      // this PR #6351 implemented similiar fusion-pattern but only for CUDA, and can only fuse conv-add-relu, while we can fuse more activation.
      transformers.emplace_back(std::make_unique<ConvAddActivationFusion>(cpu_ep));
#endif
    } break;

    default:
      ORT_THROW("Unsupported optimization level: ", static_cast<int>(level));
  }

  FilterTransformers(transformers, rules_and_transformers_to_disable);

  return transformers;
}

#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

InlinedVector<std::unique_ptr<GraphTransformer>> GenerateTransformersForMinimalBuild(
    TransformerLevel level,
    const SessionOptions& session_options,
    const SatApplyContextVariant& apply_context,
    const IExecutionProvider& cpu_execution_provider,
    const InlinedHashSet<std::string>& rules_and_transformers_to_disable) {
  InlinedVector<std::unique_ptr<GraphTransformer>> transformers;
  const bool saving = std::holds_alternative<SatRuntimeOptimizationSaveContext>(apply_context);

  switch (level) {
    case TransformerLevel::Level1:
      break;
    case TransformerLevel::Level2: {
#if !defined(DISABLE_CONTRIB_OPS)
      const bool disable_quant_qdq =
          session_options.config_options.GetConfigOrDefault(kOrtSessionOptionsDisableQuantQDQ, "0") == "1";
      const bool qdq_is_int8_allowed =
          session_options.config_options.GetConfigOrDefault(kOrtSessionOptionsQDQIsInt8Allowed,
                                                            QDQIsInt8Allowed() ? "1" : "0") == "1";

      // runtime optimizations only support CPU EP now
      const InlinedHashSet<std::string_view> cpu_ep = {onnxruntime::kCpuExecutionProvider};

      if (!disable_quant_qdq) {
        transformers.emplace_back(std::make_unique<QDQSelectorActionTransformer>(qdq_is_int8_allowed, apply_context));
      }

      transformers.emplace_back(std::make_unique<ConvActivationFusion>(cpu_ep, apply_context));
#else   // !defined(DISABLE_CONTRIB_OPS)
      ORT_UNUSED_PARAMETER(apply_context);
#endif  // !defined(DISABLE_CONTRIB_OPS)

      const bool enable_quant_qdq_cleanup =
          session_options.config_options.GetConfigOrDefault(kOrtSessionOptionsEnableQuantQDQCleanup, "0") == "1";
      if (!saving && enable_quant_qdq_cleanup) {
        transformers.emplace_back(std::make_unique<QDQFinalCleanupTransformer>());
      }

      break;
    }
    case TransformerLevel::Level3: {
      // currently the only level 3 optimizer is the NhwcTransformer which is fully supported at runtime
      if (!saving) {
#ifndef DISABLE_CONTRIB_OPS
        auto cpu_allocator = cpu_execution_provider.GetAllocator(0, OrtMemTypeDefault);
        transformers.emplace_back(std::make_unique<NhwcTransformer>(std::move(cpu_allocator)));
#else
        ORT_UNUSED_PARAMETER(cpu_execution_provider);
#endif
      }
    } break;
    default:
      ORT_THROW("Unsupported optimization level: ", static_cast<int>(level));
  }

  FilterTransformers(transformers, rules_and_transformers_to_disable);

  return transformers;
}

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

}  // namespace onnxruntime::optimizer_utils
