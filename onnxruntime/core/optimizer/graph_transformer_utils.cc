// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/graph_transformer_utils.h"

#include "core/mlas/inc/mlas.h"
#include "core/optimizer/attention_fusion.h"
#include "core/optimizer/bias_gelu_fusion.h"
#include "core/optimizer/bias_softmax_fusion.h"
#include "core/optimizer/cast_elimination.h"
#include "core/optimizer/common_subexpression_elimination.h"
#include "core/optimizer/constant_folding.h"
#include "core/optimizer/conv_activation_fusion.h"
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
#include "core/optimizer/gemm_transpose_fusion.h"
#include "core/optimizer/identity_elimination.h"
#include "core/optimizer/layer_norm_fusion.h"
#include "core/optimizer/matmul_add_fusion.h"
#include "core/optimizer/matmul_integer_to_float.h"
#include "core/optimizer/matmul_scale_fusion.h"
#include "core/optimizer/nchwc_transformer.h"
#include "core/optimizer/nhwc_transformer.h"
#include "core/optimizer/not_where_fusion.h"
#include "core/optimizer/relu_clip_fusion.h"
#include "core/optimizer/reshape_fusion.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/shape_to_initializer.h"
#include "core/optimizer/skip_layer_norm_fusion.h"
#include "core/optimizer/slice_elimination.h"
#include "core/optimizer/unsqueeze_elimination.h"
#include "core/optimizer/qdq_transformer/qdq_propagation.h"
#include "core/optimizer/qdq_transformer/qdq_s8_to_u8.h"
#include "core/optimizer/qdq_transformer/qdq_transformer.h"
#include "core/optimizer/qdq_transformer/relu_quantizelinear.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/optimizer/matmul_transpose_fusion.h"
#include "core/optimizer/bias_dropout_fusion.h"

namespace onnxruntime {
class IExecutionProvider;

namespace optimizer_utils {

std::string GenerateRuleBasedTransformerName(TransformerLevel level) {
  return "Level" + std::to_string(static_cast<uint32_t>(level)) + "_RuleBasedTransformer";
}

std::vector<std::unique_ptr<RewriteRule>> GenerateRewriteRules(
    TransformerLevel level,
    const std::unordered_set<std::string>& rules_to_disable) {
  std::vector<std::unique_ptr<RewriteRule>> rules;
  switch (level) {
    case TransformerLevel::Level1:
      rules.push_back(std::make_unique<EliminateIdentity>());
      rules.push_back(std::make_unique<EliminateSlice>());
      rules.push_back(std::make_unique<UnsqueezeElimination>());
      rules.push_back(std::make_unique<EliminateDropout>());
      rules.push_back(std::make_unique<ExpandElimination>());
      rules.push_back(std::make_unique<CastElimination>());
      rules.push_back(std::make_unique<DivMulFusion>());
      rules.push_back(std::make_unique<FuseReluClip>());
      rules.push_back(std::make_unique<GemmTransposeFusion>());
      rules.push_back(std::make_unique<NotWhereFusion>());
      rules.push_back(std::make_unique<ShapeToInitializer>());
      rules.push_back(std::make_unique<ConvAddFusion>());
      rules.push_back(std::make_unique<ConvMulFusion>());
      rules.push_back(std::make_unique<ConvBNFusion>());
      rules.push_back(std::make_unique<ReluQuantFusion>());
      break;

    case TransformerLevel::Level2:
      // No level2 rules available today
      break;

    case TransformerLevel::Level3:
      break;

    default:
      ORT_ENFORCE(false, "Unsupported level" + std::to_string(static_cast<uint32_t>(level)));
  }

  if (rules_to_disable.empty()) {
    return rules;
  } else {
    std::vector<std::unique_ptr<RewriteRule>> filtered_list;
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
    const std::unordered_set<std::string>& rules_to_disable,
    const std::unordered_set<std::string>& compatible_execution_providers) {
  auto rewrite_rules_to_register = GenerateRewriteRules(level, rules_to_disable);
  if (rewrite_rules_to_register.empty()) {
    return nullptr;
  }

  std::unique_ptr<RuleBasedGraphTransformer> rule_transformer =
      std::make_unique<RuleBasedGraphTransformer>(GenerateRuleBasedTransformerName(level),
                                                  compatible_execution_providers);
  for (auto& entry : rewrite_rules_to_register) {
    rule_transformer->Register(std::move(entry));
  }

  return rule_transformer;
}

std::vector<std::unique_ptr<GraphTransformer>> GenerateTransformers(
    TransformerLevel level,
    const SessionOptions& session_options,
    const IExecutionProvider& execution_provider, /*required by constant folding*/
    const std::unordered_set<std::string>& rules_and_transformers_to_disable) {
  std::vector<std::unique_ptr<GraphTransformer>> transformers;
  std::unique_ptr<RuleBasedGraphTransformer> rule_transformer = nullptr;
  bool disable_quant_qdq = session_options.session_configurations.GetConfigOrDefault(kOrtSessionOptionsDisableQuantQDQ, "0") == "1";
#ifndef DISABLE_CONTRIB_OPS
  bool enable_gelu_approximation = session_options.session_configurations.GetConfigOrDefault(kOrtSessionOptionsEnableGeluApproximation, "0") == "1";
#endif

  switch (level) {
    case TransformerLevel::Level1: {
      // no filtering on execution provider for L1 optimizations as they only use official ONNX operators
      transformers.emplace_back(std::make_unique<CommonSubexpressionElimination>());
      transformers.emplace_back(std::make_unique<ConstantFolding>(execution_provider, !disable_quant_qdq));
      transformers.emplace_back(std::make_unique<MatMulAddFusion>());
      transformers.emplace_back(std::make_unique<ReshapeFusion>());
      transformers.emplace_back(std::make_unique<FreeDimensionOverrideTransformer>(
          session_options.free_dimension_overrides));

      rule_transformer = GenerateRuleBasedGraphTransformer(level, rules_and_transformers_to_disable, {});
    } break;

    case TransformerLevel::Level2: {
      std::unordered_set<std::string> cpu_ep = {onnxruntime::kCpuExecutionProvider};

      // create rule based transformer consisting of all the level2 rewrite rules
      rule_transformer = GenerateRuleBasedGraphTransformer(level, rules_and_transformers_to_disable, cpu_ep);

#ifndef DISABLE_CONTRIB_OPS
      const std::unordered_set<std::string> cuda_rocm_eps = {onnxruntime::kCudaExecutionProvider,
                                                             onnxruntime::kRocmExecutionProvider};
      const std::unordered_set<std::string> cpu_cuda_rocm_eps = {onnxruntime::kCpuExecutionProvider,
                                                                 onnxruntime::kCudaExecutionProvider,
                                                                 onnxruntime::kRocmExecutionProvider};
      const std::unordered_set<std::string> cpu_cuda_rocm_acl_armnn_eps = {onnxruntime::kCpuExecutionProvider,
                                                                           onnxruntime::kCudaExecutionProvider,
                                                                           onnxruntime::kRocmExecutionProvider,
                                                                           onnxruntime::kAclExecutionProvider,
                                                                           onnxruntime::kArmNNExecutionProvider};

      if (!disable_quant_qdq) {
        transformers.emplace_back(std::make_unique<QDQS8ToU8Transformer>(cpu_ep));
        transformers.emplace_back(std::make_unique<QDQPropagationTransformer>(cpu_ep));
        transformers.emplace_back(std::make_unique<QDQTransformer>());
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
    } break;

    case TransformerLevel::Level3: {
#ifndef DISABLE_CONTRIB_OPS
      // Register the NCHWc layout transformer if supported by the platform.
      if (MlasNchwcGetBlockSize() > 1) {
        transformers.emplace_back(std::make_unique<NchwcTransformer>());
      }

      transformers.emplace_back(std::make_unique<NhwcTransformer>());
#endif
    } break;

    default:
      ORT_ENFORCE(false, "Unsupported level " + std::to_string(static_cast<uint32_t>(level)));
      break;
  }

  if (rule_transformer != nullptr) {
    transformers.emplace_back(std::move(rule_transformer));
  }

  if (rules_and_transformers_to_disable.empty()) {
    return transformers;
  } else {
    // filter out any disabled transformers
    std::vector<std::unique_ptr<GraphTransformer>> filtered_list;
    auto end = rules_and_transformers_to_disable.cend();
    std::for_each(transformers.begin(), transformers.end(),
                  [&](std::unique_ptr<GraphTransformer>& item) {
                    if ((item != nullptr) && (rules_and_transformers_to_disable.find(item->Name()) == end)) {
                      filtered_list.push_back(std::move(item));
                    }
                  });

    return filtered_list;
  }
}

}  // namespace optimizer_utils
}  // namespace onnxruntime
