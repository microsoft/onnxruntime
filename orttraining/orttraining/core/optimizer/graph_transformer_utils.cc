// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/optimizer/graph_transformer_utils.h"

#include "core/mlas/inc/mlas.h"
#include "core/optimizer/bias_gelu_fusion.h"
#include "core/optimizer/cast_elimination.h"
#include "core/optimizer/computation_reduction.h"
#include "core/optimizer/constant_folding.h"
#include "core/optimizer/conv_activation_fusion.h"
#include "core/optimizer/conv_add_fusion.h"
#include "core/optimizer/conv_bn_fusion.h"
#include "core/optimizer/conv_mul_fusion.h"
#include "core/optimizer/dropout_elimination.h"
#include "core/optimizer/embed_layer_norm_fusion.h"
#include "core/optimizer/expand_elimination.h"
#include "core/optimizer/fast_gelu_fusion.h"
#include "core/optimizer/free_dim_override_transformer.h"
#include "core/optimizer/gelu_approximation.h"
#include "core/optimizer/gelu_fusion.h"
#include "core/optimizer/gemm_activation_fusion.h"
#include "core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/identity_elimination.h"
#include "core/optimizer/layer_norm_fusion.h"
#include "core/optimizer/matmul_add_fusion.h"
#include "core/optimizer/matmul_scale_fusion.h"
#include "core/optimizer/matmul_transpose_fusion.h"
#include "core/optimizer/nchwc_transformer.h"
#include "core/optimizer/relu_clip_fusion.h"
#include "core/optimizer/reshape_fusion.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/shape_to_initializer.h"
#include "core/optimizer/skip_layer_norm_fusion.h"
#include "core/optimizer/slice_elimination.h"
#include "core/optimizer/unsqueeze_elimination.h"
#include "core/session/inference_session.h"
#include "orttraining/core/framework/distributed_run_context.h"
#include "orttraining/core/optimizer/bias_dropout_fusion.h"
#include "orttraining/core/optimizer/concat_replacement.h"
#include "orttraining/core/optimizer/insert_output_rewriter.h"
#include "orttraining/core/optimizer/localized_recompute.h"
#include "orttraining/core/optimizer/megatron_transformer.h"
#include "orttraining/core/optimizer/nonzero_shape_setter.h"

namespace onnxruntime {
namespace training {
namespace transformer_utils {

std::vector<std::unique_ptr<GraphTransformer>> GeneratePreTrainingTransformers(
    TransformerLevel level,
    const std::unordered_set<std::string>& weights_to_train,
    const TrainingSession::TrainingConfiguration::GraphTransformerConfiguration& config,
    const std::vector<std::string>& transformers_and_rules_to_enable) {
  std::vector<std::unique_ptr<GraphTransformer>> transformers;
  std::unique_ptr<RuleBasedGraphTransformer> rule_transformer = nullptr;

  // MUST be empty here, because this is called before partition, so the node's execution type is not decided yet.
  // If we give values here, the check in transformer will fail.
  std::unordered_set<std::string> compatible_eps = {};

  switch (level) {
    case TransformerLevel::Level1: {
      rule_transformer =
          onnxruntime::make_unique<RuleBasedGraphTransformer>(optimizer_utils::GenerateRuleBasedTransformerName(level),
                                                              compatible_eps);
      rule_transformer->Register(make_unique<InsertMaxPoolOutput>());
      rule_transformer->Register(make_unique<AdjustBatchNormOutputs>());
      rule_transformer->Register(make_unique<UnsqueezeElimination>());
      rule_transformer->Register(make_unique<ExpandElimination>());
      rule_transformer->Register(make_unique<CastElimination>());
      rule_transformer->Register(make_unique<NonZeroShapeSetter>());
      rule_transformer->Register(make_unique<InsertSoftmaxCrossEntropyLossOutput>());
      if (config.gelu_checkpoint) {
        rule_transformer->Register(make_unique<GeluRecompute>());
      }
      if (config.attn_dropout_checkpoint) {
        rule_transformer->Register(make_unique<AttentionDropoutRecompute>());
      }

      transformers.emplace_back(onnxruntime::make_unique<GeluFusion>(compatible_eps));
      transformers.emplace_back(onnxruntime::make_unique<LayerNormFusion>(compatible_eps));
      transformers.emplace_back(onnxruntime::make_unique<FastGeluFusion>(compatible_eps));

      transformers.emplace_back(onnxruntime::make_unique<BiasGeluFusion>(compatible_eps));

      if (config.enable_gelu_approximation) {
        transformers.emplace_back(onnxruntime::make_unique<GeluApproximation>(compatible_eps));
      }

      transformers.emplace_back(onnxruntime::make_unique<ConstantFolding>(compatible_eps, weights_to_train));
      auto horizontal_parallel_size = training::DistributedRunContext::GroupSize(training::WorkerGroupType::HorizontalParallel);
      if (horizontal_parallel_size > 1) {
        LOGS_DEFAULT(WARNING) << horizontal_parallel_size << "-way horizontal model parallel is enabled";
        transformers.emplace_back(onnxruntime::make_unique<MegatronTransformer>(
            training::DistributedRunContext::RankInGroup(training::WorkerGroupType::HorizontalParallel),
            horizontal_parallel_size, compatible_eps));
      }
      transformers.emplace_back(onnxruntime::make_unique<ComputationReductionTransformer>(compatible_eps));
    } break;

    case TransformerLevel::Level2: {
      // Put ReshapeFusion as level-2 optimization after all level-1 graph rewriters are run.
      transformers.emplace_back(onnxruntime::make_unique<ReshapeFusion>(compatible_eps));
      rule_transformer =
          onnxruntime::make_unique<RuleBasedGraphTransformer>(optimizer_utils::GenerateRuleBasedTransformerName(level),
                                                              compatible_eps);
      rule_transformer->Register(onnxruntime::make_unique<ConcatReplacement>());
    } break;

    case TransformerLevel::Level3: {
    } break;

    default:
      ORT_ENFORCE(false, "Unsupported level " + std::to_string(static_cast<uint32_t>(level)));
      break;
  }

  // if the custom list to enable transformers\rules is empty then return the default generated transformers and rules
  // otherwise generate a filtered list based on the provided custom list.
  // Note that some rule-based transformers are depending on some custom transformers,
  // e.g., ExpandElimination and CastElimination are depending on ConstantFolding to fold the constant first,
  // so we should always push the rule-based transformer to the end, this is expecially important when transformation step is 1.
  if (transformers_and_rules_to_enable.empty()) {
    if (rule_transformer != nullptr) {
      transformers.emplace_back(std::move(rule_transformer));
    }
    return transformers;
  }
  std::vector<std::unique_ptr<GraphTransformer>> filtered_list;
  // pick custom transformers enabled for this session
  for (const auto& t_name : transformers_and_rules_to_enable) {
    std::for_each(transformers.begin(), transformers.end(),
                  [&](std::unique_ptr<GraphTransformer>& item) {
                    if ((item != nullptr) && (item->Name() == t_name)) {
                      filtered_list.push_back(std::move(item));
                    }
                  });
  }
  // If the rule-based transformer is not empty, it should be included in the custom transformer list below.
  if (rule_transformer != nullptr) {
    filtered_list.emplace_back(std::move(rule_transformer));
  }
  return filtered_list;
}

std::vector<std::unique_ptr<GraphTransformer>> GenerateTransformers(
    TransformerLevel level,
    const std::unordered_set<std::string>& weights_to_train,
    gsl::span<const FreeDimensionOverride> free_dimension_overrides,
    const std::vector<std::string>& transformers_and_rules_to_enable) {
  std::vector<std::unique_ptr<GraphTransformer>> transformers;
  std::unique_ptr<RuleBasedGraphTransformer> rule_transformer = nullptr;
  switch (level) {
    case TransformerLevel::Level1: {
      std::unordered_set<std::string> l1_execution_providers = {};

      // TODO hack - constant folding currently doesn't work after mixed precision transformation so it's disabled for now
      //             ORT uses CPU kernels to evaluate constant values but some of them don't support fp16
      //transformers.emplace_back(onnxruntime::make_unique<ConstantFolding>(l1_execution_providers));
      transformers.emplace_back(onnxruntime::make_unique<MatMulAddFusion>(l1_execution_providers));
      transformers.emplace_back(onnxruntime::make_unique<FreeDimensionOverrideTransformer>(free_dimension_overrides));
      transformers.emplace_back(onnxruntime::make_unique<MatmulTransposeFusion>(l1_execution_providers));
      transformers.emplace_back(onnxruntime::make_unique<BiasDropoutFusion>(l1_execution_providers));
      transformers.emplace_back(onnxruntime::make_unique<MatMulScaleFusion>(l1_execution_providers, weights_to_train));

      rule_transformer = optimizer_utils::GenerateRuleBasedGraphTransformer(level, transformers_and_rules_to_enable, l1_execution_providers);
    } break;

    case TransformerLevel::Level2: {
      std::unordered_set<std::string> cpu_execution_providers = {onnxruntime::kCpuExecutionProvider};

      // create rule based transformer consisting of all the level2 rewrite rules
      rule_transformer = optimizer_utils::GenerateRuleBasedGraphTransformer(level, transformers_and_rules_to_enable, cpu_execution_providers);

      transformers.emplace_back(onnxruntime::make_unique<GemmActivationFusion>(cpu_execution_providers));
      transformers.emplace_back(onnxruntime::make_unique<ConvActivationFusion>(cpu_execution_providers));
    } break;

    case TransformerLevel::Level3: {
      // Register the NCHWc layout transformer if supported by the platform.
      if (MlasNchwcGetBlockSize() > 1) {
        transformers.emplace_back(onnxruntime::make_unique<NchwcTransformer>());
      }
    } break;

    default:
      ORT_ENFORCE(false, "Unsupported level " + std::to_string(static_cast<uint32_t>(level)));
      break;
  }

  // if the custom list to enable transformers\rules is empty then return the default generated transformers and rules
  // otherwise generate a filtered list based on the provided custom list.
  if (transformers_and_rules_to_enable.empty()) {
    if (rule_transformer != nullptr) {
      transformers.emplace_back(std::move(rule_transformer));
    }
    return transformers;
  }
  std::vector<std::unique_ptr<GraphTransformer>> filtered_list;
  // If the rule-based transformer is not empty, it should be included in the custom transformer list below.
  if (rule_transformer != nullptr) {
    filtered_list.emplace_back(std::move(rule_transformer));
  }
  // pick custom transformers enabled for this session
  for (const auto& t_name : transformers_and_rules_to_enable) {
    std::for_each(transformers.begin(), transformers.end(),
                  [&](std::unique_ptr<GraphTransformer>& item) {
                    if ((item != nullptr) && (item->Name() == t_name)) {
                      filtered_list.push_back(std::move(item));
                    }
                  });
  }
  return filtered_list;
}

}  // namespace transformer_utils
}  // namespace training
}  // namespace onnxruntime
