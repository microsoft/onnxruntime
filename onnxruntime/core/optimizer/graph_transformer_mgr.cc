// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/rule_based_graph_transformer.h"
using namespace onnxruntime;
using namespace ::onnxruntime::common;

namespace onnxruntime {

common::Status GraphTransformerManager::ApplyTransformers(Graph& graph, std::vector<std::string> providers, const TransformerLevel& level) const {
  const auto& transformers = level_to_transformer_map_.find(level);
  if (transformers == level_to_transformer_map_.end()) {
    Status(ONNXRUNTIME, FAIL, "No transformers registered for this level" + level);
  }

  // sort the providers. This makes it easy to find intersection of requested providers for this run
  // and registered providers for a transformer
  std::sort(providers.begin(), providers.end());

  for (unsigned step = 0; step < steps_; ++step) {
    bool graph_changed = false;
    for (const auto& transformer : transformers->second) {
      const auto compatible_providers = GenerateCompatibleProvidersList(providers, transformer->Name());
      bool modified = false;
      ORT_RETURN_IF_ERROR(transformer->Apply(graph, modified, compatible_providers));
      graph_changed = graph_changed || modified;
    }
    if (!graph_changed) {
      break;
    }
  }

  return Status::OK();
}

common::Status GraphTransformerManager::ApplyTransformers(Graph& graph, std::vector<std::string> providers, const TransformerLevel& level, const std::vector<std::string>& transformers) const {
  std::vector<std::string> rewriteRules;
  std::vector<const GraphTransformer*> standalone_transformers;
  for (const auto& name : transformers) {
    const auto& entry = transformers_info_.find(name);
    if (entry == transformers_info_.end()) {
      continue;
    }

    const auto& transformer_info = entry->second;
    if (transformer_info.level != level) {
      continue;
    }

    if (transformer_info.isRewriteRule) {
      rewriteRules.push_back(name);
    } else if (!transformer_info.isRuleBasedTransformer) {
      standalone_transformers.push_back(transformer_info.transformer);
    } else {
      return Status(ONNXRUNTIME, FAIL,
                    "Rule based transformer cannot be specified in the filter. Use levels from session options instead. Error encountered while processing " + name);
    }
  }

  // sort the providers. This makes it easy to find intersection of requested providers for this run
  // and registered providers for a transformer
  std::sort(providers.begin(), providers.end());

  for (unsigned step = 0; step < steps_; ++step) {
    bool graph_changed = false;
    for (const auto& transformer : standalone_transformers) {
      bool modified = true;
      const auto compatible_providers = GenerateCompatibleProvidersList(providers, transformer->Name());
      transformer->Apply(graph, modified, compatible_providers);
      graph_changed = graph_changed || modified;
    }

    if (!rewriteRules.empty()) {
      const auto& rewriterule_entry = transformers_info_.find(rewriteRules.at(0));
      const auto& wrapper = rewriterule_entry->second;
      bool modified = false;
      const auto compatible_providers = GenerateCompatibleProvidersList(providers, wrapper.transformer->Name());
      //TODO Add interface in rule based transformer to accept a list of rewrite rules.
      // Once that interface is ready - change the line below to something like : transformer->Apply(graph, modified, providers, rewriteRules); 
      wrapper.transformer->Apply(graph, modified, providers);
      graph_changed = graph_changed || modified;
    }

    if (!graph_changed) {
      break;
    }
  }

  return Status::OK();
}

common::Status GraphTransformerManager::Register(std::unique_ptr<GraphTransformer> transformer, const TransformerLevel& level, std::vector<std::string>&& provider) {
  return Register(std::move(transformer), level, std::move(provider), false);
}

common::Status GraphTransformerManager::Register(const std::string& op_type, std::unique_ptr<RewriteRule> rewrite_rule, const TransformerLevel& level) {
  // 1 Rule based transformer is registered for every level
  // If this transformer is not present then create a new rule based transformer for this level
  // and then register this rewrite rule
  std::string name = "";
  if (level == TransformerLevel::Optional_L1) {
    name = l1_rule_based_transformer_;
    if (transformers_info_.find(name) == transformers_info_.end()) {
      // Create and register L1 rule based transformer
      auto graph_rewrite_rules = std::make_unique<TopDownRuleBasedTransformer>(name, "Top down transformer");
      Register(std::move(graph_rewrite_rules), level, {"", onnxruntime::kCpuExecutionProvider}, true);
    }

  } else if (level == TransformerLevel::Optional_L2) {
    name = l2_rule_based_transformer_;
    if (transformers_info_.find(name) == transformers_info_.end()) {
      // Create and register L1 rule based transformer
      auto graph_rewrite_rules = std::make_unique<TopDownRuleBasedTransformer>(name, "Top down transformer");
      Register(std::move(graph_rewrite_rules), level, {onnxruntime::kCpuExecutionProvider}, true);
    }

  } else {
    return Status(ONNXRUNTIME, FAIL, "Requested level not supported " + level);
  }

  // Get transformer for this level
  auto transformer = dynamic_cast<RuleBasedGraphTransformer*>(transformers_info_[name].transformer);
  if (transformer == nullptr) {
    return Status(ONNXRUNTIME, FAIL, "Error while fetching rule based transformer for level" + level);
  }
  // Register rewrite rule to rule based transformer for this level
  const auto& rewrite_rule_name = rewrite_rule->Name();
  TransformerInfo meta_data{level, false, true, {}, transformer};
  transformers_info_[rewrite_rule_name] = std::move(meta_data);
  transformer->Register(op_type, std::move(rewrite_rule));

  return Status::OK();
}

common::Status GraphTransformerManager::Register(std::unique_ptr<GraphTransformer> transformer, const TransformerLevel& level, std::vector<std::string>&& provider, bool isRulebasedTransformer) {
  const auto& name = transformer->Name();
  if (transformers_info_.find(name) != transformers_info_.end()) {
    return Status(ONNXRUNTIME, FAIL, "This transformer is already registered " + name);
  }

  std::sort(provider.begin(), provider.end());
  TransformerInfo transformer_info{level, false, false, std::move(provider), transformer.get()};
  transformers_info_[name] = std::move(transformer_info);
  level_to_transformer_map_[level].push_back(std::move(transformer));
  return Status::OK();
}

}  // namespace onnxruntime
