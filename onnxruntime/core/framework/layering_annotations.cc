// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#include "core/graph/constants.h"
#include "core/common/narrow.h"
#include "core/common/parse_string.h"
#include "core/common/string_utils.h"
#include "core/framework/layering_annotations.h"
#include "core/framework/ortmemoryinfo.h"
#include "core/session/abi_devices.h"
#include "core/framework/execution_providers.h"
#include "core/graph/graph.h"

#include <algorithm>
#include <limits>

namespace onnxruntime {

common::Status LayeringRules::FromConfigString(const std::string& config_value, LayeringRules& rules) {
  rules.rules.clear();
  if (config_value.empty()) {
    return common::Status::OK();
  }

  // Track seen annotations to reject duplicates.
  // Separate sets for exact and prefix match annotations.
  InlinedHashSet<std::string> seen_exact_annotations;
  InlinedHashSet<std::string> seen_prefix_annotations;

  auto entries = utils::SplitString(config_value, ";");
  for (const auto& e : entries) {
    auto entry = utils::TrimString(e);
    if (entry.empty()) {
      continue;
    }

    const size_t open_paren = entry.find('(');
    const size_t close_paren = entry.find(')');

    if (open_paren == std::string::npos) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid layering config: Missing '(' in entry: ", entry);
    }
    if (close_paren == std::string::npos) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid layering config: Missing ')' in entry: ", entry);
    }
    if (close_paren < open_paren) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid layering config: ')' comes before '(' in entry: ", entry);
    }

    std::string device = entry.substr(0, open_paren);
    device = utils::TrimString(device);

    if (device.empty()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid layering config: Empty device name in entry: ", entry);
    }

    std::string annotations_list = entry.substr(open_paren + 1, close_paren - open_paren - 1);
    auto annotations = utils::SplitString(annotations_list, ",");
    for (auto& a : annotations) {
      auto ann = utils::TrimString(a);
      if (ann.empty()) {
        continue;
      }

      bool prefix_match = true;
      if (ann[0] == '=') {
        prefix_match = false;
        ann = ann.substr(1);
        ann = utils::TrimString(ann);
      }

      if (ann.empty()) {
        continue;
      }

      // Check for duplicate annotation (same annotation string and match type)
      auto& seen_set = prefix_match ? seen_prefix_annotations : seen_exact_annotations;
      auto [it, inserted] = seen_set.insert(ann);
      if (!inserted) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Invalid layering config: Duplicate ", (prefix_match ? "prefix" : "exact"),
                               " match annotation '", ann, "' found in entry: ", entry);
      }

      rules.rules.push_back({device, std::move(ann), prefix_match});
    }
  }

  return common::Status::OK();
}

LayeringRuleMatcher::LayeringRuleMatcher(const LayeringRules& rules) {
  for (size_t i = 0; i < rules.rules.size(); ++i) {
    const auto& rule = rules.rules[i];
    ORT_ENFORCE(!rule.annotation.empty(), "Layering rule annotation cannot be empty");
    if (rule.prefix_match) {
      AddPrefixRule(rule.annotation, i);
    } else {
      AddExactRule(rule.annotation, i);
    }
  }
}

std::optional<size_t> LayeringRuleMatcher::Match(const std::string& node_annotation) const {
  std::optional<size_t> best_match = std::nullopt;

  // 1. Check Prefix Matches via Trie. Prefix have priority over exact matches.
  const TrieNode* current = &root_;

  // No empty annotations
  // so we omit checking the root.

  for (char c : node_annotation) {
    if (best_match && *best_match == 0) {
      // Optimization: If we already found index 0, we can't do better.
      return best_match;
    }

    auto child_it = current->children.find(c);
    if (child_it == current->children.end()) {
      break;
    }
    current = child_it->second.get();
    if (current->rule_index) {
      UpdateBestMatch(best_match, *current->rule_index);
    }
  }

  if (best_match) {
    return best_match;
  }

  // 2. Check Exact Matches (fallback)
  auto it = exact_match_rules_.find(node_annotation);
  if (it != exact_match_rules_.end()) {
    best_match = it->second;
  }

  return best_match;
}

namespace {
bool CaseInsensitiveCompare(std::string_view a, std::string_view b) {
  return std::equal(a.begin(), a.end(), b.begin(), b.end(),
                    [](char c1, char c2) {
                      return std::tolower(static_cast<unsigned char>(c1)) ==
                             std::tolower(static_cast<unsigned char>(c2));
                    });
}

bool TryParseIndex(const std::string& str, uint32_t& index) {
  if (str.empty()) return false;
  return TryParseStringWithClassicLocale(str, index);
}

// Sentinel value representing an unknown/unavailable device type.
// Used when an OrtEpDevice has neither hardware info nor memory info,
// so we cannot determine the actual device type.
constexpr OrtDevice::DeviceType kDeviceTypeUnknown = static_cast<OrtDevice::DeviceType>(-1);

// Normalized view of an EP's device properties used by the matching logic.
// All fields are non-owning references or value types.
struct EpDeviceView {
  std::string_view ep_name;
  OrtDevice::DeviceType device_type;  // OrtDevice::CPU, GPU, NPU, FPGA, or kDeviceTypeUnknown
  uint32_t vendor_id;
  OrtDevice::DeviceId device_id;
  bool has_device_ordinal;         // true when device_id is a runtime ordinal (from device_memory_info)
  std::string_view vendor_string;  // from OrtHardwareDevice::vendor (empty if unavailable)
};

bool MatchEpDevice(const EpDeviceView& ep,
                   std::string_view target_type_str,
                   std::string_view target_specifier,
                   std::string_view target_full) {
  // "cpu"
  if (CaseInsensitiveCompare(target_type_str, "cpu")) {
    return ep.ep_name == kCpuExecutionProvider ||
           ep.device_type == OrtDevice::CPU;
  }
  // "gpu"
  if (CaseInsensitiveCompare(target_type_str, "gpu")) {
    if (target_specifier.empty()) {
      if (ep.device_type == OrtDevice::GPU) return true;
      // Heuristic fallback for common GPU EPs if hardware info is missing
      return ep.ep_name == kCudaExecutionProvider || ep.ep_name == kCudaExecutionProviderPluginAlias ||
             ep.ep_name == kDmlExecutionProvider;
    }
    // "gpu:<vendor>" or "gpu:<index>"
    if (ep.device_type == OrtDevice::GPU) {
      uint32_t index = std::numeric_limits<uint32_t>::max();
      if (TryParseIndex(std::string(target_specifier), index)) {
        // Only match by ordinal index when the device_id is known to be a runtime
        // ordinal (sourced from device_memory_info). OrtHardwareDevice::device_id is
        // a PCI hardware-type identifier, not a device instance ordinal.
        return ep.has_device_ordinal &&
               ep.device_id == static_cast<OrtDevice::DeviceId>(index);
      }
      // gpu:<vendor>
      if (!ep.vendor_string.empty() && CaseInsensitiveCompare(ep.vendor_string, target_specifier)) {
        return true;
      }
      if (CaseInsensitiveCompare(target_specifier, "nvidia") &&
          ep.vendor_id == OrtDevice::VendorIds::NVIDIA) return true;
      if (CaseInsensitiveCompare(target_specifier, "amd") &&
          ep.vendor_id == OrtDevice::VendorIds::AMD) return true;
      if (CaseInsensitiveCompare(target_specifier, "intel") &&
          ep.vendor_id == OrtDevice::VendorIds::INTEL) return true;
      // Heuristic: gpu:nvidia -> CUDA
      if (CaseInsensitiveCompare(target_specifier, "nvidia") &&
          (ep.ep_name == kCudaExecutionProvider || ep.ep_name == kCudaExecutionProviderPluginAlias)) return true;
    }
    return false;
  }
  // "accelerator" (not cpu)
  if (CaseInsensitiveCompare(target_type_str, "accelerator")) {
    // Match if the EP is not a known CPU provider and its device type
    // is not definitively CPU. Unknown device type (no HW/mem info)
    // is treated as a potential accelerator.
    return ep.ep_name != kCpuExecutionProvider && ep.device_type != OrtDevice::CPU;
  }
  // "npu"
  if (CaseInsensitiveCompare(target_type_str, "npu")) {
    if (ep.device_type == OrtDevice::NPU) return true;
    return ep.ep_name == kQnnExecutionProvider || ep.ep_name == kVitisAIExecutionProvider;
  }
  // "fpga"
  if (CaseInsensitiveCompare(target_type_str, "fpga")) {
    return ep.device_type == OrtDevice::FPGA;
  }
  // "cuda"
  if (CaseInsensitiveCompare(target_type_str, "cuda")) {
    return ep.ep_name == kCudaExecutionProvider || ep.ep_name == kCudaExecutionProviderPluginAlias;
  }
  // "dml"
  if (CaseInsensitiveCompare(target_type_str, "dml")) {
    return ep.ep_name == kDmlExecutionProvider;
  }
  // Fallback: exact EP name match
  return ep.ep_name == target_full;
}

void ParseDeviceTarget(const std::string& target_full,
                       std::string& target_type_str,
                       std::string& target_specifier) {
  const auto colon_pos = target_full.find(':');
  target_type_str = (colon_pos == std::string::npos) ? target_full : target_full.substr(0, colon_pos);
  target_specifier = (colon_pos != std::string::npos) ? target_full.substr(colon_pos + 1) : std::string();
}

}  // namespace

std::optional<std::string> EpLayeringMatcher::Match(gsl::span<const OrtEpDevice* const> ep_devices,
                                                    const LayerAnnotation& rule) {
  std::string target_type_str, target_specifier;
  ParseDeviceTarget(rule.device, target_type_str, target_specifier);

  for (const auto* ep_device_ptr : ep_devices) {
    if (!ep_device_ptr) continue;
    const OrtEpDevice& ep_device = *ep_device_ptr;

    // Build normalized view from OrtEpDevice.
    // Device type comes from either the hardware device or the memory info,
    // with hardware device taking priority. If neither is available,
    // device_type is set to kDeviceTypeUnknown.
    OrtDevice::DeviceType device_type = kDeviceTypeUnknown;
    bool has_hw = ep_device.device != nullptr;
    if (has_hw) {
      // Map OrtHardwareDeviceType to OrtDevice::DeviceType
      switch (ep_device.device->type) {
        case OrtHardwareDeviceType_GPU:
          device_type = OrtDevice::GPU;
          break;
        case OrtHardwareDeviceType_NPU:
          device_type = OrtDevice::NPU;
          break;
        case OrtHardwareDeviceType_CPU:
          device_type = OrtDevice::CPU;
          break;
        default:
          device_type = kDeviceTypeUnknown;
          break;
      }
    } else if (ep_device.device_memory_info) {
      device_type = ep_device.device_memory_info->device.Type();
    }

    EpDeviceView view{
        ep_device.ep_name,
        device_type,
        has_hw ? ep_device.device->vendor_id : 0u,
        // Use the device ordinal from device_memory_info (set by the EP factory to
        // a runtime device ordinal such as a CUDA ordinal). OrtHardwareDevice::device_id
        // is a PCI hardware-type identifier and must not be used for index-based matching.
        ep_device.device_memory_info
            ? ep_device.device_memory_info->device.Id()
            : OrtDevice::DeviceId{},
        /*has_device_ordinal=*/ep_device.device_memory_info != nullptr,
        has_hw ? std::string_view(ep_device.device->vendor) : std::string_view{}};

    if (MatchEpDevice(view, target_type_str, target_specifier, rule.device)) {
      return std::string(ep_device.ep_name);
    }
  }
  return std::nullopt;
}

std::optional<std::string> EpLayeringMatcher::Match(const ExecutionProviders& providers,
                                                    const LayerAnnotation& rule) {
  std::string target_type_str, target_specifier;
  ParseDeviceTarget(rule.device, target_type_str, target_specifier);

  for (const auto& ep_shared_ptr : providers) {
    if (!ep_shared_ptr) continue;
    const IExecutionProvider& ep = *ep_shared_ptr;
    const OrtDevice& device = ep.GetDevice();

    EpDeviceView view{
        ep.Type(),
        device.Type(),
        device.Vendor(),
        device.Id(),
        /*has_device_ordinal=*/true,  // IExecutionProvider sets device Id to a runtime ordinal
        {}};                          // no vendor string available from IExecutionProvider

    if (MatchEpDevice(view, target_type_str, target_specifier, rule.device)) {
      return std::string(ep.Type());
    }
  }
  return std::nullopt;
}

LayeringIndex LayeringIndex::Create(const Graph& graph,
                                    EpNameToLayeringIndices ep_map,
                                    LayeringIndexToEpName rule_map,
                                    LayeringRules layering_rules) {
  LayeringIndex index(std::move(layering_rules), std::move(ep_map), std::move(rule_map));
  index.ProcessGraph(graph, std::nullopt);
  return index;
}

LayeringIndex LayeringIndex::Create(const Graph& graph,
                                    EpNameToLayeringIndices ep_map,
                                    LayeringIndexToEpName rule_map,
                                    LayeringRules layering_rules,
                                    SubstringMatcher substring_matcher) {
  LayeringIndex index(std::move(layering_rules), std::move(ep_map), std::move(rule_map),
                      std::move(substring_matcher));
  index.ProcessGraph(graph, std::nullopt);
  return index;
}

Status LayeringIndex::Create(const Graph& graph,
                             const std::string& config_string,
                             const std::string& name_based_config_string,
                             gsl::span<const OrtEpDevice* const> ep_devices,
                             const ExecutionProviders& ep_providers,
                             const logging::Logger& logger,
                             std::optional<LayeringIndex>& layering_index) {
  // Annotation-based and name-based layer assignment are mutually exclusive.
  if (!config_string.empty() && !name_based_config_string.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Cannot set both 'session.layer_assignment_settings' and "
                           "'session.name_based_layer_assignment'. These options are mutually exclusive. "
                           "Use annotation-based matching for models with explicit layer annotations, "
                           "or name-based matching for models with structured node names.");
  }

  const bool is_name_based = !name_based_config_string.empty();
  const std::string& active_config = is_name_based ? name_based_config_string : config_string;

  LayeringRules rules;
  if (!active_config.empty()) {
    ORT_RETURN_IF_ERROR(LayeringRules::FromConfigString(active_config, rules));

    if (is_name_based) {
      // Reject '=' (exact-match qualifier) in name-based rules — all patterns must be substrings
      for (const auto& rule : rules.rules) {
        if (!rule.prefix_match) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Name-based layer assignment does not support the '=' (exact-match) qualifier. "
                                 "All patterns are treated as substrings. Remove the '=' prefix from pattern: '",
                                 rule.annotation, "'");
        }
      }
      LOGS(logger, INFO) << "Parsed " << rules.rules.size() << " name-based layering rules from config.";
    } else {
      LOGS(logger, INFO) << "Parsed " << rules.rules.size() << " annotation-based layering rules from config.";
    }
  }

  if (rules.rules.empty()) {
    layering_index.reset();
    return Status::OK();
  }

  // Identify which EPs satisfy which rules
  EpNameToLayeringIndices ep_map;
  LayeringIndexToEpName rule_map;

  size_t matched_rule_count = 0;

  for (size_t i = 0, lim = rules.rules.size(); i < lim; ++i) {
    const auto& rule = rules.rules[i];

    // 1. Try matching against ep_devices (from session options)
    std::optional<std::string> matched_ep;
    if (!ep_devices.empty()) {
      matched_ep = EpLayeringMatcher::Match(ep_devices, rule);
    }

    // 2. If not matched, try matching against Registered EPs
    if (!matched_ep) {
      matched_ep = EpLayeringMatcher::Match(ep_providers, rule);
    }

    if (matched_ep) {
      const std::string& ep_type = *matched_ep;
      ep_map[ep_type].insert(i);
      rule_map[i] = ep_type;
      matched_rule_count++;
      LOGS(logger, VERBOSE) << "Layering Rule " << i << " (" << rule.device << " -> " << rule.annotation
                            << ") mapped to EP: " << ep_type;
    } else {
      LOGS(logger, ERROR) << "Layering rule " << i << " (device='" << rule.device << "', annotation='" << rule.annotation
                          << "') could not be mapped to any available Execution Provider. "
                          << "If a numeric gpu index was specified (e.g. gpu:0), ensure an EP with a matching "
                          << "device ordinal is registered and reports device_memory_info.";
    }
  }

  LOGS(logger, INFO) << "LayeringIndex created. Matched " << matched_rule_count
                     << " out of " << rules.rules.size() << " rules to available Execution Providers.";

  // Build SubstringMatcher for name-based mode
  std::optional<SubstringMatcher> substring_matcher;
  if (is_name_based) {
    substring_matcher.emplace(rules);
  }

  // Create LayeringIndex — annotation mode uses matcher_ only, name-based uses substring_matcher_ only
  LayeringIndex index(std::move(rules), std::move(ep_map), std::move(rule_map),
                      std::move(substring_matcher));
  index.ProcessGraph(graph, std::nullopt);
  layering_index = std::move(index);
  return Status::OK();
}

void LayeringIndex::ProcessGraph(const Graph& graph, std::optional<size_t> parent_layer_id) {
  // 3. Create entry for this graph instance
  bool was_updated = false;
  std::optional<GraphLayeringIndex> new_index;
  GraphLayeringIndex* current_graph_index_ptr = nullptr;
  auto found = graph_index_.find(&graph);
  if (found != graph_index_.end()) {
    current_graph_index_ptr = &found->second;
  } else {
    new_index.emplace();
    current_graph_index_ptr = &(*new_index);
  }
  GraphLayeringIndex& current_graph_index = *current_graph_index_ptr;

  for (auto& node : graph.Nodes()) {
    std::optional<size_t> matched_rule_idx = std::nullopt;

    if (substring_matcher_) {
      // Name-based mode: substring matching against node name, no inheritance.
      // Node names are dense (virtually every node has one), so inheritance is
      // unnecessary — each node is matched independently by its own name.
      matched_rule_idx = substring_matcher_->Match(node.Name());
    } else {
      // Annotation-based mode: prefix/exact match against metadata annotation,
      // with subgraph inheritance for unannotated nodes.
      const std::string& annotation = node.GetLayeringAnnotation();
      if (!annotation.empty()) {
        matched_rule_idx = matcher_.Match(annotation);
      }

      // Inherit from subgraph parent node if no annotation match
      if (!matched_rule_idx && parent_layer_id) {
        matched_rule_idx = parent_layer_id;
      }
    }

    // Record assignment if we have a match
    if (matched_rule_idx) {
      const size_t rule_idx = *matched_rule_idx;

      // Only assign if this rule maps to a valid EP in our configuration
      if (layering_index_to_ep_name_.count(rule_idx)) {
        ORT_IGNORE_RETURN_VALUE(current_graph_index.node_to_layering_index_.insert_or_assign(node.Index(), rule_idx));
        ORT_IGNORE_RETURN_VALUE(current_graph_index.layer_to_node_ids_[rule_idx].insert(node.Index()));
        was_updated = true;
      } else {
        // reset since no valid EP mapping
        matched_rule_idx = std::nullopt;
      }
    }

    // Recurse for subgraphs
    if (node.ContainsSubgraph()) {
      const std::optional<size_t> subgraph_parent_assignment = matched_rule_idx;
      for (auto& [attr_name, subgraph] : node.GetAttributeNameToSubgraphMap()) {
        ProcessGraph(*subgraph, subgraph_parent_assignment);
      }
    }
  }
  if (was_updated && new_index) {
    graph_index_.emplace(&graph, std::move(*new_index));
  }
}

void LayeringIndex::Update(const Graph& graph, gsl::span<const NodeIndex> nodes) {
  // Ensure we have an entry for this graph (creating it if it doesn't exist, though typically it should)
  bool was_updated = false;
  std::optional<GraphLayeringIndex> new_index;
  GraphLayeringIndex* current_graph_index_ptr = nullptr;
  auto found = graph_index_.find(&graph);
  if (found != graph_index_.end()) {
    current_graph_index_ptr = &found->second;
  } else {
    new_index.emplace();
    current_graph_index_ptr = &(*new_index);
  }

  auto& current_graph_index = *current_graph_index_ptr;

  for (NodeIndex node_index : nodes) {
    // GetMutableNode because we want to ClearLayeringAnnotation if we use it
    const Node* node = graph.GetNode(node_index);
    if (!node) {
      continue;
    }

    std::optional<size_t> matched_rule_idx;

    if (substring_matcher_) {
      // Name-based mode: substring match against node name
      matched_rule_idx = substring_matcher_->Match(node->Name());
    } else {
      // Annotation-based mode: prefix/exact match against metadata
      const std::string& annotation = node->GetLayeringAnnotation();
      if (!annotation.empty()) {
        matched_rule_idx = matcher_.Match(annotation);
      }
    }

    if (matched_rule_idx) {
      const size_t rule_idx = *matched_rule_idx;

      // Only assign if this rule maps to a valid EP in our configuration
      if (layering_index_to_ep_name_.count(rule_idx)) {
        // Check if already assigned to a DIFFERENT rule, if so clean up old mapping
        auto prev_assign = current_graph_index.node_to_layering_index_.find(node_index);
        if (prev_assign != current_graph_index.node_to_layering_index_.end()) {
          size_t old_rule = prev_assign->second;
          if (old_rule != rule_idx) {
            current_graph_index.layer_to_node_ids_[old_rule].erase(node_index);
          }
        }

        ORT_IGNORE_RETURN_VALUE(current_graph_index.node_to_layering_index_.insert_or_assign(node_index, rule_idx));
        ORT_IGNORE_RETURN_VALUE(current_graph_index.layer_to_node_ids_[rule_idx].insert(node_index));
        was_updated = true;
      }
    }
  }
  if (was_updated && new_index) {
    graph_index_.emplace(&graph, std::move(*new_index));
  }
}

void LayeringRuleMatcher::AddExactRule(const std::string& annotation, size_t index) {
  // Only store the first occurrence (lowest index)
  exact_match_rules_.insert({annotation, index});
}

void LayeringRuleMatcher::AddPrefixRule(const std::string& annotation, size_t index) {
  TrieNode* current = &root_;
  for (char c : annotation) {
    auto p = current->children.insert({c, nullptr});
    if (p.second) {
      p.first->second = std::make_unique<TrieNode>();
    }
    current = p.first->second.get();
  }

  // Only store if strictly better (lower index) or not set
  // Since we iterate rules 0..N, if a rule index is already set for this node,
  // it corresponds to a higher priority rule, so we skip overwriting it.
  if (!current->rule_index) {
    current->rule_index = index;
  }
}

void LayeringRuleMatcher::UpdateBestMatch(std::optional<size_t>& current_best, size_t candidate) const {
  if (!current_best || candidate < *current_best) {
    current_best = candidate;
  }
}

SubstringMatcher::SubstringMatcher(const LayeringRules& rules) {
  for (size_t i = 0; i < rules.rules.size(); ++i) {
    const auto& rule = rules.rules[i];
    if (!rule.annotation.empty()) {
      patterns_.push_back({rule.annotation, i});
    }
  }
  // Sort by pattern length descending (longest first).
  // Stable sort preserves config order as tiebreaker for same-length patterns.
  std::stable_sort(patterns_.begin(), patterns_.end(),
                   [](const PatternEntry& a, const PatternEntry& b) {
                     return a.pattern.size() > b.pattern.size();
                   });
}

std::optional<size_t> SubstringMatcher::Match(std::string_view node_name) const {
  for (const auto& entry : patterns_) {
    if (node_name.find(entry.pattern) != std::string_view::npos) {
      return entry.rule_index;
    }
  }
  return std::nullopt;
}

std::optional<std::reference_wrapper<const InlinedHashSet<size_t>>>
LayeringIndex::GetLayeringRulesForThisEp(const std::string& ep_type) const {
  auto hit = ep_name_to_layering_indices_.find(ep_type);
  if (hit == ep_name_to_layering_indices_.end()) {
    return {};
  }
  return hit->second;
}

std::optional<size_t> LayeringIndex::GetNodeAssignment(const Graph& graph, NodeIndex node_id) const {
  auto hit = graph_index_.find(&graph);
  if (hit == graph_index_.end()) {
    return {};
  }

  // Nodes in subgraph that were not annotated has already inherited their
  // annotation if any from the parent node of the subgraph
  const auto& graph_layering_index = hit->second;
  auto layer_hit = graph_layering_index.node_to_layering_index_.find(node_id);
  if (layer_hit != graph_layering_index.node_to_layering_index_.end()) {
    return layer_hit->second;
  }
  return {};
}

void LayeringIndex::MakeNodeUnassigned(const Graph& graph, NodeIndex node_id) {
  auto hit = graph_index_.find(&graph);
  if (hit == graph_index_.end()) {
    return;
  }
  auto& graph_layering_index = hit->second;
  auto node_to_layer_hit = graph_layering_index.node_to_layering_index_.find(node_id);
  std::optional<size_t> layer_idx;
  if (node_to_layer_hit != graph_layering_index.node_to_layering_index_.end()) {
    // Get the layer index
    layer_idx = node_to_layer_hit->second;
    graph_layering_index.node_to_layering_index_.erase(node_to_layer_hit);
  }
  // Remove node from layer collection
  if (layer_idx) {
    auto layer_to_nodes_hit = graph_layering_index.layer_to_node_ids_.find(*layer_idx);
    if (layer_to_nodes_hit != graph_layering_index.layer_to_node_ids_.end()) {
      layer_to_nodes_hit->second.erase(node_id);
      if (layer_to_nodes_hit->second.empty()) {
        graph_layering_index.layer_to_node_ids_.erase(layer_to_nodes_hit);
      }
    }
  }
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
