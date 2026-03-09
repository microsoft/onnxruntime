// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#include "core/graph/constants.h"
#include "core/common/narrow.h"
#include "core/common/string_utils.h"
#include "core/framework/layering_annotations.h"
#include "core/framework/ortmemoryinfo.h"
#include "core/session/abi_devices.h"
#include "core/framework/execution_providers.h"
#include "core/graph/graph.h"

#include <cstdlib>
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

      bool prefix_match = false;
      if (ann[0] == '=') {
        prefix_match = true;
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
  if (str.empty() || str[0] == '-') return false;
  char* end = nullptr;
  const char* ptr = str.c_str();
  errno = 0;
  unsigned long val = std::strtoul(ptr, &end, 10);
  if (errno != 0 || end != ptr + str.size()) {
    return false;
  }
  if (val > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  index = narrow<uint32_t>(val);
  return true;
}
}  // namespace

std::optional<std::string> EpLayeringMatcher::Match(gsl::span<const OrtEpDevice* const> ep_devices,
                                                    const LayerAnnotation& rule) {
  const std::string& target_full = rule.device;
  const auto colon_pos = target_full.find(':');
  const std::string target_type_str = (colon_pos == std::string::npos) ? target_full : target_full.substr(0, colon_pos);
  // vendor or index or uuid, if present
  std::string target_specifier;
  if (colon_pos != std::string::npos) {
    target_specifier = target_full.substr(colon_pos + 1);
  }

  for (const auto* ep_device_ptr : ep_devices) {
    if (!ep_device_ptr) {
      continue;
    }
    const OrtEpDevice& ep_device = *ep_device_ptr;

    bool matched = false;

    // Helper to check device type from MemInfo if Hardware device logic fails/is absent
    auto check_mem_device_type = [&](OrtDevice::DeviceType type) -> bool {
      if (ep_device.device_memory_info) {
        return ep_device.device_memory_info->device.Type() == type;
      }
      return false;
    };

    // 1. Exact Name / Alias match
    // "cpu"
    if (CaseInsensitiveCompare(target_type_str, "cpu")) {
      if (ep_device.ep_name == kCpuExecutionProvider) {
        matched = true;
      } else if (ep_device.device && ep_device.device->type == OrtHardwareDeviceType_CPU) {
        matched = true;
      } else if (check_mem_device_type(OrtDevice::CPU)) {
        matched = true;
      }
    }  // "gpu"
    else if (CaseInsensitiveCompare(target_type_str, "gpu")) {
      // If simple "gpu"
      if (target_specifier.empty()) {
        if (ep_device.device && ep_device.device->type == OrtHardwareDeviceType_GPU) {
          matched = true;
        } else if (check_mem_device_type(OrtDevice::GPU)) {
          matched = true;
        }  // Heuristic fallback for common GPU EPs if hardware info is missing. Should we also check for TRT here?
        else if (ep_device.ep_name == kCudaExecutionProvider || ep_device.ep_name == kDmlExecutionProvider) {
          matched = true;
        }
      } else {
        // "gpu:<vendor>" or "gpu:<index>"
        if (ep_device.device && ep_device.device->type == OrtHardwareDeviceType_GPU) {
          uint32_t index = std::numeric_limits<uint32_t>::max();
          if (TryParseIndex(target_specifier, index)) {
            // gpu:<index>
            if (ep_device.device->device_id == index) {
              matched = true;
            }
          } else {
            // gpu:<vendor>
            if (CaseInsensitiveCompare(ep_device.device->vendor, target_specifier)) {
              matched = true;
            }
            // Check against vendor ID
            else if (CaseInsensitiveCompare(target_specifier, "nvidia") &&
                     ep_device.device->vendor_id == OrtDevice::VendorIds::NVIDIA) {
              matched = true;
            } else if (CaseInsensitiveCompare(target_specifier, "amd") &&
                       ep_device.device->vendor_id == OrtDevice::VendorIds::AMD) {
              matched = true;
            } else if (CaseInsensitiveCompare(target_specifier, "intel") &&
                       ep_device.device->vendor_id == OrtDevice::VendorIds::INTEL) {
              matched = true;
            }
            // Special shortcuts heuristics: gpu:nvidia -> CUDA
            else if (CaseInsensitiveCompare(target_specifier, "nvidia") &&
                     ep_device.ep_name == kCudaExecutionProvider) {
              matched = true;
            }
          }
        }
      }
    }
    // "accelerator" (not cpu)
    else if (CaseInsensitiveCompare(target_type_str, "accelerator")) {
      if (ep_device.ep_name != kCpuExecutionProvider) {
        // If we don't have HW info, assuming non-CPU EP is an accelerator.
        // If we do have HW info, check it's not CPU.
        const bool is_cpu_hw = (ep_device.device && ep_device.device->type == OrtHardwareDeviceType_CPU);
        const bool is_cpu_mem = check_mem_device_type(OrtDevice::CPU);

        if (!is_cpu_hw && !is_cpu_mem) {
          matched = true;
        }
      }
    }  // "npu"
    else if (CaseInsensitiveCompare(target_type_str, "npu")) {
      if (ep_device.device && ep_device.device->type == OrtHardwareDeviceType_NPU) {
        matched = true;
      } else if (ep_device.ep_name == kQnnExecutionProvider || ep_device.ep_name == kVitisAIExecutionProvider) {
        // Heuristic for known NPU providers if HW device info is missing
        // XXX: These can run on CPU as well, need to see if there any check that is missing.
        matched = true;
      }
    }
    // "fpga"
    else if (CaseInsensitiveCompare(target_type_str, "fpga")) {
      // No OrtHardwareDeviceType_FPGA currently, rely on OrtDevice::FPGA
      if (check_mem_device_type(OrtDevice::FPGA)) {
        matched = true;
      }
    }
    // "cuda"
    else if (CaseInsensitiveCompare(target_type_str, "cuda")) {
      if (ep_device.ep_name == kCudaExecutionProvider) {
        matched = true;
      }
    }
    // "dml"
    else if (CaseInsensitiveCompare(target_type_str, "dml")) {
      if (ep_device.ep_name == kDmlExecutionProvider) {
        matched = true;
      }
    }
    // Fallback: Exact EP name string match (e.g. "MyCustomEP")
    else if (ep_device.ep_name == target_full) {
      matched = true;
    }

    if (matched) {
      return ep_device.ep_name;
    }
  }

  return std::nullopt;
}

std::optional<std::string> EpLayeringMatcher::Match(const ExecutionProviders& providers, const LayerAnnotation& rule) {
  const std::string& target_full = rule.device;
  const auto colon_pos = target_full.find(':');
  const std::string target_type_str = (colon_pos == std::string::npos) ? target_full : target_full.substr(0, colon_pos);
  std::string target_specifier;
  if (colon_pos != std::string::npos) {
    target_specifier = target_full.substr(colon_pos + 1);
  }

  for (const auto& ep_shared_ptr : providers) {
    if (!ep_shared_ptr) {
      continue;
    }
    const IExecutionProvider& ep = *ep_shared_ptr;
    const std::string& ep_name = ep.Type();
    const OrtDevice& device = ep.GetDevice();

    bool matched = false;

    // 1. Exact Name / Alias match
    // "cpu"
    if (CaseInsensitiveCompare(target_type_str, "cpu")) {
      if (ep_name == kCpuExecutionProvider) {
        matched = true;
      } else if (device.Type() == OrtDevice::CPU) {
        matched = true;
      }
    }  // "gpu"
    else if (CaseInsensitiveCompare(target_type_str, "gpu")) {
      // If simple "gpu"
      if (target_specifier.empty()) {
        if (device.Type() == OrtDevice::GPU) {
          matched = true;
        }  // Heuristics, XXX: Should we also check for TRT here?
        else if (ep_name == kCudaExecutionProvider || ep_name == kDmlExecutionProvider) {
          matched = true;
        }
      } else {
        // "gpu:<vendor>" or "gpu:<index>"
        if (device.Type() == OrtDevice::GPU) {
          uint32_t index = std::numeric_limits<uint32_t>::max();
          if (TryParseIndex(target_specifier, index)) {
            // gpu:<index>
            if (device.Id() == static_cast<OrtDevice::DeviceId>(index)) {
              matched = true;
            }
          } else {
            // gpu:<vendor> checking against Vendor ID
            if (CaseInsensitiveCompare(target_specifier, "nvidia") &&
                device.Vendor() == OrtDevice::VendorIds::NVIDIA) {
              matched = true;
            } else if (CaseInsensitiveCompare(target_specifier, "amd") &&
                       device.Vendor() == OrtDevice::VendorIds::AMD) {
              matched = true;
            } else if (CaseInsensitiveCompare(target_specifier, "intel") &&
                       device.Vendor() == OrtDevice::VendorIds::INTEL) {
              matched = true;
            }
            // Special shortcuts heuristics: gpu:nvidia -> CUDA
            else if (CaseInsensitiveCompare(target_specifier, "nvidia") && ep_name == kCudaExecutionProvider) {
              matched = true;
            }
          }
        }
      }
    }
    // "accelerator" (not cpu)
    else if (CaseInsensitiveCompare(target_type_str, "accelerator")) {
      if (ep_name != kCpuExecutionProvider) {
        if (device.Type() != OrtDevice::CPU) {
          matched = true;
        }
      }
    }  // "npu"
    else if (CaseInsensitiveCompare(target_type_str, "npu")) {
      if (device.Type() == OrtDevice::NPU) {
        matched = true;
      } else if (ep_name == kQnnExecutionProvider || ep_name == kVitisAIExecutionProvider) {
        matched = true;
      }
    }
    // "fpga"
    else if (CaseInsensitiveCompare(target_type_str, "fpga")) {
      if (device.Type() == OrtDevice::FPGA) {
        matched = true;
      }
    }
    // "cuda"
    else if (CaseInsensitiveCompare(target_type_str, "cuda")) {
      if (ep_name == kCudaExecutionProvider) {
        matched = true;
      }
    }
    // "dml"
    else if (CaseInsensitiveCompare(target_type_str, "dml")) {
      if (ep_name == kDmlExecutionProvider) {
        matched = true;
      }
    }
    // Fallback: Exact EP name string match (e.g. "MyCustomEP")
    else if (ep_name == target_full) {
      matched = true;
    }

    if (matched) {
      return ep_name;
    }
  }

  return std::nullopt;
}

LayeringIndex LayeringIndex::Create(const Graph& graph,
                                    EpNameToLayeringIndices ep_map,
                                    LayeringIndexToEpName rule_map,
                                    LayeringRules layering_rules) {
  // 1. Create LayeringIndex instance with pre-computed maps
  LayeringIndex index(std::move(layering_rules), std::move(ep_map), std::move(rule_map));

  // 2. Traverse the graph and index nodes
  index.ProcessGraph(graph, std::nullopt);

  return index;
}

Status LayeringIndex::Create(const Graph& graph,
                             const std::string& config_string,
                             gsl::span<const OrtEpDevice* const> ep_devices,
                             const ExecutionProviders& ep_providers,
                             const logging::Logger& logger,
                             std::optional<LayeringIndex>& layering_index) {
  LayeringRules rules;
  ORT_RETURN_IF_ERROR(LayeringRules::FromConfigString(config_string, rules));

  LOGS(logger, INFO) << "Parsed " << rules.rules.size() << " layering rules from config.";

  if (rules.rules.empty()) {
    // Return no index indicating no layering
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
      // Ensure 1:1 mapping from rule index to EP type
      // Note: A rule index refers to a unique entry in LayeringRules::rules vector.
      // So 'i' is unique.
      rule_map[i] = ep_type;
      matched_rule_count++;
      LOGS(logger, VERBOSE) << "Layering Rule " << i << " (" << rule.device << " -> " << rule.annotation
                            << ") mapped to EP: " << ep_type;
    } else {
      LOGS(logger, WARNING) << "Layering Rule " << i << " (" << rule.device << " -> " << rule.annotation
                            << ") could not be mapped to any available Execution Provider.";
    }
  }

  LOGS(logger, INFO) << "LayeringIndex created. Matched " << matched_rule_count
                     << " out of " << rules.rules.size() << " rules to available Execution Providers.";

  layering_index = LayeringIndex::Create(graph, std::move(ep_map), std::move(rule_map), std::move(rules));
  return Status::OK();
}

// Process top to bottom-up assign layering indices to nodes
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

    // 4. For every node query its annotation
    const std::string& annotation = node.GetLayeringAnnotation();
    if (!annotation.empty()) {
      // If it has an annotation try to match it
      matched_rule_idx = matcher_.Match(annotation);
    }

    // 5. If node has no annotation, inherit from subgraph parent node
    if (!matched_rule_idx && parent_layer_id) {
      matched_rule_idx = parent_layer_id;
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
        // so it does not propagate to sub-graphs if any
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

    const std::string& annotation = node->GetLayeringAnnotation();
    if (!annotation.empty()) {
      auto matched_rule_idx = matcher_.Match(annotation);

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
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
