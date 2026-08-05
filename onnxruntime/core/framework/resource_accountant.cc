// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/resource_accountant.h"

#include "core/common/inlined_containers.h"
#include "core/common/narrow.h"
#include "core/common/parse_string.h"
#include "core/common/safeint.h"
#include "core/common/string_utils.h"

#include "core/framework/config_options.h"
#include "core/framework/murmurhash3.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include <fstream>
#include <optional>

namespace onnxruntime {

// Use this accountant if your resource can be counted with size_t type
// This accountant uses NodeAllocationStats to compute resource consumption per node
// which can be collected and saved to a file OR loaded from a file and used for partitioning.
// This is currently used for CUDA EP.
class SizeBasedStatsAccountant : public IResourceAccountant {
 public:
  SizeBasedStatsAccountant() = default;
  ~SizeBasedStatsAccountant() = default;

  SizeBasedStatsAccountant(size_t threshold, InlinedHashMap<std::string, NodeAllocationStats>&& node_stats)
      : IResourceAccountant(threshold), node_stats_(std::move(node_stats)) {}

  explicit SizeBasedStatsAccountant(size_t threshold) : IResourceAccountant(threshold) {}

  explicit SizeBasedStatsAccountant(InlinedHashMap<std::string, NodeAllocationStats>&& node_stats)
      : IResourceAccountant(), node_stats_(std::move(node_stats)) {}

  ResourceCount GetConsumedAmount() const noexcept override {
    return consumed_amount_;
  }

  void AddConsumedAmount(const ResourceCount& amount) noexcept override {
    if (std::holds_alternative<size_t>(amount)) {
      consumed_amount_ += std::get<size_t>(amount);
    }
  }
  void RemoveConsumedAmount(const ResourceCount& amount) noexcept override {
    if (std::holds_alternative<size_t>(amount)) {
      consumed_amount_ -= std::get<0>(amount);
    }
  }

  ResourceCount ComputeResourceCount(const Node& node) override {
    if (node_stats_) {
      const auto node_name = MakeUniqueNodeName(node);
      auto hit = node_stats_->find(node_name);
      if (hit != node_stats_->end()) {
        const auto& stats = hit->second;
        return stats.input_sizes + stats.initializers_sizes +
               stats.total_dynamic_sizes + stats.total_temp_allocations;
      }
      return static_cast<size_t>(0U);
    } else {
      const auto* graph = node.GetContainingGraph();
      if (!graph) return static_cast<size_t>(0);

      SafeInt<size_t> total_size = 0;
      for (const auto* input_def : node.InputDefs()) {
        if (!input_def->Exists()) continue;

        const auto& name = input_def->Name();
        constexpr bool check_outer_scope = true;
        const auto* tensor_proto = graph->GetInitializer(name, check_outer_scope);

        if (tensor_proto) {
          // Skip if already committed from a previous partitioning iteration
          if (committed_weights_.count(name) > 0) {
            continue;
          }

          // Skip if already pending from another node in this GetCapability pass
          if (pending_weights_.count(name) > 0) {
            continue;
          }

          size_t size = 0;
          auto status = utils::GetSizeInBytesFromTensorProto<0>(*tensor_proto, &size);

          if (status.IsOK()) {
            total_size += size;
            pending_weights_.insert(name);
            pending_weights_by_node_[node.Index()].insert(name);
          }
        }
      }

      // Account for intermediate output tensors when shape info is available.
      // GetSizeInBytesFromTensorTypeProto will only succeed when all dims are known
      // (static shape) and a valid element type is present, so dynamic outputs are
      // naturally skipped.
      SafeInt<size_t> output_size = 0;
      for (const auto* output_def : node.OutputDefs()) {
        if (!output_def->Exists() || !output_def->HasTensorOrScalarShape()) continue;
        const auto* type_proto = output_def->TypeAsProto();
        if (!type_proto || !utils::HasTensorType(*type_proto)) continue;

        size_t size = 0;
        if (utils::GetSizeInBytesFromTensorTypeProto<0>(type_proto->tensor_type(), &size).IsOK()) {
          output_size += size;
        }
      }

      // Apply a safety multiplier for workspace/temp allocations we can't see
      constexpr size_t kAdHocSafetyMultiplierPercent = 150;  // 1.5x
      SafeInt<size_t> estimated = total_size + output_size;
      return static_cast<size_t>(estimated * kAdHocSafetyMultiplierPercent / 100);
    }
  }

  void ResetPendingWeightsImpl() override {
    pending_weights_.clear();
    pending_weights_by_node_.clear();
  }

  void CommitWeightsForNode(NodeIndex node_index) override {
    auto it = pending_weights_by_node_.find(node_index);
    if (it != pending_weights_by_node_.end()) {
      for (const auto& name : it->second) {
        pending_weights_.erase(name);
      }
      committed_weights_.insert(it->second.begin(), it->second.end());
      pending_weights_by_node_.erase(it);
    }
  }

 private:
  size_t consumed_amount_ = 0;
  std::optional<InlinedHashMap<std::string, NodeAllocationStats>> node_stats_;
  // Weights committed from previous partitioning iterations.
  // These persist across GetCapability passes.
  InlinedHashSet<std::string> committed_weights_;
  // Flat set of all pending weight names for O(1) membership checks.
  InlinedHashSet<std::string> pending_weights_;
  // Same pending weights keyed by node index, used by CommitWeightsForNode.
  InlinedHashMap<NodeIndex, InlinedHashSet<std::string>> pending_weights_by_node_;
};

struct NodeStatsRecorder::Impl {
  std::filesystem::path node_stats_path;
  // This is a node name to allocation stats map
  InlinedHashMap<std::string, NodeAllocationStats> node_stats;
  // Keeps track of nodes for which input/output sizes are accounted
  InlinedHashSet<std::string> input_output_accounted;
};

NodeStatsRecorder::NodeStatsRecorder(const std::filesystem::path& node_stats_path)
    : impl_(std::make_unique<Impl>()) {
  impl_->node_stats_path = node_stats_path;
}

NodeStatsRecorder::~NodeStatsRecorder() = default;

const std::filesystem::path& NodeStatsRecorder::GetNodeStatsFileName() const noexcept {
  return impl_->node_stats_path;
}

bool NodeStatsRecorder::ShouldAccountFor(const std::string& input_output_name) const {
  return impl_->input_output_accounted.insert(input_output_name).second;
}

void NodeStatsRecorder::ResetPerRunNameDeduper() {
  impl_->input_output_accounted.clear();
}

void NodeStatsRecorder::ReportNodeStats(const std::string& node_name, const NodeAllocationStats& stats) {
  auto result = impl_->node_stats.emplace(node_name, stats);
  if (!result.second) {
    // Node already exists, update the stats
    // This may happen when the user collects stats from multiple Runs()
    result.first->second.UpdateIfGreater(stats);
  }
}

void NodeStatsRecorder::DumpStats(std::ostream& os) const {
  os << "#name,input_sizes,initializers_sizes,total_dynamic_sizes,total_temp_allocations\n";
  for (const auto& [name, stats] : impl_->node_stats) {
    os << name << "," << stats.input_sizes << "," << stats.initializers_sizes << ","
       << stats.total_dynamic_sizes << ","
       << stats.total_temp_allocations << "\n";
  }
}

void NodeStatsRecorder::DumpStats(const std::filesystem::path& model_path) const {
  auto node_stats_file = model_path;
  if (node_stats_file.has_filename()) {
    node_stats_file = node_stats_file.parent_path();
  }
  node_stats_file /= GetNodeStatsFileName();
  std::ofstream ofs(node_stats_file, std::ofstream::out);
  ORT_ENFORCE(ofs.is_open(), "Failed to open file: ", node_stats_file);
  DumpStats(ofs);
  ofs.close();
}

static Status LoadNodeAllocationStats(
    const std::filesystem::path& model_path, const std::filesystem::path& file_name,
    InlinedHashMap<std::string, NodeAllocationStats>& result) {
  InlinedHashMap<std::string, NodeAllocationStats> node_stats;
  std::filesystem::path file_path = model_path;
  if (file_path.has_filename()) {
    file_path = file_path.parent_path();
  }

  file_path /= file_name;

  std::ifstream file(file_path);
  ORT_RETURN_IF_NOT(file.is_open(), "Failed to open file ", file_path);
  std::string line;
  // Read and load a CSV file line by line
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

    auto splits = utils::SplitString(line, ",", true);
    ORT_ENFORCE(splits.size() == 5, "Invalid line in the file ", file_path, ": ", line);
    if (splits[0].empty()) {
      continue;
    }
    std::string node_name{splits[0]};
    size_t input_sizes = SafeInt<size_t>(std::stoull(std::string{splits[1]}));
    size_t initializers_sizes = SafeInt<size_t>(std::stoull(std::string{splits[2]}));
    size_t total_dynamic_sizes = SafeInt<size_t>(std::stoull(std::string{splits[3]}));
    size_t total_temp_allocations = SafeInt<size_t>(std::stoull(std::string{splits[4]}));
    const NodeAllocationStats stats = {input_sizes, initializers_sizes, total_dynamic_sizes, total_temp_allocations};
    node_stats.insert_or_assign(std::move(node_name), stats);
  }

  result.swap(node_stats);
  return Status::OK();
}

Status CreateAccountants(
    const ConfigOptions& config_options,
    const std::filesystem::path& model_path,
    std::optional<ResourceAccountantMap>& acc_map) {
  std::optional<ResourceAccountantMap> result;
  // Check if CUDA partitioning settings are provided
  const std::string resource_partitioning_settings = config_options.GetConfigOrDefault(
      kOrtSessionOptionsResourceCudaPartitioningSettings, "");

  if (!resource_partitioning_settings.empty()) {
    auto splits = utils::SplitString(resource_partitioning_settings, ",", true);
    if (splits.size() == 2) {
      auto& map = result.emplace();

      std::optional<size_t> cuda_memory_limit;
      if (!splits[0].empty()) {
        cuda_memory_limit.emplace(0U);
        ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(std::string{splits[0]}, *cuda_memory_limit));
        cuda_memory_limit = SafeInt<size_t>(*cuda_memory_limit) * 1024;  // to bytes
      }

      std::optional<InlinedHashMap<std::string, NodeAllocationStats>> loaded_stats;
      if (!splits[1].empty()) {
        loaded_stats.emplace();
        ORT_RETURN_IF_ERROR(LoadNodeAllocationStats(model_path, splits[1], *loaded_stats));
      }

      if (cuda_memory_limit && loaded_stats) {
        map.insert_or_assign(kCudaExecutionProvider,
                             std::make_unique<SizeBasedStatsAccountant>(*cuda_memory_limit,
                                                                        std::move(*loaded_stats)));
      } else if (cuda_memory_limit) {
        map.insert_or_assign(kCudaExecutionProvider,
                             std::make_unique<SizeBasedStatsAccountant>(*cuda_memory_limit));
      } else if (loaded_stats) {
        map.insert_or_assign(kCudaExecutionProvider,
                             std::make_unique<SizeBasedStatsAccountant>(std::move(*loaded_stats)));
      } else {
        map.insert_or_assign(kCudaExecutionProvider, std::make_unique<SizeBasedStatsAccountant>());
      }
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid format for: ",
                             kOrtSessionOptionsResourceCudaPartitioningSettings,
                             " : expecting comma separated fields");
    }
  }

  acc_map = std::move(result);
  return Status::OK();
}

std::string IResourceAccountant::MakeUniqueNodeName(const Node& node) {
  std::string result;

  uint32_t hash[4] = {0, 0, 0, 0};
  auto hash_str = [&hash](const std::string& str) {
    MurmurHash3::x86_128(str.data(), str.size(), hash[0], &hash);
  };

  const auto& node_name = (node.Name().empty()) ? node.OpType() : node.Name();

  for (const auto& def : node.InputDefs()) {
    hash_str(def->Name());
  }

  for (const auto& def : node.OutputDefs()) {
    hash_str(def->Name());
  }

  HashValue node_hash = hash[0] | (uint64_t(hash[1]) << 32);
  result.reserve(node_name.size() + 1 + 16);
  result.append(node_name).append("_").append(std::to_string(node_hash));

  return result;
}

ResourceCount AddResourceCounts(const ResourceCount& a, const ResourceCount& b) {
  return std::visit(
      [](auto lhs, auto rhs) -> ResourceCount {
        static_assert(std::is_same_v<decltype(lhs), decltype(rhs)>,
                      "AddResourceCounts requires both operands to hold the same type. "
                      "Handle the new ResourceCount variant member.");
        if constexpr (std::is_integral_v<decltype(lhs)>) {
          return static_cast<decltype(lhs)>(SafeInt<decltype(lhs)>(lhs) + rhs);
        } else {
          return lhs + rhs;
        }
      },
      a, b);
}

bool ResourceCountExceeds(const ResourceCount& a, const ResourceCount& b) {
  return std::visit(
      [](auto lhs, auto rhs) -> bool {
        static_assert(std::is_same_v<decltype(lhs), decltype(rhs)>,
                      "ResourceCountExceeds requires both operands to hold the same type. "
                      "Handle the new ResourceCount variant member.");
        return lhs > rhs;
      },
      a, b);
}

std::string FormatResourceCount(const ResourceCount& rc) {
  return std::visit(
      [](auto val) -> std::string { return std::to_string(val); },
      rc);
}

}  // namespace onnxruntime
