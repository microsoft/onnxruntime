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
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include <fstream>

namespace onnxruntime {

// Use this accountant if your resource can be counted with size_t type
class SizeTAccountant : public IResourceAccountant {
 public:
  SizeTAccountant() = default;
  ~SizeTAccountant() = default;

  SizeTAccountant(size_t threshold, InlinedHashMap<std::string, NodeAllocationStats>&& node_stats)
      : IResourceAccountant(threshold), node_stats_(std::move(node_stats)) {}

  explicit SizeTAccountant(InlinedHashMap<std::string, NodeAllocationStats>&& node_stats)
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

  ResourceCount ComputeResourceCount(const Node& node) const override {
    const auto node_name = MakeUniqueNodeName(node);
    auto hit = node_stats_.find(node_name);
    if (hit != node_stats_.end()) {
      const auto& stats = hit->second;
      return stats.input_sizes + stats.initializers_sizes +
             stats.total_dynamic_sizes + stats.total_temp_allocations;
    }
    return static_cast<size_t>(0U);
  }

 private:
  size_t consumed_amount_ = 0;
  InlinedHashMap<std::string, NodeAllocationStats> node_stats_;
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
    node_stats.insert_or_assign(std::move(node_name), {input_sizes, initializers_sizes,
                                                       total_dynamic_sizes, total_temp_allocations});
  }

  result.swap(node_stats);
  return Status::OK();
}

Status NodeStatsRecorder::CreateAccountants(
    const ConfigOptions& config_options,
    const std::filesystem::path& model_path,
    std::optional<ResourceAccountantMap>& acc_map) {
  // Check if CUDA partitioning settings are provided
  const std::string resource_partitioning_settings = config_options.GetConfigOrDefault(
      kOrtSessionOptionsResourceCudaPartitioningSettings, "");

  if (!resource_partitioning_settings.empty()) {
    auto splits = utils::SplitString(resource_partitioning_settings, ",", true);
    if (splits.size() == 2) {
      if (splits[1].empty()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid resource partitioning settings");
      }

      InlinedHashMap<std::string, NodeAllocationStats> loaded_stats;
      ORT_RETURN_IF_ERROR(LoadNodeAllocationStats(model_path, splits[1], loaded_stats));

      std::optional<ResourceAccountantMap> result;
      auto& map = result.emplace();

      if (!splits[0].empty()) {
        size_t cuda_memory_limit = 0;
        ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(std::string{splits[0]}, cuda_memory_limit));
        cuda_memory_limit = SafeInt<size_t>(cuda_memory_limit) * 1024;  // to bytes
        map.insert_or_assign(kCudaExecutionProvider,
                             std::make_unique<SizeTAccountant>(cuda_memory_limit,
                                                               std::move(loaded_stats)));
      } else {
        map.insert_or_assign(kCudaExecutionProvider,
                             std::make_unique<SizeTAccountant>(std::move(loaded_stats)));
      }

      acc_map = std::move(result);
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid format for: ",
                             kOrtSessionOptionsResourceCudaPartitioningSettings,
                             " : expecting comma separated fields");
    }
  }

  return Status::OK();
}

std::string IResourceAccountant::MakeUniqueNodeName(const Node& node) {
  std::string result;

  uint32_t hash[4] = {0, 0, 0, 0};
  auto hash_str = [&hash](const std::string& str) {
    MurmurHash3::x86_128(str.data(), narrow<int32_t>(str.size()), hash[0], &hash);
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

}  // namespace onnxruntime