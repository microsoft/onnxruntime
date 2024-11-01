// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/resource_accountant.h"
#include "core/common/inlined_containers.h"

#include <mutex>

namespace onnxruntime {

struct NodeStatsRecorder::Impl {
  std::filesystem::path node_stats_path_;
  // This is a node name to allocation stats map
  InlinedHashMap<std::string, NodeAllocationStats> node_stats_;
  mutable std::mutex mut_;
};

NodeStatsRecorder::NodeStatsRecorder(const std::filesystem::path& node_stats_path)
    : impl_(std::make_unique<Impl>()) {
  impl_->node_stats_path_ = node_stats_path;
}

NodeStatsRecorder::~NodeStatsRecorder() = default;

const std::filesystem::path& NodeStatsRecorder::GetNodeStatsFileName() const noexcept {
  return impl_->node_stats_path_;
}

void NodeStatsRecorder::ReportNodeStats(const std::string& node_name, const NodeAllocationStats& stats) {
  std::lock_guard lock(impl_->mut_);
  auto result = impl_->node_stats_.emplace(node_name, stats);
  if (!result.second) {
    // Node already exists, update the stats
    result.first->second.UpdateIfGreater(stats);
  }
}

void NodeStatsRecorder::DumpStats(std::ostream& os) const {
  std::lock_guard lock(impl_->mut_);
  for (const auto& [name, stats] : impl_->node_stats_) {
    os << name << "," << stats.input_sizes << "," << stats.initializers_sizes << ","
       << stats.total_dynamic_sizes << ","
       << stats.total_temp_allocations << "\n";
  }
}

}  // namespace onnxruntime