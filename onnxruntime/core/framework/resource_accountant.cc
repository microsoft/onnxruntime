// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/resource_accountant.h"

#include "core/common/inlined_containers.h"
#include "core/common/narrow.h"
#include "core/common/parse_string.h"
#include "core/common/safeint.h"
#include "core/common/string_utils.h"

#include "core/framework/config_options.h"
#include "core/framework/max_shape_override.h"
#include "core/framework/murmurhash3.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include <algorithm>
#include <fstream>
#include <optional>

namespace onnxruntime {

// Accounts for resources represented as byte counts. Per-node costs can come from
// profiling statistics, ad-hoc fallback estimation, or an operator-specific estimator.
// This is currently used by CUDA EP.
class SizeBasedResourceAccountant : public IResourceAccountant {
 public:
  SizeBasedResourceAccountant() = default;
  ~SizeBasedResourceAccountant() = default;

  SizeBasedResourceAccountant(size_t threshold, InlinedHashMap<std::string, NodeAllocationStats>&& node_stats)
      : IResourceAccountant(threshold), node_stats_(std::move(node_stats)) {}

  explicit SizeBasedResourceAccountant(size_t threshold) : IResourceAccountant(threshold) {}

  explicit SizeBasedResourceAccountant(InlinedHashMap<std::string, NodeAllocationStats>&& node_stats)
      : IResourceAccountant(), node_stats_(std::move(node_stats)) {}

  ResourceCount GetConsumedAmount() const noexcept override {
    return consumed_amount_;
  }

  void AddConsumedAmount(const ResourceCount& amount) override {
    if (std::holds_alternative<size_t>(amount)) {
      consumed_amount_ =
          static_cast<size_t>(SafeInt<size_t>(consumed_amount_) + std::get<size_t>(amount));
    }
  }
  void RemoveConsumedAmount(const ResourceCount& amount) override {
    if (std::holds_alternative<size_t>(amount)) {
      consumed_amount_ =
          static_cast<size_t>(SafeInt<size_t>(consumed_amount_) - std::get<size_t>(amount));
    }
  }

  // Computes the resource cost for a candidate node.
  //
  // If profiling statistics are available, uses their non-workspace cost and
  // profiled temporary allocations. If a Level-1 runtime workspace estimate is
  // also provided, uses the maximum of the profiled and estimated workspace.
  // Without profiling, computes known initializer/output bytes and uses the
  // Level-1 runtime workspace or, when unavailable, fallback workspace.
  // Initializer bytes are charged separately, so persistent prepack estimates
  // must include only additional allocations, not storage reused directly from
  // an initializer (for example, an offline-prepacked weight). Persistent
  // prepack estimates are additional conservative charges in both paths.
  // Initialization scratch remains diagnostic rather than part of the additive
  // hard budget. MatMulNBits tactic profiling runs synchronously while
  // each kernel is constructed, and PrePack() calls run sequentially after
  // kernel creation; their scratch buffers are therefore created and released
  // one at a time. Their true session-wide requirement is a peak, which cannot
  // be represented by the reversible per-node ResourceCount scalar.
  //
  // GetCapability may probe nodes that are not ultimately assigned to this EP,
  // so per-node weights and workspace remain pending. CommitResourcesForNode()
  // promotes them after acceptance, while ResetForNewPass() discards state from
  // rejected or superseded capabilities.
  ResourceCount ComputeResourceCount(
      const Node& node, std::optional<Level1MemoryEstimate> level1_memory_estimate) override {
    if (node_stats_) {
      const auto node_name = MakeUniqueNodeName(node);
      auto hit = node_stats_->find(node_name);
      if (hit != node_stats_->end()) {
        const auto& stats = hit->second;
        const bool has_runtime_workspace_estimator =
            level1_memory_estimate.has_value() &&
            level1_memory_estimate->runtime_workspace_bytes.has_value();
        const size_t runtime_transient_bytes =
            level1_memory_estimate.has_value() ? level1_memory_estimate->runtime_transient_bytes : 0;
        const size_t level1_workspace_bytes =
            std::max(level1_memory_estimate.has_value()
                         ? level1_memory_estimate->runtime_workspace_bytes.value_or(0)
                         : 0,
                     runtime_transient_bytes);
        const size_t selected_workspace =
            std::max(stats.total_temp_allocations, level1_workspace_bytes);
        const size_t persistent_prepack_bytes =
            level1_memory_estimate.has_value() ? level1_memory_estimate->persistent_prepack_bytes : 0;
        const size_t temporary_prepack_bytes =
            level1_memory_estimate.has_value() ? level1_memory_estimate->temporary_prepack_bytes : 0;
        pending_workspace_selection_by_node_.insert_or_assign(
            node.Index(),
            WorkspaceEstimateSelection{
                selected_workspace,
                has_runtime_workspace_estimator ? WorkspaceEstimateSource::kProfileAndEstimator
                                                : WorkspaceEstimateSource::kProfile,
                stats.total_temp_allocations,
                level1_workspace_bytes,
                persistent_prepack_bytes,
                temporary_prepack_bytes});
        const SafeInt<size_t> resource_count =
            SafeInt<size_t>(stats.input_sizes) + stats.initializers_sizes +
            stats.total_dynamic_sizes + selected_workspace +
            persistent_prepack_bytes;
        return static_cast<size_t>(resource_count);
      }

      // Preserve the established partial-profile behavior: a node absent from
      // the stats file has zero cost. Falling through to ad-hoc accounting
      // would change existing partition decisions and mix profile-based costs
      // with initializer bookkeeping that the profile path does not use.
      return static_cast<size_t>(0);
    }

    const auto* graph = node.GetContainingGraph();
    if (!graph) return static_cast<size_t>(0);
    const auto& max_shapes = GetMaxShapeInferenceResult();

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
    // When max-shape inference is available, use it to resolve dynamic outputs.
    // Otherwise, GetSizeInBytesFromTensorTypeProto will only succeed when all dims
    // are known (static shape).
    SafeInt<size_t> output_size = 0;
    for (const auto* output_def : node.OutputDefs()) {
      if (!output_def->Exists() || !output_def->HasTensorOrScalarShape()) continue;
      const auto* type_proto = output_def->TypeAsProto();
      if (!type_proto || !utils::HasTensorType(*type_proto)) continue;

      size_t size = 0;
      // Try max-shape inference first for dynamic outputs
      if (!max_shapes.Empty()) {
        if (const TensorShape* max_shape =
                max_shapes.GetShape(graph, output_def->Name())) {
          const auto& tensor_type = type_proto->tensor_type();
          if (tensor_type.has_elem_type()) {
            const SafeInt<size_t> inferred_size =
                SafeInt<size_t>(max_shape->Size()) *
                utils::GetElementSizeOfTensor(
                    static_cast<ONNX_NAMESPACE::TensorProto_DataType>(tensor_type.elem_type()));
            size = inferred_size;
          }
        }
      }
      // Fall back to static shape
      if (size == 0 &&
          !utils::GetSizeInBytesFromTensorTypeProto<0>(type_proto->tensor_type(), &size).IsOK()) {
        continue;
      }
      output_size += size;
    }

    // Use the safety-margin portion as fallback workspace when no Level-1
    // estimate is available. Max-shape inference makes tensor sizes concrete
    // but does not, by itself, make kernel workspace requirements known.
    constexpr size_t kAdHocSafetyMultiplierPercent = 150;
    SafeInt<size_t> estimated = total_size + output_size;
    const size_t fallback_workspace =
        static_cast<size_t>(estimated * (kAdHocSafetyMultiplierPercent - 100) / 100);
    const bool has_runtime_workspace_estimator =
        level1_memory_estimate.has_value() &&
        level1_memory_estimate->runtime_workspace_bytes.has_value();
    const size_t runtime_transient_bytes =
        level1_memory_estimate.has_value() ? level1_memory_estimate->runtime_transient_bytes : 0;
    const size_t level1_workspace_bytes =
        std::max(level1_memory_estimate.has_value()
                     ? level1_memory_estimate->runtime_workspace_bytes.value_or(0)
                     : 0,
                 runtime_transient_bytes);
    const size_t selected_workspace =
        has_runtime_workspace_estimator
            ? level1_workspace_bytes
            : std::max(fallback_workspace, runtime_transient_bytes);
    const bool has_estimator =
        has_runtime_workspace_estimator || runtime_transient_bytes > fallback_workspace;
    const size_t persistent_prepack_bytes =
        level1_memory_estimate.has_value() ? level1_memory_estimate->persistent_prepack_bytes : 0;
    const size_t temporary_prepack_bytes =
        level1_memory_estimate.has_value() ? level1_memory_estimate->temporary_prepack_bytes : 0;
    pending_workspace_selection_by_node_.insert_or_assign(
        node.Index(),
        WorkspaceEstimateSelection{
            selected_workspace,
            has_estimator ? WorkspaceEstimateSource::kEstimator
                          : WorkspaceEstimateSource::kFallback,
            0,
            level1_workspace_bytes,
            persistent_prepack_bytes,
            temporary_prepack_bytes});
    return static_cast<size_t>(estimated + selected_workspace +
                               persistent_prepack_bytes);
  }

  void ResetPendingResourcesImpl() override {
    pending_weights_.clear();
    pending_weights_by_node_.clear();
    pending_workspace_selection_by_node_.clear();
  }

  void CommitResourcesForNode(NodeIndex node_index) override {
    auto it = pending_weights_by_node_.find(node_index);
    if (it != pending_weights_by_node_.end()) {
      for (const auto& name : it->second) {
        pending_weights_.erase(name);
      }
      committed_weights_.insert(it->second.begin(), it->second.end());
      pending_weights_by_node_.erase(it);
    }

    auto workspace_it = pending_workspace_selection_by_node_.find(node_index);
    if (workspace_it != pending_workspace_selection_by_node_.end()) {
      CommitWorkspaceEstimate(workspace_it->second);
      pending_workspace_selection_by_node_.erase(workspace_it);
    }
  }

  WorkspaceEstimateSelection GetPendingWorkspaceEstimateSelection(
      NodeIndex node_index) const override {
    auto it = pending_workspace_selection_by_node_.find(node_index);
    return it == pending_workspace_selection_by_node_.end() ? WorkspaceEstimateSelection{} : it->second;
  }

  void AddCommittedWorkspaceEstimate(WorkspaceEstimateSelection selection) override {
    CommitWorkspaceEstimate(selection);
  }

  WorkspaceEstimateSourceCounts GetWorkspaceEstimateSourceCounts() const override {
    return workspace_source_counts_;
  }

  WorkspaceEstimateComparisonSummary GetWorkspaceEstimateComparisonSummary() const override {
    return workspace_estimate_comparison_;
  }

  size_t GetCommittedWorkspaceEstimate() const override {
    return committed_workspace_estimate_;
  }

  size_t GetCommittedPersistentPrepackEstimate() const override {
    return committed_persistent_prepack_estimate_;
  }

  size_t GetCommittedTemporaryPrepackEstimate() const override {
    return committed_temporary_prepack_estimate_;
  }

 private:
  void CommitWorkspaceEstimate(WorkspaceEstimateSelection selection) {
    const size_t new_workspace_estimate =
        static_cast<size_t>(SafeInt<size_t>(committed_workspace_estimate_) + selection.bytes);
    const size_t new_persistent_prepack_estimate =
        static_cast<size_t>(SafeInt<size_t>(committed_persistent_prepack_estimate_) +
                            selection.persistent_prepack_bytes);
    // Kernel construction and PrePack() are sequential today, so committed
    // initialization scratch is a session-wide peak rather than a sum.
    const size_t new_temporary_prepack_estimate =
        std::max(committed_temporary_prepack_estimate_, selection.temporary_prepack_bytes);

    committed_workspace_estimate_ = new_workspace_estimate;
    committed_persistent_prepack_estimate_ = new_persistent_prepack_estimate;
    committed_temporary_prepack_estimate_ = new_temporary_prepack_estimate;
    switch (selection.source) {
      case WorkspaceEstimateSource::kFallback:
        ++workspace_source_counts_.fallback;
        break;
      case WorkspaceEstimateSource::kProfile:
        ++workspace_source_counts_.profile;
        break;
      case WorkspaceEstimateSource::kEstimator:
        ++workspace_source_counts_.estimator;
        break;
      case WorkspaceEstimateSource::kProfileAndEstimator:
        ++workspace_source_counts_.profile_and_estimator;
        ++workspace_estimate_comparison_.node_count;
        workspace_estimate_comparison_.profiled_bytes =
            static_cast<size_t>(SafeInt<size_t>(workspace_estimate_comparison_.profiled_bytes) +
                                selection.profiled_bytes);
        workspace_estimate_comparison_.level1_estimated_bytes =
            static_cast<size_t>(SafeInt<size_t>(workspace_estimate_comparison_.level1_estimated_bytes) +
                                selection.level1_estimated_bytes);
        if (selection.profiled_bytes > selection.level1_estimated_bytes) {
          ++workspace_estimate_comparison_.profile_larger;
        } else if (selection.profiled_bytes < selection.level1_estimated_bytes) {
          ++workspace_estimate_comparison_.estimator_larger;
        } else {
          ++workspace_estimate_comparison_.equal;
        }
        break;
      case WorkspaceEstimateSource::kNone:
        break;
    }
  }

  size_t consumed_amount_ = 0;
  std::optional<InlinedHashMap<std::string, NodeAllocationStats>> node_stats_;
  // Weights committed from previous partitioning iterations.
  // These persist across GetCapability passes.
  InlinedHashSet<std::string> committed_weights_;

  // Initializers already counted during the current GetCapability pass. This
  // prevents a shared initializer from being charged to multiple probed nodes.
  InlinedHashSet<std::string> pending_weights_;

  // Initializers tentatively charged to each node. CommitResourcesForNode()
  // uses this mapping to move accepted-node initializers into committed_weights_.
  InlinedHashMap<NodeIndex, InlinedHashSet<std::string>> pending_weights_by_node_;

  // Selected workspace bytes and source for each probed node. Keeping them in
  // one value prevents the reported source from diverging from the selected size.
  InlinedHashMap<NodeIndex, WorkspaceEstimateSelection> pending_workspace_selection_by_node_;

  // Workspace total and source counts for nodes ultimately accepted by the EP.
  size_t committed_workspace_estimate_ = 0;
  size_t committed_persistent_prepack_estimate_ = 0;
  size_t committed_temporary_prepack_estimate_ = 0;
  WorkspaceEstimateSourceCounts workspace_source_counts_;
  WorkspaceEstimateComparisonSummary workspace_estimate_comparison_;
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
                             std::make_unique<SizeBasedResourceAccountant>(*cuda_memory_limit,
                                                                           std::move(*loaded_stats)));
      } else if (cuda_memory_limit) {
        map.insert_or_assign(kCudaExecutionProvider,
                             std::make_unique<SizeBasedResourceAccountant>(*cuda_memory_limit));
      } else if (loaded_stats) {
        map.insert_or_assign(kCudaExecutionProvider,
                             std::make_unique<SizeBasedResourceAccountant>(std::move(*loaded_stats)));
      } else {
        map.insert_or_assign(kCudaExecutionProvider, std::make_unique<SizeBasedResourceAccountant>());
      }
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid format for: ",
                             kOrtSessionOptionsResourceCudaPartitioningSettings,
                             " : expecting comma separated fields");
    }
  }

  if (result.has_value()) {
    WorkspaceEstimatorConfig estimator_config{
        config_options.GetConfigEntry(kOrtSessionOptionsCudaFpAIntBGemm),
        config_options.GetConfigEntry(kOrtSessionOptionsCudaFpAIntBProfileM)};
    for (auto& [ep_type, accountant] : *result) {
      ORT_UNUSED_PARAMETER(ep_type);
      accountant->SetWorkspaceEstimatorConfig(estimator_config);
    }
  }

  // Parse max shape overrides and attach to all accountants
  const std::string max_shape_config = config_options.GetConfigOrDefault(
      kOrtSessionOptionsMaxShapeOverride, "");
  if (!max_shape_config.empty() && result.has_value()) {
    MaxShapeOverrideMap shape_overrides;
    ORT_RETURN_IF_ERROR(ParseMaxShapeOverride(max_shape_config, shape_overrides));
    for (auto& [ep_type, accountant] : *result) {
      ORT_UNUSED_PARAMETER(ep_type);
      accountant->SetMaxShapeOverrides(shape_overrides);
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
