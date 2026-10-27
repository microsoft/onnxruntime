// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/compute_estimate_report.h"

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <optional>
#include <string_view>
#include <utility>

#include "nlohmann/json.hpp"
#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/common/path_string.h"
#include "core/common/safeint.h"
#include "core/framework/runtime_workload_collector.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "core/platform/env.h"

#if !defined(DISABLE_CONTRIB_OPS)
#include "core/framework/matmul_nbits_compute_estimate.h"
#endif

namespace onnxruntime {
namespace {

using Json = nlohmann::ordered_json;

Json ExcludedComponents() {
  return Json::array({"dequantization", "bias", "packing", "conversion", "padding", "route_specific_extra_work"});
}

#if !defined(DISABLE_CONTRIB_OPS)
const char* ReasonName(contrib::ComputeEstimateUnavailableReason reason) {
  switch (reason) {
    case contrib::ComputeEstimateUnavailableReason::UnsupportedOperator:
      return "unsupported_operator";
    case contrib::ComputeEstimateUnavailableReason::UnsupportedVersion:
      return "unsupported_version";
    case contrib::ComputeEstimateUnavailableReason::UnknownShape:
      return "unknown_shape";
    case contrib::ComputeEstimateUnavailableReason::InvalidMetadata:
      return "invalid_metadata";
    case contrib::ComputeEstimateUnavailableReason::Overflow:
      return "overflow";
    case contrib::ComputeEstimateUnavailableReason::MissingSemanticValue:
      return "missing_semantic_value";
  }
  ORT_THROW("Invalid compute estimate reason.");
}
#endif

Json InputShapes(const Node& node) {
  Json result = Json::array();
  for (const auto* input : node.InputDefs()) {
    if (input == nullptr || !input->Exists()) {
      result.push_back({{"state", "missing"}});
      continue;
    }
    const auto* shape = input->Shape();
    Json value{{"name", input->Name()}, {"state", shape ? "present_with_shape" : "present_without_shape"}};
    if (shape) {
      Json dimensions = Json::array();
      for (const auto& dim : shape->dim()) {
        if (dim.has_dim_value()) {
          dimensions.push_back(dim.dim_value());
        } else {
          dimensions.push_back(nullptr);
        }
      }
      value["dimensions"] = std::move(dimensions);
    }
    result.push_back(std::move(value));
  }
  return result;
}

// Lookup remains internal and deliberately narrow; route-dependent metrics do not belong here.
Json EstimateNode(const Node& node, std::optional<uint64_t>& operation_count) {
  operation_count.reset();
  Json result{{"availability", "unavailable"}};
  if (node.NodeType() == Node::Type::Fused) {
    result["unavailable_reason"] = "opaque_partition";
  } else if (node.ContainsSubgraph()) {
    result["unavailable_reason"] = "runtime_execution_required";
  } else if (node.Domain() != kMSDomain || node.OpType() != "MatMulNBits") {
    result["unavailable_reason"] = "unsupported_operator";
  } else {
    Json component{{"component", "matmul_core"},
                   {"metric", "dense_equivalent_ops"},
                   {"scope", "matmul_core"},
                   {"convention_version", "core_arithmetic_v1"},
                   {"input_basis", "canonical_graph_metadata"},
                   {"basis", "logical_semantics"},
                   {"execution_phase", "per_invocation"},
                   {"excluded_components", ExcludedComponents()}};
#if !defined(DISABLE_CONTRIB_OPS)
    const auto estimate = contrib::EstimateMatMulNBitsCompute(node);
    if (const auto* count = std::get_if<contrib::MatMulNBitsComputeEstimate>(&estimate)) {
      operation_count = count->dense_equivalent_ops;
      component["availability"] = "available";
      component["estimate_kind"] = "exact_under_convention";
      component["input_basis"] = "static_concrete";
      component["operation_count"] = std::to_string(count->dense_equivalent_ops);
    } else {
      component["availability"] = "unavailable";
      component["unavailable_reason"] =
          ReasonName(std::get<contrib::ComputeEstimateUnavailableReason>(estimate));
    }
#else
    component["availability"] = "unavailable";
    component["unavailable_reason"] = "estimators_disabled";
#endif
    result["availability"] = component["availability"];
    if (component.contains("unavailable_reason")) {
      result["unavailable_reason"] = component["unavailable_reason"];
    }
    result["components"] = Json::array({std::move(component)});
    Json bias{{"component", "bias"},
              {"metric", "floating_point_ops"},
              {"scope", "bias"},
              {"convention_version", "bias_add_v1"},
              {"input_basis", "canonical_graph_metadata"},
              {"basis", "logical_semantics"},
              {"execution_phase", "per_invocation"},
              {"excluded_components", Json::array({"matmul_core", "dequantization", "packing", "conversion",
                                                   "padding", "route_specific_extra_work"})}};
#if !defined(DISABLE_CONTRIB_OPS)
    const auto bias_estimate = contrib::EstimateMatMulNBitsBias(node);
    if (const auto* count = std::get_if<contrib::MatMulNBitsBiasEstimate>(&bias_estimate)) {
      const auto& inputs = node.InputDefs();
      bias["input_basis"] = inputs.size() > 5 && inputs[5] != nullptr && inputs[5]->Exists()
                                ? "static_concrete"
                                : "optional_input_absent";
      bias["availability"] = "available";
      bias["estimate_kind"] = "exact_under_convention";
      bias["operation_count"] = std::to_string(count->floating_point_ops);
    } else {
      bias["availability"] = "unavailable";
      bias["unavailable_reason"] = ReasonName(std::get<contrib::ComputeEstimateUnavailableReason>(bias_estimate));
    }
#else
    bias["availability"] = "unavailable";
    bias["unavailable_reason"] = "estimators_disabled";
#endif
    const bool core_available = operation_count.has_value();
    const bool bias_available = bias["availability"] == "available";
    result["availability"] = core_available && bias_available   ? "available"
                             : core_available || bias_available ? "partial"
                                                                : "unavailable";
    if (!bias_available && core_available) {
      result["unavailable_reason"] = bias["unavailable_reason"];
    }
    result["components"].push_back(std::move(bias));
  }
  if (!result.contains("components")) {
    result["components"] = Json::array();
  }
  result["whole_operator_work_complete"] = false;
  return result;
}

std::string EscapePathSegment(std::string_view value) {
  std::string result;
  for (char c : value) {
    if (c == '~') {
      result += "~0";
    } else if (c == '/') {
      result += "~1";
    } else {
      result += c;
    }
  }
  return result;
}

void AppendGraph(const Graph& graph, const std::string& path, Json& report) {
  uint64_t known_ops = 0;
  bool subtotal_overflow = false;
  size_t estimated_count = 0;
  size_t unavailable_count = 0;
  size_t partial_count = 0;
  struct ComponentSubtotal {
    std::string name;
    std::string metric;
    std::string convention;
    uint64_t count = 0;
    size_t estimated = 0;
    size_t unavailable = 0;
    bool overflow = false;
    Json reasons = Json::object();
  };
  InlinedVector<ComponentSubtotal> component_subtotals;
  bool has_subgraphs = false;
  Json reasons = Json::object();
  const auto graph_index = report["graphs"].size();
  report["graphs"].push_back({{"graph_path", path}, {"graph_name", graph.Name()}});

  for (const auto& node : graph.Nodes()) {
    Json entry{{"graph_path", path},
               {"node_index", node.Index()},
               {"node_name", node.Name()},
               {"domain", node.Domain()},
               {"op_type", node.OpType()},
               {"opset_version", node.SinceVersion()},
               {"provider", node.GetExecutionProviderType()},
               {"input_shapes", InputShapes(node)}};
    std::optional<uint64_t> operation_count;
    entry.update(EstimateNode(node, operation_count));
    if (operation_count.has_value()) {
      if (!subtotal_overflow && !SafeAdd(known_ops, *operation_count, known_ops)) {
        subtotal_overflow = true;
      }
    }
    if (entry["availability"] == "available") {
      ++estimated_count;
    } else {
      if (entry["availability"] == "partial") {
        ++partial_count;
      } else {
        ++unavailable_count;
      }
      const std::string reason = entry["unavailable_reason"].get<std::string>();
      reasons[reason] = reasons.value(reason, size_t{0}) + 1;
    }
    for (const auto& component : entry["components"]) {
      const auto name = component["component"].get<std::string>();
      auto it = std::find_if(component_subtotals.begin(), component_subtotals.end(),
                             [&](const auto& subtotal) { return subtotal.name == name; });
      if (it == component_subtotals.end()) {
        component_subtotals.push_back({name, component["metric"].get<std::string>(),
                                       component["convention_version"].get<std::string>()});
        it = component_subtotals.end() - 1;
      }
      if (component["availability"] == "available") {
        ++it->estimated;
        const auto count = std::stoull(component["operation_count"].get<std::string>());
        if (!it->overflow && !SafeAdd(it->count, count, it->count)) {
          it->overflow = true;
        }
      } else {
        ++it->unavailable;
        const auto reason = component["unavailable_reason"].get<std::string>();
        it->reasons[reason] = it->reasons.value(reason, size_t{0}) + 1;
      }
    }
    report["nodes"].push_back(std::move(entry));

    const auto subgraphs = node.GetAttributeNameToSubgraphMap();
    InlinedVector<std::string> attributes;
    attributes.reserve(subgraphs.size());
    for (const auto& item : subgraphs) {
      attributes.push_back(item.first);
    }
    std::sort(attributes.begin(), attributes.end());
    for (const auto& attribute : attributes) {
      has_subgraphs = true;
      AppendGraph(*subgraphs.at(attribute),
                  path + "/" + std::to_string(node.Index()) + "/" + EscapePathSegment(attribute), report);
    }
  }

  Json component_summaries = Json::array();
  bool component_overflow = false;
  for (const auto& subtotal : component_subtotals) {
    Json value{{"component", subtotal.name},
               {"metric", subtotal.metric},
               {"convention_version", subtotal.convention},
               {"basis", "logical_semantics"},
               {"execution_phase", "per_invocation"},
               {"estimated_component_count", subtotal.estimated},
               {"unavailable_component_count", subtotal.unavailable},
               {"unavailable_reason_counts", subtotal.reasons},
               {"subtotal_availability", subtotal.overflow ? "unavailable" : "available"},
               {"complete_for_declared_component_scope", subtotal.unavailable == 0 && !subtotal.overflow}};
    if (subtotal.overflow) {
      component_overflow = true;
      value["subtotal_unavailable_reason"] = "overflow";
    } else {
      value["known_operation_count"] = std::to_string(subtotal.count);
    }
    component_summaries.push_back(std::move(value));
  }
  Json summary{{"reported_node_count", estimated_count + partial_count + unavailable_count},
               {"estimated_node_count", estimated_count},
               {"partially_estimated_node_count", partial_count},
               {"unavailable_node_count", unavailable_count},
               {"component_summaries", std::move(component_summaries)},
               {"unavailable_reason_counts", std::move(reasons)},
               {"subtotal_availability", subtotal_overflow ? "unavailable" : "available"},
               {"contains_subgraphs", has_subgraphs},
               {"complete_for_declared_graph_and_component_scope",
                unavailable_count == 0 && partial_count == 0 && !subtotal_overflow && !component_overflow},
               {"whole_operator_work_complete", false},
               {"count_scope", "direct_nodes_only"},
               {"per_run_total_available", false}};
  if (subtotal_overflow) {
    summary["subtotal_unavailable_reason"] = "overflow";
  } else {
    summary["known_dense_equivalent_ops"] = std::to_string(known_ops);
  }
  report["graphs"][graph_index]["summary"] = std::move(summary);
}

}  // namespace

Status BuildComputeEstimateReport(const Graph& graph, std::string& report) {
  Status status;
  ORT_TRY {
    Json value{{"schema_version", 2},
               {"convention_version", "core_arithmetic_v1"},
               {"graph_scope", "optimized_executable"},
               {"nodes", Json::array()},
               {"graphs", Json::array()}};
#if defined(DISABLE_CONTRIB_OPS)
    value["estimator_support"] = "disabled";
#else
    value["estimator_support"] = "matmul_nbits_core_and_bias";
#endif
    AppendGraph(graph, "main", value);
    value["summary"] = value["graphs"][0]["summary"];
    value["summary"]["graph_path"] = "main";
    report = value.dump(2) + "\n";
    return Status::OK();
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to generate compute estimate report: ", ex.what());
    });
  }
  return status;
}

namespace {
Status WriteReport(const std::string& report, const std::string& utf8_path) {
  Status status;
  ORT_TRY {
    std::ofstream stream(ToPathString(utf8_path), std::ios::out | std::ios::trunc | std::ios::binary);
    ORT_RETURN_IF_NOT(stream.is_open(), "Failed to open compute estimate report: ", utf8_path);
    stream << report;
    stream.close();
    ORT_RETURN_IF_NOT(stream.good(), "Failed to write compute estimate report: ", utf8_path);
    return Status::OK();
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to write compute estimate report: ", utf8_path, ": ", ex.what());
    });
  }
  return status;
}

const char* RouteName(MatMulNBitsRuntimeRoute route) {
  switch (route) {
    case MatMulNBitsRuntimeRoute::EmptyOutput:
      return "empty_output";
    case MatMulNBitsRuntimeRoute::PackedLut:
      return "packed_lut";
    case MatMulNBitsRuntimeRoute::PackedQNBit:
      return "packed_qnbit";
    case MatMulNBitsRuntimeRoute::Unpacked:
      return "unpacked";
  }
  ORT_THROW("Invalid MatMulNBits runtime route.");
}

const char* ModeName(MatMulNBitsRuntimeComputeMode mode) {
  switch (mode) {
    case MatMulNBitsRuntimeComputeMode::NotExecuted:
      return "not_executed";
    case MatMulNBitsRuntimeComputeMode::Float32Gemm:
      return "float32_gemm";
    case MatMulNBitsRuntimeComputeMode::Lut:
      return "lut";
    case MatMulNBitsRuntimeComputeMode::SQNBitFp32:
      return "sqnbit_fp32";
    case MatMulNBitsRuntimeComputeMode::HQNBitFp16:
      return "hqnbit_fp16";
    case MatMulNBitsRuntimeComputeMode::BHQNBitBf16:
      return "bhqnbit_bf16";
    case MatMulNBitsRuntimeComputeMode::SQNBitInt8:
      return "sqnbit_int8";
    case MatMulNBitsRuntimeComputeMode::HQNBitInt8:
      return "hqnbit_int8";
  }
  ORT_THROW("Invalid MatMulNBits runtime compute mode.");
}

const char* RunStateName(RuntimeWorkloadRunState state) {
  switch (state) {
    case RuntimeWorkloadRunState::NotStarted:
      return "not_started";
    case RuntimeWorkloadRunState::Running:
      return "running";
    case RuntimeWorkloadRunState::Succeeded:
      return "succeeded";
    case RuntimeWorkloadRunState::Failed:
      return "failed";
  }
  ORT_THROW("Invalid runtime workload run state.");
}

Json RuntimeComponent(bool bias) {
  return {{"component", bias ? "bias" : "matmul_core"},
          {"metric", bias ? "floating_point_ops" : "dense_equivalent_ops"},
          {"convention_version", bias ? "bias_add_v1" : "core_arithmetic_v1"},
          {"input_basis", "runtime_metadata"},
          {"basis", "logical_semantics"},
          {"execution_phase", "per_invocation"},
          {"availability", "unavailable"}};
}
}  // namespace

Status WriteComputeEstimateReport(const Graph& graph, const std::string& utf8_path) {
  std::string report;
  ORT_RETURN_IF_ERROR(BuildComputeEstimateReport(graph, report));
  return WriteReport(report, utf8_path);
}

Status BuildComputeEstimateReport(const RuntimeWorkloadSnapshot& snapshot, std::string& report) {
  Status status;
  ORT_TRY {
    const auto& stats = snapshot.stats;
    Json value{{"schema_version", 3},
               {"report_kind", "runtime_workload"},
               {"session_id", snapshot.session_id},
               {"run_id", snapshot.run_id},
               {"run_state", RunStateName(snapshot.run_state)},
               {"coverage_scope", "instrumented_cpu_matmul_nbits_invocations_only"},
               {"whole_run_work_complete", false},
               {"whole_operator_work_complete", false},
               {"route_adjustment_applied", false},
               {"excluded_components", Json::array({"dequantization", "packing", "conversion", "padding",
                                                    "route_specific_extra_work"})},
               {"invocations", Json::array()}};
    value["collection"] = {{"collection_complete", stats.collection_complete},
                           {"max_records", snapshot.limits.max_records},
                           {"max_payload_bytes", snapshot.limits.max_payload_bytes},
                           {"retained_record_count", snapshot.records.size()},
                           {"retained_payload_bytes", stats.retained_payload_bytes},
                           {"record_limit_drops", stats.record_limit_drops},
                           {"payload_limit_drops", stats.payload_limit_drops},
                           {"allocation_failure_drops", stats.allocation_failure_drops},
                           {"metadata_failure_drops", stats.metadata_failure_drops},
                           {"run_reuse_rejections", stats.run_reuse_rejections}};
#if defined(DISABLE_CONTRIB_OPS)
    value["estimator_support"] = "disabled";
#else
    value["estimator_support"] = "matmul_nbits_core_and_bias";
#endif
    struct Subtotal {
      uint64_t count = 0;
      size_t estimated = 0;
      size_t unavailable = 0;
      bool overflow = false;
      Json reasons = Json::object();
    };
    std::array<Subtotal, 2> subtotals;
    for (const auto& record : snapshot.records) {
      const auto& metadata = record.matmul_nbits;
      Json entry{{"invocation_id", record.invocation_id},
                 {"graph_path", record.graph_path},
                 {"node_index", record.node_index},
                 {"provider", record.provider},
                 {"domain", kMSDomain},
                 {"op_type", "MatMulNBits"},
                 {"input_shape", metadata.input_shape},
                 {"output_shape", metadata.output_shape},
                 {"backend_geometry", {{"m", metadata.m}, {"n", metadata.n}, {"k", metadata.k}, {"batch_count", metadata.batch_count}}},
                 {"bias_present", metadata.bias_present},
                 {"input_element_type", metadata.input_element_type},
                 {"bits", metadata.bits},
                 {"block_size", metadata.block_size},
                 {"components", Json::array()}};
      if (metadata.route) entry["route"] = RouteName(*metadata.route);
      if (metadata.compute_mode) entry["compute_mode"] = ModeName(*metadata.compute_mode);
      size_t available = 0;
      for (size_t i = 0; i < subtotals.size(); ++i) {
        Json component = RuntimeComponent(i == 1);
        std::optional<uint64_t> count;
        if (record.provider != kCpuExecutionProvider) {
          component["unavailable_reason"] = "unsupported_provider";
        } else {
#if !defined(DISABLE_CONTRIB_OPS)
          // Full activation dimensions already include all batches; do not multiply by backend batch_count again.
          if (i == 0) {
            const auto estimate = contrib::EstimateMatMulNBitsCompute(metadata.input_shape, metadata.k, metadata.n);
            if (const auto* result = std::get_if<contrib::MatMulNBitsComputeEstimate>(&estimate)) {
              count = result->dense_equivalent_ops;
            } else {
              component["unavailable_reason"] = ReasonName(std::get<contrib::ComputeEstimateUnavailableReason>(estimate));
            }
          } else {
            const auto estimate = contrib::EstimateMatMulNBitsBias(
                metadata.input_shape, metadata.k, metadata.n, metadata.bias_present);
            if (const auto* result = std::get_if<contrib::MatMulNBitsBiasEstimate>(&estimate)) {
              count = result->floating_point_ops;
            } else {
              component["unavailable_reason"] = ReasonName(std::get<contrib::ComputeEstimateUnavailableReason>(estimate));
            }
          }
#else
          component["unavailable_reason"] = "estimators_disabled";
#endif
        }
        auto& subtotal = subtotals[i];
        if (count) {
          ++available;
          ++subtotal.estimated;
          component["availability"] = "available";
          component["estimate_kind"] = "exact_under_convention";
          component["operation_count"] = std::to_string(*count);
          if (!subtotal.overflow && !SafeAdd(subtotal.count, *count, subtotal.count)) {
            subtotal.overflow = true;
          }
        } else {
          ++subtotal.unavailable;
          const auto reason = component["unavailable_reason"].get<std::string>();
          subtotal.reasons[reason] = subtotal.reasons.value(reason, size_t{0}) + 1;
        }
        entry["components"].push_back(std::move(component));
      }
      entry["availability"] = available == 2 ? "available" : available == 1 ? "partial"
                                                                            : "unavailable";
      value["invocations"].push_back(std::move(entry));
    }
    value["component_summaries"] = Json::array();
    for (size_t i = 0; i < subtotals.size(); ++i) {
      const auto& subtotal = subtotals[i];
      Json summary = RuntimeComponent(i == 1);
      summary.erase("availability");
      summary["estimated_component_count"] = subtotal.estimated;
      summary["unavailable_component_count"] = subtotal.unavailable;
      summary["unavailable_reason_counts"] = subtotal.reasons;
      summary["count_scope"] = "retained_successful_invocations";
      summary["subtotal_availability"] = subtotal.overflow ? "unavailable" : "available";
      summary["complete_for_captured_component_scope"] =
          snapshot.run_state == RuntimeWorkloadRunState::Succeeded && stats.collection_complete &&
          subtotal.unavailable == 0 && !subtotal.overflow;
      if (subtotal.overflow) {
        summary["subtotal_unavailable_reason"] = "overflow";
      } else {
        summary["known_operation_count"] = std::to_string(subtotal.count);
      }
      value["component_summaries"].push_back(std::move(summary));
    }
    report = value.dump(2) + "\n";
    return Status::OK();
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to generate runtime compute estimate report: ", ex.what());
    });
  }
  return status;
}

Status WriteComputeEstimateReport(const RuntimeWorkloadSnapshot& snapshot, const std::string& utf8_path) {
  std::string report;
  ORT_RETURN_IF_ERROR(BuildComputeEstimateReport(snapshot, report));
  return WriteReport(report, utf8_path);
}

Status ExportRuntimeComputeEstimateReport(const RuntimeWorkloadCollector& collector, const std::string& utf8_prefix) {
  Status status;
  ORT_TRY {
    RuntimeWorkloadSnapshot snapshot;
    ORT_RETURN_IF_ERROR(collector.GetSnapshot(snapshot));
    ORT_RETURN_IF_NOT(snapshot.run_state == RuntimeWorkloadRunState::Succeeded ||
                          snapshot.run_state == RuntimeWorkloadRunState::Failed,
                      "Runtime compute estimate export requires a completed run.");
    static const auto process_token =
        std::to_string(Env::Default().GetSelfPid()) + "_" +
        std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    const auto path = utf8_prefix + "_" + process_token + "_session-" + std::to_string(snapshot.session_id) +
                      "_run-" + std::to_string(snapshot.run_id) + ".json";
    return WriteComputeEstimateReport(snapshot, path);
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to export runtime compute estimate report: ", ex.what());
    });
  }
  return status;
}

}  // namespace onnxruntime
#endif
