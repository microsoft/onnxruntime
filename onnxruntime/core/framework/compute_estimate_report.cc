// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/compute_estimate_report.h"

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <fstream>
#include <optional>
#include <string_view>
#include <utility>

#include "nlohmann/json.hpp"
#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/common/path_string.h"
#include "core/common/safeint.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"

#if !defined(DISABLE_CONTRIB_OPS)
#include "contrib_ops/cpu/quantization/matmul_nbits_compute_estimate.h"
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

Status WriteComputeEstimateReport(const Graph& graph, const std::string& utf8_path) {
  Status status;
  ORT_TRY {
    std::string report;
    ORT_RETURN_IF_ERROR(BuildComputeEstimateReport(graph, report));
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

}  // namespace onnxruntime
#endif
