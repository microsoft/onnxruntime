#if !defined(ORT_MINIMAL_BUILD)

#include "test/unittest_util/base_tester.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <set>

#include "gtest/gtest.h"
#include "core/common/safeint.h"
#include "test/ep_op_conformance/runner/case_io.h"

namespace onnxruntime::test {
namespace {

void ValidateInlineTensor(const ONNX_NAMESPACE::TensorProto& tensor) {
  ORT_ENFORCE(tensor.data_type() == ONNX_NAMESPACE::TensorProto::FLOAT && !tensor.has_segment() &&
                  tensor.data_location() != ONNX_NAMESPACE::TensorProto::EXTERNAL && tensor.external_data_size() == 0,
              "EP op conformance export requires inline dense float32 tensors");
  SafeInt<size_t> count = 1;
  for (const auto dimension : tensor.dims()) {
    ORT_ENFORCE(dimension >= 0, "EP op conformance export requires concrete tensor dimensions");
    count *= static_cast<size_t>(dimension);
  }
  if (tensor.has_raw_data()) {
    const uint16_t endian_check = 1;
    ORT_ENFORCE(*reinterpret_cast<const unsigned char*>(&endian_check) == 1,
                "EP op conformance export requires little-endian raw tensor data");
    ORT_ENFORCE(tensor.float_data_size() == 0 && tensor.raw_data().size() == count * sizeof(float),
                "EP op conformance export found invalid raw tensor data");
  } else {
    ORT_ENFORCE(static_cast<size_t>(tensor.float_data_size()) == static_cast<size_t>(count),
                "EP op conformance export found invalid tensor data");
  }
}

ONNX_NAMESPACE::TensorProto ExportTensor(const std::string& name, const OrtValue& value) {
  ORT_ENFORCE(!name.empty() && value.IsAllocated() && value.IsTensor(),
              "EP op conformance export requires named, allocated tensors; missing optional values are unsupported");
  const auto& tensor = value.Get<Tensor>();
  ORT_ENFORCE(tensor.IsDataType<float>() && tensor.Location().device.Type() == OrtDevice::CPU,
              "EP op conformance export requires CPU-authored float32 tensors");
  ONNX_NAMESPACE::TensorProto result;
  result.set_name(name);
  result.set_data_type(ONNX_NAMESPACE::TensorProto::FLOAT);
  for (const auto dimension : tensor.Shape().GetDims()) {
    result.add_dims(dimension);
  }
  const auto* values = tensor.Data<float>();
  for (int64_t index = 0; index < tensor.Shape().Size(); ++index) {
    result.add_float_data(values[index]);
  }
  ValidateInlineTensor(result);
  return result;
}

std::string Serialize(const google::protobuf::MessageLite& message) {
  std::string bytes;
  ORT_ENFORCE(message.SerializeToString(&bytes), "EP op conformance export serialization failed");
  return bytes;
}

}  // namespace

void BaseTester::ExportEpOpConformanceCase(const Model& model, const std::unordered_map<std::string, OrtValue>& feeds) {
  ORT_ENFORCE(num_run_calls_ == 1, "EP op conformance export requires exactly one run per dataset");
  ORT_ENFORCE(verify_output_ && !custom_output_verifier_,
              "EP op conformance export requires the standard output verifier");
  ORT_ENFORCE(ctx_.expect_result == ExpectResult::kExpectSuccess && ctx_.expected_failure_string.empty(),
              "EP op conformance export does not support expected failures");
  ORT_ENFORCE(custom_schema_registries_.empty() && custom_session_registries_.empty(),
              "EP op conformance export does not support custom registries");
  ORT_ENFORCE(!custom_session_options_configured_ && !custom_run_options_configured_ &&
                  !custom_resolve_options_configured_ && ctx_.run_options == nullptr,
              "EP op conformance export does not support customized session, run, or resolve options");
  ORT_ENFORCE(!ctx_.resolve_options.override_types && !ctx_.resolve_options.no_proto_sync_required &&
                  ctx_.resolve_options.initializer_names_to_preserve == nullptr,
              "EP op conformance export does not support customized resolve options");
  const auto& options = ctx_.session_options;
  ORT_ENFORCE(options.graph_optimization_level == TransformerLevel::Default,
              "EP op conformance export requires unoptimized simple cases; use Run() without custom session options");
  ORT_ENFORCE(options.execution_mode == ExecutionMode::ORT_SEQUENTIAL &&
                  options.execution_order == ExecutionOrder::DEFAULT && !options.use_deterministic_compute &&
                  !use_determinism_ && !add_prepacked_shared_container_to_sessions_ &&
                  options.config_options.GetConfigOptionsMap().empty() && options.free_dimension_overrides.empty() &&
                  options.initializers_to_share_map.empty() && !options.custom_create_thread_fn &&
                  !options.custom_join_thread_fn && !options.custom_thread_creation_options &&
                  !options.user_logging_function && !options.user_logging_param && !options.custom_op_libs &&
                  !options.enable_profiling && options.optimized_model_filepath.empty() &&
                  !options.ep_selection_policy.enable && !options.has_explicit_ep_context_gen_options &&
                  !options.ep_context_data_read_func && test_allow_released_onnx_opset_only_,
              "EP op conformance export requires baseline session semantics");

  auto proto = model.ToProto();
  ORT_ENFORCE(proto.has_graph() && proto.functions_size() == 0 && proto.training_info_size() == 0,
              "EP op conformance export does not support functions or training graphs");
  auto& graph = *proto.mutable_graph();
  ORT_ENFORCE(graph.sparse_initializer_size() == 0 && graph.node_size() > 0,
              "EP op conformance export requires a nonempty dense graph");

  std::set<std::string> initializer_names;
  for (const auto& initializer : graph.initializer()) {
    ValidateInlineTensor(initializer);
    ORT_ENFORCE(!initializer.name().empty() && initializer_names.insert(initializer.name()).second,
                "EP op conformance export requires unique named initializers");
  }

  std::set<std::string> node_names;
  for (const auto& node : graph.node()) {
    if (!node.name().empty()) {
      ORT_ENFORCE(node_names.insert(node.name()).second, "EP op conformance export requires unique node names");
    }
  }

  auto required_nodes = ep_op_conformance::Json::array();
  size_t node_index = 0;
  for (auto& node : *graph.mutable_node()) {
    ORT_ENFORCE(node.domain().empty() || node.domain() == "ai.onnx",
                "EP op conformance export only supports standard ONNX operators");
    if (node.name().empty()) {
      std::string name;
      do {
        name = "conformance_node_" + std::to_string(node_index++);
      } while (!node_names.insert(name).second);
      node.set_name(name);
    }
    required_nodes.push_back(node.name());
    for (const auto& attribute : node.attribute()) {
      ORT_ENFORCE(attribute.type() != ONNX_NAMESPACE::AttributeProto::GRAPH &&
                      attribute.type() != ONNX_NAMESPACE::AttributeProto::GRAPHS &&
                      !attribute.has_g() && attribute.graphs_size() == 0 &&
                      !attribute.has_sparse_tensor() && attribute.sparse_tensors_size() == 0,
                  "EP op conformance export does not support nested graphs or sparse attributes");
      if (attribute.has_t()) {
        ValidateInlineTensor(attribute.t());
      }

      for (const auto& tensor : attribute.tensors()) {
        ValidateInlineTensor(tensor);
      }
    }
  }

  std::vector<std::pair<std::string, std::string>> files;
  auto input_files = ep_op_conformance::Json::array();
  auto output_files = ep_op_conformance::Json::array();
  auto comparisons = ep_op_conformance::Json::object();
  std::set<std::string> input_names;
  for (const auto& input : input_data_) {
    ORT_ENFORCE(input.def.Exists() && input.def.TypeAsProto() && input.def.TypeAsProto()->has_tensor_type(),
                "EP op conformance export does not support missing or optional inputs");
    ExportTensor(input.def.Name(), input.data);
    ORT_ENFORCE(input_names.insert(input.def.Name()).second, "EP op conformance export requires unique input names");
  }
  std::set<std::string> graph_inputs;
  for (const auto& input : graph.input()) {
    ORT_ENFORCE(input.type().has_tensor_type() &&
                    input.type().tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto::FLOAT,
                "EP op conformance export requires float32 graph inputs");
    if (initializer_names.count(input.name())) {
      continue;
    }

    ORT_ENFORCE(feeds.count(input.name()) == 1, "EP op conformance export is missing a required input");
    graph_inputs.insert(input.name());
    const auto path = "input_" + std::to_string(input_files.size()) + ".pb";
    files.emplace_back(path, Serialize(ExportTensor(input.name(), feeds.at(input.name()))));
    input_files.push_back(path);
  }

  ORT_ENFORCE(graph_inputs.size() == feeds.size(), "EP op conformance export found unexpected feeds");
  std::set<std::string> output_names;
  for (const auto& output : output_data_) {
    ORT_ENFORCE(output.def.Exists() && output.def.TypeAsProto() && output.def.TypeAsProto()->has_tensor_type() &&
                    !output.validation_params.sort_output,
                "EP op conformance export does not support missing, optional, or sorted outputs");
    const auto& name = output.def.Name();
    ORT_ENFORCE(output_names.insert(name).second, "EP op conformance export requires unique output names");
    const auto path = "output_" + std::to_string(output_files.size()) + ".pb";
    files.emplace_back(path, Serialize(ExportTensor(name, output.data)));
    output_files.push_back(path);
#if defined(ENABLE_TRAINING)
    constexpr float default_absolute = 1e-3f;
#else
    constexpr float default_absolute = 1e-5f;
#endif
    const ep_op_conformance::Comparison policy{output.validation_params.absolute_error.value_or(default_absolute),
                                               output.validation_params.relative_error.value_or(1e-4f), true};
    ep_op_conformance::ValidateComparison(policy);
    comparisons[name] = {{"atol", policy.absolute}, {"rtol", policy.relative}, {"nan_equal", policy.nan_equal}};
  }
  ORT_ENFORCE(!output_names.empty() && output_names.size() == static_cast<size_t>(graph.output_size()),
              "EP op conformance export requires authored expected data for every graph output");
  for (const auto& output : graph.output()) {
    ORT_ENFORCE(output_names.count(output.name()) && output.type().has_tensor_type() &&
                    output.type().tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto::FLOAT,
                "EP op conformance export is missing a float32 output target");
  }

  const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
  ORT_ENFORCE(test_info != nullptr, "EP op conformance export requires Google Test provenance");
  std::string source_file = test_info->file();
  std::replace(source_file.begin(), source_file.end(), '\\', '/');
  const auto test_root = source_file.find("onnxruntime/test/");
  source_file = test_root == std::string::npos ? std::filesystem::path(source_file).filename().generic_string()
                                               : source_file.substr(test_root);
  ep_op_conformance::Json manifest;
  manifest["schema_version"] = 1;
  manifest["id"] = ep_op_conformance_export_id_;
  manifest["model"] = "model.onnx";
  manifest["required_nodes"] = required_nodes;
  manifest["datasets"] = ep_op_conformance::Json::array({{{"inputs", input_files}, {"outputs", output_files}}});
  manifest["comparison"] = comparisons;
  manifest["source"] = {{"suite", test_info->test_suite_name()}, {"test", test_info->name()}, {"file", source_file}, {"line", test_info->line()}};
  files.emplace_back("model.onnx", Serialize(proto));
  files.emplace_back("case.json", manifest.dump(2));

  const std::filesystem::path directory(ep_op_conformance_export_directory_);
  ORT_ENFORCE(!std::filesystem::exists(directory), "EP op conformance export directory already exists");
  if (!directory.parent_path().empty()) {
    std::filesystem::create_directories(directory.parent_path());
  }

  ORT_ENFORCE(std::filesystem::create_directory(directory), "EP op conformance export directory already exists");
  ORT_TRY {
    for (const auto& file : files) {
      ep_op_conformance::WriteFile(directory / file.first, file.second);
    }
  }
  ORT_CATCH(...) {
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    ORT_RETHROW;
  }
}

}  // namespace onnxruntime::test

#endif