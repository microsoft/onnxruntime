#pragma once

#include "comparison.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>

#include <nlohmann/json.hpp>
#include <onnx/onnx-ml.pb.h>

namespace onnxruntime::test::ep_op_conformance {

using Json = nlohmann::json;
namespace fs = std::filesystem;

inline std::string ReadFile(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("Cannot open " + path.string());
  }

  std::ostringstream contents;
  contents << stream.rdbuf();
  if (stream.bad()) {
    throw std::runtime_error("Cannot read " + path.string());
  }

  return contents.str();
}

inline void WriteFile(const fs::path& path, const std::string& bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  stream.close();
  if (!stream) {
    throw std::runtime_error("Cannot write " + path.string());
  }
}

inline fs::path CasePath(const fs::path& root, const std::string& relative) {
  const fs::path path(relative);
  if (relative.empty() || path.is_absolute() || path.has_root_name()) {
    throw std::runtime_error("Case paths must be relative");
  }

  for (const auto& component : path) {
    if (component == "..") {
      throw std::runtime_error("Case paths cannot escape case directory");
    }
  }

  const auto base = fs::canonical(root);
  const auto resolved = fs::canonical(root / path);
  auto base_part = base.begin();
  auto resolved_part = resolved.begin();
  for (; base_part != base.end(); ++base_part, ++resolved_part) {
    if (resolved_part == resolved.end() || *base_part != *resolved_part) {
      throw std::runtime_error("Case path resolves outside case directory");
    }
  }

  return resolved;
}

inline void CheckFields(const Json& object, const std::set<std::string>& allowed) {
  if (!object.is_object()) {
    throw std::runtime_error("Expected JSON object");
  }

  for (auto field = object.begin(); field != object.end(); ++field) {
    if (!allowed.count(field.key())) {
      throw std::runtime_error("Unsupported field: " + field.key());
    }
  }
}

struct FloatTensor {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<float> values;
};

inline FloatTensor ReadTensor(const onnx::TensorProto& proto, bool require_name = true) {
  if (proto.data_type() != onnx::TensorProto::FLOAT || (require_name && proto.name().empty()) ||
      proto.has_segment() || proto.data_location() == onnx::TensorProto::EXTERNAL || proto.external_data_size() ||
      proto.int32_data_size() || proto.int64_data_size() || proto.uint64_data_size() || proto.double_data_size() ||
      proto.string_data_size()) {
    throw std::runtime_error("Pilot requires named, inline, dense float32 TensorProto");
  }

  FloatTensor tensor{proto.name(), {proto.dims().begin(), proto.dims().end()}, {}};
  size_t count = 1;
  for (const auto dimension : tensor.shape) {
    if (dimension < 0 || (dimension && count > std::numeric_limits<size_t>::max() / static_cast<size_t>(dimension))) {
      throw std::runtime_error("Invalid tensor shape");
    }
    count *= static_cast<size_t>(dimension);
  }
  if (count > std::numeric_limits<size_t>::max() / sizeof(float)) {
    throw std::runtime_error("Tensor too large");
  }

  if (proto.has_raw_data()) {
    if (proto.float_data_size() || proto.raw_data().size() != count * sizeof(float)) {
      throw std::runtime_error("Invalid raw tensor data size");
    }

    const uint16_t endian_check = 1;
    if (*reinterpret_cast<const unsigned char*>(&endian_check) != 1) {
      throw std::runtime_error("Pilot raw float data requires little-endian host");
    }

    tensor.values.resize(count);
    if (count) {
      std::memcpy(tensor.values.data(), proto.raw_data().data(), count * sizeof(float));
    }
  } else {
    if (static_cast<size_t>(proto.float_data_size()) != count) {
      throw std::runtime_error("Invalid tensor element count");
    }

    tensor.values.assign(proto.float_data().begin(), proto.float_data().end());
  }

  return tensor;
}

inline FloatTensor ReadTensor(const fs::path& path) {
  onnx::TensorProto proto;
  if (!proto.ParseFromString(ReadFile(path))) {
    throw std::runtime_error("Invalid TensorProto: " + path.string());
  }

  return ReadTensor(proto);
}

inline std::set<std::string> ValidateModel(const onnx::ModelProto& model) {
  if (!model.has_graph() || model.functions_size() || model.training_info_size()) {
    throw std::runtime_error("Pilot does not support functions or training graphs");
  }

  const auto& graph = model.graph();
  if (graph.node_size() == 0 || graph.sparse_initializer_size()) {
    throw std::runtime_error("Pilot requires a nonempty dense graph");
  }

  for (const auto* values : {&graph.input(), &graph.output(), &graph.value_info()}) {
    for (const auto& value : *values) {
      if (!value.type().has_tensor_type() || value.type().tensor_type().elem_type() != onnx::TensorProto::FLOAT) {
        throw std::runtime_error("Pilot requires float32 tensor model values");
      }
    }
  }

  for (const auto& initializer : graph.initializer()) {
    ReadTensor(initializer);
  }

  std::set<std::string> names;
  for (const auto& node : graph.node()) {
    if ((!node.domain().empty() && node.domain() != "ai.onnx") || node.name().empty() ||
        !names.insert(node.name()).second) {
      throw std::runtime_error("Pilot requires uniquely named standard-domain nodes");
    }

    for (const auto& attribute : node.attribute()) {
      if (attribute.has_g() || attribute.graphs_size() || attribute.has_sparse_tensor() ||
          attribute.sparse_tensors_size() || attribute.type() == onnx::AttributeProto::GRAPH ||
          attribute.type() == onnx::AttributeProto::GRAPHS) {
        throw std::runtime_error("Pilot does not support nested graphs or sparse attributes");
      }

      if (attribute.has_t()) {
        ReadTensor(attribute.t(), false);
      }

      for (const auto& tensor : attribute.tensors()) {
        ReadTensor(tensor, false);
      }
    }
  }

  return names;
}

struct Dataset {
  std::vector<FloatTensor> inputs;
  std::vector<FloatTensor> outputs;
};

struct Case {
  Json manifest;
  std::string model;
  std::vector<Dataset> datasets;
};

inline Comparison ReadComparison(const Json& policy) {
  CheckFields(policy, {"atol", "rtol", "nan_equal"});
  Comparison result{policy.at("atol").get<double>(), policy.at("rtol").get<double>(),
                    policy.at("nan_equal").get<bool>()};
  ValidateComparison(result);

  return result;
}

inline Case ReadCase(const fs::path& directory) {
  Case result;
  result.manifest = Json::parse(ReadFile(directory / "case.json"));
  const auto& manifest = result.manifest;
  CheckFields(manifest, {"schema_version", "id", "model", "required_nodes", "datasets",
                         "comparison", "source"});
  if (!manifest.at("schema_version").is_number_integer() || manifest.at("schema_version") != 1 ||
      manifest.at("id").get<std::string>().empty()) {
    throw std::runtime_error("Unsupported case version or empty ID");
  }

  if (!manifest.at("source").is_object() || manifest.at("source").empty()) {
    throw std::runtime_error("Case must retain source provenance");
  }

  const auto nodes = manifest.at("required_nodes").get<std::vector<std::string>>();
  if (nodes.empty()) {
    throw std::runtime_error("Required nodes cannot be empty");
  }

  std::set<std::string> required;
  for (const auto& node : nodes) {
    if (node.empty() || !required.insert(node).second) {
      throw std::runtime_error("Invalid required node identity");
    }
  }

  result.model = ReadFile(CasePath(directory, manifest.at("model").get<std::string>()));
  onnx::ModelProto model;
  if (!model.ParseFromString(result.model) || !model.has_graph()) {
    throw std::runtime_error("Invalid ONNX model");
  }

  const auto model_nodes = ValidateModel(model);
  for (const auto& node : required) {
    if (!model_nodes.count(node)) {
      throw std::runtime_error("Required node absent from model: " + node);
    }
  }

  if (!manifest.at("datasets").is_array() || manifest.at("datasets").empty()) {
    throw std::runtime_error("Case has no datasets");
  }

  for (const auto& dataset : manifest.at("datasets")) {
    CheckFields(dataset, {"inputs", "outputs"});
    Dataset data;
    for (const auto& kind : {std::string("inputs"), std::string("outputs")}) {
      auto& tensors = kind == "inputs" ? data.inputs : data.outputs;
      std::set<std::string> names;
      for (const auto& file : dataset.at(kind).get<std::vector<std::string>>()) {
        auto tensor = ReadTensor(CasePath(directory, file));
        if (!names.insert(tensor.name).second) {
          throw std::runtime_error("Duplicate tensor name");
        }

        if (kind == "outputs") {
          ReadComparison(manifest.at("comparison").at(tensor.name));
        }

        tensors.push_back(std::move(tensor));
      }
    }

    if (data.outputs.size() != static_cast<size_t>(model.graph().output_size()) || data.outputs.empty()) {
      throw std::runtime_error("Every model output must have expected data");
    }

    std::set<std::string> output_names;
    for (const auto& output : data.outputs) {
      output_names.insert(output.name);
    }

    for (const auto& output : model.graph().output()) {
      if (!output_names.count(output.name())) {
        throw std::runtime_error("Missing expected output: " + output.name());
      }
    }

    if (manifest.at("comparison").size() != output_names.size()) {
      throw std::runtime_error("Unexpected comparison entry");
    }

    result.datasets.push_back(std::move(data));
  }

  return result;
}

}  // namespace onnxruntime::test::ep_op_conformance