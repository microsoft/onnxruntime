// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <limits>
#include <memory>
#include <tuple>

#include "core/common/common.h"
#include "core/framework/error_code_helper.h"
#include "core/session/ort_apis.h"

struct OrtGraphSerializationOptions {
  OrtGraphExternalInitializerPolicy policy = ORT_GRAPH_EXTERNAL_INITIALIZER_PRESERVE_REFERENCES;
  size_t minimum_size = 0;
  OrtGetInitializerLocationFunc location = nullptr;
  void* location_state = nullptr;
  size_t chunk_size = 64 * 1024;
};

#if !defined(ORT_MINIMAL_BUILD)
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include "core/framework/tensorprotoutils.h"
#include "core/graph/ep_api_types.h"
#include "core/graph/model.h"
#include "core/platform/env.h"

namespace {
using namespace onnxruntime;
using namespace ONNX_NAMESPACE;

// Only collections without semantic ordering are sorted.
template <typename T, typename Key>
void Sort(T* values, Key key) {
  std::sort(values->begin(), values->end(), [&](const auto& a, const auto& b) { return key(a) < key(b); });
}

void CanonicalTensor(TensorProto& tensor) {
  Sort(tensor.mutable_external_data(), [](const auto& entry) { return entry.key(); });
  Sort(tensor.mutable_metadata_props(), [](const auto& entry) { return entry.key(); });
}

Status BuildView(const GraphViewer& view, GraphProto& result, bool standalone) {
  result.set_name(view.Name());
  result.set_doc_string(view.Description());
  *result.mutable_metadata_props() = view.GetGraph().GetMetadataForSerialization();
  InlinedHashSet<std::string> defined;
  InlinedHashSet<std::string> needed;
  for (const auto* input : view.GetInputsIncludingInitializers()) {
    *result.add_input() = input->ToProto();
    defined.insert(input->Name());
  }
  for (const auto* output : view.GetOutputs()) {
    *result.add_output() = output->ToProto();
    needed.insert(output->Name());
  }
  InlinedHashMap<std::string, const TensorProto*> initializers;
  for (const auto& [name, tensor] : view.GetAllInitializedTensors()) initializers.emplace(name, tensor);
  for (auto index : view.GetNodesInTopologicalOrder()) {
    const auto& node = *view.GetNode(index);
    auto& output = *result.add_node();
    node.ToProto(output, false);
    for (const auto* arg : node.InputDefs())
      if (arg->Exists()) needed.insert(arg->Name());
    for (const auto* arg : node.ImplicitInputDefs())
      if (arg->Exists()) needed.insert(arg->Name());
    for (const auto* arg : node.OutputDefs())
      if (arg->Exists()) defined.insert(arg->Name());
    for (auto& attr : *output.mutable_attribute()) {
      if (const auto* subgraph = node.GetGraphAttribute(attr.name())) {
        attr.clear_g();
        ORT_RETURN_IF_ERROR(BuildView(GraphViewer(*subgraph), *attr.mutable_g(), false));
      }
    }
  }
  for (const auto* info : view.GetValueInfo()) {
    if (defined.count(info->Name()) || needed.count(info->Name())) *result.add_value_info() = info->ToProto();
  }
  if (standalone) {
    InlinedVector<std::string> captures;
    for (const auto& name : needed) {
      if (initializers.count(name)) continue;
      if (const auto* tensor = view.GetConstantInitializer(name, true)) {
        initializers.emplace(name, tensor);
      } else if (!defined.count(name)) {
        captures.push_back(name);
      }
    }
    std::sort(captures.begin(), captures.end());
    for (const auto& name : captures) {
      const auto* arg = view.GetGraph().GetNodeArgIncludingParentGraphs(name);
      ORT_RETURN_IF_NOT(arg, "Missing captured value: ", name);
      *result.add_input() = arg->ToProto();
    }
  }
  InlinedVector<std::string> names;
  for (const auto& [name, tensor] : initializers) names.push_back(name);
  std::sort(names.begin(), names.end());
  for (const auto& name : names) {
    const Graph* scope = &view.GetGraph();
    while (scope && !scope->IsInitializedTensor(name)) scope = scope->ParentGraph();
#if !defined(DISABLE_SPARSE_TENSORS)
    if (scope && scope->IsSparseInitializer(name)) {
      const auto* sparse = scope->GetSparseInitializerProto(name);
      if (sparse) {
        *result.add_sparse_initializer() = *sparse;
      } else {
        ORT_RETURN_IF_ERROR(utils::DenseTensorToSparseTensorProto(*initializers.at(name), scope->ModelPath(),
                                                                  *result.add_sparse_initializer()));
      }
      continue;
    }
#endif
    *result.add_initializer() = *initializers.at(name);
  }
  return Status::OK();
}

// Preflight the complete model before any user callback can have side effects.
bool HasFileData(const GraphProto& graph);

bool HasFileData(const AttributeProto& attr) {
  if (attr.has_g() && HasFileData(attr.g())) return true;
  for (const auto& graph : attr.graphs())
    if (HasFileData(graph)) return true;
  return false;
}

bool HasFileData(const GraphProto& graph) {
  for (const auto& tensor : graph.initializer())
    if (utils::HasExternalDataInFile(tensor)) return true;
  for (const auto& sparse : graph.sparse_initializer()) {
    if (utils::HasExternalDataInFile(sparse.values()) || utils::HasExternalDataInFile(sparse.indices())) return true;
  }
  for (const auto& node : graph.node())
    for (const auto& attr : node.attribute())
      if (HasFileData(attr)) return true;
  return false;
}

bool HasFileData(const ModelProto& model) {
  if (HasFileData(model.graph())) return true;
  for (const auto& function : model.functions()) {
    for (const auto& attr : function.attribute_proto())
      if (HasFileData(attr)) return true;
    for (const auto& node : function.node())
      for (const auto& attr : node.attribute())
        if (HasFileData(attr)) return true;
  }
  return false;
}

Status ProcessTensor(TensorProto& tensor, const std::filesystem::path& path,
                     const OrtGraphSerializationOptions& options, bool callback_allowed) {
  const bool file = utils::HasExternalDataInFile(tensor);
  if (file && options.policy == ORT_GRAPH_EXTERNAL_INITIALIZER_PRESERVE_REFERENCES) {
    CanonicalTensor(tensor);
    return Status::OK();
  }
  if (file && options.policy == ORT_GRAPH_EXTERNAL_INITIALIZER_FAIL_IF_PRESENT)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "External initializer present: ", tensor.name());
  const bool callback = callback_allowed && options.location &&
                        tensor.data_type() != TensorProto::STRING &&
                        (!file || options.policy == ORT_GRAPH_EXTERNAL_INITIALIZER_REPACKAGE_VIA_CALLBACK);
  if (!file && !utils::HasExternalDataInMemory(tensor) && !callback) {
    CanonicalTensor(tensor);
    return Status::OK();
  }
  OrtValue value;
  if (file) {
    ORT_RETURN_IF_ERROR(utils::GetExtDataFromTensorProto(Env::Default(), path, tensor, value));
  } else {
    ORT_RETURN_IF_ERROR(utils::TensorProtoToOrtValue(Env::Default(), path, tensor,
                                                     CPUAllocator::DefaultInstance(), value));
  }
  const auto& data = value.Get<Tensor>();
  if (callback && data.SizeInBytes() >= options.minimum_size) {
    std::unique_ptr<ExternalDataInfo> original;
    if (file) ORT_RETURN_IF_ERROR(ExternalDataInfo::Create(tensor.external_data(), original));
    OrtExternalInitializerInfo* location = nullptr;
    auto status = ToStatusAndRelease(options.location(options.location_state, tensor.name().c_str(), &value,
                                                      static_cast<OrtExternalInitializerInfo*>(original.get()),
                                                      &location));
    ORT_RETURN_IF(location && location == static_cast<OrtExternalInitializerInfo*>(original.get()),
                  "Initializer callback must return a new external info instance.");
    std::unique_ptr<OrtExternalInitializerInfo> holder(location);
    ORT_RETURN_IF_ERROR(status);
    if (location) {
      auto replacement = utils::TensorToTensorProto(data, tensor.name());
      replacement.clear_raw_data();
      ExternalDataInfo::SetExternalLocationToProto(location->GetRelPath(), location->GetOffset(),
                                                   location->GetLength(), replacement);
      replacement.set_doc_string(tensor.doc_string());
      *replacement.mutable_metadata_props() = tensor.metadata_props();
      tensor = std::move(replacement);
      CanonicalTensor(tensor);
      return Status::OK();
    }
  }
  auto replacement = utils::TensorToTensorProto(data, tensor.name());
  replacement.set_doc_string(tensor.doc_string());
  *replacement.mutable_metadata_props() = tensor.metadata_props();
  tensor = std::move(replacement);
  CanonicalTensor(tensor);
  return Status::OK();
}

Status ProcessGraph(GraphProto& graph, const std::filesystem::path& path,
                    const OrtGraphSerializationOptions& options);

Status ProcessAttribute(AttributeProto& attr, const std::filesystem::path& path,
                        const OrtGraphSerializationOptions& options) {
  if (attr.has_g()) ORT_RETURN_IF_ERROR(ProcessGraph(*attr.mutable_g(), path, options));
  for (auto& graph : *attr.mutable_graphs()) ORT_RETURN_IF_ERROR(ProcessGraph(graph, path, options));
  return Status::OK();
}

Status ProcessGraph(GraphProto& graph, const std::filesystem::path& path,
                    const OrtGraphSerializationOptions& options) {
  Sort(graph.mutable_value_info(), [](const auto& entry) { return entry.name(); });
  Sort(graph.mutable_metadata_props(), [](const auto& entry) { return entry.key(); });
  Sort(graph.mutable_initializer(), [](const auto& entry) { return entry.name(); });
  Sort(graph.mutable_sparse_initializer(), [](const auto& entry) { return entry.values().name(); });
  for (auto& tensor : *graph.mutable_initializer()) ORT_RETURN_IF_ERROR(ProcessTensor(tensor, path, options, true));
  for (auto& sparse : *graph.mutable_sparse_initializer()) {
    ORT_RETURN_IF_ERROR(ProcessTensor(*sparse.mutable_values(), path, options, false));
    ORT_RETURN_IF_ERROR(ProcessTensor(*sparse.mutable_indices(), path, options, false));
  }
  for (auto& node : *graph.mutable_node()) {
    Sort(node.mutable_attribute(), [](const auto& entry) { return entry.name(); });
    Sort(node.mutable_metadata_props(), [](const auto& entry) { return entry.key(); });
    for (auto& attr : *node.mutable_attribute()) {
      ORT_RETURN_IF_ERROR(ProcessAttribute(attr, path, options));
    }
  }
  return Status::OK();
}

Status BuildModel(const OrtGraph* graph, const OrtGraphSerializationOptions& options, ModelProto& model) {
  if (!graph) return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "graph must not be null");
  const auto* ep = EpGraph::ToInternal(graph);
  if (!ep) return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                  "Model-editor graphs do not support ONNX serialization");
  ORT_RETURN_IF(options.policy == ORT_GRAPH_EXTERNAL_INITIALIZER_REPACKAGE_VIA_CALLBACK && !options.location,
                "REPACKAGE_VIA_CALLBACK requires an initializer location callback");
  const auto& view = ep->GetGraphViewer();
  model = view.GetGraph().GetModel().ToProtoWithoutGraph();
  ORT_RETURN_IF_ERROR(BuildView(view, *model.mutable_graph(), true));
  ORT_RETURN_IF(options.policy == ORT_GRAPH_EXTERNAL_INITIALIZER_FAIL_IF_PRESENT && HasFileData(model),
                "External initializers are forbidden by FAIL_IF_PRESENT");
  ORT_RETURN_IF_ERROR(ProcessGraph(*model.mutable_graph(), view.GetGraph().ModelPath(), options));
  Sort(model.mutable_metadata_props(), [](const auto& entry) { return entry.key(); });
  Sort(model.mutable_opset_import(), [](const auto& entry) { return entry.domain(); });
  Sort(model.mutable_functions(), [](const auto& entry) {
    return std::make_tuple(entry.domain(), entry.name(), entry.overload());
  });
  for (auto& function : *model.mutable_functions()) {
    Sort(function.mutable_opset_import(), [](const auto& entry) { return entry.domain(); });
    Sort(function.mutable_metadata_props(), [](const auto& entry) { return entry.key(); });
    Sort(function.mutable_attribute(), [](const auto& entry) { return entry; });
    Sort(function.mutable_attribute_proto(), [](const auto& entry) { return entry.name(); });
    for (auto& attr : *function.mutable_attribute_proto()) {
      ORT_RETURN_IF_ERROR(ProcessAttribute(attr, view.GetGraph().ModelPath(), options));
    }
    for (auto& node : *function.mutable_node()) {
      Sort(node.mutable_attribute(), [](const auto& entry) { return entry.name(); });
      for (auto& attr : *node.mutable_attribute()) {
        ORT_RETURN_IF_ERROR(ProcessAttribute(attr, view.GetGraph().ModelPath(), options));
      }
    }
  }
  ORT_RETURN_IF(model.ByteSizeLong() > static_cast<size_t>(std::numeric_limits<int>::max()),
                "Serialized model exceeds protobuf's size limit");
  return Status::OK();
}

class CallbackStream final : public google::protobuf::io::ZeroCopyOutputStream {
 public:
  CallbackStream(OrtWriteBufferFunc func, void* state, size_t chunk)
      : func_(func), state_(state), buffer_(std::min(chunk, size_t{64 * 1024})) {}
  bool Next(void** data, int* size) override {
    if (!Flush().IsOK()) return false;
    *data = buffer_.data();
    *size = static_cast<int>(buffer_.size());
    pending_ = *size;
    count_ += pending_;
    return true;
  }
  void BackUp(int count) override {
    pending_ -= count;
    count_ -= count;
  }
  int64_t ByteCount() const override { return count_; }
  Status Flush() {
    if (status_.IsOK() && pending_) status_ = ToStatusAndRelease(func_(state_, buffer_.data(), pending_));
    pending_ = 0;
    return status_;
  }

 private:
  OrtWriteBufferFunc func_;
  void* state_;
  InlinedVector<char> buffer_;
  int pending_ = 0;
  int64_t count_ = 0;
  Status status_;
};
}  // namespace
#endif

using namespace onnxruntime;

ORT_API(void, OrtApis::ReleaseGraphSerializationOptions, OrtGraphSerializationOptions* options) { delete options; }

ORT_API_STATUS_IMPL(OrtApis::CreateGraphSerializationOptions, OrtGraphSerializationOptions** out) {
  API_IMPL_BEGIN
  if (!out) return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "out must not be null");
  *out = new OrtGraphSerializationOptions();
  return nullptr;
  API_IMPL_END
}
ORT_API_STATUS_IMPL(OrtApis::GraphSerializationOptions_SetExternalInitializerPolicy,
                    OrtGraphSerializationOptions* options, OrtGraphExternalInitializerPolicy policy) {
  API_IMPL_BEGIN
  if (!options || policy < ORT_GRAPH_EXTERNAL_INITIALIZER_PRESERVE_REFERENCES ||
      policy > ORT_GRAPH_EXTERNAL_INITIALIZER_FAIL_IF_PRESENT)
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid options or initializer policy");
  options->policy = policy;
  return nullptr;
  API_IMPL_END
}
ORT_API_STATUS_IMPL(OrtApis::GraphSerializationOptions_SetMinimumExternalInitializerSize,
                    OrtGraphSerializationOptions* options, size_t min_bytes) {
  API_IMPL_BEGIN
  if (!options) return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "options must not be null");
  options->minimum_size = min_bytes;
  return nullptr;
  API_IMPL_END
}
ORT_API_STATUS_IMPL(OrtApis::GraphSerializationOptions_SetInitializerLocationFunc,
                    OrtGraphSerializationOptions* options, OrtGetInitializerLocationFunc func, void* state) {
  API_IMPL_BEGIN
  if (!options) return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "options must not be null");
  options->location = func;
  options->location_state = func ? state : nullptr;
  return nullptr;
  API_IMPL_END
}
ORT_API_STATUS_IMPL(OrtApis::GraphSerializationOptions_SetMaximumWriteChunkSize,
                    OrtGraphSerializationOptions* options, size_t chunk) {
  API_IMPL_BEGIN
  if (!options) return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "options must not be null");
  options->chunk_size = chunk ? chunk : 64 * 1024;
  return nullptr;
  API_IMPL_END
}
ORT_API_STATUS_IMPL(OrtApis::Graph_SerializeToOnnxModelWithWriteFunc, const OrtGraph* graph,
                    const OrtGraphSerializationOptions* options, OrtWriteBufferFunc func, void* state) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (!func) return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "write_func must not be null");
  ONNX_NAMESPACE::ModelProto model;
  const OrtGraphSerializationOptions defaults;
  const auto& settings = options ? *options : defaults;
  ORT_API_RETURN_IF_STATUS_NOT_OK(BuildModel(graph, settings, model));
  CallbackStream stream(func, state, settings.chunk_size);
  bool success;
  {
    google::protobuf::io::CodedOutputStream coded(&stream);
    coded.SetSerializationDeterministic(true);
    success = model.SerializeToCodedStream(&coded) && !coded.HadError();
  }
  ORT_API_RETURN_IF_STATUS_NOT_OK(stream.Flush());
  if (!success) return OrtApis::CreateStatus(ORT_FAIL, "Model serialization failed");
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(graph);
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(func);
  ORT_UNUSED_PARAMETER(state);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Graph serialization is unavailable in minimal builds");
#endif
  API_IMPL_END
}
ORT_API_STATUS_IMPL(OrtApis::Graph_SerializeToOnnxModel, const OrtGraph* graph,
                    const OrtGraphSerializationOptions* options, OrtAllocator* allocator,
                    void** bytes, size_t* length) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (!allocator || !allocator->Alloc || !allocator->Free || !bytes || !length)
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "allocator and output pointers must be valid");
  ONNX_NAMESPACE::ModelProto model;
  ORT_API_RETURN_IF_STATUS_NOT_OK(BuildModel(graph, options ? *options : OrtGraphSerializationOptions{}, model));
  const auto size = model.ByteSizeLong();
  void* buffer = allocator->Alloc(allocator, size);
  if (!buffer) return OrtApis::CreateStatus(ORT_FAIL, "Model buffer allocation failed");
  auto free_buffer = [allocator](void* p) { allocator->Free(allocator, p); };
  std::unique_ptr<void, decltype(free_buffer)> holder(buffer, free_buffer);
  bool success;
  {
    google::protobuf::io::ArrayOutputStream stream(buffer, static_cast<int>(size));
    google::protobuf::io::CodedOutputStream coded(&stream);
    coded.SetSerializationDeterministic(true);
    success = model.SerializeToCodedStream(&coded) && !coded.HadError();
  }
  if (!success) return OrtApis::CreateStatus(ORT_FAIL, "Model serialization failed");
  *bytes = holder.release();
  *length = size;
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(graph);
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(allocator);
  ORT_UNUSED_PARAMETER(bytes);
  ORT_UNUSED_PARAMETER(length);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Graph serialization is unavailable in minimal builds");
#endif
  API_IMPL_END
}
