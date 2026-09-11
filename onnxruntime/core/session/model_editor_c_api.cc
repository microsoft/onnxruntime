// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <variant>

#include "core/common/inlined_containers.h"
#include "core/common/make_string.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/ort_value.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "core/graph/model_editor_api_types.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/inference_session.h"
#include "core/session/model_editor_api.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"
#include "core/session/utils.h"

using namespace onnxruntime;

ORT_API_STATUS_IMPL(OrtModelEditorAPI::CreateValueInfo, _In_ const char* name, _In_ const OrtTypeInfo* type_info,
                    _Outptr_ OrtValueInfo** value_info) {
  API_IMPL_BEGIN
  if (name == nullptr || *name == '\0') {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "name cannot be null or empty string");
  }

  if (type_info == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "type_info cannot be null");
  }

  if (type_info->type != ONNX_TYPE_TENSOR) {
    return OrtApis::CreateStatus(ORT_FAIL, "Only tensor types are supported currently");
  }

  if (type_info->tensor_type_info == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "tensor_type_info cannot be null");
  }

  auto vi = std::make_unique<onnxruntime::ModelEditorValueInfo>();
  vi->name = name;
  vi->type_info = type_info->Clone();

  *value_info = vi.release()->ToExternal();

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelEditorAPI::CreateNode, const char* operator_name, const char* domain_name,
                    _In_ const char* node_name,
                    _In_reads_(input_names_len) const char* const* input_names, size_t input_names_len,
                    _In_reads_(output_names_len) const char* const* output_names, size_t output_names_len,
                    _In_reads_(attribs_len) _Inout_opt_ OrtOpAttr** attributes, _In_opt_ size_t attribs_len,
                    _Outptr_ OrtNode** node) {
  API_IMPL_BEGIN
  auto n = std::make_unique<onnxruntime::ModelEditorNode>();
  n->operator_name = operator_name;
  n->domain_name = domain_name == kOnnxDomainAlias ? kOnnxDomain : domain_name;
  n->node_name = node_name;

  n->input_names.reserve(input_names_len);
  for (size_t i = 0; i < input_names_len; ++i) {
    n->input_names.push_back(input_names[i]);
  }

  n->output_names.reserve(output_names_len);
  for (size_t i = 0; i < output_names_len; ++i) {
    n->output_names.push_back(output_names[i]);
  }

  if (attributes != nullptr) {
    // Check for duplicate pointers in the attributes array to prevent double-free
    onnxruntime::InlinedHashSet<const OrtOpAttr*> seen;
    seen.reserve(attribs_len);
    for (size_t i = 0; i < attribs_len; ++i) {
      if (attributes[i] == nullptr) {
        return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "attributes cannot contain null entries");
      }

      if (!seen.insert(attributes[i]).second) {
        return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                     "Duplicate OrtOpAttr pointer found in attributes array. "
                                     "Each OrtOpAttr can only appear once.");
      }
    }

    n->attributes.reserve(attribs_len);
    for (size_t i = 0; i < attribs_len; ++i) {
      n->attributes.push_back(*reinterpret_cast<const ONNX_NAMESPACE::AttributeProto*>(attributes[i]));
      // take ownership. as we took a copy that means releasing the original value
      OrtApis::ReleaseOpAttr(attributes[i]);
      attributes[i] = nullptr;
    }
  }

  *node = n.release()->ToExternal();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelEditorAPI::CreateGraph, _Outptr_ OrtGraph** graph) {
  API_IMPL_BEGIN
  auto g = std::make_unique<onnxruntime::ModelEditorGraph>();

  // do some reserves to reduce reallocation. if we had a hint about sizes upfront that would be optimal
  g->initializers.reserve(32);
  g->external_initializers.reserve(32);
  g->nodes.reserve(64);

  *graph = g.release()->ToExternal();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelEditorAPI::SetGraphInputs, _Inout_ OrtGraph* ort_graph,
                    _In_reads_(inputs_len) _Inout_ OrtValueInfo** inputs, _In_ size_t inputs_len) {
  API_IMPL_BEGIN
  if (ort_graph == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "graph cannot be null");
  }

  if (inputs == nullptr && inputs_len != 0) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "inputs cannot be null when inputs_len is non-zero");
  }

  onnxruntime::ModelEditorGraph* graph = onnxruntime::ModelEditorGraph::ToInternal(ort_graph);

  if (graph == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Invalid OrtGraph variant for use in the OrtModelEditorApi");
  }

  // Strong exception safety: validate every entry and pre-allocate the new vector before mutating any
  // observable state. If anything below throws or returns an error, the existing graph->inputs and the
  // caller's `inputs` array are left untouched, so ownership is never partially transferred.
  //
  // Duplicate-pointer guard: reject (a) the same OrtValueInfo* appearing twice in `inputs[]` and
  // (b) any pointer already owned by graph->inputs or graph->outputs. Without this the swap below
  // (which destroys the displaced old vector) would double-free pointers that survive into the new
  // committed vector.
  onnxruntime::InlinedHashSet<const OrtValueInfo*> already_owned;
  already_owned.reserve(graph->inputs.size() + graph->outputs.size());
  for (const auto& vi : graph->inputs) already_owned.insert(vi.get());
  for (const auto& vi : graph->outputs) already_owned.insert(vi.get());

  onnxruntime::InlinedHashSet<const OrtValueInfo*> seen;
  seen.reserve(inputs_len);
  onnxruntime::InlinedVector<onnxruntime::ModelEditorValueInfo*> validated;
  validated.reserve(inputs_len);
  for (size_t i = 0; i < inputs_len; ++i) {
    if (inputs[i] == nullptr) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "inputs cannot contain null entries");
    }
    if (already_owned.count(inputs[i]) != 0) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                   "This OrtValueInfo pointer has already been added to the graph. "
                                   "Each OrtValueInfo must only be added once.");
    }
    if (!seen.insert(inputs[i]).second) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                   "Duplicate OrtValueInfo pointer found in inputs array. "
                                   "Each OrtValueInfo can only appear once.");
    }

    onnxruntime::ModelEditorValueInfo* input = onnxruntime::ModelEditorValueInfo::ToInternal(inputs[i]);
    if (input == nullptr) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                   "Invalid OrtValueInfo variant for use in the OrtModelEditorApi");
    }
    validated.push_back(input);
  }

  onnxruntime::InlinedVector<std::unique_ptr<onnxruntime::ModelEditorValueInfo,
                                             onnxruntime::OrtValueInfoDeleter>>
      new_inputs;
  new_inputs.reserve(validated.size());  // last operation that may throw

  // Commit phase: only noexcept operations from here on.
  for (auto* p : validated) {
    new_inputs.emplace_back(p);  // noexcept: capacity already reserved, unique_ptr ctor is noexcept
  }
  graph->inputs.swap(new_inputs);
  for (size_t i = 0; i < inputs_len; ++i) {
    inputs[i] = nullptr;
  }

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelEditorAPI::SetGraphOutputs, _Inout_ OrtGraph* ort_graph,
                    _In_reads_(outputs_len) _Inout_ OrtValueInfo** outputs, _In_ size_t outputs_len) {
  API_IMPL_BEGIN
  if (ort_graph == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "graph cannot be null");
  }

  if (outputs == nullptr && outputs_len != 0) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "outputs cannot be null when outputs_len is non-zero");
  }

  onnxruntime::ModelEditorGraph* graph = onnxruntime::ModelEditorGraph::ToInternal(ort_graph);

  if (graph == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Invalid OrtGraph variant for use in the OrtModelEditorApi");
  }

  // Strong exception safety + duplicate-pointer guard: see SetGraphInputs above for rationale.
  onnxruntime::InlinedHashSet<const OrtValueInfo*> already_owned;
  already_owned.reserve(graph->inputs.size() + graph->outputs.size());
  for (const auto& vi : graph->inputs) already_owned.insert(vi.get());
  for (const auto& vi : graph->outputs) already_owned.insert(vi.get());

  onnxruntime::InlinedHashSet<const OrtValueInfo*> seen;
  seen.reserve(outputs_len);
  onnxruntime::InlinedVector<onnxruntime::ModelEditorValueInfo*> validated;
  validated.reserve(outputs_len);
  for (size_t i = 0; i < outputs_len; ++i) {
    if (outputs[i] == nullptr) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "outputs cannot contain null entries");
    }
    if (already_owned.count(outputs[i]) != 0) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                   "This OrtValueInfo pointer has already been added to the graph. "
                                   "Each OrtValueInfo must only be added once.");
    }
    if (!seen.insert(outputs[i]).second) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                   "Duplicate OrtValueInfo pointer found in outputs array. "
                                   "Each OrtValueInfo can only appear once.");
    }

    onnxruntime::ModelEditorValueInfo* output = onnxruntime::ModelEditorValueInfo::ToInternal(outputs[i]);
    if (output == nullptr) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                   "Invalid OrtValueInfo variant for use in the OrtModelEditorApi");
    }
    validated.push_back(output);
  }

  onnxruntime::InlinedVector<std::unique_ptr<onnxruntime::ModelEditorValueInfo,
                                             onnxruntime::OrtValueInfoDeleter>>
      new_outputs;
  new_outputs.reserve(validated.size());  // last operation that may throw

  // Commit phase: only noexcept operations from here on.
  for (auto* p : validated) {
    new_outputs.emplace_back(p);  // noexcept: capacity already reserved, unique_ptr ctor is noexcept
  }
  graph->outputs.swap(new_outputs);
  for (size_t i = 0; i < outputs_len; ++i) {
    outputs[i] = nullptr;
  }

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelEditorAPI::AddInitializerToGraph, _Inout_ OrtGraph* ort_graph, _In_ const char* name,
                    _Inout_ OrtValue* tensor, bool data_is_external) {
  API_IMPL_BEGIN
  if (ort_graph == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "graph cannot be null");
  }

  if (name == nullptr || *name == '\0') {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "name cannot be null or empty");
  }

  if (tensor == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "tensor cannot be null");
  }

  onnxruntime::ModelEditorGraph* graph = onnxruntime::ModelEditorGraph::ToInternal(ort_graph);

  if (graph == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Invalid OrtGraph variant for use in the OrtModelEditorApi");
  }

  if (!tensor->IsTensor()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Only Tensor is currently supported.");
  }

  if (!tensor->IsAllocated()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Tensor must be allocated.");
  }

  const auto& t = tensor->Get<onnxruntime::Tensor>();
  if (t.Location().device.Type() != OrtDevice::CPU) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Only CPU based tensors are currently supported.");
  }

  if (data_is_external) {
    // When data_is_external is true, LoadFromModelEditorApiModel encodes `tensor`'s buffer pointer into
    // the resulting TensorProto as an in-memory external-data reference (kTensorProtoNativeEndianMemoryAddressTag).
    // That encoding is an internal ORT mechanism, not something callers can construct themselves, so this
    // function is the only legitimate producer of it. Reject small tensors here to ensure we never emit
    // an in-memory reference for data that is small enough to be needed by ONNX shape inferencing
    // (which does not understand external data and consumes the tensor's bytes inline).
    // e.g. Reshape's `shape`, Reduce's `axes`, Slice's `starts`/`ends`/`steps`, Clip's `min`/`max`, etc.
    if (t.SizeInBytes() <= onnxruntime::utils::kSmallTensorExternalDataThreshold) {
      const std::string msg = onnxruntime::MakeString(
          "data_is_external=true requires the tensor to be larger than ",
          onnxruntime::utils::kSmallTensorExternalDataThreshold,
          " bytes (got ", t.SizeInBytes(),
          " bytes). For smaller tensors, copy the data into an ORT-allocated tensor via "
          "CreateTensorAsOrtValue and pass data_is_external=false.");
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, msg.c_str());
    }
  }

  // Reject duplicate name first (pure read, no state change needed on rejection).
  auto& target_map = data_is_external ? graph->external_initializers : graph->initializers;
  auto& other_map = data_is_external ? graph->initializers : graph->external_initializers;
  if (target_map.count(name) != 0 || other_map.count(name) != 0) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "An initializer with this name has already been added to the graph.");
  }

  // Duplicate-pointer check: set::insert returns {iterator, inserted=false} when the pointer is
  // already owned. If insert throws bad_alloc, no state has changed and the caller still owns `tensor`.
  auto [ptr_it, ptr_inserted] = graph->initializer_ptrs.insert(tensor);
  if (!ptr_inserted) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "This OrtValue pointer has already been added to the graph. "
                                 "Each OrtValue must only be added once.");
  }

  // Insert the owning entry. operator[] is the only remaining throwing call; on bad_alloc the map
  // is unchanged and the caller still owns `tensor`. Roll back the set entry and convert the
  // exception into a Status rather than letting it propagate.
  ORT_TRY {
    target_map[name].reset(tensor);  // takes ownership on success
  }
  ORT_CATCH(const std::exception& e) {
    OrtStatus* status = nullptr;
    ORT_HANDLE_EXCEPTION([&]() {
      graph->initializer_ptrs.erase(ptr_it);
      status = OrtApis::CreateStatus(ORT_FAIL, e.what());
    });
    return status;
  }

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelEditorAPI::AddNodeToGraph, _Inout_ OrtGraph* ort_graph, _Inout_ OrtNode* ort_node) {
  API_IMPL_BEGIN
  if (ort_graph == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "graph cannot be null");
  }

  if (ort_node == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "node cannot be null");
  }

  onnxruntime::ModelEditorGraph* graph = onnxruntime::ModelEditorGraph::ToInternal(ort_graph);

  if (graph == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Invalid OrtGraph variant for use in the OrtModelEditorApi");
  }

  onnxruntime::ModelEditorNode* node = onnxruntime::ModelEditorNode::ToInternal(ort_node);

  if (node == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Invalid OrtNode variant for use in the OrtModelEditorApi");
  }

  // Combined duplicate-pointer check + insert via the set's return value. If insert throws bad_alloc,
  // no state is changed and the caller still owns `ort_node`.
  auto [ptr_it, ptr_inserted] = graph->node_ptrs.insert(node);
  if (!ptr_inserted) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "This OrtNode pointer has already been added to the graph. "
                                 "Each OrtNode must only be added once.");
  }

  // Take ownership via emplace_back. If it throws bad_alloc, roll back the set entry and convert
  // the exception into a Status; vector's amortized exponential growth handles capacity.
  ORT_TRY {
    graph->nodes.emplace_back(node);  // takes ownership on success
  }
  ORT_CATCH(const std::exception& e) {
    OrtStatus* status = nullptr;
    ORT_HANDLE_EXCEPTION([&]() {
      graph->node_ptrs.erase(ptr_it);
      status = OrtApis::CreateStatus(ORT_FAIL, e.what());
    });
    return status;
  }
  // Assign id only after successful insertion — avoids mutating the node on failure.
  node->id = graph->nodes.size() - 1;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelEditorAPI::CreateModel,
                    _In_reads_(opset_entries_len) const char* const* domain_names,
                    _In_reads_(opset_entries_len) const int* opset_versions,
                    size_t opset_entries_len,
                    _Outptr_ OrtModel** model) {
  API_IMPL_BEGIN
  auto m = std::make_unique<OrtModel>();
  for (size_t i = 0; i < opset_entries_len; ++i) {
    m->domain_to_version[domain_names[i]] = opset_versions[i];
  }

  *model = m.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelEditorAPI::AddGraphToModel, _Inout_ OrtModel* model, _Inout_ OrtGraph* graph) {
  API_IMPL_BEGIN

  if (model == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "model cannot be null");
  }

  if (graph == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "graph cannot be null");
  }

  if (onnxruntime::ModelEditorGraph::ToInternal(graph) == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Invalid OrtGraph variant for use in the OrtModelEditorApi");
  }

  // Reject if the model already owns a graph. Each OrtModel may have at most one graph; without this
  // check, a second call would silently delete the previously-owned graph via unique_ptr assignment.
  if (model->graph != nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Model already has a graph. Each OrtModel can only have one graph.");
  }

  model->graph.reset(graph);  // take ownership; destruction routes through OrtApis::ReleaseGraph
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelEditorAPI::CreateSessionFromModel, _In_ const OrtEnv* env, _In_ const OrtModel* model,
                    _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out) {
  API_IMPL_BEGIN

  std::unique_ptr<onnxruntime::InferenceSession> sess;
  OrtStatus* status = nullptr;
  *out = nullptr;

  ORT_TRY {
    sess = std::make_unique<onnxruntime::InferenceSession>(
        options == nullptr ? onnxruntime::SessionOptions() : options->value,
        env->GetEnvironment());

    ORT_API_RETURN_IF_STATUS_NOT_OK(sess->Load(*model));

    ORT_API_RETURN_IF_ERROR(InitializeSession(options, *sess));

    *out = reinterpret_cast<OrtSession*>(sess.release());
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = OrtApis::CreateStatus(ORT_FAIL, e.what());
    });
  }

  return status;

  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelEditorAPI::CreateModelEditorSession,
                    _In_ const OrtEnv* env, _In_ const ORTCHAR_T* model_path, _In_ const OrtSessionOptions* options,
                    _Outptr_ OrtSession** out) {
  API_IMPL_BEGIN
  std::unique_ptr<onnxruntime::InferenceSession> session;
  OrtStatus* status = nullptr;
  *out = nullptr;

  ORT_TRY {
    ORT_API_RETURN_IF_ERROR(CreateSessionAndLoadModel(options, env, model_path, nullptr, 0, session));
    *out = reinterpret_cast<OrtSession*>(session.release());
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = OrtApis::CreateStatus(ORT_FAIL, e.what());
    });
  }

  return status;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelEditorAPI::CreateModelEditorSessionFromArray, _In_ const OrtEnv* env,
                    _In_ const void* model_data, size_t model_data_length,
                    _In_ const OrtSessionOptions* options,
                    _Outptr_ OrtSession** out) {
  API_IMPL_BEGIN
  std::unique_ptr<onnxruntime::InferenceSession> session;
  OrtStatus* status = nullptr;
  *out = nullptr;

  ORT_TRY {
    ORT_API_RETURN_IF_ERROR(CreateSessionAndLoadModel(options, env, nullptr, model_data, model_data_length, session));
    *out = reinterpret_cast<OrtSession*>(session.release());
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = OrtApis::CreateStatus(ORT_FAIL, e.what());
    });
  }

  return status;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelEditorAPI::SessionGetOpsetForDomain, _In_ const OrtSession* ort_session,
                    _In_ const char* domain, _Out_ int* opset) {
  const auto& session = *reinterpret_cast<const ::onnxruntime::InferenceSession*>(ort_session);
  const auto& domain_opset_map = session.GetModel().MainGraph().DomainToVersionMap();

  auto it = domain_opset_map.find(domain);
  if (it == domain_opset_map.cend()) {
    return OrtApis::CreateStatus(ORT_FAIL, "Domain not used by model.");
  }

  *opset = it->second;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtModelEditorAPI::ApplyModelToModelEditorSession,
                    _In_ OrtSession* session, _In_ OrtModel* model) {
  API_IMPL_BEGIN
  auto sess = reinterpret_cast<onnxruntime::InferenceSession*>(session);
  ORT_API_RETURN_IF_STATUS_NOT_OK(sess->ApplyUpdates(*model));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelEditorAPI::FinalizeModelEditorSession, _In_ OrtSession* session,
                    _In_ const OrtSessionOptions* options,
                    _Inout_ OrtPrepackedWeightsContainer* prepacked_weights_container) {
  API_IMPL_BEGIN
  auto sess = reinterpret_cast<onnxruntime::InferenceSession*>(session);
  ORT_API_RETURN_IF_ERROR(InitializeSession(options, *sess, prepacked_weights_container));
  return nullptr;
  API_IMPL_END
}

static constexpr OrtModelEditorApi ort_model_editor_api = {
    // NOTE: The C# bindings depend on the API order within this struct so all additions must be at the end,
    // and no functions can be removed (the implementation needs to change to return an error).

    &OrtModelEditorAPI::CreateTensorTypeInfo,
    &OrtModelEditorAPI::CreateSparseTensorTypeInfo,
    &OrtModelEditorAPI::CreateMapTypeInfo,
    &OrtModelEditorAPI::CreateSequenceTypeInfo,
    &OrtModelEditorAPI::CreateOptionalTypeInfo,

    &OrtModelEditorAPI::CreateValueInfo,

    &OrtModelEditorAPI::CreateNode,

    &OrtModelEditorAPI::CreateGraph,
    &OrtModelEditorAPI::SetGraphInputs,
    &OrtModelEditorAPI::SetGraphOutputs,
    &OrtModelEditorAPI::AddInitializerToGraph,
    &OrtModelEditorAPI::AddNodeToGraph,

    &OrtModelEditorAPI::CreateModel,
    &OrtModelEditorAPI::AddGraphToModel,

    &OrtModelEditorAPI::CreateSessionFromModel,

    &OrtModelEditorAPI::CreateModelEditorSession,
    &OrtModelEditorAPI::CreateModelEditorSessionFromArray,
    &OrtModelEditorAPI::SessionGetOpsetForDomain,
    &OrtModelEditorAPI::ApplyModelToModelEditorSession,
    &OrtModelEditorAPI::FinalizeModelEditorSession,
    // End of Version 22 - DO NOT MODIFY ABOVE
};

// checks that we don't violate the rule that the functions must remain in the slots they were originally assigned
static_assert(offsetof(OrtModelEditorApi, FinalizeModelEditorSession) / sizeof(void*) == 19,
              "Size of version 22 API cannot change");  // initial version in ORT 1.22

ORT_API(const OrtModelEditorApi*, OrtModelEditorAPI::GetModelEditorApi) {
  return &ort_model_editor_api;
}

#endif  // !defined(ORT_MINIMAL_BUILD)
