// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <variant>

#include "core/framework/error_code_helper.h"
#include "core/framework/ort_value.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/framework/tensor_type_and_shape.h"
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

  auto vi = std::make_unique<OrtValueInfo>();
  vi->name = name;
  vi->type_info = type_info->Clone();

  *value_info = vi.release();

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
  auto n = std::make_unique<OrtNode>();
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
    n->attributes.reserve(attribs_len);
    for (size_t i = 0; i < attribs_len; ++i) {
      n->attributes.push_back(*reinterpret_cast<const ONNX_NAMESPACE::AttributeProto*>(attributes[i]));
      // take ownership. as we took a copy that means releasing the original value
      OrtApis::ReleaseOpAttr(attributes[i]);
      attributes[i] = nullptr;
    }
  }

  *node = n.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelEditorAPI::CreateGraph, _Outptr_ OrtGraph** graph) {
  API_IMPL_BEGIN
  auto g = std::make_unique<OrtGraph>();

  // do some reserves to reduce reallocation. if we had a hint about sizes upfront that would be optimal
  g->initializers.reserve(32);
  g->external_initializers.reserve(32);
  g->nodes.reserve(64);

  *graph = g.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelEditorAPI::SetGraphInputs, _In_ OrtGraph* graph,
                    _In_reads_(inputs_len) _In_ OrtValueInfo** inputs, _In_ size_t inputs_len) {
  API_IMPL_BEGIN
  graph->inputs.clear();
  for (size_t i = 0; i < inputs_len; ++i) {
    if (inputs[i] == nullptr) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "inputs cannot contain null entries");
    }

    graph->inputs.push_back(std::unique_ptr<OrtValueInfo>(inputs[i]));  // take ownership
    inputs[i] = nullptr;
  }

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelEditorAPI::SetGraphOutputs, _In_ OrtGraph* graph,
                    _In_reads_(outputs_len) _In_ OrtValueInfo** outputs, _In_ size_t outputs_len) {
  API_IMPL_BEGIN
  graph->outputs.clear();
  for (size_t i = 0; i < outputs_len; ++i) {
    if (outputs[i] == nullptr) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "outputs cannot contain null entries");
    }

    graph->outputs.push_back(std::unique_ptr<OrtValueInfo>(outputs[i]));  // take ownership
    outputs[i] = nullptr;
  }

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelEditorAPI::AddInitializerToGraph, _In_ OrtGraph* graph, _In_ const char* name,
                    _Inout_ OrtValue* tensor, bool data_is_external) {
  API_IMPL_BEGIN
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
    // enforce that an external initializer is not used if the data size is < 128 bytes.
    // the reason for this is to avoid potential shape inferencing errors if this initializer is providing an
    // input involved in that. the ONNX shape inferencing does not support external data for those values.
    // e.g. Reshape's `shape` input, Reduce's `axes', Slice's `starts`, `ends`, `steps`, Clip's `min`, `max`, etc.
    if (t.SizeInBytes() < 128) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                   "External initializer should only be used for data >= 128 bytes. "
                                   "Please use CreateTensorAsOrtValue instead.");
    }

    graph->external_initializers[name] = std::unique_ptr<OrtValue>(tensor);  // take ownership
  } else {
    graph->initializers[name] = std::unique_ptr<OrtValue>(tensor);  // take ownership
  }

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtModelEditorAPI::AddNodeToGraph, _In_ OrtGraph* graph, _Inout_ OrtNode* node) {
  API_IMPL_BEGIN
  graph->nodes.push_back(std::unique_ptr<OrtNode>(node));  // take ownership
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

ORT_API_STATUS_IMPL(OrtModelEditorAPI::AddGraphToModel, _In_ OrtModel* model, _Inout_ OrtGraph* graph) {
  API_IMPL_BEGIN

  if (graph == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "graph cannot be null");
  }

  model->graph = std::unique_ptr<OrtGraph>(graph);  // take ownership
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
};

// checks that we don't violate the rule that the functions must remain in the slots they were originally assigned
static_assert(offsetof(OrtModelEditorApi, FinalizeModelEditorSession) / sizeof(void*) == 19,
              "Size of version 21 API cannot change");  // initial version in ORT 1.21

ORT_API(const OrtModelEditorApi*, OrtModelEditorAPI::GetModelEditorApi) {
  return &ort_model_editor_api;
}

#endif  // !defined(ORT_MINIMAL_BUILD)
