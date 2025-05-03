// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/utils.h"

#include "core/framework/error_code_helper.h"
#include "core/framework/execution_provider.h"
#include "core/framework/provider_options.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/environment.h"
#include "core/session/inference_session.h"
#include "core/session/inference_session_utils.h"
#include "core/session/ep_factory_internal.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/ep_library_plugin.h"
#include "core/session/ep_library_provider_bridge.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/provider_policy_context.h"

using namespace onnxruntime;
#if !defined(ORT_MINIMAL_BUILD)
namespace {
// temporary implementation for testing. EP to 'select' is specified in config option
Status TestAutoSelectEPsImpl(const Environment& env, InferenceSession& sess, const std::string& ep_to_select) {
  const auto& execution_devices = env.GetOrtEpDevices();

  // Create OrtSessionOptions for the CreateEp call.
  // Once the InferenceSession is created, its SessionOptions is the source of truth and contains all the values from
  // the user provided OrtSessionOptions. We do a copy for simplicity. The OrtSessionOptions instance goes away
  // once we exit this function.
  auto& session_options = sess.GetMutableSessionOptions();
  OrtSessionOptions ort_so;
  ort_so.value = session_options;
  const auto& session_logger = sess.GetLogger();
  const OrtLogger& api_session_logger = *session_logger->ToExternal();

  for (const auto* ep_device : execution_devices) {
    if (ep_device->ep_name != ep_to_select) {
      continue;
    }

    // get internal factory if available.
    EpFactoryInternal* internal_factory = env.GetEpFactoryInternal(ep_device->ep_factory);

    // in the real implementation multiple devices can be assigned to an EP
    // in our current test-able setup it's 1:1
    std::vector<const OrtHardwareDevice*> devices{ep_device->device};
    std::vector<const OrtKeyValuePairs*> ep_metadata{&ep_device->ep_metadata};

    // add ep_options to SessionOptions with prefix.
    // preserve any user provided values.
    const std::string ep_options_prefix = OrtSessionOptions::GetProviderOptionPrefix(ep_device->ep_name.c_str());
    for (const auto& [key, value] : ep_device->ep_options.entries) {
      auto prefixed_key = ep_options_prefix + key;
      if (session_options.config_options.configurations.count(key) == 0) {
        // add the default value with prefix
        ORT_RETURN_IF_ERROR(session_options.config_options.AddConfigEntry(prefixed_key.c_str(), value.c_str()));
      }
    }

    std::unique_ptr<IExecutionProvider> ep;

    if (internal_factory) {
      // this is a factory we created and registered. internal or provider bridge EP.
      OrtStatus* status = internal_factory->CreateIExecutionProvider(
          devices.data(), ep_metadata.data(), devices.size(), &ort_so, &api_session_logger, &ep);

      if (status != nullptr) {
        return ToStatus(status);
      }
    } else {
      // in the real setup we need an IExecutionProvider wrapper implementation that uses the OrtEp internally,
      // and we would add that IExecutionProvider to the InferenceSession.
      ORT_NOT_IMPLEMENTED("IExecutionProvider that wraps OrtEp has not been implemented.");

      /*
      OrtEp* api_ep = nullptr;
      auto status = ep_device->ep_factory->CreateEp(
          ep_device->ep_factory, devices.data(), ep_metadata.data(), devices.size(),
          &ort_so, &api_session_logger, &api_ep);

      if (status != nullptr) {
        return ToStatus(status);
      }
      */
    }

    ORT_RETURN_IF_ERROR(sess.RegisterExecutionProvider(std::move(ep)));

    // once we have the EP and one device that's enough for test purposes.
    break;
  }

  return Status::OK();
}
}  // namespace
#endif  // !defined(ORT_MINIMAL_BUILD)

common::Status CopyStringToOutputArg(std::string_view str, const char* err_msg, char* out, size_t* size) {
  const size_t str_len = str.size();
  const size_t req_size = str_len + 1;

  if (out == nullptr) {  // User is querying the total output buffer size
    *size = req_size;
    return onnxruntime::common::Status::OK();
  }

  if (*size >= req_size) {  // User provided a buffer of sufficient size
    std::memcpy(out, str.data(), str_len);
    out[str_len] = '\0';
    *size = req_size;
    return onnxruntime::common::Status::OK();
  }

  // User has provided a buffer that is not large enough
  *size = req_size;
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, err_msg);
}

// provider either model_path, or modal_data + model_data_length.
OrtStatus* CreateSessionAndLoadModel(_In_ const OrtSessionOptions* options,
                                     _In_ const OrtEnv* env,
                                     _In_opt_z_ const ORTCHAR_T* model_path,
                                     _In_opt_ const void* model_data,
                                     size_t model_data_length,
                                     std::unique_ptr<onnxruntime::InferenceSession>& sess) {
  // quick check here to decide load path. InferenceSession will provide error message for invalid values.
  // TODO: Could move to a helper
  const Env& os_env = Env::Default();  // OS environment (!= ORT environment)
  bool load_config_from_model =
      os_env.GetEnvironmentVar(inference_session_utils::kOrtLoadConfigFromModelEnvVar) == "1";

  // If ep.context_enable is set, then ep.context_file_path is expected, otherwise ORT don't know where to generate the _ctx.onnx file
  if (options && model_path == nullptr) {
    EpContextModelGenerationOptions ep_ctx_gen_options = options->value.GetEpContextGenerationOptions();

    // This is checked by the OrtCompileApi's CompileModel() function, but we check again here in case
    // the user used the older SessionOptions' configuration entries to generate a compiled model.
    if (ep_ctx_gen_options.enable &&
        ep_ctx_gen_options.output_model_file_path.empty() &&
        ep_ctx_gen_options.output_model_buffer_ptr == nullptr) {
      return OrtApis::CreateStatus(ORT_FAIL,
                                   "Inference session was configured with EPContext model generation enabled but "
                                   "without a valid location (e.g., file or buffer) for the output model. "
                                   "Please specify a valid ep.context_file_path via SessionOption configs "
                                   "or use the OrtCompileApi to compile a model to a file or buffer.");
    }
  }

  if (load_config_from_model) {
#if !defined(ORT_MINIMAL_BUILD)
    if (model_path != nullptr) {
      sess = std::make_unique<onnxruntime::InferenceSession>(
          options == nullptr ? onnxruntime::SessionOptions() : options->value,
          env->GetEnvironment(),
          model_path);
    } else {
      sess = std::make_unique<onnxruntime::InferenceSession>(
          options == nullptr ? onnxruntime::SessionOptions() : options->value,
          env->GetEnvironment(),
          model_data, static_cast<int>(model_data_length));
    }
#else
    return OrtApis::CreateStatus(ORT_FAIL, "Loading config from ONNX models is not supported in this build.");
#endif
  } else {
    sess = std::make_unique<onnxruntime::InferenceSession>(
        options == nullptr ? onnxruntime::SessionOptions() : options->value,
        env->GetEnvironment());
  }

#if !defined(ORT_MINIMAL_BUILD)
  // TEMPORARY for testing. Manually specify the EP to select.
  auto auto_select_ep_name = sess->GetSessionOptions().config_options.GetConfigEntry("test.ep_to_select");
  if (auto_select_ep_name) {
    ORT_API_RETURN_IF_STATUS_NOT_OK(TestAutoSelectEPsImpl(env->GetEnvironment(), *sess, *auto_select_ep_name));
  }

  // if there are no providers registered, and there's an ep selection policy set, do auto ep selection
  if (options != nullptr && options->provider_factories.empty() && options->value.ep_selection_policy.enable) {
    ProviderPolicyContext context;
    ORT_API_RETURN_IF_STATUS_NOT_OK(context.SelectEpsForSession(env->GetEnvironment(), *options, *sess));
  }
#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  // Add custom domains
  if (options && !options->custom_op_domains_.empty()) {
    ORT_API_RETURN_IF_STATUS_NOT_OK(sess->AddCustomOpDomains(options->custom_op_domains_));
  }
#endif

  // Finish load
  if (load_config_from_model) {
#if !defined(ORT_MINIMAL_BUILD)
    ORT_API_RETURN_IF_STATUS_NOT_OK(sess->Load());
#endif
  } else {
    if (model_path != nullptr) {
      ORT_API_RETURN_IF_STATUS_NOT_OK(sess->Load(model_path));
    } else {
      ORT_API_RETURN_IF_STATUS_NOT_OK(sess->Load(model_data, static_cast<int>(model_data_length)));
    }
  }

  return nullptr;
}

OrtStatus* InitializeSession(_In_ const OrtSessionOptions* options,
                             _In_ onnxruntime::InferenceSession& sess,
                             _Inout_opt_ OrtPrepackedWeightsContainer* prepacked_weights_container) {
  const logging::Logger* session_logger = sess.GetLogger();
  ORT_ENFORCE(session_logger != nullptr,
              "Session logger is invalid, but should have been initialized during session construction.");

  // we need to disable mem pattern if DML is one of the providers since DML doesn't have the concept of
  // byte addressable memory
  std::vector<std::unique_ptr<IExecutionProvider>> provider_list;
  if (options) {
    for (auto& factory : options->provider_factories) {
      auto provider = factory->CreateProvider(*options, *session_logger->ToExternal());
      provider_list.push_back(std::move(provider));
    }
  }

  // register the providers
  for (auto& provider : provider_list) {
    if (provider) {
      ORT_API_RETURN_IF_STATUS_NOT_OK(sess.RegisterExecutionProvider(std::move(provider)));
    }
  }

  if (prepacked_weights_container != nullptr) {
    ORT_API_RETURN_IF_STATUS_NOT_OK(sess.AddPrePackedWeightsContainer(
        reinterpret_cast<PrepackedWeightsContainer*>(prepacked_weights_container)));
  }

  ORT_API_RETURN_IF_STATUS_NOT_OK(sess.Initialize());

  return nullptr;
}

namespace onnxruntime {
#if !defined(ORT_MINIMAL_BUILD)
Status LoadPluginOrProviderBridge(const std::string& registration_name,
                                  const ORTCHAR_T* library_path,
                                  std::unique_ptr<EpLibrary>& ep_library,
                                  std::vector<EpFactoryInternal*>& internal_factories) {
  // If the `library_path` is absolute, use it as-is. Otherwise follow the precedent of ProviderLibrary::Load and make
  // it absolute by combining it with the OnnxRuntime location.
  std::filesystem::path resolved_library_path{library_path};

  if (!resolved_library_path.is_absolute()) {
    resolved_library_path = Env::Default().GetRuntimePath() / std::move(resolved_library_path);
  }

  // if it's a provider bridge library we need to create ProviderLibrary first to ensure the dependencies are loaded
  // like the onnxruntime_provider_shared library.
  auto provider_library = std::make_unique<ProviderLibrary>(resolved_library_path.native().c_str(),
                                                            true,
                                                            ProviderLibraryPathType::Absolute);
  bool is_provider_bridge = provider_library->Load() == Status::OK();  // library has GetProvider
  LOGS_DEFAULT(INFO) << "Loading EP library: " << library_path
                     << (is_provider_bridge ? " as a provider bridge" : " as a plugin");

  // create EpLibraryPlugin to ensure CreateEpFactories and ReleaseEpFactory are available
  auto ep_library_plugin = std::make_unique<EpLibraryPlugin>(registration_name, std::move(resolved_library_path));
  ORT_RETURN_IF_ERROR(ep_library_plugin->Load());

  if (is_provider_bridge) {
    // wrap the EpLibraryPlugin with EpLibraryProviderBridge to add to directly create an IExecutionProvider
    auto ep_library_provider_bridge = std::make_unique<EpLibraryProviderBridge>(std::move(provider_library),
                                                                                std::move(ep_library_plugin));
    ORT_RETURN_IF_ERROR(ep_library_provider_bridge->Load());
    internal_factories = ep_library_provider_bridge->GetInternalFactories();
    ep_library = std::move(ep_library_provider_bridge);
  } else {
    ep_library = std::move(ep_library_plugin);
  }

  return Status::OK();
}
#endif
}  // namespace onnxruntime
