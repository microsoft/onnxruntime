// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/utils.h"

#include <memory>
#include <utility>

#include "core/framework/error_code_helper.h"
#include "core/framework/execution_provider.h"
#include "core/framework/provider_options.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/environment.h"
#include "core/session/inference_session.h"
#include "core/session/inference_session_utils.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"
#include "core/session/onnxruntime_ep_device_ep_metadata_keys.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "core/session/plugin_ep/ep_factory_internal.h"
#include "core/session/plugin_ep/ep_plugin_provider_interfaces.h"
#include "core/session/plugin_ep/ep_library_plugin.h"
#include "core/session/plugin_ep/ep_library_provider_bridge.h"
#include "core/session/model_compilation_options.h"
#include "core/session/provider_policy_context.h"
#endif  // !defined(ORT_MINIMAL_BUILD)

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
    for (const auto& [key, value] : ep_device->ep_options.Entries()) {
      auto prefixed_key = ep_options_prefix + key;
      if (session_options.config_options.configurations.count(key) == 0) {
        // add the default value with prefix
        ORT_RETURN_IF_ERROR(session_options.config_options.AddConfigEntry(prefixed_key.c_str(), value.c_str()));
      }
    }

    std::unique_ptr<IExecutionProvider> ep;

    if (internal_factory) {
      // this is a factory we created and registered. internal or provider bridge EP.
      ORT_RETURN_IF_ERROR(ToStatusAndRelease(internal_factory->CreateIExecutionProvider(
          devices.data(), ep_metadata.data(), devices.size(), &ort_so, &api_session_logger, &ep)));
    } else {
      // in the real setup we need an IExecutionProvider wrapper implementation that uses the OrtEp internally,
      // and we would add that IExecutionProvider to the InferenceSession.
      ORT_NOT_IMPLEMENTED("IExecutionProvider that wraps OrtEp has not been implemented.");

      /*
      OrtEp* api_ep = nullptr;
      ORT_RETURN_IF_ERROR(ToStatusAndRelease(ep_device->ep_factory->CreateEp(
          ep_device->ep_factory, devices.data(), ep_metadata.data(), devices.size(),
          &ort_so, &api_session_logger, &api_ep)));
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

// Internal function that creates an InferenceSession and loads the model.
// Caller should provide either model_path, or modal_data + model_data_length.
static OrtStatus* CreateSessionAndLoadModelImpl(_In_ const OrtSessionOptions* options,
                                                const onnxruntime::Environment& env,
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
          env,
          model_path);
    } else {
      sess = std::make_unique<onnxruntime::InferenceSession>(
          options == nullptr ? onnxruntime::SessionOptions() : options->value,
          env,
          model_data, static_cast<int>(model_data_length));
    }
#else
    return OrtApis::CreateStatus(ORT_FAIL, "Loading config from ONNX models is not supported in this build.");
#endif
  } else {
    sess = std::make_unique<onnxruntime::InferenceSession>(
        options == nullptr ? onnxruntime::SessionOptions() : options->value,
        env);
  }

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

// Creates an InferenceSession and loads the model.
// Caller should provide either model_path, or modal_data + model_data_length.
OrtStatus* CreateSessionAndLoadModel(_In_ const OrtSessionOptions* options,
                                     _In_ const OrtEnv* env,
                                     _In_opt_z_ const ORTCHAR_T* model_path,
                                     _In_opt_ const void* model_data,
                                     size_t model_data_length,
                                     std::unique_ptr<onnxruntime::InferenceSession>& sess) {
  return CreateSessionAndLoadModelImpl(options, env->GetEnvironment(), model_path, model_data, model_data_length, sess);
}

#if !defined(ORT_MINIMAL_BUILD)
static const char* GetCompatibilityStatusString(OrtCompiledModelCompatibility status) {
  switch (status) {
    case OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL:
      return "SUPPORTED_OPTIMAL";
    case OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION:
      return "SUPPORTED_PREFER_RECOMPILATION";
    case OrtCompiledModelCompatibility_EP_UNSUPPORTED:
      return "UNSUPPORTED";
    case OrtCompiledModelCompatibility_EP_NOT_APPLICABLE:
      return "NOT_APPLICABLE";
    default:
      return "UNKNOWN";
  }
}

static Status ValidateCompiledModelCompatibility(InferenceSession& sess) {
  // Get model metadata
  auto [status, model_metadata] = sess.GetModelMetadata();
  if (!status.IsOK() || !model_metadata) {
    // No metadata available, skip validation
    return Status::OK();
  }

  // Check if user wants to fail on suboptimal models
  bool fail_on_suboptimal = sess.GetSessionOptions().config_options.GetConfigEntry(
                                kOrtSessionOptionsFailOnSuboptimalCompiledModel) == "1";

  const auto& custom_metadata = model_metadata->custom_metadata_map;
  const auto& registered_provider_types = sess.GetRegisteredProviderTypes();

  // Access the execution providers through the session state (available after Initialize)
  const auto& execution_providers = sess.GetSessionState().GetExecutionProviders();

  for (const auto& ep_type : registered_provider_types) {
    // Construct the full metadata key using the prefix + EP type
    const std::string metadata_key = std::string(kOrtModelMetadata_EpCompatibilityInfoPrefix) + ep_type;

    auto metadata_it = custom_metadata.find(metadata_key);
    if (metadata_it != custom_metadata.end()) {
      const std::string& compatibility_info = metadata_it->second;

      // Get the actual EP instance to call validation
      const IExecutionProvider* ep = execution_providers.Get(ep_type);

      if (ep != nullptr) {
        // Call the EP's validation method (virtual method with default implementation)
        OrtCompiledModelCompatibility compatibility_status;
        Status validation_result = ep->ValidateCompiledModelCompatibilityInfo(
            compatibility_info, compatibility_status);

        if (validation_result.IsOK()) {
          // Log the compatibility status
          const char* status_str = GetCompatibilityStatusString(compatibility_status);
          LOGS(*sess.GetLogger(), INFO)
              << "EP " << ep_type << " compiled model compatibility: " << status_str;

          // Enforce compatibility based on status
          switch (compatibility_status) {
            case OrtCompiledModelCompatibility_EP_NOT_APPLICABLE:
            case OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL:
              // Continue execution
              break;

            case OrtCompiledModelCompatibility_EP_UNSUPPORTED:
              // Always fail for unsupported models
              return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                     "Compiled model is not supported by execution provider: " + ep_type);

            case OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION:
              // Behavior depends on user setting
              if (fail_on_suboptimal) {
                return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                       "Compiled model is suboptimal for execution provider: " + ep_type +
                                           ". Recompilation recommended for better performance.");
              }
              // Otherwise continue with warning
              LOGS(*sess.GetLogger(), WARNING)
                  << "EP " << ep_type << " reports compiled model is supported but suboptimal. "
                  << "Consider recompiling for better performance.";
              break;

            default:
              // Handle any unknown status values
              LOGS(*sess.GetLogger(), WARNING)
                  << "EP " << ep_type << " returned unknown compatibility status: " << compatibility_status;
              break;
          }
        } else {
          // Validation failed - this should cause session initialization to fail
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "Failed to validate compiled model compatibility for EP " + ep_type +
                                     ": " + validation_result.ErrorMessage());
        }
      }
    } else {
      // No compatibility info found for this EP - normal for non-compiled models
      LOGS(*sess.GetLogger(), VERBOSE)
          << "No compiled model compatibility info found for EP " << ep_type;
    }
  }

  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD)

OrtStatus* InitializeSession(_In_ const OrtSessionOptions* options,
                             _In_ onnxruntime::InferenceSession& sess,
                             _Inout_opt_ OrtPrepackedWeightsContainer* prepacked_weights_container) {
  const logging::Logger* session_logger = sess.GetLogger();
  ORT_ENFORCE(session_logger != nullptr,
              "Session logger is invalid, but should have been initialized during session construction.");

  const bool has_provider_factories = options != nullptr && !options->provider_factories.empty();

  if (has_provider_factories) {
    std::vector<std::unique_ptr<IExecutionProvider>> provider_list;
    for (auto& factory : options->provider_factories) {
      auto provider = factory->CreateProvider(*options, *session_logger->ToExternal());
      provider_list.push_back(std::move(provider));
    }

    // register the providers
    for (auto& provider : provider_list) {
      if (provider) {
        ORT_API_RETURN_IF_STATUS_NOT_OK(sess.RegisterExecutionProvider(std::move(provider)));
      }
    }
  }
#if !defined(ORT_MINIMAL_BUILD)
  else {
    // TEMPORARY for testing. Manually specify the EP to select.
    auto auto_select_ep_name = sess.GetSessionOptions().config_options.GetConfigEntry("test.ep_to_select");
    if (auto_select_ep_name) {
      ORT_API_RETURN_IF_STATUS_NOT_OK(TestAutoSelectEPsImpl(sess.GetEnvironment(), sess, *auto_select_ep_name));
    }

    // if there are no providers registered, and there's an ep selection policy set, do auto ep selection.
    // note: the model has already been loaded so model metadata should be available to the policy delegate callback.
    if (options != nullptr && options->value.ep_selection_policy.enable) {
      ProviderPolicyContext context;
      ORT_API_RETURN_IF_STATUS_NOT_OK(context.SelectEpsForSession(sess.GetEnvironment(), *options, sess));
    }
  }
#endif  // !defined(ORT_MINIMAL_BUILD)

  if (prepacked_weights_container != nullptr) {
    ORT_API_RETURN_IF_STATUS_NOT_OK(sess.AddPrePackedWeightsContainer(
        reinterpret_cast<PrepackedWeightsContainer*>(prepacked_weights_container)));
  }

  ORT_API_RETURN_IF_STATUS_NOT_OK(sess.Initialize());

#if !defined(ORT_MINIMAL_BUILD)
  // Validate compiled model compatibility for all registered execution providers
  // This must be done after Initialize() so the session state is available
  ORT_API_RETURN_IF_STATUS_NOT_OK(ValidateCompiledModelCompatibility(sess));
#endif  // !defined(ORT_MINIMAL_BUILD)

  return nullptr;
}

namespace onnxruntime {
#if !defined(ORT_MINIMAL_BUILD)
Status CompileModel(const Environment& env, const ModelCompilationOptions& model_compile_options) {
  ORT_RETURN_IF_ERROR(model_compile_options.Check());

  std::unique_ptr<onnxruntime::InferenceSession> session;
  const OrtSessionOptions* session_options = &model_compile_options.GetSessionOptions();

  if (model_compile_options.InputModelComesFromFile()) {
    PathString input_model_path = ToPathString(model_compile_options.GetInputModelPath());
    ORT_RETURN_IF_ERROR(ToStatusAndRelease(CreateSessionAndLoadModelImpl(session_options, env,
                                                                         input_model_path.c_str(),
                                                                         nullptr, 0, session)));
  } else {
    ORT_RETURN_IF_ERROR(
        ToStatusAndRelease(CreateSessionAndLoadModelImpl(session_options, env, nullptr,
                                                         model_compile_options.GetInputModelData(),
                                                         model_compile_options.GetInputModelDataSize(),
                                                         session)));
  }

  ORT_RETURN_IF_ERROR(ToStatusAndRelease(InitializeSession(session_options, *session)));
  return Status::OK();
}

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

Status CreateIExecutionProviderFactoryForEpDevices(const Environment& env,
                                                   SessionOptions& session_options,
                                                   gsl::span<const OrtEpDevice* const> ep_devices,
                                                   gsl::span<const char* const> ep_option_keys,
                                                   gsl::span<const char* const> ep_option_vals,
                                                   /*output*/ std::unique_ptr<IExecutionProviderFactory>& out) {
  if (ep_devices.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Must provide one or more OrtEpDevice instances.");
  }

  const size_t num_ep_options = ep_option_keys.size();
  if (ep_option_vals.size() != num_ep_options) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Must provide the same number of keys and values for EP options.");
  }

  const auto& ep_name = ep_devices[0]->ep_name;
  OrtEpFactory* ep_factory = ep_devices[0]->ep_factory;
  bool all_match = std::all_of(ep_devices.begin() + 1, ep_devices.end(),
                               [&ep_name, &ep_factory](const OrtEpDevice* ep_device) {
                                 return (ep_device->ep_name == ep_name) && (ep_device->ep_factory == ep_factory);
                               });
  if (!all_match) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "All OrtEpDevice values in ep_devices must have the same execution provider.");
  }

  for (const OrtEpDevice* ep_device : ep_devices) {
    // add the options to the session options with the EP prefix.
    // first add the default values with prefix followed by user specified values so those win
    const std::string prefix = OrtSessionOptions::GetProviderOptionPrefix(ep_device->ep_name.c_str());
    auto& config_options = session_options.config_options;
    for (const auto& [key, value] : ep_device->ep_options.Entries()) {
      ORT_RETURN_IF_ERROR(config_options.AddConfigEntry((prefix + key).c_str(), value.c_str()));
    }

    for (size_t j = 0; j < num_ep_options; ++j) {
      if (ep_option_keys[j] == nullptr) {
        continue;
      }

      ORT_RETURN_IF_ERROR(config_options.AddConfigEntry((prefix + ep_option_keys[j]).c_str(), ep_option_vals[j]));
    }
  }

  EpFactoryInternal* internal_factory = env.GetEpFactoryInternal(ep_factory);

  if (internal_factory) {
    out = std::make_unique<InternalExecutionProviderFactory>(*internal_factory, ep_devices);
  } else {
    out = std::make_unique<PluginExecutionProviderFactory>(*ep_factory, ep_devices);
  }

  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD)
}  // namespace onnxruntime
