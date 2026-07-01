// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep_factory.h"

#include <cassert>
#include <optional>
#include <string>
#include <string_view>

#include "compatibility_combine.h"
#include "ep.h"
#include "ep_allocator.h"
#include "ep_arena.h"
#include "ep_data_transfer.h"
#include "ep_stream_support.h"

#include "core/session/onnxruntime_ep_device_ep_metadata_keys.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

ExampleEpFactory::ExampleEpFactory(const char* ep_name, ApiPtrs apis, const OrtLogger& default_logger)
    : OrtEpFactory{},
      ApiPtrs(apis),
      default_logger_{default_logger},
      ep_name_{ep_name},
      default_memory_info_{nullptr},
      readonly_memory_info_{nullptr},
      host_accessible_memory_info_{nullptr} {
  ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.
  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;
  GetVendorId = GetVendorIdImpl;
  GetVersion = GetVersionImpl;

  GetSupportedDevices = GetSupportedDevicesImpl;

  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;

  CreateAllocator = CreateAllocatorImpl;
  ReleaseAllocator = ReleaseAllocatorImpl;

  CreateDataTransfer = CreateDataTransferImpl;

  IsStreamAware = IsStreamAwareImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
  GetHardwareDeviceIncompatibilityDetails = GetHardwareDeviceIncompatibilityDetailsImpl;

  CreateExternalResourceImporterForDevice = CreateExternalResourceImporterForDeviceImpl;

  GetNumCustomOpDomains = GetNumCustomOpDomainsImpl;
  GetCustomOpDomains = GetCustomOpDomainsImpl;
  ValidateCompiledModelCompatibilityInfo = ValidateCompiledModelCompatibilityInfoImpl;

  // setup the OrtMemoryInfo instances required by the EP.
  // We pretend the device the EP is running on is GPU.
  default_memory_info_ = Ort::MemoryInfo{"ExampleEP GPU",
                                         OrtMemoryInfoDeviceType_GPU,
                                         /*vendor*/ 0xBE57, /* device_id */ 0,
                                         OrtDeviceMemoryType_DEFAULT,
                                         /*alignment*/ 0,
                                         // it is invalid to use OrtArenaAllocator as that is reserved for the internal
                                         // ORT Arena implementation
                                         OrtAllocatorType::OrtDeviceAllocator};

  // create data transfer for the device
  const OrtMemoryDevice* device = ep_api.MemoryInfo_GetMemoryDevice(default_memory_info_);
  data_transfer_impl_ = std::make_unique<ExampleDataTransfer>(apis, device);

  // create read-only allocator for use with initializers. same info as DEFAULT memory apart from the allocator type.
  readonly_memory_info_ = Ort::MemoryInfo{"ExampleEP GPU readonly",
                                          OrtMemoryInfoDeviceType_GPU,
                                          /*vendor*/ 0xBE57, /* device_id */ 0,
                                          OrtDeviceMemoryType_DEFAULT,
                                          /*alignment*/ 0,
                                          OrtAllocatorType::OrtReadOnlyAllocator};

  // HOST_ACCESSIBLE memory example. use the non-CPU device type so it's clear which device the memory is also
  // accessible from. we infer from the type of HOST_ACCESSIBLE that it's CPU accessible.
  host_accessible_memory_info_ = Ort::MemoryInfo{"ExampleEP GPU pinned",
                                                 OrtMemoryInfoDeviceType_GPU,
                                                 /*vendor*/ 0xBE57, /* device_id */ 0,
                                                 OrtDeviceMemoryType_HOST_ACCESSIBLE,
                                                 /*alignment*/ 0,
                                                 OrtAllocatorType::OrtDeviceAllocator};
  // Custom Op Domains
  custom_op_domains_[0] = Ort::CustomOpDomain{"test"};
  custom_op_domains_[1] = Ort::CustomOpDomain{"test2"};

  std::vector<std::unique_ptr<ExampleEpCustomOp>> created_custom_op_list;
  created_custom_op_list.push_back(std::make_unique<ExampleEpCustomOp>(ep_name_.c_str(), this));
  created_custom_op_list.back().get()->SetName("Custom_Mul");
  custom_op_domains_[0].Add(created_custom_op_list.back().get());

  std::vector<std::unique_ptr<ExampleEpCustomOp>> created_custom_op_list_2;
  created_custom_op_list_2.push_back(std::make_unique<ExampleEpCustomOp>(ep_name_.c_str(), this));
  created_custom_op_list_2.back().get()->SetName("Custom_Mul2");
  custom_op_domains_[1].Add(created_custom_op_list_2.back().get());

  created_custom_op_lists_[0] = std::move(created_custom_op_list);
  created_custom_op_lists_[1] = std::move(created_custom_op_list_2);
}

/*static*/
const char* ORT_API_CALL ExampleEpFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const ExampleEpFactory*>(this_ptr);
  return factory->ep_name_.c_str();
}

/*static*/
const char* ORT_API_CALL ExampleEpFactory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const ExampleEpFactory*>(this_ptr);
  return factory->vendor_.c_str();
}

/*static*/
uint32_t ORT_API_CALL ExampleEpFactory::GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const ExampleEpFactory*>(this_ptr);
  return factory->vendor_id_;
}

/*static*/
const char* ORT_API_CALL ExampleEpFactory::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const ExampleEpFactory*>(this_ptr);
  return factory->ep_version_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL ExampleEpFactory::GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                                  const OrtHardwareDevice* const* devices,
                                                                  size_t num_devices,
                                                                  OrtEpDevice** ep_devices,
                                                                  size_t max_ep_devices,
                                                                  size_t* p_num_ep_devices) noexcept {
  size_t& num_ep_devices = *p_num_ep_devices;
  auto* factory = static_cast<ExampleEpFactory*>(this_ptr);

  for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
    // C API
    const OrtHardwareDevice& device = *devices[i];
    if (factory->ort_api.HardwareDevice_Type(&device) == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
      // these can be returned as nullptr if you have nothing to add.
      OrtKeyValuePairs* ep_metadata = nullptr;
      OrtKeyValuePairs* ep_options = nullptr;
      factory->ort_api.CreateKeyValuePairs(&ep_metadata);
      factory->ort_api.CreateKeyValuePairs(&ep_options);

      // random example using made up values
      factory->ort_api.AddKeyValuePair(ep_metadata, "supported_devices", "CrackGriffin 7+");
      // Example os_driver_version. A real EP would read the OS driver version from the device.
      // The format is a 4-part dot-separated version matching the DXCore DriverVersion property.
      factory->ort_api.AddKeyValuePair(ep_metadata, kOrtEpDevice_EpMetadataKey_OSDriverVersion, "31.0.101.1000");
      factory->ort_api.AddKeyValuePair(ep_options, "run_really_fast", "true");

      // OrtEpDevice copies ep_metadata and ep_options.
      OrtEpDevice* ep_device = nullptr;
      auto* status = factory->ort_api.GetEpApi()->CreateEpDevice(factory, &device, ep_metadata, ep_options,
                                                                 &ep_device);

      factory->ort_api.ReleaseKeyValuePairs(ep_metadata);
      factory->ort_api.ReleaseKeyValuePairs(ep_options);

      if (status != nullptr) {
        return status;
      }

      // register the allocator info required by the EP.
      // OrtReadOnlyAllocator + OrtDeviceMemoryType_DEFAULT allocator for use with initializers is optional.
      // OrtDeviceMemoryType_HOST_ACCESSIBLE is also optional and exposes CPU-accessible memory on the EP device.
      RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, factory->default_memory_info_));
      RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, factory->readonly_memory_info_));
      RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, factory->host_accessible_memory_info_));

      ep_devices[num_ep_devices++] = ep_device;
    }

    // C++ API equivalent. Throws on error.
    //{
    //  Ort::ConstHardwareDevice device(devices[i]);
    //  if (device.Type() == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
    //    Ort::KeyValuePairs ep_metadata;
    //    Ort::KeyValuePairs ep_options;
    //    ep_metadata.Add("supported_devices", "CrackGriffin 7+");
    //    ep_metadata.Add(kOrtEpDevice_EpMetadataKey_OSDriverVersion, "31.0.101.1000");
    //    ep_options.Add("run_really_fast", "true");
    //    Ort::EpDevice ep_device{*this_ptr, device, ep_metadata.GetConst(), ep_options.GetConst()};
    //    ep_devices[num_ep_devices++] = ep_device.release();
    //  }
    //}
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL ExampleEpFactory::CreateEpImpl(OrtEpFactory* this_ptr,
                                                       const OrtHardwareDevice* const* /*devices*/,
                                                       const OrtKeyValuePairs* const* /*ep_metadata*/,
                                                       size_t num_devices,
                                                       const OrtSessionOptions* session_options,
                                                       const OrtLogger* logger,
                                                       OrtEp** ep) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto* factory = static_cast<ExampleEpFactory*>(this_ptr);
  *ep = nullptr;

  if (num_devices != 1) {
    // we only registered for CPU and only expected to be selected for one CPU
    // if you register for multiple devices (e.g. CPU, GPU and maybe NPU) you will get an entry for each device
    // the EP has been selected for.
    return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                         "Example EP only supports selection for one device.");
  }

  // Create the execution provider
  RETURN_IF_ERROR(factory->ort_api.Logger_LogMessage(logger,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                     "Creating Example EP", ORT_FILE, __LINE__, __FUNCTION__));

  // use properties from the device and ep_metadata if needed
  // const OrtHardwareDevice* device = devices[0];
  // const OrtKeyValuePairs* ep_metadata = ep_metadata[0];

  // Create EP configuration from session options, if needed.
  // Note: should not store a direct reference to the session options object as its lifespan is not guaranteed.
  std::string ep_context_enable;
  std::string ep_context_embed_mode;
  std::string ep_context_output_model_path;
  std::string weightless_ep_context_nodes_enable;
  RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(*session_options, kOrtSessionOptionEpContextEnable, "0",
                                                 ep_context_enable));
  RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(*session_options, kOrtSessionOptionEpContextEmbedMode, "0",
                                                 ep_context_embed_mode));
  RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(*session_options, kOrtSessionOptionEpContextFilePath, "",
                                                 ep_context_output_model_path));
  RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(*session_options, kOrtSessionOptionEpEnableWeightlessEpContextNodes,
                                                 "0", weightless_ep_context_nodes_enable));

  ExampleEp::Config config = {};
  config.enable_ep_context = ep_context_enable == "1";
  config.embed_ep_context_in_model = ep_context_embed_mode == "1";
  config.ep_context_output_model_path = std::move(ep_context_output_model_path);
  config.enable_weightless_ep_context_nodes = weightless_ep_context_nodes_enable == "1";

  // The EpContextConfig wrapper extracts the EPContext callbacks from the session options and owns the handle. It
  // throws if the experimental functions are unavailable or extraction fails; EXCEPTION_TO_RETURNED_STATUS_END
  // converts that (and any other exception thrown in this function) into an OrtStatus.
  auto dummy_ep = std::make_unique<ExampleEp>(
      *factory, factory->ep_name_, config, *logger,
      Ort::Experimental::EpContextConfig{Ort::ConstSessionOptions{session_options}});
  *ep = dummy_ep.release();
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

/*static*/
void ORT_API_CALL ExampleEpFactory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
  ExampleEp* dummy_ep = static_cast<ExampleEp*>(ep);
  delete dummy_ep;
}

/*static*/
OrtStatus* ORT_API_CALL ExampleEpFactory::CreateAllocatorImpl(OrtEpFactory* this_ptr,
                                                              const OrtMemoryInfo* memory_info,
                                                              const OrtKeyValuePairs* allocator_options,
                                                              OrtAllocator** allocator) noexcept {
  auto& factory = *static_cast<ExampleEpFactory*>(this_ptr);
  *allocator = nullptr;

  bool is_default_allocator = memory_info == factory.default_memory_info_;
  bool is_readonly_allocator = memory_info == factory.readonly_memory_info_;
  bool is_host_accessible_allocator = memory_info == factory.host_accessible_memory_info_;

  if (!is_default_allocator && !is_readonly_allocator && !is_host_accessible_allocator) {
    return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                        "INTERNAL ERROR! Unknown memory info provided to CreateAllocator. "
                                        "Value did not come directly from an OrtEpDevice returned by this factory.");
  }

  // NOTE: The factory implementation is free to return a shared OrtAllocator* instance instead of creating a new
  //       allocator on each call. To do this have an allocator instance as an OrtEpFactory class member and make
  //       ReleaseAllocatorImpl a no-op.
  //
  // NOTE: EP should implement its own arena logic. ep_arena.cc/h is provided as a reference and we use it here for
  //       device memory. `allocator_options` can be used for arena configuration and there is a helper in ep_arena.h
  //       to convert from OrtKeyValuePairs to the same arena config settings that ORT uses.
  //       You are of course free to have completely different settings.

  // the read-only allocator is used for initializers. we don't need an arena for that.
  // host-accessible memory is also returned via a plain non-arena allocator.
  if (is_readonly_allocator || is_host_accessible_allocator) {
    auto simple_allocator = std::make_unique<CustomAllocator>(memory_info, factory);
    *allocator = simple_allocator.release();
    return nullptr;
  }

  // create/use the shared arena based allocator
  std::lock_guard<std::mutex> lock{factory.mutex_};

  if (!factory.arena_allocator_) {
    AllocatorUniquePtr ep_allocator = std::make_unique<CustomAllocator>(memory_info, factory);

    // initial shared allocator in environment does not have allocator options.
    // if the user calls CreateSharedAllocator they can provide options to configure the arena differently.
    factory.arena_allocator_using_default_settings_ = allocator_options == nullptr;
    RETURN_IF_ERROR(ArenaAllocator::CreateOrtArenaAllocator(std::move(ep_allocator), allocator_options,
                                                            factory.ort_api,
                                                            factory.default_logger_, factory.arena_allocator_));

  } else {
    if (factory.arena_allocator_using_default_settings_ && allocator_options) {
      // potential change in arena settings. up to EP author to determine how to handle this.
      // we should not get here if replacing the shared allocator in the environment, as we free the existing one
      // before replacing it. i.e. ReleaseAllocatorImpl should have been called, and arena_allocator_ should be null.
    }
  }

  ++factory.num_arena_users_;
  *allocator = factory.arena_allocator_.get();

  return nullptr;
}

/*static*/
void ORT_API_CALL ExampleEpFactory::ReleaseAllocatorImpl(OrtEpFactory* this_ptr, OrtAllocator* allocator) noexcept {
  auto& factory = *static_cast<ExampleEpFactory*>(this_ptr);
  std::lock_guard<std::mutex> lock{factory.mutex_};

  if (allocator == factory.arena_allocator_.get()) {
    if (--factory.num_arena_users_ == 0) {
      factory.arena_allocator_ = nullptr;
    }
  } else {
    delete static_cast<CustomAllocator*>(allocator);
  }
}

/*static*/
OrtStatus* ORT_API_CALL ExampleEpFactory::CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                                 OrtDataTransferImpl** data_transfer) noexcept {
  auto& factory = *static_cast<ExampleEpFactory*>(this_ptr);
  *data_transfer = factory.data_transfer_impl_.get();

  return nullptr;
}

/*static*/
bool ORT_API_CALL ExampleEpFactory::IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return true;  // the example EP implements stream synchronization.
}

/*static*/
OrtStatus* ORT_API_CALL ExampleEpFactory::CreateSyncStreamForDeviceImpl(OrtEpFactory* this_ptr,
                                                                        const OrtMemoryDevice* memory_device,
                                                                        const OrtKeyValuePairs* stream_options,
                                                                        OrtSyncStreamImpl** stream) noexcept {
  auto& factory = *static_cast<ExampleEpFactory*>(this_ptr);
  *stream = nullptr;

  // we only need stream synchronization on the device stream
  if (factory.ep_api.MemoryDevice_GetMemoryType(memory_device) == OrtDeviceMemoryType_DEFAULT) {
    auto sync_stream = std::make_unique<StreamImpl>(factory, /*OrtEp**/ nullptr, stream_options);
    *stream = sync_stream.release();
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL ExampleEpFactory::GetNumCustomOpDomainsImpl(OrtEpFactory* this_ptr,
                                                                    _Out_ size_t* num_domains) noexcept {
  auto* factory = static_cast<ExampleEpFactory*>(this_ptr);
  *num_domains = factory->custom_op_domains_.size();

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL ExampleEpFactory::GetCustomOpDomainsImpl(
    OrtEpFactory* this_ptr,
    _Outptr_result_maybenull_ OrtCustomOpDomain** domains,
    _Out_ size_t num_domains) noexcept {
  auto* factory = static_cast<ExampleEpFactory*>(this_ptr);

  // The `num_domains` should be 2 as ORT calls GetNumCustomOpDomainsImpl() to get the number prior to
  // call this function.
  gsl::span<OrtCustomOpDomain*> domains_span(domains, num_domains);
  domains_span[0] = factory->custom_op_domains_[0];
  domains_span[1] = factory->custom_op_domains_[1];

  return nullptr;
}

OrtStatusPtr ExampleEpCustomOp::CreateKernelV2(const OrtApi& /*api*/,
                                               const OrtKernelInfo* /*info*/,
                                               void** op_kernel) const {
  std::string node_input_0 = "X";
  std::string node_input_1 = "W";
  auto custom_kernel_op = std::make_unique<CustomMulKernel>(factory_->ort_api,
                                                            factory_->default_logger_,
                                                            float_initializers_,
                                                            node_input_0,
                                                            node_input_1);
  *op_kernel = custom_kernel_op.release();
  return nullptr;
}

OrtStatusPtr ExampleEpCustomOp::KernelComputeV2(void* op_kernel, OrtKernelContext* context) const {
  return static_cast<CustomMulKernel*>(op_kernel)->ComputeV2(context);
}

OrtStatus* ORT_API_CALL ExampleEpFactory::GetHardwareDeviceIncompatibilityDetailsImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* hw,
    OrtDeviceEpIncompatibilityDetails* details) noexcept {
  auto& factory = *static_cast<ExampleEpFactory*>(this_ptr);

  // Example: This EP only supports CPU devices. Report incompatibility for non-CPU devices.
  OrtHardwareDeviceType device_type = factory.ort_api.HardwareDevice_Type(hw);

  if (device_type != OrtHardwareDeviceType_CPU) {
    // Report that the device type is not supported
    uint32_t reasons = OrtDeviceEpIncompatibility_DEVICE_INCOMPATIBLE;
    return factory.ep_api.DeviceEpIncompatibilityDetails_SetDetails(
        details,
        reasons,
        static_cast<int32_t>(device_type),  // Use device type as the error code for testing
        "ExampleEP only supports CPU devices");
  }

  // Device is compatible - details are already initialized with default values by ORT
  return nullptr;
}

OrtStatus* ORT_API_CALL ExampleEpFactory::CreateExternalResourceImporterForDeviceImpl(
    OrtEpFactory* this_ptr,
    const OrtEpDevice* /*ep_device*/,
    OrtExternalResourceImporterImpl** out_importer) noexcept {
  auto& factory = *static_cast<ExampleEpFactory*>(this_ptr);

  if (out_importer == nullptr) {
    return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                        "out_importer cannot be nullptr");
  }

  // Create the external resource importer
  // NOTE: For production multi-GPU EPs, you should capture ep_device in the importer
  // to enable proper device validation and support multiple physical devices.
  // This example EP only supports a single device, so we don't store it.
  auto importer = std::make_unique<ExampleExternalResourceImporter>(factory);
  *out_importer = importer.release();

  return nullptr;
}

namespace {

// Field keys for the example EP's compatibility string format
// "<ep_name>;version=X;ort_api_version=Y;hardware_architecture=Z". Using named constants (rather than literal
// offsets) keeps the parser robust if a key is renamed.
constexpr std::string_view kVersionKey = "version=";
constexpr std::string_view kOrtApiVersionKey = "ort_api_version=";
constexpr std::string_view kHardwareArchKey = "hardware_architecture=";

// Extracts the value of `key` from a ';'-delimited "k=v;k=v;..." string, or std::nullopt if the key is absent.
// The key must appear at the start of a field (string start or right after a ';'), so "version=" does NOT match
// the "version=" embedded in "ort_api_version=", and the value ends at the next ';' if present (otherwise the rest of the string).
std::optional<std::string> GetField(const std::string& info, std::string_view key) {
  for (size_t search = 0;;) {
    size_t pos = info.find(key.data(), search, key.size());
    if (pos == std::string::npos) {
      return std::nullopt;
    }
    const bool at_field_start = (pos == 0) || (info[pos - 1] == ';');
    if (at_field_start) {
      size_t value_start = pos + key.size();
      size_t value_end = info.find(';', value_start);
      return value_end != std::string::npos ? info.substr(value_start, value_end - value_start)
                                            : info.substr(value_start);
    }
    search = pos + 1;
  }
}

// Computes the compatibility verdict of a single compatibility string against a single hardware device.
// The compatibility string is opaque to ORT; only the EP that produced it knows how to interpret it.
//
// The architecture a compiled artifact targets is device-specific, so a real multi-device EP derives the expected
// value from `device` (e.g., via HardwareDevice_Type/VendorId/Metadata) and the verdict can differ per device.
// This example demonstrates that by mapping the hardware device type to an arch label. `device` may be nullptr
// (when num_devices == 0), in which case the EP's default configuration is used.
OrtCompiledModelCompatibility ComputeCompatibilityForDevice(const OrtApi& ort_api,
                                                            const std::string& ep_version,
                                                            const std::string& info,
                                                            const OrtHardwareDevice* device) {
  std::optional<std::string> compiled_ep_version = GetField(info, kVersionKey);
  if (!compiled_ep_version.has_value()) {
    // Our prefix but an unparseable string -> the artifact is ours but unusable.
    return OrtCompiledModelCompatibility_EP_UNSUPPORTED;
  }

  // Different EP version - might work but prefer recompilation.
  if (*compiled_ep_version != ep_version) {
    return OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION;
  }

  // Check ORT API version if present. Different ORT version - might still work but prefer recompilation.
  if (std::optional<std::string> ort_version = GetField(info, kOrtApiVersionKey); ort_version.has_value()) {
    if (*ort_version != std::to_string(ORT_API_VERSION)) {
      return OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION;
    }
  }

  // Check hardware architecture compatibility if present in the string. The expected arch is derived from the
  // target device (CPU -> "arch1"); a mismatch means the artifact was built for a different device.
  if (std::optional<std::string> hardware_arch = GetField(info, kHardwareArchKey); hardware_arch.has_value()) {
    std::string expected_arch = "arch1";  // default / num_devices == 0
    if (device != nullptr) {
      switch (ort_api.HardwareDevice_Type(device)) {
        case OrtHardwareDeviceType_GPU:
          expected_arch = "arch2";
          break;
        case OrtHardwareDeviceType_NPU:
          expected_arch = "arch3";
          break;
        default:  // CPU and any other type
          expected_arch = "arch1";
          break;
      }
    }
    // Different hardware architecture - might still work but prefer recompilation.
    if (*hardware_arch != expected_arch) {
      return OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION;
    }
  }

  // Everything matches - the compiled model is fully compatible with this device.
  return OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL;
}

}  // namespace

OrtStatus* ORT_API_CALL ExampleEpFactory::ValidateCompiledModelCompatibilityInfoImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* const* devices,
    size_t num_devices,
    const char* compatibility_info,
    OrtCompiledModelCompatibility* model_compatibility) noexcept {
  auto& factory = *static_cast<ExampleEpFactory*>(this_ptr);

  if (model_compatibility == nullptr) {
    return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "model_compatibility cannot be nullptr");
  }

  // The compatibility string is opaque to ORT and is interpreted only by the EP that produced it. An empty string,
  // or one that was not produced by this EP, means we have no opinion: report EP_NOT_APPLICABLE.
  if (compatibility_info == nullptr || compatibility_info[0] == '\0') {
    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    return nullptr;
  }

  // The expected format is "<ep_name>;version=<semver>;ort_api_version=<ORT_API_VERSION>;hardware_architecture=<arch>".
  std::string info(compatibility_info);
  if (info.find(factory.ep_name_ + ";") != 0) {
    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    return nullptr;
  }

  // `devices` are the hardware devices this EP would run the model on *together* (e.g., multi-adapter or multi-GPU).
  // Because we must return a single verdict for the whole set, evaluate the string against each device and combine
  // the per-device verdicts: EP_NOT_APPLICABLE is neutral and otherwise the worst verdict wins, so the result is
  // EP_SUPPORTED_OPTIMAL only if every device the EP has an opinion on is optimal. See the documentation for
  // ValidateCompiledModelCompatibilityInfo in onnxruntime_ep_c_api.h.
  if (num_devices == 0) {
    // No specific devices supplied; evaluate against the EP's own configuration.
    *model_compatibility =
        ComputeCompatibilityForDevice(factory.ort_api, factory.GetEpVersionString(), info, /*device*/ nullptr);
    return nullptr;
  }

  OrtCompiledModelCompatibility combined = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
  for (size_t i = 0; i < num_devices; ++i) {
    combined = example_ep::CombineCompatibility(
        combined, ComputeCompatibilityForDevice(factory.ort_api, factory.GetEpVersionString(), info, devices[i]));
    if (combined == OrtCompiledModelCompatibility_EP_UNSUPPORTED) {
      break;  // worst possible verdict; no need to evaluate the remaining devices
    }
  }

  *model_compatibility = combined;
  return nullptr;
}
