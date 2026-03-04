// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep_factory.h"

#include <cassert>

#include "ep.h"
#include "ep_allocator.h"
#include "ep_arena.h"
#include "ep_data_transfer.h"
#include "ep_stream_support.h"

ExampleEpFactory::ExampleEpFactory(const char* ep_name, ApiPtrs apis, const OrtLogger& default_logger)
    : OrtEpFactory{},
      ApiPtrs(apis),
      default_logger_{default_logger},
      ep_name_{ep_name},
      default_memory_info_{nullptr},
      readonly_memory_info_{nullptr} {
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
  auto host_accessible_memory_info = Ort::MemoryInfo{"ExampleEP GPU pinned",
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
      // registering OrtMemoryInfo for host accessible memory would be done in an additional call.
      // OrtReadOnlyAllocator + OrtDeviceMemoryType_DEFAULT allocator for use with initializers is optional.
      RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, factory->default_memory_info_));
      RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, factory->readonly_memory_info_));

      ep_devices[num_ep_devices++] = ep_device;
    }

    // C++ API equivalent. Throws on error.
    //{
    //  Ort::ConstHardwareDevice device(devices[i]);
    //  if (device.Type() == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
    //    Ort::KeyValuePairs ep_metadata;
    //    Ort::KeyValuePairs ep_options;
    //    ep_metadata.Add("supported_devices", "CrackGriffin 7+");
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
  RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(*session_options, "ep.context_enable", "0", ep_context_enable));

  ExampleEp::Config config = {};
  config.enable_ep_context = ep_context_enable == "1";

  auto dummy_ep = std::make_unique<ExampleEp>(*factory, factory->ep_name_, config, *logger);

  *ep = dummy_ep.release();
  return nullptr;
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

  if (!is_default_allocator && !is_readonly_allocator) {
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
  if (is_readonly_allocator) {
    auto read_only_allocator = std::make_unique<CustomAllocator>(memory_info, factory);
    *allocator = read_only_allocator.release();
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

OrtStatus* ORT_API_CALL ExampleEpFactory::ValidateCompiledModelCompatibilityInfoImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* const* /*devices*/,
    size_t /*num_devices*/,
    const char* compatibility_info,
    OrtCompiledModelCompatibility* model_compatibility) noexcept {
  auto& factory = *static_cast<ExampleEpFactory*>(this_ptr);

  if (model_compatibility == nullptr) {
    return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "model_compatibility cannot be nullptr");
  }

  // Parse the compatibility info to check if it matches our current configuration.
  // The expected format is "ExampleEP;version=0.1.0;ort_api_version=24".
  // For this example implementation, we simply check if the string starts with our EP name.

  if (compatibility_info == nullptr || compatibility_info[0] == '\0') {
    *model_compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    return nullptr;
  }

  std::string info(compatibility_info);
  std::string expected_prefix = factory.ep_name_ + ";";

  if (info.find(expected_prefix) != 0) {
    // The compatibility info doesn't match our EP
    *model_compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    return nullptr;
  }

  // Parse version parts: "ExampleEP;version=X;ort_api_version=Y"
  // Look for "version=" and extract the value
  size_t version_pos = info.find("version=");
  size_t ort_version_pos = info.find("ort_api_version=");

  if (version_pos == std::string::npos) {
    // Invalid format
    *model_compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    return nullptr;
  }

  // Extract EP version (between "version=" and the next ";")
  size_t version_start = version_pos + 8;  // length of "version="
  size_t version_end = info.find(';', version_start);
  std::string ep_version = (version_end != std::string::npos)
                               ? info.substr(version_start, version_end - version_start)
                               : info.substr(version_start);

  // Check if the EP version matches our version
  if (ep_version != factory.ep_version_) {
    // Different EP version - might work but prefer recompilation
    *model_compatibility = OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION;
    return nullptr;
  }

  // Check ORT API version if present
  if (ort_version_pos != std::string::npos) {
    size_t ort_version_start = ort_version_pos + 16;  // length of "ort_api_version="
    std::string ort_version = info.substr(ort_version_start);
    std::string current_ort_version = std::to_string(ORT_API_VERSION);
    if (ort_version != current_ort_version) {
      // Different ORT version - might still work but prefer recompilation
      *model_compatibility = OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION;
      return nullptr;
    }
  }

  // Everything matches - the compiled model is fully compatible
  *model_compatibility = OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL;
  return nullptr;
}
