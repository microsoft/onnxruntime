// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>

#include "onnxruntime_c_api.h"
#include "ep_arena.h"
#include "ep_data_transfer.h"
#include "../plugin_ep_utils.h"

// This is a placeholder for "compile-based" plugin EP to provide custom op domains to ORT.
// Please note that this is not for "kernel registration" plugin EP to register kernels.
struct PluginEpCustomKernel {
  PluginEpCustomKernel(const OrtKernelInfo* /*info*/, void* compute_stream)
      : compute_stream_(compute_stream) {
  }

  void Compute(OrtKernelContext* /*context*/) {
    // The implementation is in plugin EP's compiled bits. No need to implement it here.
  };

 private:
  void* compute_stream_;
};

struct PluginEpCustomOp : Ort::CustomOpBase<PluginEpCustomOp, PluginEpCustomKernel> {
  explicit PluginEpCustomOp(const char* provider, void* compute_stream) : provider_(provider),
                                                                          compute_stream_(compute_stream) {
  }

  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* info) const {
    return new PluginEpCustomKernel(info, compute_stream_);
  };

  const char* GetName() const { return name_; };

  void SetName(const char* name) { name_ = name; };

  const char* GetExecutionProviderType() const { return provider_; };

  size_t GetInputTypeCount() const { return num_inputs_; };

  void SetInputTypeCount(size_t num) { num_inputs_ = num; };

  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED; };

  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  };

  size_t GetOutputTypeCount() const { return num_outputs_; };

  void SetOutputTypeCount(size_t num) { num_outputs_ = num; };

  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED; };

  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  };

  bool GetVariadicInputHomogeneity() const {
    return false;  // heterogenous
  }

  bool GetVariadicOutputHomogeneity() const {
    return false;  // heterogeneous
  }

 private:
  const char* provider_ = nullptr;
  void* compute_stream_ = nullptr;
  const char* name_ = nullptr;
  size_t num_inputs_ = 1;   // set to 1 to match with default min_arity for variadic input
  size_t num_outputs_ = 1;  // set to 1 to match with default min_arity for variadic output
};

/// <summary>
/// Example EP factory that can create an OrtEp and return information about the supported hardware devices.
/// </summary>
class ExampleEpFactory : public OrtEpFactory, public ApiPtrs {
 public:
  ExampleEpFactory(const char* ep_name, ApiPtrs apis, const OrtLogger& default_logger);

  OrtDataTransferImpl* GetDataTransfer() const {
    return data_transfer_impl_.get();
  }

  // Get the shared arena allocator if created.
  ArenaAllocator* GetArenaAllocator() const {
    return arena_allocator_.get();
  }

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) noexcept;

  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) noexcept;
  static uint32_t ORT_API_CALL GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept;

  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                         const OrtHardwareDevice* const* devices,
                                                         size_t num_devices,
                                                         OrtEpDevice** ep_devices,
                                                         size_t max_ep_devices,
                                                         size_t* p_num_ep_devices) noexcept;

  static OrtStatus* ORT_API_CALL CreateEpImpl(OrtEpFactory* this_ptr,
                                              const OrtHardwareDevice* const* /*devices*/,
                                              const OrtKeyValuePairs* const* /*ep_metadata*/,
                                              size_t num_devices,
                                              const OrtSessionOptions* session_options,
                                              const OrtLogger* logger,
                                              OrtEp** ep) noexcept;

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept;

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(OrtEpFactory* this_ptr,
                                                     const OrtMemoryInfo* memory_info,
                                                     const OrtKeyValuePairs* /*allocator_options*/,
                                                     OrtAllocator** allocator) noexcept;

  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* /*this*/, OrtAllocator* allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                        OrtDataTransferImpl** data_transfer) noexcept;

  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(OrtEpFactory* this_ptr,
                                                               const OrtMemoryDevice* memory_device,
                                                               const OrtKeyValuePairs* stream_options,
                                                               OrtSyncStreamImpl** stream) noexcept;

  static OrtStatus* ORT_API_CALL CreateCustomOpDomainsImpl(OrtEpFactory* this_ptr,
                                                           _Outptr_result_maybenull_ OrtCustomOpDomain** out,
                                                           _Out_ size_t* num_domains) noexcept;

  const OrtLogger& default_logger_;        // default logger for the EP factory
  const std::string ep_name_;              // EP name
  const std::string vendor_{"Contoso"};    // EP vendor name
  const uint32_t vendor_id_{0xB357};       // EP vendor ID
  const std::string ep_version_{"0.1.0"};  // EP version

  // CPU allocator so we can control the arena behavior. optional as ORT always provides a CPU allocator if needed.
  Ort::MemoryInfo default_memory_info_;
  Ort::MemoryInfo readonly_memory_info_;  // used for initializers

  bool arena_allocator_using_default_settings_{true};
  std::unique_ptr<ArenaAllocator> arena_allocator_;  // shared device allocator that uses an arena
  uint32_t num_arena_users_{0};
  std::mutex mutex_;  // mutex to protect arena_allocator_ and num_arena_users_

  std::unique_ptr<ExampleDataTransfer> data_transfer_impl_;  // data transfer implementation for this factory

  // std::unique_ptr<OrtCustomOpDomain, std::function<void(OrtCustomOpDomain*)>> custom_op_domain_;
  Ort::CustomOpDomain custom_op_domain_;
  std::vector<std::unique_ptr<PluginEpCustomOp>> created_custom_op_list_;
};
