// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>

#include "onnxruntime_c_api.h"
#include "ep_arena.h"
#include "ep_data_transfer.h"
#include "../plugin_ep_utils.h"
#include "ep.h"

// Plugin EPs can provide two types of custom ops:
//
// 1. A full OrtCustomOp with a concrete kernel implementation
//    - This Example EP demonstrates this approach.
//    - In GetCapability(), it calls EpGraphSupportInfo_AddSingleNode() to inform ORT
//      that the custom node should NOT be fused or compiled. Instead, ORT should invoke
//      the custom node's Compute() function at runtime.
//
// 2. A "placeholder" OrtCustomOp with an empty kernel implementation
//    - A compile-based Plugin EP can supply an OrtCustomOp whose CustomKernel::Compute()
//      does nothing. The purpose is to satisfy model validation during model loading by
//      registering the custom op as a valid operator in the session.
//    - In GetCapability(), the EP should call EpGraphSupportInfo_AddNodesToFuse() to
//      notify ORT that this custom node should be fused and compiled by the EP.
//    - In Compile(), the EP executes its compiled bits to perform inference for
//      the fused custom node.
//
// Note: Approach #2 is suitable for plugin TRT RTX EP to support TRT plugins.

struct CustomMulKernel : MulKernel {
  CustomMulKernel(const OrtApi& ort_api,
                  const OrtLogger& logger,
                  const std::unordered_map<std::string, FloatInitializer>& float_initializers,
                  std::string input0_name,
                  std::string input1_name) : MulKernel(ort_api, logger, float_initializers,
                                                       input0_name, input1_name) {
  }
};

struct ExampleEpCustomOp : Ort::CustomOpBase<ExampleEpCustomOp, CustomMulKernel> {
  explicit ExampleEpCustomOp(const char* provider, ExampleEpFactory* factory) : provider_(provider),
                                                                                factory_(factory) {
  }

  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const;

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
  const char* name_ = nullptr;
  size_t num_inputs_ = 1;   // set to 1 to match with default min_arity for variadic input
  size_t num_outputs_ = 1;  // set to 1 to match with default min_arity for variadic output
  ExampleEpFactory* factory_ = nullptr;
  std::unordered_map<std::string, FloatInitializer> float_initializers_;
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

  const OrtLogger& default_logger_;  // default logger for the EP factory

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

  static OrtStatus* ORT_API_CALL GetNumCustomOpDomainsImpl(OrtEpFactory* this_ptr,
                                                           _Out_ size_t* num_domains) noexcept;

  static OrtStatus* ORT_API_CALL CreateCustomOpDomainsImpl(OrtEpFactory* this_ptr,
                                                           _Outptr_result_maybenull_ OrtCustomOpDomain** domains,
                                                           _Out_ size_t num_domains) noexcept;

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
  std::vector<Ort::CustomOpDomain> custom_op_domains_{2};
  std::vector<std::vector<std::unique_ptr<ExampleEpCustomOp>>> created_custom_op_lists_{2};
};
