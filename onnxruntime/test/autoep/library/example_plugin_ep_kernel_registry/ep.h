// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

class ExampleKernelEpFactory;

/// <summary>
/// Example EP that uses kernel registration.
/// </summary>
class ExampleKernelEp : public OrtEp {
 public:
  struct Config {
    bool enable_prepack_weight_sharing = false;
  };

  ExampleKernelEp(ExampleKernelEpFactory& factory, const Config& config, const OrtLogger& logger);
  ~ExampleKernelEp();

  const OrtApi& GetOrtApi() const { return ort_api_; }
  const OrtEpApi& GetEpApi() const { return ep_api_; }
  const Config& GetConfig() const { return config_; }

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL GetKernelRegistryImpl(
      _In_ OrtEp* this_ptr,
      _Outptr_result_maybenull_ const OrtKernelRegistry** kernel_registry) noexcept;

  static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                                   OrtEpGraphSupportInfo* graph_support_info) noexcept;

  ExampleKernelEpFactory& factory_;
  const OrtApi& ort_api_;
  const OrtEpApi& ep_api_;
  std::string name_;
  Config config_;
  const OrtLogger& logger_;
};
