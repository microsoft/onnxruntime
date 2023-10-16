// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

// 1st-party libs/headers.
#include "core/framework/execution_provider.h"
#include "core/framework/customregistry.h"
#include "core/session/onnxruntime_c_api.h"
// XXX: Once the unified EP is ready, we probably need to
// decommission current separate EPs such as Vitis AI EP,
// MIGraphX EP, and ZenDNN EP from ONNXRuntime, but they
// will still work as libraries (most likely shared libraries)
// to serve the unified EP.
#include "core/providers/vitisai/vitisai_execution_provider.h"
#include "./amd_unified_execution_provider_info.h"

// Standard libs.
#include <sstream>
#include <algorithm>


namespace onnxruntime {

std::vector<std::string> SplitStr(const std::string& str, char delim,
    size_t start_pos = 0) {
  std::vector<std::string> res;
  std::stringstream ss(start_pos == 0 ? str : str.substr(start_pos));
  std::string item;

  while (std::getline(ss, item, delim)) {
    res.push_back(item);
  }

  return res;
}

std::vector<std::string> ParseDevicesStrRepr(const std::string& devices_str) {
  size_t colon_prefix_pos = devices_str.find(':');
  auto devices = SplitStr(devices_str, ',',
      colon_prefix_pos == std::string::npos ? 0 : colon_prefix_pos + 1);

  const std::string device_options[] = {"CPU", "GPU", "FGGA"};
  for (auto& d : device_options) {
    if (std::find(devices.begin(), devices.end(), d) == devices.end()) {
      ORT_THROW("Invalid device string: " + devices_str);
    }
  }

  return devices;
}

// Logical representation of AMD devices CPU/GPU/IPU/FPGA etc.
class AMDUnifiedExecutionProvider : public IExecutionProvider {
 public:
  explicit AMDUnifiedExecutionProvider(const AMDUnifiedExecutionProviderInfo&);
  ~AMDUnifiedExecutionProvider() = default;

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer&, const IKernelLookup&) const override;

  common::Status Compile(const std::vector<FusedNodeAndGraph>&,
      std::vector<NodeComputeInfo>&) override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  // XXX: We only have a limited set of downstream EPs, such as
  // Vitis AI EP, MIGraphX EP, ZenDNN EP, and ROCm EP, so it's fine
  // that we simply enumerate all of the related methods here.
  static void CreateDownstreamEP_VitisAI(const ProviderOptions&);
  // TODO: More EPs like MIGraphX EP, ZenDNN EP, etc.
  //static void CreateDownstreamEP_MIGraphX(const ProviderOptions&);
  //static void CreateDownstreamEP_ZenDNN(const ProviderOptions&);

 private:
  void CreateKernelRegistry();

  std::vector<std::unique_ptr<ComputeCapability>> CombineDownstreamCapabilites(
      const onnxruntime::GraphViewer&, const IKernelLookup&) const;

  common::Status CombineDownstreamCompilation(
      const std::vector<FusedNodeAndGraph>&, std::vector<NodeComputeInfo>&);

  AMDUnifiedExecutionProviderInfo ep_info_;
  std::vector<OrtCustomOpDomain*> custom_op_domains_;
  std::shared_ptr<KernelRegistry> kernel_registry_;
  std::vector<std::string> amd_unified_optypes_;

  // XXX: We only have a limited set of downstream EPs, such as
  // Vitis AI EP, MIGraphX EP, ZenDNN EP, and ROCm EP, so it's fine
  // that we simply enumerate the downstream EPs here.
  static std::unique_ptr<IExecutionProvider> vitisai_ep_;
  // TODO: More EPs like MIGraphX EP, ZenDNN EP, etc.
  //static std::unique_ptr<IExecutionProvider> migraphx_ep_;
  //static std::unique_ptr<IExecutionProvider> zendnn_ep_;
};

}  // namespace onnxruntime
