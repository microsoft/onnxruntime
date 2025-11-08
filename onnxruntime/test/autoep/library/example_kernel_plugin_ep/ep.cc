// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep.h"

#include <gsl/span>
#include <array>
#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ep_factory.h"
#include "../plugin_ep_utils.h"

KernelEp::KernelEp(KernelEpFactory& factory, const OrtLogger& logger)
    : OrtEp{},  // explicitly call the struct ctor to ensure all optional values are default initialized
      factory_{factory},
      ort_api_{factory.GetOrtApi()},
      ep_api_{factory.GetEpApi()},
      name_{factory.GetEpName()},
      logger_{logger} {
  ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.

  // Initialize the execution provider's function table
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  GetKernelRegistry = GetKernelRegistryImpl;

  // This is not a compiling EP, so don't need the following
  Compile = nullptr;
  ReleaseNodeComputeInfos = nullptr;

  // ignore status for now
  OrtStatus* status = ort_api_.Logger_LogMessage(&logger_,
                                                 OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                 ("KernelEp has been created with name " + name_).c_str(),
                                                 ORT_FILE, __LINE__, __FUNCTION__);
  Ort::Status _ignored{status};
}

KernelEp::~KernelEp() = default;

/*static*/
const char* ORT_API_CALL KernelEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const KernelEp*>(this_ptr);
  return ep->name_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL KernelEp::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* ort_graph,
                                                    OrtEpGraphSupportInfo* graph_support_info) noexcept {
  try {
    KernelEp* ep = static_cast<KernelEp*>(this_ptr);

    Ort::ConstGraph graph{ort_graph};
    std::vector<Ort::ConstNode> nodes = graph.GetNodes();

    if (nodes.empty()) {
      return nullptr;  // No nodes to process
    }

    for (const auto& node : nodes) {
      const OrtKernelDef* kernel_def = nullptr;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_LookUpKernel(graph_support_info, node, &kernel_def));

      // Claim nodes for which we have a registered kernel
      if (kernel_def != nullptr) {
        RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddSingleNode(graph_support_info, node));
      }
    }
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL KernelEp::GetKernelRegistryImpl(
    _In_ OrtEp* this_ptr,
    _Outptr_result_maybenull_ const OrtKernelRegistry** kernel_registry) noexcept {
  KernelEp* ep = static_cast<KernelEp*>(this_ptr);

  *kernel_registry = nullptr;

  // Get the cached kernel registry from parent factory to avoid recreating the kernel registry for every EP instance.
  RETURN_IF_ERROR(ep->factory_.GetKernelRegistryForEp(*ep, kernel_registry));
  return nullptr;
}
