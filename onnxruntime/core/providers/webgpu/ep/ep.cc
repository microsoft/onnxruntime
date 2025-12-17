// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep.h"

#include "factory.h"

#include "core/framework/run_options.h"
#include "core/framework/kernel_registry.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#include "core/session/plugin_ep/ep_kernel_registration.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"

namespace onnxruntime {
namespace webgpu {
namespace ep {

using onnxruntime::ep::Api;

// Constructor
Ep::Ep(IExecutionProvider* impl, Factory& factory, const OrtLogger& logger, const Config& config)
    : onnxruntime::ep::Ep{impl, config.cpu_allocator, config.device_allocator},
      factory_{factory},
      logger_{logger},
      config_{config} {
  ort_version_supported = ORT_API_VERSION;

  // Initialize the execution provider's function table
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  GetKernelRegistry = GetKernelRegistryImpl;
  Compile = nullptr;  // Per-kernel EP does not use Compile
  ReleaseNodeComputeInfos = nullptr;
  GetPreferredDataLayout = GetPreferredDataLayoutImpl;
  ShouldConvertDataLayoutForOp = ShouldConvertDataLayoutForOpImpl;
  SetDynamicOptions = nullptr;  // Not implemented
  OnRunStart = OnRunStartImpl;
  OnRunEnd = OnRunEndImpl;
  CreateAllocator = CreateAllocatorImpl;
  CreateSyncStreamForDevice = nullptr;          // Not stream aware
  GetCompiledModelCompatibilityInfo = nullptr;  // Not a compiled EP
}

// OrtEp interface implementations
const char* ORT_API_CALL Ep::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const Ep*>(this_ptr);
  return ep->factory_.GetName(&ep->factory_);
}

OrtStatus* ORT_API_CALL Ep::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                              OrtEpGraphSupportInfo* graph_support_info) noexcept {
  try {
    Ep* ep = static_cast<Ep*>(this_ptr);
    Ort::ConstGraph ort_graph{graph};

    // Get all nodes in the graph
    std::vector<Ort::ConstNode> all_nodes = ort_graph.GetNodes();

    if (all_nodes.empty()) {
      return nullptr;  // No nodes to process
    }

    // For each node, check if we have a registered kernel for it
    for (const auto& node : all_nodes) {
      const OrtKernelDef* kernel_def = nullptr;
      RETURN_IF_ERROR(Api().ep.EpGraphSupportInfo_LookUpKernel(graph_support_info, node, &kernel_def));

      // If we have a kernel definition for this node, mark it as supported
      if (kernel_def != nullptr) {
        RETURN_IF_ERROR(Api().ep.EpGraphSupportInfo_AddSingleNode(graph_support_info, node));
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

OrtStatus* ORT_API_CALL Ep::GetKernelRegistryImpl(
    _In_ OrtEp* this_ptr,
    _Outptr_result_maybenull_ const OrtKernelRegistry** kernel_registry) noexcept {
  try {
    *kernel_registry = nullptr;

    // For the WebGPU EP, delegate to the CreateKernelRegistry function
    // which properly constructs a registry using only public APIs
    auto* ep = static_cast<Ep*>(this_ptr);
    const char* ep_name = ep->factory_.GetName(&ep->factory_);

    auto& webgpu_ep = *ep->EpImpl();

    *kernel_registry = *onnxruntime::webgpu::GetKernelRegistry(webgpu_ep.IsGraphCaptureEnabled()).get();
    return nullptr;
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }
}

OrtStatus* ORT_API_CALL Ep::GetPreferredDataLayoutImpl(_In_ OrtEp* this_ptr,
                                                       _Out_ OrtEpDataLayout* preferred_data_layout) noexcept {
  // Delegate to the underlying WebGPU EP's GetPreferredLayout()
  // DataLayout enum values map 1:1 to OrtEpDataLayout (NCHW=0, NHWC=1)
  auto* ep = static_cast<Ep*>(this_ptr);
  *preferred_data_layout = static_cast<OrtEpDataLayout>(ep->EpImpl()->GetPreferredLayout());
  return nullptr;
}

OrtStatus* ORT_API_CALL Ep::ShouldConvertDataLayoutForOpImpl(_In_ OrtEp* this_ptr,
                                                             _In_z_ const char* domain,
                                                             _In_z_ const char* op_type,
                                                             _In_ OrtEpDataLayout target_data_layout,
                                                             _Outptr_ int* should_convert) noexcept {
  // DataLayout enum values map 1:1 to OrtEpDataLayout (NCHW=0, NHWC=1)
  auto* ep = static_cast<Ep*>(this_ptr);
  auto result = ep->EpImpl()->ShouldConvertDataLayoutForOp(domain, op_type,
                                                           static_cast<DataLayout>(target_data_layout));
  if (result.has_value()) {
    *should_convert = result.value() ? 1 : 0;
  } else {
    *should_convert = -1;
  }
  return nullptr;
}

OrtStatus* ORT_API_CALL Ep::OnRunStartImpl(_In_ OrtEp* this_ptr,
                                           _In_ const OrtRunOptions* run_options) noexcept {
  onnxruntime::RunOptions options{};
  // currently only option "gpu_graph_id" is used
  auto graph_annotation_str = Api().ort.GetRunConfigEntry(run_options, kOrtRunOptionsConfigCudaGraphAnnotation);
  if (graph_annotation_str != nullptr) {
    options.config_options.AddConfigEntry(kOrtRunOptionsConfigCudaGraphAnnotation, graph_annotation_str);
  }
  auto* ep = static_cast<Ep*>(this_ptr);
  auto status = ep->EpImpl()->OnRunStart(options);
  if (!status.IsOK()) {
    return Api().ort.CreateStatus(static_cast<OrtErrorCode>(status.Code()),
                                  status.ErrorMessage().c_str());
  }
  return nullptr;
}

OrtStatus* ORT_API_CALL Ep::OnRunEndImpl(_In_ OrtEp* this_ptr,
                                         _In_ const OrtRunOptions* run_options,
                                         _In_ bool sync_stream) noexcept {
  auto* ep = static_cast<Ep*>(this_ptr);
  auto status = ep->EpImpl()->OnRunEnd(sync_stream, {});
  if (!status.IsOK()) {
    return Api().ort.CreateStatus(static_cast<OrtErrorCode>(status.Code()),
                                  status.ErrorMessage().c_str());
  }
  return nullptr;
}

OrtStatus* ORT_API_CALL Ep::CreateAllocatorImpl(_In_ OrtEp* this_ptr,
                                                _In_ const OrtMemoryInfo* memory_info,
                                                _Outptr_result_maybenull_ OrtAllocator** allocator) noexcept {
  auto* ep = static_cast<Ep*>(this_ptr);
  if (memory_info && memory_info->alloc_type == OrtReadOnlyAllocator) {
    *allocator = new onnxruntime::ep::Allocator(ep->config_.initializer_allocator);
  } else {
    *allocator = new onnxruntime::ep::Allocator(ep->config_.device_allocator);
  }
  return nullptr;
}

}  // namespace ep
}  // namespace webgpu
}  // namespace onnxruntime
