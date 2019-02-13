#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/allocatormgr.h"
#include "openvino_graph.h"

namespace onnxruntime {

constexpr const char* OPENVINO = "OpenVINO";
using namespace InferenceEngine;

// Information needed to construct OpenVINO execution providers.
struct OpenVINOExecutionProviderInfo {
  const char* device { "CPU_FP32" };

  explicit OpenVINOExecutionProviderInfo(const char* dev) :
      device(dev) {
  }
  OpenVINOExecutionProviderInfo() {
  }
};

struct OpenVINOEPFunctionState {
  AllocateFunc allocate_func = nullptr;
  DestroyFunc destroy_func = nullptr;
  AllocatorHandle allocator_handle = nullptr;
  std::shared_ptr<openvino_ep::OpenVINOGraph> openvino_graph = nullptr;
};

class OpenVINOExecutionProvider: public IExecutionProvider {

public:
  explicit OpenVINOExecutionProvider(OpenVINOExecutionProviderInfo& info);

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
      const std::vector<const KernelRegistry*>& kernel_registries) const
          override;

  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
      std::vector<NodeComputeInfo>& node_compute_funcs) override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override {
    return std::make_shared<KernelRegistry>();
  }

  common::Status CopyTensor(const Tensor& src, Tensor& dst) const override {

    // Todo: Copy for now. May optimize later to avoid copy.
    size_t bytes = src.DataType()->Size() * src.Shape().Size();
    const void* src_data = src.DataRaw();
    void* dst_data = dst.MutableDataRaw();
    memcpy(dst_data, src_data, bytes);

    return Status::OK();
  }

  const void* GetExecutionHandle() const noexcept override {
    return nullptr;
  }

private:

  OpenVINOExecutionProviderInfo info_;
};

}
