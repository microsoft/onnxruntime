// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/trt/trt_kernel.h"

namespace onnxruntime{

// Information needed to construct trt execution providers.
struct TRTExecutionProviderInfo{
    int device_id{0};
};

// Information to construct kernel function state.
struct TRTFuncState {
  AllocateFunc test_allocate_func = nullptr;
  DestroyFunc test_release_func = nullptr;
  AllocatorHandle allocator = nullptr;
  nvinfer1::ICudaEngine* engine = nullptr;
  nvinfer1::IExecutionContext* context = nullptr;
  std::vector<std::vector<int>> input_info;
  std::vector<std::vector<int>> output_info;
  std::vector<std::vector<int64_t>> output_shapes;
};

// Logical device representation.
class TRTExecutionProvider : public IExecutionProvider{
public:
    TRTExecutionProvider();
    virtual ~TRTExecutionProvider();

    std::vector<std::unique_ptr<ComputeCapability>>
            GetCapability(const onnxruntime::GraphViewer& graph,
                          const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

    common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
        std::vector<NodeComputeInfo>& node_compute_funcs) override;

    std::string Type() const override{
        return onnxruntime::kTRTExecutionProvider;
    }

    Status CopyTensor(const Tensor& src, Tensor& dst) const override;

    const void* GetExecutionHandle() const noexcept override{
        return nullptr;
    }

    std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;


private:
    int device_id_;
    std::unordered_map<std::string, std::shared_ptr<nvinfer1::ICudaEngine>> engines_;
    std::unordered_map<std::string, std::shared_ptr<nvinfer1::IExecutionContext>> contexts_;
    std::unordered_map<std::string, std::vector<std::vector<int>>> input_info_;
    std::unordered_map<std::string, std::vector<std::vector<int>>> output_info_;
    std::unordered_map<std::string, std::vector<std::vector<int64_t>>> output_shapes_;
};

}  // namespace onnxruntime


