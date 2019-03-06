// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>

#include "core/graph/graph.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/framework/execution_frame.h"
#include "core/framework/mlvalue_name_idx_map.h"
#include "core/framework/ml_value.h"
#include "core/framework/callback.h"

namespace onnxruntime {

class OptimizerExecutionFrame final : public IExecutionFrame {
 public:
  class Info {
   public:
    Info(const std::vector<const Node*>& nodes,
         const InitializedTensorSet& initialized_tensor_set);
    ~Info() {
      for (auto& kvp : deleter_for_initialized_tensors_) {
        kvp.second.f(kvp.second.param);
      }
    }
    AllocatorPtr GetAllocator(const OrtAllocatorInfo& info) const {
      return cpu_execution_provider_->GetAllocator(info.id, info.mem_type);
    }

    AllocatorPtr GetAllocator() const {
      return allocator_ptr_;
    }

    const MLValueNameIdxMap& GetMLValueNameIdxMap() const noexcept { return mlvalue_name_idx_map_; }
    const std::unordered_map<int, const NodeArg*>& GetMLValueIdxNodeArgMap() const noexcept { return mlvalue_idx_nodearg_map_; }
    const std::unordered_map<int, MLValue>& GetInitializers() const noexcept { return initializers_; }
    const NodeIndexInfo& GetNodeIndexInfo() const { return *node_index_info_; }
    int GetMLValueIndex(const std::string& name) const {
      int index = -1;
      if (mlvalue_name_idx_map_.GetIdx(name, index) == Status::OK()) {
        return index;
      }
      return -1;
    }

    const OpKernel* GetKernel(NodeIndex node_id) const;

   private:
    // The optimizer is running on CPU execution provider by default.
    std::unique_ptr<CPUExecutionProvider> cpu_execution_provider_;
    const int device_id_{0};
    const OrtMemType mem_type_{OrtMemTypeDefault};
    AllocatorPtr allocator_ptr_;

    // MLValues for optimizer
    MLValueNameIdxMap mlvalue_name_idx_map_;
    std::unordered_map<int, const NodeArg*> mlvalue_idx_nodearg_map_;
    std::unordered_map<int, MLValue> initializers_;
    std::unordered_map<int, std::unique_ptr<char[]>> buffer_for_initialized_tensors_;
    // This data structure is for unintializing string tensors and
    // munmap memory region and close file descriptor
    std::unordered_map<int, OrtCallback> deleter_for_initialized_tensors_;
    std::unique_ptr<NodeIndexInfo> node_index_info_;

    std::unordered_map<onnxruntime::NodeIndex, std::unique_ptr<OpKernel>> kernels_;
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Info);
  };

  OptimizerExecutionFrame(const Info& info,
                          const std::vector<int>& fetch_mlvalue_idxs);

  ~OptimizerExecutionFrame() = default;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OptimizerExecutionFrame);

  AllocatorPtr GetAllocatorImpl(const OrtAllocatorInfo& info) const override;
  Status CreateNodeOutputMLValueImpl(MLValue& mlvalue, int mlvalue_idx, const TensorShape* shape) override;

  const Info& info_;
};

}  // namespace onnxruntime