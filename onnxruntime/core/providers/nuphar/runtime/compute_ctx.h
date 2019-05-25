// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/common/common.h"
#include "core/framework/allocator.h"
#include "core/framework/data_types.h"
#include "core/framework/func_api.h"
#include "core/framework/func_kernel.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "gsl/gsl_util"

#include "core/providers/nuphar/runtime/handle.h"
#include "core/providers/nuphar/compiler/nuphar_func_ctx.h"
#include "core/providers/nuphar/runtime/control_flow/loop_exec_ctx.h"

// TODO change name space
namespace onnxruntime {
namespace tvm_codegen {

class LoopExecCtx;

using DataAllocFunc = std::function<void*(size_t)>;

// TODO: split to header and source, and keep only inline function in header
// this class contains ORT compute context from NodeComputeInfo
class NupharComputeCtx {
 public:
  explicit NupharComputeCtx(
      const nuphar::NupharRuntimeHandle* handle,
      std::unordered_map<std::string, int64_t>& realized_dims,
      const NupharFuncInfo& func_info,
      DataAllocFunc data_alloc_func);

  void Bind(OpKernelContext* op_kernel_ctx);

  inline const void* InputData(int index) const {
    return op_kernel_ctx_->Input<Tensor>(index)->DataRaw();
  }

  inline const int64_t* InputShape(int index) const {
    const auto* t = op_kernel_ctx_->Input<Tensor>(index);
    if (t == nullptr)
      return nullptr;

    const auto& s = t->Shape();
    return s.GetDims().data();
  }

  inline size_t InputShapeRank(int index) const {
    const auto* t = op_kernel_ctx_->Input<Tensor>(index);
    if (t == nullptr)
      return 0;

    const auto& s = t->Shape();
    return s.NumDimensions();
  }

  inline IAllocatorUniquePtr<void> AllocateData(const TensorShape& shape, MLDataType dtype) {
    return IAllocator::MakeUniquePtr<void>(handle_->allocator, shape.Size() * dtype->Size());
  }

  inline IAllocatorUniquePtr<void> AllocateData(const int64_t* shape, const size_t rank, MLDataType dtype) {
    int64_t total_size = dtype->Size();
    for (size_t i = 0; i < rank; ++i) {
      total_size *= shape[i];
    }
    return IAllocator::MakeUniquePtr<void>(handle_->allocator, total_size);
  }

  inline void* OutputData(int index, const TensorShape& shape) {
    auto t = op_kernel_ctx_->Output(index, shape);
    return t->MutableDataRaw();
  }

  inline int InputCount() const {
    return op_kernel_ctx_->InputCount();
  }

  inline int OutputCount() const {
    return op_kernel_ctx_->OutputCount();
  }

  inline OpKernelContext* GetOpKernelContext() const {
    return op_kernel_ctx_;
  }

  inline const nuphar::NupharRuntimeHandle* GetRuntimeHandle() const {
    return handle_;
  }

  inline std::unordered_map<std::string, int64_t>& GetRealizedDims() {
    return realized_dims_;
  }

  inline const NupharFuncInfo& GetNupharFuncInfo() const {
    return func_info_;
  }

  inline std::vector<TVMValue>& GetTVMValues() {
    return lvalues_;
  }

  inline std::vector<DLTensor>& GetDLTensors() {
    return dl_tensors_;
  }

  inline DLTensor& GetDLTensor(int index) {
    ORT_ENFORCE_DEBUG(index < dl_tensors_.size());
    return dl_tensors_[index];
  }

  bool HasInitialized() const {
    return dl_tensors_.size() > 0;
  }

  inline std::vector<std::vector<int64_t>>& GetDLOutputShapes() {
    return dl_output_shapes_;
  }

  inline LoopExecCtx* GetControlFlowCtx() {
    ORT_ENFORCE_DEBUG(nullptr != loop_cf_ctx_);
    return loop_cf_ctx_.get();
  }

 private:
  DataAllocFunc data_alloc_func_;

  // runtime handle
  const nuphar::NupharRuntimeHandle* handle_;

  // realized_dim
  std::unordered_map<std::string, int64_t>& realized_dims_;
  // Static function info
  const NupharFuncInfo& func_info_;

  // PackedFunc's argurments
  // NupharComputeCtx has the ownership
  std::vector<TVMValue> lvalues_;
  std::vector<DLTensor> dl_tensors_;
  std::vector<std::vector<int64_t>> dl_output_shapes_;

  OpKernelContext* op_kernel_ctx_;

  // LoopState
  std::unique_ptr<LoopExecCtx> loop_cf_ctx_;
};

}  // namespace tvm_codegen
}  // namespace onnxruntime
