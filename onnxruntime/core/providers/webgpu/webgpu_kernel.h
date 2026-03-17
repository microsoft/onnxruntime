// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/compute_context.h"

#include "core/framework/op_kernel.h"
#include "core/providers/webgpu/numpy_io.h"

namespace onnxruntime {

class WebGpuExecutionProvider;
namespace webgpu {

// -----------------------------------------------------------------------
// Base class for WebGPU kernels
// -----------------------------------------------------------------------
class WebGpuKernel : public OpKernel {
 public:
  explicit WebGpuKernel(const OpKernelInfo& info);

  Status Compute(OpKernelContext* p_op_kernel_context) const override;

  virtual Status ComputeInternal(ComputeContext& context) const = 0;

  // call with
  // NpyTensor<MLFloat16>(hidden_state, "/tmp/hidden_state.npy", context);

  template <typename T>
  void NpyTensor(const Tensor* t, std::string file, ComputeContext& context) const {
    auto t_cpu = context.CreateCPUTensor(t->DataType(), t->Shape());
    ORT_THROW_IF_ERROR(Info().GetDataTransferManager().CopyTensor(*t, t_cpu));

    std::vector<size_t> dims;
    auto dims1 = t_cpu.Shape().GetDims();
    for (uint64_t i=0; i<dims1.size(); i++) {
      dims.push_back(dims1[i]);
    }
    auto a = numpy_io::NumpyArray<T>(dims);
    for (int64_t i = 0; i < t_cpu.Shape().Size(); i++) {
      a.data[i] = static_cast<T>(t_cpu.Data<T>()[i]);
    }
    numpy_io::write_numpy_array(file, a);
  }


  // Overrides OpKernel::PrePack to handle constant tensor pre-processing for WebGPU kernels.
  // This method creates a ComputeContextBase and delegates to PrePackInternal.
  //
  // NOTE: Currently, ORT does not allow using prepacked weights in non-CPU EPs, so the
  // prepacked_weights parameter is not passed to PrePackInternal. Kernel implementations
  // that support prepacking should manage their own storage.
  Status PrePack(const Tensor& tensor,
                 int input_idx,
                 AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  // Virtual method that allows derived kernels to pre-process constant tensors during initialization.
  //
  // This method is called during kernel initialization when constant tensors are available,
  // allowing kernels to perform operations like tensor transposition or format conversion
  // before the first Compute call.
  //
  // @param context       The WebGPU compute context base providing access to the execution environment.
  // @param tensor        The constant tensor to potentially pre-process.
  // @param input_idx     The index of this input in the kernel's input list.
  // @param alloc         The allocator to use for any new tensor allocations (prepack allocator).
  // @param is_packed     Output parameter. Set to true if the tensor was pre-packed/processed,
  //                      false otherwise. The default implementation sets this to false.
  //
  // @return Status::OK() on success, or an error status on failure.
  virtual Status PrePackInternal(ComputeContextBase& context,
                                 const Tensor& tensor,
                                 int input_idx,
                                 AllocatorPtr alloc,
                                 /*out*/ bool& is_packed);

 private:
  const WebGpuExecutionProvider& ep_;
  WebGpuContext& webgpu_context_;
};

}  // namespace webgpu
}  // namespace onnxruntime
