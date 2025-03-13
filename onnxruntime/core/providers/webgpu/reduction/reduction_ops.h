// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/optional.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/cpu/reduction/reduction_kernel_base.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
namespace onnxruntime {
namespace webgpu {
// reduceOpSpecificCode is a 3-element array of strings that represent the op specific code for the reduce operation.
// The first element is the loop header, the second element is the loop body, and the third element is the loop footer.
// The loop header is the code that is executed before the loop starts. The loop body is the code that is executed for each element in the loop.
// The loop footer is the code that is executed after the loop ends. The loop body should contain the code that accumulates the result of the reduction and
// the loop footer should contain the code that assigins output_value the result of the reduction.
typedef std::array<std::string, 3> ReduceOpSpecificCode;
class ReduceKernelProgram final : public Program<ReduceKernelProgram> {
 public:
  ReduceKernelProgram(std::string name, bool keepdims, bool no_op_with_empty_axes, const InlinedVector<uint32_t>& axes, ReduceOpSpecificCode code, bool is_input_empty) : Program{name}, keepdims_(keepdims), no_op_with_empty_axes_(no_op_with_empty_axes), axes_(axes.begin(), axes.end()), code_(code), is_input_empty_(is_input_empty) {}
  Status GenerateShaderCode(ShaderHelper& wgpuShaderModuleAddRef) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"no_op_with_empty_axes", ProgramUniformVariableDataType::Uint32},
                                          {"reduce_axes", ProgramUniformVariableDataType::Uint32});

 private:
  const bool keepdims_;
  const bool no_op_with_empty_axes_;
  InlinedVector<uint32_t> axes_;
  ReduceOpSpecificCode code_;
  bool is_input_empty_;
};

template <bool allow_multi_axes = true>
class ReduceKernel : public WebGpuKernel, public ReduceKernelBase<allow_multi_axes> {
 protected:
  using ReduceKernelBase<allow_multi_axes>::axes_;
  using ReduceKernelBase<allow_multi_axes>::noop_with_empty_axes_;
  using ReduceKernelBase<allow_multi_axes>::keepdims_;
  using ReduceKernelBase<allow_multi_axes>::select_last_index_;

  ReduceKernel(const OpKernelInfo& info, std::string name, bool allow_empty_input = false, optional<int64_t> keepdims_override = {})
      : WebGpuKernel(info),
        ReduceKernelBase<allow_multi_axes>(info, keepdims_override),
        name_(name),
        allow_empty_input_(allow_empty_input) {
  }
  Status ComputeInternal(ComputeContext& ctx) const;
  virtual ReduceOpSpecificCode GetOpSpecificCode(const Tensor* input_tensor) const = 0;

  Status CheckInput(const Tensor* input_tensor) const {
    ORT_ENFORCE(input_tensor != nullptr && (input_tensor->Shape().Size() > 0 || allow_empty_input_), "Input tensor cannot be null or empty");
    return Status::OK();
  }

 private:
  std::string name_;
  bool allow_empty_input_;
};

class ReduceMean final : public ReduceKernel<true> {
 public:
  ReduceMean(const OpKernelInfo& info) : ReduceKernel<true>(info, "ReduceMean", true) {}
  ReduceOpSpecificCode GetOpSpecificCode(const Tensor* input_tensor) const override;
};

class ReduceMax final : public ReduceKernel<true> {
 public:
  ReduceMax(const OpKernelInfo& info) : ReduceKernel<true>(info, "ReduceMax") {}
  ReduceOpSpecificCode GetOpSpecificCode(const Tensor* input_tensor) const override;
};

class ReduceSum final : public ReduceKernel<true> {
 public:
  ReduceSum(const OpKernelInfo& info) : ReduceKernel<true>(info, "ReduceSum", true) {}
  ReduceOpSpecificCode GetOpSpecificCode(const Tensor* input_tensor) const override;
};

}  // namespace webgpu
}  // namespace onnxruntime
