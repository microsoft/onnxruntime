// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/optional.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/cpu/reduction/reduction_kernel_base.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include <string>
#include <unordered_map>

namespace onnxruntime {
namespace webgpu {
// reduceOpSpecificCode is a struct of three strings that represent the op specific code for the reduce operation.
// The first element is the loop header, the second element is the loop body, and the third element is the loop footer.
// The loop header is the code that is executed before the loop starts. The loop body is the code that is executed for each element in the loop.
// The loop footer is the code that is executed after the loop ends. The loop body should contain the code that accumulates the result of the reduction and
// the loop footer should contain the code that assigins output_value the result of the reduction.
typedef struct ReduceOpSpecificCode {
  std::string loop_header_;
  std::string loop_body_;
  std::string loop_footer_;
} ReduceOpSpecificCode;

enum class ReduceOpType {
  Max,
  Min,
  Mean,
  Sum,
  Prod,
  SumSquare,
  LogSumExp,
  L1,
  L2,
  LogSum,
  ArgMax,
  ArgMax_select_last_index,
  ArgMin,
  ArgMin_select_last_index,
};

ReduceOpType StringToReduceOp(std::string name);

class ReduceNaiveProgram final : public Program<ReduceNaiveProgram> {
 public:
  ReduceNaiveProgram(std::string name, ReduceOpType reduce_op_type, bool keepdims, bool no_op_with_empty_axes, const InlinedVector<uint32_t>& axes, bool is_input_empty) : Program{name}, keepdims_(keepdims), no_op_with_empty_axes_(no_op_with_empty_axes), axes_(axes.begin(), axes.end()), is_input_empty_(is_input_empty), reduce_op_type_(reduce_op_type) {}
  Status GenerateShaderCode(ShaderHelper& wgpuShaderModuleAddRef) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"no_op_with_empty_axes", ProgramUniformVariableDataType::Uint32},
                                          {"reduce_size", ProgramUniformVariableDataType::Uint32});

 private:
  const bool keepdims_;
  const bool no_op_with_empty_axes_;
  InlinedVector<uint32_t> axes_;
  bool is_input_empty_;
  const ReduceOpType reduce_op_type_;
};

class ReduceSharedProgram final : public Program<ReduceSharedProgram> {
 public:
  ReduceSharedProgram(std::string name, ReduceOpType reduce_op_type, uint32_t worgroup_size) : Program(name), reduce_op_type_(reduce_op_type), workgroup_size_(worgroup_size) {
    if (reduce_op_type_ == ReduceOpType::ArgMax || reduce_op_type_ == ReduceOpType::ArgMin || reduce_op_type_ == ReduceOpType::ArgMax_select_last_index || reduce_op_type_ == ReduceOpType::ArgMin_select_last_index) {
      ORT_THROW("ReduceSharedProgram: ArgMax/ArgMin is not supported in WebGPU yet.");
    }
  }
  Status GenerateShaderCode(ShaderHelper& wgpuShaderModuleAddRef) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"reduceSize", ProgramUniformVariableDataType::Uint32});

 private:
  const ReduceOpType reduce_op_type_;
  uint32_t workgroup_size_;
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
};

class ReduceMax final : public ReduceKernel<true> {
 public:
  ReduceMax(const OpKernelInfo& info) : ReduceKernel<true>(info, "ReduceMax") {}
};

class ReduceMin final : public ReduceKernel<true> {
 public:
  ReduceMin(const OpKernelInfo& info) : ReduceKernel<true>(info, "ReduceMin") {}
};

class ReduceSum final : public ReduceKernel<true> {
 public:
  ReduceSum(const OpKernelInfo& info) : ReduceKernel<true>(info, "ReduceSum", true) {}
};

class ReduceProd final : public ReduceKernel<true> {
 public:
  ReduceProd(const OpKernelInfo& info) : ReduceKernel<true>(info, "ReduceProd", true) {}
};

class ReduceL1 final : public ReduceKernel<true> {
 public:
  ReduceL1(const OpKernelInfo& info) : ReduceKernel<true>(info, "ReduceL1", true) {}
};

class ReduceL2 final : public ReduceKernel<true> {
 public:
  ReduceL2(const OpKernelInfo& info) : ReduceKernel<true>(info, "ReduceL2", true) {}
};

class ReduceLogSum final : public ReduceKernel<true> {
 public:
  ReduceLogSum(const OpKernelInfo& info) : ReduceKernel<true>(info, "ReduceLogSum", true) {}
};

class ReduceSumSquare final : public ReduceKernel<true> {
 public:
  ReduceSumSquare(const OpKernelInfo& info) : ReduceKernel<true>(info, "ReduceSumSquare", true) {}
};

class ReduceLogSumExp final : public ReduceKernel<true> {
 public:
  ReduceLogSumExp(const OpKernelInfo& info) : ReduceKernel<true>(info, "ReduceLogSumExp", true) {}
};

class ArgMin final : public ReduceKernel<false> {
 public:
  ArgMin(const OpKernelInfo& info) : ReduceKernel<false>(info, "ArgMin", true) {}
};

class ArgMax final : public ReduceKernel<false> {
 public:
  ArgMax(const OpKernelInfo& info) : ReduceKernel<false>(info, "ArgMax", true) {}
};

}  // namespace webgpu
}  // namespace onnxruntime
