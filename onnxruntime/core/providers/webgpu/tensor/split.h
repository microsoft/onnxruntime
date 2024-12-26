// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/split.h"

namespace onnxruntime {
namespace webgpu {

class SplitProgram final : public Program<SplitProgram> {
 public:
  SplitProgram(const uint32_t axis) : Program{"Split"}, axis_{axis} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"input_size", ProgramUniformVariableDataType::Uint32},
                                          {"sizes_in_split_axis", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t axis_;
};

class Split : public WebGpuKernel, public SplitBase {
 public:
  Split(const OpKernelInfo& info, uint32_t opset) : WebGpuKernel(info), SplitBase(info, opset) {
    std::vector<int32_t> split_sizes;
    // Check if split_sizes is provided as an attribute.
    if (split_sizes_.size() > 0) {
      ORT_ENFORCE(split_sizes_.size() == info.node().OutputDefs().size(), "Number of outputs (",
                  info.node().OutputDefs().size(), ") does not match split_sizes (", split_sizes_.size(), ")");
      split_sizes.resize(split_sizes_.size());
      for (size_t i = 0; i < split_sizes_.size(); ++i) {
        split_sizes[i] = gsl::narrow_cast<int32_t>(split_sizes_[i]);
      }
    } else if (info.GetInputCount() < 2) {
      // No valid split_sizes is providede as an attribute or input tensor. In this case, we try to compute it from input, output shapes and
      // num_outputs.

      // Handle negative axis.
      const auto num_dimensions = gsl::narrow_cast<int64_t>(info.node().InputDefs()[0]->Shape()->dim_size());
      const auto axis = HandleNegativeAxis(axis_, num_dimensions);

      auto total_split_size = info.node().InputDefs()[0]->Shape()->dim(gsl::narrow_cast<int32_t>(axis)).dim_value();
      int64_t split_size_sum = 0;
      if (num_outputs_ >= 0) {
        ORT_ENFORCE(num_outputs_ == gsl::narrow_cast<int64_t>(info.node().OutputDefs().size()),
                    "Invalid num_outputs value of ", num_outputs_, ". Size of dimension being split is ",
                    info.node().OutputDefs().size());
      }

      // Compute split_sizes from the output shapes.
      for (auto output : info.node().OutputDefs()) {
        auto split_size = output->Shape()->dim(gsl::narrow_cast<int32_t>(axis)).dim_value();
        split_sizes.push_back(gsl::narrow_cast<int32_t>(split_size));
        split_size_sum += split_size;
      }
      ORT_ENFORCE(split_size_sum == total_split_size, "Sum of split sizes (", split_size_sum,
                  ") does not match input size (", total_split_size, ")");
    }
  }

 protected:
  Status ComputeInternal(ComputeContext& context) const override;
};

class Split_1 final : public Split {
 public:
  Split_1(const OpKernelInfo& info) : Split(info, 1) {}
};

class Split_2_10 final : public Split {
 public:
  Split_2_10(const OpKernelInfo& info) : Split(info, 2) {}
};

class Split_11_12 final : public Split {
 public:
  Split_11_12(const OpKernelInfo& info) : Split(info, 11) {}
};

class Split_13_17 final : public Split {
 public:
  Split_13_17(const OpKernelInfo& info) : Split(info, 13) {}
};

class Split_18 final : public Split {
 public:
  Split_18(const OpKernelInfo& info) : Split(info, 18) {}
};

}  // namespace webgpu
}  // namespace onnxruntime
