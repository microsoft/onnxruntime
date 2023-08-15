// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef PRINT_TOLERANCE_ERRORS

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace contrib {

class PrintToleranceErrors : public OpKernel {
 public:
  PrintToleranceErrors(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<std::string>("node_name", &node_name_).IsOK());
    ORT_ENFORCE(info.GetAttr<std::string>("node_type", &node_type_).IsOK());
    ORT_ENFORCE(info.GetAttr<std::string>("execution_provider", &execution_provider_).IsOK());
    ORT_ENFORCE(info.GetAttr<int64_t>("node_output_index", &node_output_index_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* float16_input = context->Input<Tensor>(0);
    const Tensor* float32_input = context->Input<Tensor>(1);

    auto float16_elements = float16_input->DataAsSpan<MLFloat16>();
    auto float32_elements = float32_input->DataAsSpan<float>();

    if (float16_elements.size() != float32_elements.size()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "float16_input and float32_input need to have the same number of elements.",
                             " float16_input: ", float16_elements.size(), " elements",
                             " float32_input: ", float32_elements.size(), " elements");
    }

    constexpr float relative_tolerance = 1e-2f;
    constexpr float absolute_tolerance = 1e-2f;
    float biggest_absolute_error = 0;
    float biggest_relative_error = 0;
    float biggest_float16_value = 0;
    float biggest_float32_value = 0;
    bool has_tolerance_error = false;

    for (int i = 0; i < float16_elements.size(); ++i) {
      float float16_element = static_cast<float>(float16_elements[i]);
      float float32_element = float32_elements[i];
      float relative_error = std::abs((float16_element - float32_element) / float32_element);
      float absolute_error = std::abs(float16_element - float32_element);

      if (relative_error > relative_tolerance && absolute_error > absolute_tolerance) {
        if (absolute_error > biggest_absolute_error) {
          biggest_float16_value = float16_element;
          biggest_float32_value = float32_element;
        }

        biggest_relative_error = std::max(biggest_relative_error, std::abs((biggest_float16_value - biggest_float32_value) / biggest_float32_value));
        biggest_absolute_error = std::max(biggest_absolute_error, std::abs(biggest_float16_value - biggest_float32_value));
        has_tolerance_error = true;
      }
    }

    if (has_tolerance_error) {
      printf("%s (%s) produced values with relative errors higher than relative tolerance %f and absolute tolerance %f for output %lld on %s. Biggest relative error: %f. Biggest absolute error: %f. float16 value: %f, float32 value: %f\n",
             node_name_.c_str(),
             node_type_.c_str(),
             relative_tolerance,
             absolute_tolerance,
             node_output_index_,
             execution_provider_.c_str(),
             biggest_relative_error,
             biggest_absolute_error,
             biggest_float16_value,
             biggest_float32_value);
    }

    Tensor* Y = context->Output(0, float16_input->Shape());
    CopyCpuTensor(float16_input, Y);

    return Status::OK();
  }

 private:
  std::string node_name_;
  std::string node_type_;
  std::string execution_provider_;
  int64_t node_output_index_;
};

ONNX_CPU_OPERATOR_MS_KERNEL(
    PrintToleranceErrors,
    1,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    PrintToleranceErrors);
}  // namespace contrib
}  // namespace onnxruntime

#endif
