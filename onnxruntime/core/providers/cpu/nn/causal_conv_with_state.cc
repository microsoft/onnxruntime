// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/platform/threadpool.h"
#include "core/providers/causal_conv_with_state_helper.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

template <typename T>
class CausalConvWithState final : public OpKernel {
 public:
  explicit CausalConvWithState(const OpKernelInfo& info)
      : OpKernel(info),
        dilation_(info.GetAttrOrDefault<int64_t>("dilation", 1)),
        activation_(info.GetAttrOrDefault<std::string>("activation", "none")) {
    ORT_ENFORCE(activation_ == "none" || activation_ == "silu" || activation_ == "swish",
                "CausalConvWithState activation '", activation_,
                "' is unsupported. Use 'none', 'silu', or 'swish'.");
  }

  Status Compute(OpKernelContext* context) const override {
    const auto& input = context->RequiredInput<Tensor>(0);
    const auto& weight = context->RequiredInput<Tensor>(1);
    const auto* bias = context->Input<Tensor>(2);
    const auto* past = context->Input<Tensor>(3);
    CausalConvWithStateParameters p;
    ORT_RETURN_IF_ERROR(PrepareCausalConvWithState(input, weight, bias, past, dilation_, p));
    auto& output = context->RequiredOutput(0, input.Shape());
    auto& present = context->RequiredOutput(1, p.state_shape);
    if (p.batch == 0 || p.channels == 0) return Status::OK();

    const T* x = input.Data<T>();
    const T* w = weight.Data<T>();
    const T* b = bias ? bias->Data<T>() : nullptr;
    const T* previous = past ? past->Data<T>() : nullptr;
    T* y = output.MutableData<T>();
    T* next = present.MutableData<T>();
    const int64_t rows = SafeInt<int64_t>(p.batch) * p.channels;
    const auto length = narrow<Eigen::Index>(p.length);
    const auto padded_length = narrow<Eigen::Index>(SafeInt<int64_t>(p.history) + p.length);
    concurrency::ThreadPool::TryParallelFor(
        context->GetOperatorThreadPool(), narrow<std::ptrdiff_t>(rows),
        static_cast<double>(p.length) * static_cast<double>(p.kernel),
        [&](std::ptrdiff_t first, std::ptrdiff_t last) {
          Eigen::ArrayXf padded(length > 1 ? padded_length : 0), sums(length > 1 ? length : 0);
          for (auto row = first; row < last; ++row) {
            // State keeps raw chronological samples, including the taps skipped by dilation.
            for (int64_t s = 0; s < p.history; ++s) {
              const int64_t source = p.length + s;
              next[row * p.history + s] = source < p.history
                                              ? (previous ? previous[row * p.history + source] : T{})
                                              : x[row * p.length + source - p.history];
            }
            if (p.length == 0) continue;
            const int64_t channel = row % p.channels;
            if (p.length == 1) {
              float sum = 0;
              for (int64_t tap = 0; tap < p.kernel - 1; ++tap) {
                const float sample = previous ? static_cast<float>(previous[row * p.history + tap * dilation_]) : 0.0f;
                sum += sample * static_cast<float>(w[channel * p.kernel + tap]);
              }
              sum += static_cast<float>(x[row]) * static_cast<float>(w[channel * p.kernel + p.kernel - 1]);
              if (b) sum += static_cast<float>(b[channel]);
              const T convolution = static_cast<T>(sum);
              const float value = static_cast<float>(convolution);
              y[row] = activation_ == "none" ? convolution : static_cast<T>(value / (1.0f + std::exp(-value)));
              continue;
            }
            for (int64_t s = 0; s < p.history; ++s) {
              padded[s] = previous ? static_cast<float>(previous[row * p.history + s]) : 0.0f;
            }
            for (int64_t t = 0; t < p.length; ++t) {
              padded[p.history + t] = static_cast<float>(x[row * p.length + t]);
            }
            // Vectorize over contiguous positions; no per-output history/input branch.
            sums.setZero();
            for (int64_t tap = 0; tap < p.kernel; ++tap) {
              sums += padded.segment(tap * dilation_, length) * static_cast<float>(w[channel * p.kernel + tap]);
            }
            if (b) sums += static_cast<float>(b[channel]);
            for (int64_t t = 0; t < p.length; ++t) {
              // ONNX Conv produces T before the float32 SiLU function body.
              const T convolution = static_cast<T>(sums[t]);
              const float value = static_cast<float>(convolution);
              y[row * p.length + t] = activation_ == "none"
                                          ? convolution
                                          : static_cast<T>(value / (1.0f + std::exp(-value)));
            }
          }
        });
    return Status::OK();
  }

 private:
  const int64_t dilation_;
  const std::string activation_;
};

#define REGISTER_CAUSAL_CONV29(T)                                                                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(CausalConvWithState, kOnnxDomain, 29, T,                                  \
                                kCpuExecutionProvider,                                                    \
                                KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
                                CausalConvWithState<T>);

REGISTER_CAUSAL_CONV29(float)
REGISTER_CAUSAL_CONV29(MLFloat16)
REGISTER_CAUSAL_CONV29(BFloat16)
#undef REGISTER_CAUSAL_CONV29

}  // namespace onnxruntime
