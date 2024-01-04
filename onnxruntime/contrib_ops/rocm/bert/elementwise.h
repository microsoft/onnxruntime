// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename Fn, typename T>
Status LaunchElementwiseKernel(RocmTuningContext* tuning_ctx, Stream* stream,
                               const T* input, int input_length,
                               const T* bias, int bias_length,
                               T* output);

// The following is LaunchElementwiseKernel implementation detail. Their interfaces are exposed for kernel explorer.
namespace internal {

template <typename T>
struct ElementwiseParams : OpParams {
  ElementwiseParams(RocmTuningContext* tuning_ctx, onnxruntime::Stream* stream,
                    const T* input, const T* bias, T* output, int input_length, int bias_length)
      : OpParams(tuning_ctx, stream),
        input(input),
        bias(bias),
        output(output),
        input_length(input_length),
        bias_length(bias_length) {}

  std::string Signature() const override {
    std::string sig = std::to_string(input_length) + "_" + std::to_string(bias_length);
    return sig;
  }

  const T* input;
  const T* bias;
  T* output;
  int input_length;
  int bias_length;
};

template <typename Fn, typename T, int ThreadsPerBlock, int VecSize>
class ElementwiseOp {
 public:
  Status operator()(const ElementwiseParams<T>* params);
  Status IsSupported(const ElementwiseParams<T>* params);
};

template <typename Fn, typename T>
Status ElementwiseStaticSelection(const ElementwiseParams<T>* params);

template <typename Fn, typename T>
class ElementwiseTunableOp : public TunableOp<ElementwiseParams<T>> {
 public:
  ElementwiseTunableOp();
};

}  // namespace internal

#define ELEMENTWISE_FWD_DECL(FnName, T) \
  namespace functor {                   \
  struct FnName;                        \
  }

ELEMENTWISE_FWD_DECL(FastGeLU, float);
ELEMENTWISE_FWD_DECL(FastGeLU, half);
ELEMENTWISE_FWD_DECL(FastGeLU, BFloat16);

ELEMENTWISE_FWD_DECL(GeLU, float);
ELEMENTWISE_FWD_DECL(GeLU, half);
ELEMENTWISE_FWD_DECL(GeLU, BFloat16);

ELEMENTWISE_FWD_DECL(ReLU, float);
ELEMENTWISE_FWD_DECL(ReLU, half);
ELEMENTWISE_FWD_DECL(ReLU, BFloat16);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
