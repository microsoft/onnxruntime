// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include "contrib_ops/rocm/bert/skip_layer_norm_impl_kernel.h"
#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/tunable/rocm_tunable.h"

using onnxruntime::rocm::CeilDiv;

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
struct SkipLayerNormParams : OpParams {
  SkipLayerNormParams(RocmTuningContext* tuning_ctx, hipStream_t stream, T* output, T* skip_input_bias_add_output, const T* input,
                      const T* skip, const T* gamma, const T* beta,
                      const T* bias, float epsilon, int ld, int element_count)
      : OpParams(tuning_ctx, stream), output(output), skip_input_bias_add_output(skip_input_bias_add_output), input(input), skip(skip),
        gamma(gamma), beta(beta), bias(bias), epsilon(epsilon), ld(ld), element_count(element_count) {}

  std::string Signature() const override {
    std::string sig = std::to_string(ld) + "_" + std::to_string(element_count);
    return sig;
  }

  T* output;
  T* skip_input_bias_add_output;
  const T* input;
  const T* skip;
  const T* gamma;
  const T* beta;
  const T* bias;
  float epsilon;
  int ld;
  int element_count;
};

template <typename T, int ThreadsPerBlock, int VecSize>
Status SkipLayerNormSmallOp(const SkipLayerNormParams<T>* params) {
  // Loosen the hard constraint for ld (hidden_size) to include more possible *Small kernels,
  // which could offer better performance in some combinations of ThreadsPerBlock and VecSize.
  TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
      !((params->ld <= 8192 && params->ld % VecSize == 0 &&
         params->ld <= ThreadsPerBlock * VecSize && params->ld > (ThreadsPerBlock - GPU_WARP_SIZE) * VecSize)));
  SkipLayerNormKernelSmall<T, ThreadsPerBlock, VecSize><<<dim3(CeilDiv(params->element_count, params->ld)),
                                                          dim3(ThreadsPerBlock),
                                                          0, params->stream>>>(
      params->ld, params->input, params->skip,
      params->beta, params->gamma, params->bias, maybe2half<T>(params->epsilon), params->output, params->skip_input_bias_add_output,
      (params->bias == nullptr) ? false : true, (params->skip_input_bias_add_output == nullptr) ? false : true);
  return HIP_CALL(hipGetLastError());
}

template <typename T, int ThreadsPerBlock, int VecSize>
Status SkipLayerNormRegularOp(const SkipLayerNormParams<T>* params) {
  TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
      !((params->ld > 0 && params->ld % VecSize == 0 &&
         (params->ld >= ThreadsPerBlock * VecSize ||
          (params->ld < GPU_WARP_SIZE && params->ld > (ThreadsPerBlock - GPU_WARP_SIZE) * VecSize)))));
  SkipLayerNormKernelVec<T, ThreadsPerBlock, VecSize><<<dim3(CeilDiv(params->element_count, params->ld)),
                                                        dim3(ThreadsPerBlock),
                                                        0, params->stream>>>(
      params->ld, params->input, params->skip,
      params->beta, params->gamma, params->bias, maybe2half<T>(params->epsilon), params->output, params->skip_input_bias_add_output,
      (params->bias == nullptr) ? false : true, (params->skip_input_bias_add_output == nullptr) ? false : true);
  return HIP_CALL(hipGetLastError());
}

template <typename T>
Status SkipLayerNormStaticSelection(const SkipLayerNormParams<T>* params) {
  bool hasBias = (params->bias == nullptr) ? false : true;
  bool hasSkipInputBiasAdditionOutput = (params->skip_input_bias_add_output == nullptr) ? false : true;
  const int grid_size = params->element_count / params->ld;
  const int block_size = 256;

#define LAUNCH_SKIPLAYERNORM_SMALL_FORWARD(ELEMENTS, TPB, ILP)                              \
  if (params->ld <= ELEMENTS) {                                                             \
    SkipLayerNormKernelSmall<T, TPB, ILP><<<grid_size, TPB, 0, params->stream>>>(           \
        params->ld, params->input, params->skip, params->beta, params->gamma, params->bias, \
        maybe2half<T>(params->epsilon), params->output, params->skip_input_bias_add_output, \
        hasBias, hasSkipInputBiasAdditionOutput);                                           \
    break;                                                                                  \
  }
  if (0 == (params->ld % 4)) {
    do {
      LAUNCH_SKIPLAYERNORM_SMALL_FORWARD(32, 32, 1)
      LAUNCH_SKIPLAYERNORM_SMALL_FORWARD(64, 32, 2)
      LAUNCH_SKIPLAYERNORM_SMALL_FORWARD(128, 32, 4)
      LAUNCH_SKIPLAYERNORM_SMALL_FORWARD(384, 96, 4)
      LAUNCH_SKIPLAYERNORM_SMALL_FORWARD(768, 192, 4)
      LAUNCH_SKIPLAYERNORM_SMALL_FORWARD(1024, 256, 4)

      SkipLayerNormKernel<T, block_size><<<grid_size, block_size, 0, params->stream>>>(
          params->ld, params->input, params->skip, params->beta, params->gamma, params->bias,
          maybe2half<T>(params->epsilon), params->output, params->skip_input_bias_add_output);
    } while (0);
  } else {
    do {
      LAUNCH_SKIPLAYERNORM_SMALL_FORWARD(32, 32, 1)
      LAUNCH_SKIPLAYERNORM_SMALL_FORWARD(64, 64, 1)
      LAUNCH_SKIPLAYERNORM_SMALL_FORWARD(128, 128, 1)
      LAUNCH_SKIPLAYERNORM_SMALL_FORWARD(384, 384, 1)

      SkipLayerNormKernel<T, block_size><<<grid_size, block_size, 0, params->stream>>>(
          params->ld, params->input, params->skip, params->beta, params->gamma, params->bias,
          maybe2half<T>(params->epsilon), params->output, params->skip_input_bias_add_output);
    } while (0);
  }
  return HIP_CALL(hipPeekAtLastError());
}  // namespace rocm

#define ADD_OP_FOR_ALL_VEC_SIZE(name, threads_per_block) \
  this->RegisterOp(name<T, threads_per_block, 1>);       \
  this->RegisterOp(name<T, threads_per_block, 2>);       \
  this->RegisterOp(name<T, threads_per_block, 4>);       \
  this->RegisterOp(name<T, threads_per_block, 8>);       \
  this->RegisterOp(name<T, threads_per_block, 16>);

#define ADD_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(name) \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 64)                         \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 128)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 192)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 256)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 320)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 384)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 448)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 512)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 576)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 640)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 704)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 768)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 832)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 896)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 1024)

template <typename T>
class SkipLayerNormTunableOp : public TunableOp<SkipLayerNormParams<T>> {
 public:
  SkipLayerNormTunableOp() {
    this->RegisterOp(SkipLayerNormStaticSelection<T>);
    ADD_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(SkipLayerNormSmallOp)
    ADD_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(SkipLayerNormRegularOp)

    // NOTE: the 1st kernel is SkipLayerNorm Original implementation.
    this->SetDefaultId(0);
  }
};

#undef ADD_OP_FOR_ALL_VEC_SIZE
#undef ADD_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
