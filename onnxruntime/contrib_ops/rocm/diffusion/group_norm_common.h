// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using onnxruntime::rocm::CeilDiv;

int32_t findMaxDivisor(int32_t n, int32_t maxAllowedDivisor) {
  int32_t maxDivisor = -1;
  for (int32_t i = 1; i <= std::sqrt(n); i++) {
    if (n % i == 0) {
      int32_t divisor1 = n / i;
      int32_t divisor2 = i;

      if (divisor1 > maxDivisor && divisor1 < maxAllowedDivisor) {
        maxDivisor = divisor1;
      }
      if (divisor2 > maxDivisor && divisor2 < maxAllowedDivisor) {
        maxDivisor = divisor2;
      }
    }
  }
  return maxDivisor;
}

template <typename T>
struct GroupNormNHWCParams : OpParams {
  GroupNormNHWCParams(RocmTuningContext* tuning_ctx, onnxruntime::Stream* stream, T* dst, float* redBuffer, const T* src, const float* gamma,
                      const float* beta, int32_t n, int32_t h, int32_t w, int32_t c, int32_t groups, float epsilon, bool withSwish)
      : OpParams(tuning_ctx, stream), dst(dst), src(src), gamma(gamma), beta(beta), redBuffer(redBuffer), epsilon(epsilon), n(n), h(h), w(w), c(c), groups(groups), withSwish(withSwish) {
    int32_t maxBlocksPerHW = 1024;
    switch (c) {
      case 960:
      case 1920:
        cPerBlock = 480;
        break;
      case 512:
      case 256:
        cPerBlock = 256;
        break;
      case 128:
        cPerBlock = 128;
        break;
      default:
        cPerBlock = 320;
    }

    hw = h * w;
    const int32_t blocksPerHW = findMaxDivisor(hw, maxBlocksPerHW);
    hwPerBlock = CeilDiv(hw, blocksPerHW);
    cPerGroup = c / groups;
    hwc = hw * c;
    invHWC = 1.F / (float)(hw * cPerGroup);
    groupsPerBlock = cPerBlock / cPerGroup;
  }

  std::string Signature() const override {
    std::string swish_suffix = withSwish ? "_Swish" : "_Pass";
    std::string sig = std::to_string(n) + "_" + std::to_string(h * w) + "_" + std::to_string(c) + "_" + std::to_string(groups) + swish_suffix;
    return sig;
  }

  // The output buffer. Layout NHWC.
  T* dst;
  // The input buffer. Layout NHWC.
  T const* src;
  // The gamma scaling factor.
  float const* gamma;
  // The beta term to add in GN.
  float const* beta;
  // The temporary buffer to do the global parallel reduction. Size:
  // BLOCKS_PER_BATCH x C x 2.
  float* redBuffer;
  float epsilon;

  // The number of instances in the batch.
  int32_t n;
  // The height and width of each activation map.
  int32_t h;
  int32_t w;
  // The number of channels.
  int32_t c;
  // The number of groups.
  int32_t groups;
  // Do we apply the Swish activation function?
  bool withSwish;

  // Precomputed values and parameters to control the execution of the kernels.

  // The number of activations per instance (h * w) and the number of
  // activations per block.
  int32_t hw;
  int32_t hwPerBlock;
  // The number of channels per group and blocks per activation in the C
  // dimension.
  int32_t cPerBlock;
  int32_t cPerGroup;

  // The precomputed stride between instances.
  int32_t hwc;
  // The inverse of hwc in floats (to compute mean/var).
  float invHWC;
  // The precomputed number of groups per block.
  int32_t groupsPerBlock;
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
