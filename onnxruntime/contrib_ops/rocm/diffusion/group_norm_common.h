#pragma once

#include <string>
#include <vector>

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
struct GroupNormNHWCParams : OpParams {
  GroupNormNHWCParams(RocmTuningContext* tuning_ctx, hipStream_t stream, T* dst, const T* src, const float* gamma,
                      const float* beta, int32_t n, int32_t h, int32_t w, int32_t c, int32_t groups, bool withSwish)
      : OpParams(tuning_ctx, stream), dst(dst), src(src), gamma(gamma), beta(beta), n(n), h(h), w(w), c(c), groups(groups), withSwish(withSwish) {}

  std::string Signature() const override {
    std::string sig = std::to_string(n) + "_" + std::to_string(h) + "_" + std::to_string(w) + "_" + std::to_string(c) + "_" + std::to_string(groups);
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
