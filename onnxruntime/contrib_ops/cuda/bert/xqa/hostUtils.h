#pragma once
#include <cuda_runtime.h>

inline cudaLaunchConfig_t makeLaunchConfig(
    dim3 const& gridDim, dim3 const& ctaDim, size_t dynShmBytes, cudaStream_t stream, bool usePDL) {
  static cudaLaunchAttribute pdlAttr;
  pdlAttr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  pdlAttr.val.programmaticStreamSerializationAllowed = (usePDL ? 1 : 0);

  cudaLaunchConfig_t cfg{gridDim, ctaDim, dynShmBytes, stream, &pdlAttr, 1};
  return cfg;
}
