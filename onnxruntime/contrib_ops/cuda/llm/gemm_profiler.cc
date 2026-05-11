#if USE_FPA_INTB_GEMM
#include "contrib_ops/cuda/llm/gemm_profiler.h"
#include "contrib_ops/cuda/llm/common/logger.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm/fpA_intB_gemm.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

#include <cstddef>

namespace onnxruntime::llm::kernels::weight_only {

// Explicit instantiation for existing use case
template class GemmPluginProfiler<onnxruntime::llm::cutlass_extensions::CutlassGemmConfig,
                                  std::shared_ptr<onnxruntime::llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface>, GemmIdCore,
                                  GemmIdCoreHash>;

}  // namespace onnxruntime::llm::kernels::weight_only
#endif
