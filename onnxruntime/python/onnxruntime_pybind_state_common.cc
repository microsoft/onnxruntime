#include "onnxruntime_pybind_exceptions.h"
#include "onnxruntime_pybind_state_common.h"

#include "core/framework/arena_extend_strategy.h"

namespace onnxruntime {
namespace python {
namespace py = pybind11;

const std::string onnxruntime::python::SessionObjectInitializer::default_logger_id = "Default";

#ifdef USE_OPENVINO
// TODO remove deprecated global config
std::string openvino_device_type;
#endif

#ifdef USE_NUPHAR
// TODO remove deprecated global config
std::string nuphar_settings;
#endif


// TODO remove deprecated global config
OrtDevice::DeviceId cuda_device_id = 0;
// TODO remove deprecated global config
size_t gpu_mem_limit = std::numeric_limits<size_t>::max();

#if defined(USE_CUDA) || defined(USE_ROCM)
#ifdef USE_CUDA
// TODO remove deprecated global config
OrtCudnnConvAlgoSearch cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE;
// TODO remove deprecated global config
bool do_copy_in_default_stream = true;
onnxruntime::CUDAExecutionProviderExternalAllocatorInfo external_allocator_info{};
#endif

#ifdef USE_ROCM
#include "core/providers/rocm/rocm_execution_provider.h"
#include "core/providers/rocm/rocm_allocator.h"
onnxruntime::ROCMExecutionProviderExternalAllocatorInfo external_allocator_info{};
#endif

// TODO remove deprecated global config
onnxruntime::ArenaExtendStrategy arena_extend_strategy = onnxruntime::ArenaExtendStrategy::kNextPowerOfTwo;
#endif

}  // namespace python
}  // namespace onnxruntime
