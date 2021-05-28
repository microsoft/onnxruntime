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

#ifdef ENABLE_TRAINING

void DlpackCapsuleDestructor(PyObject* data) {
  DLManagedTensor* dlmanged_tensor = (DLManagedTensor*)PyCapsule_GetPointer(data, "dltensor");
  if (dlmanged_tensor) {
    // the dlmanged_tensor has not been consumed, call deleter ourselves.
    dlmanged_tensor->deleter(const_cast<DLManagedTensor*>(dlmanged_tensor));
  } else {
    // the dlmanged_tensor has been consumed,
    // PyCapsule_GetPointer has set an error indicator.
    PyErr_Clear();
  }
}

py::object ToDlpack(OrtValue& ort_value) {
  DLManagedTensor* dlmanaged_tensor = dlpack::OrtValueToDlpack(ort_value);
  return py::reinterpret_steal<py::object>(
      PyCapsule_New(dlmanaged_tensor, "dltensor", DlpackCapsuleDestructor));
}

OrtValue FromDlpack(py::object dlpack_tensor, const bool is_bool_tensor) {
  DLManagedTensor* dlmanaged_tensor = (DLManagedTensor*)PyCapsule_GetPointer(dlpack_tensor.ptr(), "dltensor");
  OrtValue ort_value = dlpack::DlpackToOrtValue(dlmanaged_tensor, is_bool_tensor);
  // Make sure this capsule will never be used again.
  PyCapsule_SetName(dlpack_tensor.ptr(), "used_dltensor");
  return ort_value;
}

#endif

}  // namespace python
}  // namespace onnxruntime
