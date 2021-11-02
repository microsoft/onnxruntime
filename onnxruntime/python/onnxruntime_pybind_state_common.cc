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

#ifdef USE_CUDA
// TODO remove deprecated global config
OrtCudnnConvAlgoSearch cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
// TODO remove deprecated global config
bool do_copy_in_default_stream = true;
onnxruntime::CUDAExecutionProviderExternalAllocatorInfo external_allocator_info{};
// TODO remove deprecated global config
onnxruntime::ArenaExtendStrategy arena_extend_strategy = onnxruntime::ArenaExtendStrategy::kNextPowerOfTwo;
#endif

#ifdef USE_ROCM
// TODO remove deprecated global config
bool miopen_conv_exhaustive_search = false;
// TODO remove deprecated global config
bool do_copy_in_default_stream = true;
onnxruntime::ROCMExecutionProviderExternalAllocatorInfo external_allocator_info{};
// TODO remove deprecated global config
onnxruntime::ArenaExtendStrategy arena_extend_strategy = onnxruntime::ArenaExtendStrategy::kNextPowerOfTwo;
#endif

#ifdef ENABLE_TRAINING

static void DlpackCapsuleDestructor(PyObject* data) {
  DLManagedTensor* dlmanged_tensor = reinterpret_cast<DLManagedTensor*>(
      PyCapsule_GetPointer(data, "dltensor"));
  if (dlmanged_tensor) {
    // The dlmanged_tensor has not been consumed, call deleter ourselves.
    dlmanged_tensor->deleter(const_cast<DLManagedTensor*>(dlmanged_tensor));
  } else {
    // The dlmanged_tensor has been consumed,
    // PyCapsule_GetPointer has set an error indicator.
    PyErr_Clear();
  }
}

// Allocate a new Capsule object, which takes the ownership of OrtValue.
// Caller is responsible for releasing.
// This function calls OrtValueToDlpack(...).
PyObject* ToDlpack(OrtValue ort_value) {
  DLManagedTensor* dlmanaged_tensor = dlpack::OrtValueToDlpack(ort_value);
  return PyCapsule_New(dlmanaged_tensor, "dltensor", DlpackCapsuleDestructor);
}

// Consume a Capsule object and claims the ownership of its underlying tensor to
// create a OrtValue. This function calls DlpackToOrtValue(...) to do the conversion.
OrtValue FromDlpack(PyObject* dlpack_tensor, const bool is_bool_tensor) {
  // Extract DLPack tensor pointer from the capsule carrier.
  DLManagedTensor* dlmanaged_tensor = (DLManagedTensor*)PyCapsule_GetPointer(dlpack_tensor, "dltensor");
  OrtValue ort_value = dlpack::DlpackToOrtValue(dlmanaged_tensor, is_bool_tensor);
  // Make sure this capsule will never be used again.
  PyCapsule_SetName(dlpack_tensor, "used_dltensor");
  return ort_value;
}

#endif

std::unique_ptr<OrtValue> PySparseTensor::AsOrtValue() const {
  if (instance_) {
    auto ort_value = std::make_unique<OrtValue>();
    auto ml_type = DataTypeImpl::GetType<SparseTensor>();
    py::object this_object = py::cast(*this);
    // Create an std::function deleter that captures and ref-counts this PySparseTensor
    ort_value->Init(instance_.get(), ml_type, [object = std::move(this_object)](void*) {});
    return ort_value;
  }

  assert(ort_value_.IsAllocated());
  return std::make_unique<OrtValue>(ort_value_);
}

PySparseTensor::~PySparseTensor() {
  // pybind11 will deref and potentially destroy its objects
  // that we use to hold a reference and it may throw python errors
  // so we want to do it in a controlled manner
  auto None = py::none();
  for (auto& obj : backing_storage_) {
    try {
      obj = None;
    } catch (py::error_already_set& ex) {
      // we need it mutable to properly log and discard it
      ex.discard_as_unraisable(__func__);
    }
  }
}

}  // namespace python
}  // namespace onnxruntime
