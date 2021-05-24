#include "onnxruntime_pybind_exceptions.h"
#include "onnxruntime_pybind_state_common.h"

#include "core/framework/arena_extend_strategy.h"

namespace onnxruntime {
namespace python {
namespace py = pybind11;

const std::string onnxruntime::python::SessionObjectInitializer::default_logger_id = "Default";

#ifdef USE_OPENVINO
#include "core/providers/openvino/openvino_provider_factory.h"
// TODO remove deprecated global config
std::string openvino_device_type;
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
const onnxruntime::ROCMExecutionProviderExternalAllocatorInfo external_allocator_info{};
#endif

// TODO remove deprecated global config
onnxruntime::ArenaExtendStrategy arena_extend_strategy = onnxruntime::ArenaExtendStrategy::kNextPowerOfTwo;
#endif


void ThrowIfPyErrOccured() {
  if (PyErr_Occurred()) {
    PyObject *ptype, *pvalue, *ptraceback;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);

    PyObject* pStr = PyObject_Str(ptype);
    std::string sType = py::reinterpret_borrow<py::str>(pStr);
    Py_XDECREF(pStr);
    pStr = PyObject_Str(pvalue);
    sType += ": ";
    sType += py::reinterpret_borrow<py::str>(pStr);
    Py_XDECREF(pStr);
    throw Fail(sType);
  }
}

void RegisterExceptions(pybind11::module& m) {
  pybind11::register_exception<Fail>(m, "Fail");
  pybind11::register_exception<InvalidArgument>(m, "InvalidArgument");
  pybind11::register_exception<NoSuchFile>(m, "NoSuchFile");
  pybind11::register_exception<NoModel>(m, "NoModel");
  pybind11::register_exception<EngineError>(m, "EngineError");
  pybind11::register_exception<RuntimeException>(m, "RuntimeException");
  pybind11::register_exception<InvalidProtobuf>(m, "InvalidProtobuf");
  pybind11::register_exception<ModelLoaded>(m, "ModelLoaded");
  pybind11::register_exception<NotImplemented>(m, "NotImplemented");
  pybind11::register_exception<InvalidGraph>(m, "InvalidGraph");
  pybind11::register_exception<EPFail>(m, "EPFail");
}

void OrtPybindThrowIfError(onnxruntime::common::Status status) {
  std::string msg = status.ToString();
  if (!status.IsOK()) {
    switch (status.Code()) {
      case onnxruntime::common::StatusCode::FAIL:
        throw Fail(std::move(msg));
      case onnxruntime::common::StatusCode::INVALID_ARGUMENT:
        throw InvalidArgument(std::move(msg));
      case onnxruntime::common::StatusCode::NO_SUCHFILE:
        throw NoSuchFile(std::move(msg));
      case onnxruntime::common::StatusCode::NO_MODEL:
        throw NoModel(std::move(msg));
      case onnxruntime::common::StatusCode::ENGINE_ERROR:
        throw EngineError(std::move(msg));
      case onnxruntime::common::StatusCode::RUNTIME_EXCEPTION:
        throw RuntimeException(std::move(msg));
      case onnxruntime::common::StatusCode::INVALID_PROTOBUF:
        throw InvalidProtobuf(std::move(msg));
      case onnxruntime::common::StatusCode::NOT_IMPLEMENTED:
        throw NotImplemented(std::move(msg));
      case onnxruntime::common::StatusCode::INVALID_GRAPH:
        throw InvalidGraph(std::move(msg));
      case onnxruntime::common::StatusCode::EP_FAIL:
        throw EPFail(std::move(msg));
      default:
        throw std::runtime_error(std::move(msg));
    }
  }
}

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
