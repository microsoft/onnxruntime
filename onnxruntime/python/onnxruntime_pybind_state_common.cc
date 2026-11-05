#include "onnxruntime_pybind_exceptions.h"
#include "onnxruntime_pybind_state_common.h"

#include "core/framework/arena_extend_strategy.h"

namespace onnxruntime {
namespace python {
namespace py = pybind11;

const std::string onnxruntime::python::SessionObjectInitializer::default_logger_id = "Default";

#if defined(USE_OPENVINO) || defined(USE_OPENVINO_PROVIDER_INTERFACE)
// TODO remove deprecated global config
std::string openvino_device_type;
#endif

// TODO remove deprecated global config
OrtDevice::DeviceId cuda_device_id = 0;
// TODO remove deprecated global config
size_t gpu_mem_limit = std::numeric_limits<size_t>::max();

#if defined(USE_CUDA) || defined(USE_CUDA_PROVIDER_INTERFACE)
// TODO remove deprecated global config
OrtCudnnConvAlgoSearch cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
// TODO remove deprecated global config
bool do_copy_in_default_stream = true;
// TODO remove deprecated global config
onnxruntime::cuda::TunableOpInfo tunable_op{};
onnxruntime::CUDAExecutionProviderExternalAllocatorInfo external_allocator_info{};
#endif

#if defined(USE_MIGRAPHX) || defined(USE_CUDA) || defined(USE_CUDA_PROVIDER_INTERFACE)
// TODO remove deprecated global config
onnxruntime::ArenaExtendStrategy arena_extend_strategy = onnxruntime::ArenaExtendStrategy::kNextPowerOfTwo;
#endif

#ifdef USE_MIGRAPHX
namespace migraphx::external {
void* alloc_fn{nullptr};
void* free_fn{nullptr};
void* empty_cache_fn{nullptr};
}  // namespace migraphx::external
#endif

#if defined(ENABLE_DLPACK)

void DlpackCapsuleDestructor(PyObject* data) {
  // Do not disturb a pending exception while releasing an unconsumed capsule.
  if (PyCapsule_IsValid(data, "dltensor")) {
    auto* tensor = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(data, "dltensor"));
    tensor->deleter(tensor);
  }
}

namespace {
void SynchronizeDlpackDevice(const DLDevice& device) {
  if (device.device_type != kDLCUDA) {
    return;
  }
#if defined(USE_CUDA) || defined(USE_CUDA_PROVIDER_INTERFACE)
  if (auto* provider = TryGetProviderInfo_CUDA()) {
    ORT_THROW_IF_ERROR(provider->SynchronizeDevice(device.device_id));
    return;
  }
#endif
  throw py::buffer_error(
      "CUDA DLPack interchange requires an available CUDA provider to synchronize the tensor's device. "
      "Install a CUDA-enabled ONNX Runtime package and load its CUDA provider. For an already synchronized "
      "export, call __dlpack__(stream=-1) explicitly.");
}
}  // namespace

PyObject* ToDlpack(OrtValue& ort_value, py::object stream) {
  const auto device = dlpack::GetDlpackDevice(ort_value, ort_value.Get<Tensor>().Location().device.Id());
  bool synchronize = true;
  if (!stream.is_none()) {
    if (device.device_type != kDLCUDA) {
      throw py::buffer_error("CPU DLPack exports require stream=None; omit the stream argument.");
    }
    if (!PyLong_Check(stream.ptr()) || PyBool_Check(stream.ptr())) {
      throw py::type_error("A CUDA DLPack stream must be an integer: -1, 1, 2, or a positive stream handle.");
    }
    const auto stream_id = PyLong_AsLongLong(stream.ptr());
    if (PyErr_Occurred()) {
      throw py::error_already_set();
    }
    if (stream_id == 0 || stream_id < -1) {
      throw py::value_error(
          "Invalid CUDA DLPack stream: use 1 for the legacy default, 2 for the per-thread default, "
          "a positive stream handle, or -1 to opt out of synchronization.");
    }
    synchronize = stream_id != -1;
  }
  // OrtValue does not record its producer stream, so a device-wide fence is
  // necessary before handing its storage to any consumer stream.
  if (synchronize) {
    SynchronizeDlpackDevice(device);
  }
  DLManagedTensor* dlmanaged_tensor = dlpack::OrtValueToDlpack(ort_value);
  PyObject* capsule = PyCapsule_New(dlmanaged_tensor, "dltensor", DlpackCapsuleDestructor);
  if (!capsule) {
    dlmanaged_tensor->deleter(dlmanaged_tensor);
    throw py::error_already_set();
  }
  return capsule;
}

OrtValue FromDlpack(py::object data) {
  if (!PyCapsule_CheckExact(data.ptr())) {
    auto method = py::reinterpret_steal<py::object>(PyObject_GetAttrString(data.ptr(), "__dlpack__"));
    if (method) {
      data = method();
    } else if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
      PyErr_Clear();
    } else {
      throw py::error_already_set();
    }
  }
  PyObject* dlpack_tensor = data.ptr();
  if (!PyCapsule_IsValid(dlpack_tensor, "dltensor")) {
    throw py::type_error("from_dlpack expected a valid DLPack capsule named 'dltensor'");
  }

  // Extract DLPack tensor pointer from the capsule carrier.
  DLManagedTensor* dlmanaged_tensor = (DLManagedTensor*)PyCapsule_GetPointer(dlpack_tensor, "dltensor");
  // The producer's default-stream handoff alone is insufficient for ORT's
  // session-specific streams. Fence before importing both objects and raw capsules.
  SynchronizeDlpackDevice(dlmanaged_tensor->dl_tensor.device);
  OrtValue ort_value = dlpack::DlpackToOrtValue(dlmanaged_tensor);
  // Make sure this capsule will never be used again.
  PyCapsule_SetName(dlpack_tensor, "used_dltensor");
  return ort_value;
}

#endif

#if !defined(DISABLE_SPARSE_TENSORS)
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
#endif  // !defined(DISABLE_SPARSE_TENSORS)

}  // namespace python
}  // namespace onnxruntime
