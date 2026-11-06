#include <exception>

#include "onnxruntime_pybind_exceptions.h"
#include "onnxruntime_pybind_state_common.h"

#include "core/framework/arena_extend_strategy.h"

namespace onnxruntime {
namespace python {
namespace py = pybind11;

const std::string onnxruntime::python::SessionObjectInitializer::default_logger_id = "Default";

void ReleasePythonObject(PyObject* object) noexcept {
  // After interpreter shutdown, retaining the reference is safer than touching Python.
  if (object == nullptr || !Py_IsInitialized()) {
    return;
  }
  if (PyGILState_Check()) {
    Py_DECREF(object);
  } else {
    ReleasePythonObjectOnCleanupThread(object);
  }
}

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
#if !defined(ORT_MINIMAL_BUILD)
Environment::DeviceInterop GetPluginDeviceInterop(int device_type, int device_id,
                                                  Environment::DeviceInteropCapability capability) {
  if (device_type != kDLCUDA && device_type != kDLROCM) {
    return {};
  }
  const auto vendor = device_type == kDLCUDA ? OrtDevice::VendorIds::NVIDIA : OrtDevice::VendorIds::AMD;
  return GetOrtEnv()->GetEnvironment().GetDeviceInterop(
      OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, vendor, gsl::narrow<OrtDevice::DeviceId>(device_id)),
      capability);
}
#endif

template <typename SetDevice>
py::object ExportDlpackOnDevice(const py::object& method, bool no_sync, int device_id,
                                int original_device, SetDevice set_device) {
  if (original_device == device_id) {
    return no_sync ? method(py::arg("stream") = -1) : method();
  }
  set_device(device_id);
  py::object capsule;
  std::exception_ptr error;
  ORT_TRY {
    capsule = no_sync ? method(py::arg("stream") = -1) : method();
  }
  ORT_CATCH(...) {
    error = std::current_exception();
  }
  // Restore on both success and producer failure, and surface restoration errors.
  set_device(original_device);
  if (error) {
    std::rethrow_exception(error);
  }
  return capsule;
}

py::object ExportDlpack(const py::object& method, bool no_sync, const std::pair<int, int>& device) {
#if !defined(ORT_MINIMAL_BUILD)
  if (auto interop = GetPluginDeviceInterop(device.first, device.second,
                                            Environment::DeviceInteropCapability::CurrentDevice);
      interop.factory != nullptr) {
    auto* factory = interop.factory;
    int original_device;
    Ort::ThrowOnError(factory->GetCurrentDevice(factory, &original_device));
    return ExportDlpackOnDevice(method, no_sync, device.second, original_device,
                                [factory](int id) { Ort::ThrowOnError(factory->SetCurrentDevice(factory, id)); });
  }
#endif
  if (device.first == kDLCUDA) {
#if defined(USE_CUDA) || defined(USE_CUDA_PROVIDER_INTERFACE)
    if (auto* provider = TryGetProviderInfo_CUDA()) {
      int original_device;
      Ort::ThrowOnError(provider->GetCurrentGpuDeviceId(&original_device));
      return ExportDlpackOnDevice(method, no_sync, device.second, original_device,
                                  [provider](int id) { Ort::ThrowOnError(provider->SetCurrentGpuDeviceId(id)); });
    }
#endif
  } else if (device.first == kDLROCM) {
#if defined(USE_MIGRAPHX) || defined(USE_MIGRAPHX_PROVIDER_INTERFACE)
    if (auto* provider = TryGetProviderInfo_MIGraphX()) {
      return ExportDlpackOnDevice(method, no_sync, device.second, provider->GetCurrentGpuDeviceId(),
                                  [provider](int id) { provider->SetCurrentGpuDeviceId(id); });
    }
#endif
  }
  return no_sync ? method(py::arg("stream") = -1) : method();
}

py::object GetOptionalDlpackAttribute(const py::object& data, const char* name) {
  auto attribute = py::reinterpret_steal<py::object>(PyObject_GetAttrString(data.ptr(), name));
  if (attribute) {
    return attribute;
  }
  py::error_already_set error;
  if (!error.matches(PyExc_AttributeError)) {
    throw error;
  }
  // AttributeError from an existing descriptor is a producer failure, not an
  // absent protocol method. Static lookup does not invoke that descriptor again.
  auto declared = py::module_::import("inspect").attr("getattr_static")(data, name, py::none());
  if (!declared.is_none()) {
    throw error;
  }
  return py::none();
}

void SynchronizeDlpackDevice(const DLDevice& device) {
  if (device.device_type == kDLCPU) {
    return;
  }
#if !defined(ORT_MINIMAL_BUILD)
  if (auto interop = GetPluginDeviceInterop(device.device_type, device.device_id,
                                            Environment::DeviceInteropCapability::Synchronize);
      interop.factory != nullptr) {
    py::gil_scoped_release release;
    Ort::ThrowOnError(interop.factory->SynchronizeDevice(interop.factory, device.device_id));
    return;
  }
#endif
  if (device.device_type == kDLCUDA) {
#if defined(USE_CUDA) || defined(USE_CUDA_PROVIDER_INTERFACE)
    if (auto* provider = TryGetProviderInfo_CUDA()) {
      py::gil_scoped_release release;
      ORT_THROW_IF_ERROR(provider->SynchronizeDevice(device.device_id));
      return;
    }
#endif
  } else if (device.device_type == kDLROCM) {
#if defined(USE_MIGRAPHX) || defined(USE_MIGRAPHX_PROVIDER_INTERFACE)
    if (auto* provider = TryGetProviderInfo_MIGraphX()) {
      py::gil_scoped_release release;
      ORT_THROW_IF_ERROR(provider->SynchronizeDevice(device.device_id));
      return;
    }
#endif
  }
  throw py::buffer_error(
      "DLPack synchronization requires device-wide synchronization from a matching provider "
      "(CUDA for NVIDIA, MIGraphX for AMD). Install and load a provider implementing this capability, "
      "or explicitly opt out only after arranging stream ordering: "
      "__dlpack__(stream=-1), from_dlpack(..., synchronize=False), or to_dlpacks(..., synchronize=False).");
}
}  // namespace

PyObject* ToDlpack(OrtValue& ort_value, py::object stream) {
  OrtValue owner = ort_value;
  ORT_ENFORCE(owner.IsTensor(), "DLPack export requires a dense tensor OrtValue; initialize the value before exporting.");
  const auto device = dlpack::GetDlpackDevice(owner, owner.Get<Tensor>().Location().device.Id());
  bool synchronize = true;
  if (!stream.is_none()) {
    if (device.device_type != kDLCUDA && device.device_type != kDLROCM) {
      throw py::buffer_error("CPU DLPack exports require stream=None; omit the stream argument.");
    }
    if (!PyLong_Check(stream.ptr()) || PyBool_Check(stream.ptr())) {
      throw py::type_error("A GPU DLPack stream must be an integer: -1, a default-stream identifier, or a stream handle.");
    }
    const auto stream_id = PyLong_AsLongLong(stream.ptr());
    if (PyErr_Occurred()) {
      throw py::error_already_set();
    }
    if ((device.device_type == kDLCUDA && stream_id == 0) || stream_id < -1) {
      throw py::value_error(
          "Invalid GPU DLPack stream: CUDA accepts 1 or 2 for default streams; ROCm accepts 0 or 1. "
          "Use a positive stream handle, or -1 to opt out of synchronization.");
    }
    synchronize = stream_id != -1;
  }
  // OrtValue does not record its producer stream, so a device-wide fence is
  // necessary before handing its storage to any consumer stream.
  if (synchronize) {
    SynchronizeDlpackDevice(device);
  }
  DLManagedTensor* dlmanaged_tensor = dlpack::OrtValueToDlpack(owner);
  PyObject* capsule = PyCapsule_New(dlmanaged_tensor, "dltensor", DlpackCapsuleDestructor);
  if (!capsule) {
    dlmanaged_tensor->deleter(dlmanaged_tensor);
    throw py::error_already_set();
  }
  return capsule;
}

OrtValue FromDlpack(py::object data, bool synchronize) {
  if (!PyCapsule_CheckExact(data.ptr())) {
    auto method = GetOptionalDlpackAttribute(data, "__dlpack__");
    if (!method.is_none()) {
      auto device_method = GetOptionalDlpackAttribute(data, "__dlpack_device__");
      if (!device_method.is_none()) {
        const auto device = device_method().cast<std::pair<int, int>>();
        const bool no_sync = !synchronize && (device.first == kDLCUDA || device.first == kDLROCM);
        data = ExportDlpack(method, no_sync, device);
      } else {
        data = method();
      }
    }
  }
  PyObject* dlpack_tensor = data.ptr();
  if (!PyCapsule_IsValid(dlpack_tensor, "dltensor")) {
    throw py::type_error("from_dlpack expected a valid DLPack capsule named 'dltensor'");
  }

  // Extract DLPack tensor pointer from the capsule carrier.
  DLManagedTensor* dlmanaged_tensor = (DLManagedTensor*)PyCapsule_GetPointer(dlpack_tensor, "dltensor");
  OrtValue ort_value = dlpack::DlpackToOrtValue(dlmanaged_tensor);
  // Claim exclusive ownership under the GIL before the native wait releases it.
  // The OrtValue also releases the producer if synchronization subsequently fails.
  PyCapsule_SetName(dlpack_tensor, "used_dltensor");
  // The producer's default-stream handoff alone is insufficient for ORT's
  // session-specific streams. Fence before importing both objects and raw capsules.
  if (synchronize) {
    SynchronizeDlpackDevice(dlmanaged_tensor->dl_tensor.device);
  }
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
