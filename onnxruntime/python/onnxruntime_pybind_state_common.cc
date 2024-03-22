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

// TODO remove deprecated global config
OrtDevice::DeviceId cuda_device_id = 0;
// TODO remove deprecated global config
size_t gpu_mem_limit = std::numeric_limits<size_t>::max();

#ifdef USE_CUDA
// TODO remove deprecated global config
OrtCudnnConvAlgoSearch cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
// TODO remove deprecated global config
bool do_copy_in_default_stream = true;
// TODO remove deprecated global config
onnxruntime::cuda::TunableOpInfo tunable_op{};
onnxruntime::CUDAExecutionProviderExternalAllocatorInfo external_allocator_info{};
// TODO remove deprecated global config
onnxruntime::ArenaExtendStrategy arena_extend_strategy = onnxruntime::ArenaExtendStrategy::kNextPowerOfTwo;
#endif

#ifdef USE_ROCM
// TODO remove deprecated global config
bool miopen_conv_exhaustive_search = false;
// TODO remove deprecated global config
bool do_copy_in_default_stream = true;
// TODO remove deprecated global config
onnxruntime::rocm::TunableOpInfo tunable_op{};
onnxruntime::ROCMExecutionProviderExternalAllocatorInfo external_allocator_info{};
// TODO remove deprecated global config
onnxruntime::ArenaExtendStrategy arena_extend_strategy = onnxruntime::ArenaExtendStrategy::kNextPowerOfTwo;
#endif


static at::Device ToTorchDevice(OrtDevice const& device) {
  switch (device.Type()) {
    case OrtDevice::CPU:
      return at::Device(OrtDevice::CPU);
    case OrtDevice::GPU:
      return at::Device(OrtDevice::CUDA, device.Id());
    case OrtDevice::FPGA:
    case OrtDevice::NPU:
      return at::Device(OrtDevice::ORT, device.Id());
    default:
      ORT_THROW("Undefined device type:" + device.ToString());
  }
}

static OrtDevice::DeviceType ToOrtDeviceType(at::Device const& device) {
    switch (device.type()) {
    case c10::Device::Type::CPU:
      return OrtDevice::CPU;
    case c10::Device::Type::CUDA:
      return OrtDevice::GPU;
    case c10::Device::Type::ORT:
      return OrtDevice::NPU;
    default:
      ORT_THROW("Undefined device type:" + device.str());
  }
}

static OrtMemoryInfo ToOrtMemoryInfo(at::Device const& device) {
  auto device_id = device.index();
  OrtDevice::DeviceType device_type = ToOrtDeviceType(device);
  OrtAllocatorType alloc_type = (device_type != OrtDevice::CPU)? OrtAllocatorType::OrtDeviceAllocator : OrtAllocatorType::OrtArenaAllocator;
  OrtMemoryInfo info("torch",
                      alloc_type,
                      OrtDevice(device_type, OrtDevice::MemType::DEFAULT, device_id),
                      device_id,
                      OrtMemTypeDefault)
  return info;

}

static at::ScalarType ToTorchDataType(int32_t elem_type) {
  switch (elem_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return ScalarType::Double;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return ScalarType::Float;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      return ScalarType::Char;
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      return ScalarType::Short;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return ScalarType::Int;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return ScalarType::Long;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return ScalarType::Half;
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      return ScalarType::BFloat16;
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      return ScalarType::Bool;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return ScalarType::Byte;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
    default:
      ORT_THROW("Unexpected data type of ", elem_type);
  }
}

static onnxruntime::MLDataType ToOrtDataType(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat:
      return onnxruntime::DataTypeImpl::GetType<float>();
    case at::kDouble:
      return onnxruntime::DataTypeImpl::GetType<double>();
    case at::kHalf:
      return onnxruntime::DataTypeImpl::GetType<onnxruntime::MLFloat16>();
    case at::kBFloat16:
      return onnxruntime::DataTypeImpl::GetType<onnxruntime::BFloat16>();
    case at::kInt:
      return onnxruntime::DataTypeImpl::GetType<int>();
    case at::kShort:
      return onnxruntime::DataTypeImpl::GetType<int16_t>();
    case at::kLong:
      return onnxruntime::DataTypeImpl::GetType<int64_t>();
    case at::kBool:
      return onnxruntime::DataTypeImpl::GetType<bool>();
    default:
      ORT_THROW("Unsupport aten scalar type: ", dtype);
  }
}

struct OrtManagedTensor : OrtValue {
  using OrtValue::OrtValue;
};

at::Tensor ToTorch(OrtValue& ort_value) {
  ORT_ENFORCE(ort_value.IsTensor(), "Only tensor type OrtValues are supported");
  Tensor& tensor = *ort_value.GetMutable<Tensor>();
  OrtManagedTensor* ort_managed_tensor(new OrtManagedTensor(ortvalue)); // increment ref count by 1
  at::Device device = ToTorchDevice(tensor.Location().device);
  at::ScalarType dtype = ToTorchType(tensor.GetElementType());
  at::IntArrayRef shape(tensor.Shape().GetDims().begin(), tensor.Shape().NumDimensions());

  auto deleter = [ort_managed_tensor](void* /*self*/) {
    delete ort_managed_tensor;
  };

  if (tensor.IsContiguous()) {
    return at::from_blob(tensor.MutableDataRaw(),
        std::move(shape),
        deleter,
        at::device(device).dtype(dtype));
  } else {
    return at::from_blob(
        tensor.MutableDataRaw(),
        std::move(shape),
        at::IntArrayRef(tensor.Strides().begin(), tensor.Shape().NumDimensions()),
        deleter,
        at::device(device).dtype(dtype),
        { device });
  }
}

OrtValue FromTorch(const at::Tensor& torch_tensor) {

  if (torch_tensor.is_sparse()) {
    throw std::runtime_error("FromTorch: sparse tensor is not supported");
  }

  if (torch_tensor.is_quantized()) {
    throw std::runtime_error("FromTorch: quantized tensor is not supported");
  }

  OrtMemoryInfo info = ToOrtMemoryInfo(torch_tensor.device());
  auto element_type = ToOrtDataType(torch_tensor.scalar_type());

  OrtValue ort_tensor;

  onnxruntime::Tensor::InitOrtValue(
      element_type,
      onnxruntime::TensorShape(torch_tensor.sizes().vec()),
      torch_tensor.data_ptr(),
      info,
      ort_tensor,
      0L,  // offset.
      tensor.strides().vec());
  return ort_tensor;

}


#ifdef ENABLE_TRAINING

void DlpackCapsuleDestructor(PyObject* data) {
  DLManagedTensor* dlmanaged_tensor = reinterpret_cast<DLManagedTensor*>(PyCapsule_GetPointer(data, "dltensor"));
  if (dlmanaged_tensor) {
    // The dlmanaged_tensor has not been consumed, call deleter ourselves.
    dlmanaged_tensor->deleter(const_cast<DLManagedTensor*>(dlmanaged_tensor));
  } else {
    // The dlmanaged_tensor has been consumed,
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
