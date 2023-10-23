// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/cerr_sink.h"
#include "core/common/optional.h"
#include "core/framework/allocator.h"
#include "core/framework/session_options.h"
#include "core/session/environment.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/inference_session.h"
#ifdef ENABLE_TRAINING
#include "core/dlpack/dlpack_converter.h"
#endif

#include "onnxruntime_pybind.h"  // must use this for the include of <pybind11/pybind11.h>

// execution provider factory creator headers
struct OrtStatus {
  OrtErrorCode code;
  char msg[1];  // a null-terminated string
};

#define BACKEND_DEVICE BACKEND_PROC BACKEND_DNNL BACKEND_OPENVINO BACKEND_TVM BACKEND_OPENBLAS BACKEND_MIGRAPHX BACKEND_ACL BACKEND_ARMNN BACKEND_DML BACKEND_CANN
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/providers.h"
#include "core/providers/provider_factory_creators.h"
#include "core/providers/tensorrt/tensorrt_provider_options.h"

#if defined(USE_CUDA) || defined(USE_ROCM)
#define BACKEND_PROC "GPU"
#else
#define BACKEND_PROC "CPU"
#endif

#if USE_DNNL
#define BACKEND_DNNL "-DNNL"
#else
#define BACKEND_DNNL ""
#endif

#if USE_MIGRAPHX
#define BACKEND_MIGRAPHX "-MIGRAPHX"
#else
#define BACKEND_MIGRAPHX ""
#endif

#ifdef USE_OPENVINO
#if OPENVINO_CONFIG_CPU_FP32
#define BACKEND_OPENVINO "-OPENVINO_CPU_FP32"

#elif OPENVINO_CONFIG_CPU_FP16
#define BACKEND_OPENVINO "-OPENVINO_CPU_FP16"

#elif OPENVINO_CONFIG_GPU_FP32
#define BACKEND_OPENVINO "-OPENVINO_GPU_FP32"

#elif OPENVINO_CONFIG_GPU_FP16
#define BACKEND_OPENVINO "-OPENVINO_GPU_FP16"

#elif OPENVINO_CONFIG_NPU_FP16
#define BACKEND_OPENVINO "-OPENVINO_NPU_FP16"

#elif OPENVINO_CONFIG_NPU_U8
#define BACKEND_OPENVINO "-OPENVINO_NPU_U8"

#elif OPENVINO_CONFIG_MULTI
#define BACKEND_OPENVINO "-OPENVINO_MULTI"

#elif OPENVINO_CONFIG_AUTO
#define BACKEND_OPENVINO "-OPENVINO_AUTO"

#elif OPENVINO_CONFIG_HETERO
#define BACKEND_OPENVINO "-OPENVINO_HETERO"
#endif
#else
#define BACKEND_OPENVINO ""
#endif

#ifdef USE_TVM
#define BACKEND_TVM "-TVM"
#else
#define BACKEND_TVM ""
#endif

#if USE_VITISAI
#define BACKEND_VITISAI "-VITISAI"
#include "core/providers/vitisai/vitisai_execution_provider.h"
#else
#define BACKEND_VITISAI ""
#endif

#if USE_OPENBLAS
#define BACKEND_OPENBLAS "-OPENBLAS"
#else
#define BACKEND_OPENBLAS ""
#endif

#if USE_ACL
#define BACKEND_ACL "-ACL"
#else
#define BACKEND_ACL ""
#endif

#if USE_ARMNN
#define BACKEND_ARMNN "-ARMNN"
#else
#define BACKEND_ARMNN ""
#endif

#if USE_DML
#define BACKEND_DML "-DML"
#else
#define BACKEND_DML ""
#endif

#if USE_CANN
#define BACKEND_CANN "-CANN"
#else
#define BACKEND_CANN ""
#endif

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_factory.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"
#endif
#ifdef USE_ROCM
#include "core/providers/rocm/rocm_provider_factory.h"
#include "core/providers/rocm/rocm_execution_provider_info.h"
#endif
#ifdef USE_TENSORRT
#include "core/providers/tensorrt/tensorrt_provider_factory.h"
#endif
#ifdef USE_MIGRAPHX
#include "core/providers/migraphx/migraphx_provider_factory.h"
#endif
#ifdef USE_OPENVINO
#include "core/providers/openvino/openvino_provider_factory.h"
// TODO remove deprecated global config
namespace onnxruntime {
ProviderInfo_OpenVINO* GetProviderInfo_OpenVINO();
namespace python {
extern std::string openvino_device_type;
}
}  // namespace onnxruntime
#endif
#ifdef USE_TVM
#include "core/providers/tvm/tvm_ep_options.h"
#endif
#ifdef USE_ACL
#include "core/providers/acl/acl_provider_factory.h"
#endif
#ifdef USE_ARMNN
#include "core/providers/armnn/armnn_provider_factory.h"
#endif
#ifdef USE_DML
#include "core/providers/dml/dml_provider_factory.h"
#endif
#ifdef USE_CANN
#include "core/providers/cann/cann_provider_factory.h"
#include "core/providers/cann/cann_execution_provider_info.h"
#endif

#ifdef USE_CUDA
namespace onnxruntime {
ProviderInfo_CUDA* TryGetProviderInfo_CUDA();
ProviderInfo_CUDA& GetProviderInfo_CUDA();
namespace python {
// TODO remove deprecated global config
extern OrtCudnnConvAlgoSearch cudnn_conv_algo_search;
// TODO remove deprecated global config
extern bool do_copy_in_default_stream;
// TODO remove deprecated global config
extern onnxruntime::cuda::TunableOpInfo tunable_op;
extern onnxruntime::CUDAExecutionProviderExternalAllocatorInfo external_allocator_info;
extern onnxruntime::ArenaExtendStrategy arena_extend_strategy;
}  // namespace python
}  // namespace onnxruntime
#endif

#ifdef USE_TENSORRT
namespace onnxruntime {
ProviderInfo_TensorRT* TryGetProviderInfo_TensorRT();
ProviderInfo_TensorRT& GetProviderInfo_TensorRT();
}  // namespace onnxruntime
#endif

#ifdef USE_CANN
namespace onnxruntime {
ProviderInfo_CANN* TryGetProviderInfo_CANN();
ProviderInfo_CANN& GetProviderInfo_CANN();
}  // namespace onnxruntime
#endif

#ifdef USE_ROCM
namespace onnxruntime {
ProviderInfo_ROCM* TryGetProviderInfo_ROCM();
ProviderInfo_ROCM& GetProviderInfo_ROCM();
namespace python {
// TODO remove deprecated global config
extern bool miopen_conv_exhaustive_search;
// TODO remove deprecated global config
extern bool do_copy_in_default_stream;
// TODO remove deprecated global config
extern onnxruntime::rocm::TunableOpInfo tunable_op;
extern onnxruntime::ROCMExecutionProviderExternalAllocatorInfo external_allocator_info;
extern onnxruntime::ArenaExtendStrategy arena_extend_strategy;
}  // namespace python
}  // namespace onnxruntime
#endif

#include "core/providers/dnnl/dnnl_provider_factory.h"
#include "core/providers/shared_library/provider_host_api.h"

namespace onnxruntime {
#if !defined(SHARED_PROVIDER) && !defined(DISABLE_SPARSE_TENSORS)
class SparseTensor;
#endif
namespace python {

using ExecutionProviderRegistrationFn = std::function<void(InferenceSession*,
                                                           const std::vector<std::string>&,
                                                           const ProviderOptionsMap&)>;

// TODO remove deprecated global config
extern OrtDevice::DeviceId cuda_device_id;
// TODO remove deprecated global config
extern size_t gpu_mem_limit;

using PySessionOptions = OrtSessionOptions;

// Thin wrapper over internal C++ InferenceSession to accommodate custom op library management for the Python user
struct PyInferenceSession {
  PyInferenceSession(std::shared_ptr<Environment> env, const PySessionOptions& so)
      : env_(std::move(env)) {
    sess_ = std::make_unique<InferenceSession>(so.value, *env_);
  }

#if !defined(ORT_MINIMAL_BUILD)
  PyInferenceSession(std::shared_ptr<Environment> env, const PySessionOptions& so, const std::string& arg, bool is_arg_file_name)
      : env_(std::move(env)) {
    if (is_arg_file_name) {
      // Given arg is the file path. Invoke the corresponding ctor().
      sess_ = std::make_unique<InferenceSession>(so.value, *env_, arg);
    } else {
      // Given arg is the model content as bytes. Invoke the corresponding ctor().
      std::istringstream buffer(arg);
      sess_ = std::make_unique<InferenceSession>(so.value, *env_, buffer);
    }
  }
#endif

  InferenceSession* GetSessionHandle() const { return sess_.get(); }

  virtual ~PyInferenceSession() = default;

 protected:
  PyInferenceSession(std::shared_ptr<Environment> env, std::unique_ptr<InferenceSession> sess)
      : env_(std::move(env)), sess_(std::move(sess)) {
  }

 private:
  std::shared_ptr<Environment> env_;
  std::unique_ptr<InferenceSession> sess_;
};

inline const PySessionOptions& GetDefaultCPUSessionOptions() {
  static PySessionOptions so;
  return so;
}

inline AllocatorPtr& GetAllocator() {
  static AllocatorPtr alloc = std::make_shared<CPUAllocator>();
  return alloc;
}

#if !defined(DISABLE_SPARSE_TENSORS)
// This class exposes SparseTensor to Python
// The class serves two major purposes
// - to be able to map numpy arrays memory and use it on input, this serves as a reference holder
//   so incoming arrays do not disappear. To this end we create an instance of SparseTensor
//   on top of the user provided numpy arrays and create a duplicate of py::objects for those
//   numpy array for ref-counting purposes and store it here.
//
// - to be able to expose SparseTensor returned from run method. We get an OrtValue from run()
//   and store a copy of it in ort_value_. The OrtValue shared_ptr ref-counting will make sure
//   the memory stays around.
//
//   An object of the class must never have both instance_ and ort_value_ have data at the same time.
class PySparseTensor {
 public:
  /// <summary>
  /// Use this constructor when you created a SparseTensor instance which is backed
  /// by python array storage and it important that they stay alive while this object is
  /// alive
  /// </summary>
  /// <param name="instance">a fully constructed and populated instance of SparseTensor</param>
  /// <param name="storage">a collection reference guards</param>
  PySparseTensor(std::unique_ptr<SparseTensor>&& instance,
                 std::vector<pybind11::object>&& storage)
      : instance_(std::move(instance)), backing_storage_(std::move(storage)), ort_value_() {
  }

  /// <summary>
  /// Same as above but no backing storage as SparseTensor owns the memory
  /// </summary>
  /// <param name="instance"></param>
  explicit PySparseTensor(std::unique_ptr<SparseTensor>&& instance)
      : instance_(std::move(instance)), backing_storage_(), ort_value_() {
  }

  /// <summary>
  /// Edge case when we can not copy memory on GPU and therefore
  /// can not own it.
  /// </summary>
  /// <param name="ort_value"></param>
  explicit PySparseTensor(const OrtValue& ort_value)
      : instance_(), backing_storage_(), ort_value_(ort_value) {}

  PySparseTensor(const PySparseTensor&) = delete;
  PySparseTensor& operator=(const PySparseTensor&) = delete;

  PySparseTensor(PySparseTensor&& o) noexcept {
    *this = std::move(o);
  }

  PySparseTensor& operator=(PySparseTensor&& o) noexcept {
    instance_ = std::move(o.instance_);
    backing_storage_ = std::move(o.backing_storage_);
    ort_value_ = std::move(o.ort_value_);
    return *this;
  }

  ~PySparseTensor();

  const SparseTensor& Instance() const {
    if (instance_) {
      return *instance_;
    }
    return ort_value_.Get<SparseTensor>();
  }

  std::unique_ptr<OrtValue> AsOrtValue() const;

 private:
  // instance_ represents data that comes as input. Thus we depend on numpy
  // arrays that own the underlying memory to stay around. We store copies
  // of py::objects for those arrays in backing_storage_ as an extra ref-count.

  // If we have and are able to copy from the OrtValue returned by run() to CPU, then this owns the data
  // and backing_storage_ is empty.
  std::unique_ptr<SparseTensor> instance_;
  std::vector<pybind11::object> backing_storage_;

  // We create a copy of OrtValue when we obtain it from a run method.
  OrtValue ort_value_;
};
#endif  // !defined(DISABLE_SPARSE_TENSORS)

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
// You can attempt to make 'onnxruntime::python::SessionObjectInitializer::Get' constexpr
#pragma warning(disable : 26497)
#endif
class SessionObjectInitializer {
 public:
  typedef const PySessionOptions& Arg1;
  // typedef logging::LoggingManager* Arg2;
  static const std::string default_logger_id;
  operator Arg1() {
    return GetDefaultCPUSessionOptions();
  }

  // operator Arg2() {
  //   static LoggingManager default_logging_manager{std::unique_ptr<ISink>{new CErrSink{}},
  //                                                 Severity::kWARNING, false, LoggingManager::InstanceType::Default,
  //                                                 &default_logger_id};
  //   return &default_logging_manager;
  // }

  static SessionObjectInitializer Get() {
    return SessionObjectInitializer();
  }
};
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
std::shared_ptr<Environment> GetEnv();

// Initialize an InferenceSession.
// Any provider_options should have entries in matching order to provider_types.
void InitializeSession(InferenceSession* sess,
                       ExecutionProviderRegistrationFn ep_registration_fn,
                       const std::vector<std::string>& provider_types = {},
                       const ProviderOptionsVector& provider_options = {},
                       const std::unordered_set<std::string>& disabled_optimizer_names = {});

// Checks if PyErrOccured, fetches status and throws.
void ThrowIfPyErrOccured();

void addOrtValueMethods(pybind11::module& m);

void addIoBindingMethods(pybind11::module& m);

void addSparseTensorMethods(pybind11::module& m);

void addGlobalSchemaFunctions(pybind11::module& m);

void addOpKernelSubmodule(pybind11::module& m);

void addOpSchemaSubmodule(pybind11::module& m);

const char* GetDeviceName(const OrtDevice& device);

bool IsCudaDeviceIdValid(const onnxruntime::logging::Logger& logger, int id);

AllocatorPtr GetCudaAllocator(OrtDevice::DeviceId id);

bool CheckIfTensor(const std::vector<const NodeArg*>& def_list,
                   const std::string& name,
                   /*out*/ ONNX_NAMESPACE::TypeProto& type_proto);

#ifdef ENABLE_TRAINING

// Allocate a new Capsule object, which takes the ownership of OrtValue.
// Caller is responsible for releasing.
// This function calls OrtValueToDlpack(...).
PyObject* ToDlpack(OrtValue ort_value);

// Consume a Capsule object and claims the ownership of its underlying tensor to
// create a OrtValue. This function calls DlpackToOrtValue(...) to do the conversion.
OrtValue FromDlpack(PyObject* dlpack_tensor, const bool is_bool_tensor);

// Destructor for Capsule object holding a DLPack structure.
void DlpackCapsuleDestructor(PyObject* data);

#endif

}  // namespace python

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(const OrtTensorRTProviderOptions* params);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(const OrtTensorRTProviderOptionsV2* params);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(int device_id);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_MIGraphX(const OrtMIGraphXProviderOptions* params);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_MIGraphX(int device_id);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Cuda(const OrtCUDAProviderOptions* params);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Dnnl(const OrtDnnlProviderOptions* params);
#ifdef USE_TVM
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tvm(const tvm::TvmEPOptions& info);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tvm(const char* params);
#endif
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_VITISAI(const char* backend_type, int device_id,
                                                                                  const char* export_runtime_module,
                                                                                  const char* load_runtime_module);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_ACL(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_ArmNN(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_DML(int device_id);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nnapi(
    uint32_t flags, const optional<std::string>& partitioning_stop_ops_list);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Rknpu();
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CoreML(uint32_t flags);
constexpr const char* kDefaultExecutionProviderEntry = "GetProvider";
}  // namespace onnxruntime
