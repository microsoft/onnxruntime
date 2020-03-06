// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_pybind_exceptions.h"
#include "onnxruntime_pybind_mlvalue.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL onnxruntime_python_ARRAY_API
#include <numpy/arrayobject.h>

#include "core/framework/data_types_internal.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/severity.h"
#include "core/framework/TensorSeq.h"
#include "core/framework/session_options.h"
#include "core/framework/bfc_arena.h"

#if USE_CUDA
#define BACKEND_PROC "GPU"
#else
#define BACKEND_PROC "CPU"
#endif

#if USE_OPENMP
#define BACKEND_OPENMP "-OPENMP"
#else
#define BACKEND_OPENMP ""
#endif

#if USE_DNNL
#define BACKEND_DNNL "-DNNL"
#include "core/providers/dnnl/dnnl_execution_provider.h"
#else
#define BACKEND_DNNL ""
#endif

#if USE_MKLML
#define BACKEND_MKLML "-MKL-ML"
#else
#define BACKEND_MKLML ""
#endif

#if USE_NGRAPH
#define BACKEND_NGRAPH "-NGRAPH"
#include "core/providers/ngraph/ngraph_execution_provider.h"
#else
#define BACKEND_NGRAPH ""
#endif

#if OPENVINO_CONFIG_CPU_FP32
#define BACKEND_OPENVINO "-OPENVINO_CPU_FP32"

#elif OPENVINO_CONFIG_GPU_FP32
#define BACKEND_OPENVINO "-OPENVINO_GPU_FP32"

#elif OPENVINO_CONFIG_GPU_FP16
#define BACKEND_OPENVINO "-OPENVINO_GPU_FP16"

#elif OPENVINO_CONFIG_MYRIAD
#define BACKEND_OPENVINO "-OPENVINO_MYRIAD"

#elif OPENVINO_CONFIG_VAD_M
#define BACKEND_OPENVINO "-OPENVINO_VAD_M"

#else
#define BACKEND_OPENVINO ""
#endif

#ifdef USE_NUPHAR
#define BACKEND_NUPHAR "-NUPHAR"
#else
#define BACKEND_NUPHAR ""
#endif

#if USE_OPENBLAS
#define BACKEND_OPENBLAS "-OPENBLAS"
#else
#define BACKEND_OPENBLAS ""
#endif

#define BACKEND_DEVICE BACKEND_PROC BACKEND_DNNL BACKEND_MKLML BACKEND_NGRAPH BACKEND_OPENVINO BACKEND_NUPHAR BACKEND_OPENBLAS
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/providers.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/providers/cpu/cpu_provider_factory.h"

#ifdef ENABLE_TRAINING
#include "core/session/IOBinding.h"
#include "orttraining/core/session/training_session.h"
#include "orttraining/core/graph/optimizer_config.h"
#include "orttraining/core/framework/mpi_setup.h"
#include "orttraining/core/framework/data_transfer_utils.h"
#endif

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_factory.h"
int cuda_device_id = 0;
size_t cuda_mem_limit = std::numeric_limits<size_t>::max();
onnxruntime::ArenaExtendStrategy arena_extend_strategy = onnxruntime::ArenaExtendStrategy::kNextPowerOfTwo;
#endif
#ifdef USE_TENSORRT
#include "core/providers/tensorrt/tensorrt_provider_factory.h"
#endif
#ifdef USE_DNNL
#include "core/providers/dnnl/dnnl_provider_factory.h"
#endif
#ifdef USE_NGRAPH
#include "core/providers/ngraph/ngraph_provider_factory.h"
#endif
#ifdef USE_OPENVINO
#include "core/providers/openvino/openvino_provider_factory.h"
#endif
#ifdef USE_NUPHAR
#include "core/providers/nuphar/nuphar_provider_factory.h"
std::string nuphar_settings;
#endif

namespace onnxruntime {
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CPU(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CUDA(OrtDevice::DeviceId device_id,
                                                                               size_t cuda_mem_limit,
                                                                               onnxruntime::ArenaExtendStrategy arena_extend_strategy);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(int device_id);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Dnnl(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_NGraph(const char* ng_backend_type);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_OpenVINO(const char* device);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nuphar(bool, const char*);
}  // namespace onnxruntime

#if defined(_MSC_VER)
#pragma warning(disable : 4267 4996 4503 4003)
#endif  // _MSC_VER

#include <iterator>

#if defined(_MSC_VER)
#pragma warning(disable : 4267 4996 4503 4003)
#endif  // _MSC_VER

namespace onnxruntime {
using namespace training;
namespace python {

namespace py = pybind11;
using namespace onnxruntime;
using namespace onnxruntime::logging;

static AllocatorPtr& GetAllocator() {
  static AllocatorPtr alloc = std::make_shared<TAllocator>();
  return alloc;
}

static const SessionOptions& GetDefaultCPUSessionOptions() {
  static SessionOptions so;
  return so;
}

template <typename T>
void AddNonTensor(OrtValue& val, std::vector<py::object>& pyobjs) {
  pyobjs.push_back(py::cast(val.Get<T>()));
}

void GetPyObjFromTensor(const Tensor& rtensor, py::object& obj, const DataTransferManager* data_transfer_manager = nullptr) {
  std::vector<npy_intp> npy_dims;
  const TensorShape& shape = rtensor.Shape();

  for (size_t n = 0; n < shape.NumDimensions(); ++n) {
    npy_dims.push_back(shape[n]);
  }

  MLDataType dtype = rtensor.DataType();
  const int numpy_type = OnnxRuntimeTensorToNumpyType(dtype);
  obj = py::reinterpret_steal<py::object>(PyArray_SimpleNew(
      shape.NumDimensions(), npy_dims.data(), numpy_type));

  void* outPtr = static_cast<void*>(
      PyArray_DATA(reinterpret_cast<PyArrayObject*>(obj.ptr())));

  if (numpy_type != NPY_OBJECT) {
    //if it is not cpu tensor, need to copy to host
    if (rtensor.Location().device.Type() != OrtDevice::CPU) {
      if (!data_transfer_manager)
        throw std::runtime_error("GetPyObjFromTensor: data transfer manager is needed when convert non-CPU tensor to numpy array");
      static const OrtMemoryInfo cpu_alloc_info{onnxruntime::CPU, OrtDeviceAllocator};
      std::vector<char> tensor_data_buffer{};
      tensor_data_buffer.resize(rtensor.SizeInBytes());
      ORT_THROW_IF_ERROR(CopyTensorDataToByteSpan(
          *data_transfer_manager, rtensor, cpu_alloc_info, gsl::make_span(tensor_data_buffer)));
      memcpy(outPtr, tensor_data_buffer.data(), dtype->Size() * shape.Size());
    } else
      memcpy(outPtr, rtensor.DataRaw(dtype), dtype->Size() * shape.Size());
  } else {
    // Handle string type.
    py::object* outObj = static_cast<py::object*>(outPtr);
    const std::string* src = rtensor.template Data<std::string>();
    for (int i = 0; i < rtensor.Shape().Size(); i++, src++) {
      outObj[i] = py::cast(*src);
    }
  }
}

static std::string GetDeviceName(const OrtDevice& device) {
  switch (device.Type()) {
    case OrtDevice::CPU:
      return CPU;
    case OrtDevice::GPU:
      return CUDA;
    case OrtDevice::FPGA:
      return "FPGA";
    default:
      throw std::runtime_error("Unknow device type:" + std::to_string(device.Type()));
  }
}

struct TrainingParameters {
  std::string loss_output_name;
  std::unordered_set<std::string> weights_to_train;
  std::unordered_set<std::string> weights_not_to_train;
  onnxruntime::training::TrainingSession::ImmutableWeights immutable_weights;

  // optimizer
  std::string training_optimizer_name;
  std::string loss_scale_input_name;
  std::string scaled_loss_output_name;
  std::string lr_params_feed_name = "Learning_Rate";
  std::unordered_map<std::string, std::unordered_map<std::string, float>> optimizer_attributes_map;
  bool use_fp16_moments = false;

  bool use_mixed_precision = false;
  bool allreduce_post_accumulation = false;
  float loss_scale = 0.0f;
  int world_rank = -1;
  int world_size = 1;
  int local_rank = -1;
  int local_size = 1;
  int gradient_accumulation_steps = 1;
  int data_parallel_size = 1;
  int horizontal_parallel_size = 1;
  bool partition_optimizer = false;
};

template <>
void AddNonTensor<TensorSeq>(OrtValue& val, std::vector<py::object>& pyobjs) {
  const auto& seq_tensors = val.Get<TensorSeq>();
  py::list py_list;
  for (const auto& rtensor : seq_tensors) {
    py::object obj;
    GetPyObjFromTensor(rtensor, obj);
    py_list.append(obj);
  }
  pyobjs.push_back(py_list);
}

void AddNonTensorAsPyObj(OrtValue& val, std::vector<py::object>& pyobjs) {
  // Should be in sync with core/framework/datatypes.h
  auto val_type = val.Type();
  if (val_type->IsTensorSequenceType()) {
    AddNonTensor<TensorSeq>(val, pyobjs);
  } else {
    utils::ContainerChecker c_checker(val_type);
    if (c_checker.IsMap()) {
      if (c_checker.IsMapOf<std::string, std::string>()) {
        AddNonTensor<MapStringToString>(val, pyobjs);
      } else if (c_checker.IsMapOf<std::string, int64_t>()) {
        AddNonTensor<MapStringToInt64>(val, pyobjs);
      } else if (c_checker.IsMapOf<std::string, float>()) {
        AddNonTensor<MapStringToFloat>(val, pyobjs);
      } else if (c_checker.IsMapOf<std::string, double>()) {
        AddNonTensor<MapStringToDouble>(val, pyobjs);
      } else if (c_checker.IsMapOf<int64_t, std::string>()) {
        AddNonTensor<MapInt64ToString>(val, pyobjs);
      } else if (c_checker.IsMapOf<int64_t, int64_t>()) {
        AddNonTensor<MapInt64ToInt64>(val, pyobjs);
      } else if (c_checker.IsMapOf<int64_t, float>()) {
        AddNonTensor<MapInt64ToFloat>(val, pyobjs);
      } else if (c_checker.IsMapOf<int64_t, double>()) {
        AddNonTensor<MapInt64ToDouble>(val, pyobjs);
      }
    } else {
      if (c_checker.IsSequenceOf<std::map<std::string, float>>()) {
        AddNonTensor<VectorMapStringToFloat>(val, pyobjs);
      } else if (c_checker.IsSequenceOf<std::map<int64_t, float>>()) {
        AddNonTensor<VectorMapInt64ToFloat>(val, pyobjs);
      } else {
        throw std::runtime_error("Output is a non-tensor type which is not supported.");
      }
    }
  }
}

void AddTensorAsPyObj(OrtValue& val, std::vector<py::object>& pyobjs) {
  const Tensor& rtensor = val.Get<Tensor>();
  py::object obj;
  GetPyObjFromTensor(rtensor, obj);
  pyobjs.push_back(obj);
}

class SessionObjectInitializer {
 public:
  typedef const SessionOptions& Arg1;
  typedef logging::LoggingManager* Arg2;
  operator Arg1() {
    return GetDefaultCPUSessionOptions();
  }

  operator Arg2() {
    static std::string default_logger_id{"Default"};
    static LoggingManager default_logging_manager{std::unique_ptr<ISink>{new CErrSink{}},
                                                  Severity::kWARNING, false, LoggingManager::InstanceType::Default,
                                                  &default_logger_id};
    return &default_logging_manager;
  }

  static SessionObjectInitializer Get() {
    return SessionObjectInitializer();
  }
};

inline void RegisterExecutionProvider(InferenceSession* sess, onnxruntime::IExecutionProviderFactory& f) {
  auto p = f.CreateProvider();
  OrtPybindThrowIfError(sess->RegisterExecutionProvider(std::move(p)));
}

// ordered by default priority. highest to lowest.
const std::vector<std::string>& GetAllProviders() {
  static std::vector<std::string> all_providers = {kTensorrtExecutionProvider, kCudaExecutionProvider, kDnnlExecutionProvider,
                                                   kNGraphExecutionProvider, kOpenVINOExecutionProvider, kNupharExecutionProvider,
                                                   kCpuExecutionProvider};
  return all_providers;
}

const std::vector<std::string>& GetAvailableProviders() {
  auto InitializeProviders = []() {
    std::vector<std::string> available_providers = {kCpuExecutionProvider};
#ifdef USE_TENSORRT
    available_providers.push_back(kTensorrtExecutionProvider);
#endif
#ifdef USE_CUDA
    available_providers.push_back(kCudaExecutionProvider);
#endif
#ifdef USE_DNNL
    available_providers.push_back(kDnnlExecutionProvider);
#endif
#ifdef USE_NGRAPH
    available_providers.push_back(kNGraphExecutionProvider);
#endif
#ifdef USE_OPENVINO
    available_providers.push_back(kOpenVINOExecutionProvider);
#endif
#ifdef USE_NUPHAR
    available_providers.push_back(kNupharExecutionProvider);
#endif
    return available_providers;
  };
  static std::vector<std::string> available_providers = InitializeProviders();
  return available_providers;
}

void RegisterExecutionProviders(InferenceSession* sess, const std::vector<std::string>& provider_types) {
  for (const std::string& type : provider_types) {
    if (type == kCpuExecutionProvider) {
      RegisterExecutionProvider(sess, *onnxruntime::CreateExecutionProviderFactory_CPU(sess->GetSessionOptions().enable_cpu_mem_arena));
    } else if (type == kTensorrtExecutionProvider) {
#ifdef USE_TENSORRT
      RegisterExecutionProvider(sess, *onnxruntime::CreateExecutionProviderFactory_Tensorrt(0));
#endif
    } else if (type == kCudaExecutionProvider) {
#ifdef USE_CUDA
      RegisterExecutionProvider(sess, *onnxruntime::CreateExecutionProviderFactory_CUDA(cuda_device_id, cuda_mem_limit, arena_extend_strategy));
      cuda_device_id = 0;
      cuda_mem_limit = static_cast<size_t>(INT_MAX);
      arena_extend_strategy = ArenaExtendStrategy::kNextPowerOfTwo;
#endif
    } else if (type == kDnnlExecutionProvider) {
#ifdef USE_DNNL
      RegisterExecutionProvider(sess, *onnxruntime::CreateExecutionProviderFactory_Dnnl(sess->GetSessionOptions().enable_cpu_mem_arena));
#endif
    } else if (type == kNGraphExecutionProvider) {
#if USE_NGRAPH
      RegisterExecutionProvider(sess, *onnxruntime::CreateExecutionProviderFactory_NGraph("CPU"));
#endif
    } else if (type == kOpenVINOExecutionProvider) {
#ifdef USE_OPENVINO
      RegisterExecutionProvider(sess, *onnxruntime::CreateExecutionProviderFactory_OpenVINO("CPU"));
#endif
    } else if (type == kNupharExecutionProvider) {
#if USE_NUPHAR
      RegisterExecutionProvider(sess, *onnxruntime::CreateExecutionProviderFactory_Nuphar(true, nuphar_settings.c_str()));
      nuphar_settings.clear();  // clear nuphar_settings after use to avoid it being accidentally passed on to next session
#endif
    } else {
      // unknown provider
      throw std::runtime_error("Unknown Provider Type: " + type);
    }
  }
}

void InitializeSession(InferenceSession* sess, const std::vector<std::string>& provider_types) {
  if (provider_types.empty()) {
    // use default registration priority.
    RegisterExecutionProviders(sess, GetAllProviders());
  } else {
    RegisterExecutionProviders(sess, provider_types);
  }
  OrtPybindThrowIfError(sess->Initialize());
}

// TODO: this method does not handle parallel optimization.
static void ConfigureSessionForTraining(
    training::TrainingSession* sess, TrainingParameters& parameters) {
#ifdef USE_HOROVOD
  // this condition block is temporary.
  // For now, nccl allreduce kernel only implements for allreduce_post_accumulation
  // hovorod allreduce kernel only implements for not allreduce_post_accumulation.
  bool use_nccl = parameters.allreduce_post_accumulation;
  if (!use_nccl && parameters.world_size > 1) {
    auto mpi_context = setup_horovod();
    ORT_ENFORCE(parameters.horizontal_parallel_size <= mpi_context.world_size);
    ORT_ENFORCE(parameters.data_parallel_size <= mpi_context.world_size);
    if (mpi_context.world_size % parameters.horizontal_parallel_size != 0) {
      throw std::runtime_error("Cannot split horizontal parallel group because world_size is not divisible");
    }

    auto data_group_size = mpi_context.world_size / parameters.horizontal_parallel_size;
    if (data_group_size != parameters.data_parallel_size) {
      std::cout << "WARNING: data_parallel_size is not correct, tuned automatically to "
                << data_group_size << std::endl;
      parameters.data_parallel_size = data_group_size;
    }
    std::cout << "mpi_context.world_rank: " << mpi_context.world_rank << std::endl;
    std::cout << "mpi_context.local_rank: " << mpi_context.local_rank << std::endl;
    std::cout << "mpi_context.world_size: " << mpi_context.world_size << std::endl;
    std::cout << "mpi_context.local_size: " << mpi_context.local_size << std::endl;
    parameters.local_size = mpi_context.local_size;
    parameters.local_rank = mpi_context.local_rank;
  }
#endif

  training::TrainingSession::TrainingConfiguration config{};
  config.weight_names_to_train = parameters.weights_to_train;
  config.weight_names_to_not_train = parameters.weights_not_to_train;
  config.immutable_weights = parameters.immutable_weights;

  config.set_gradients_as_graph_outputs = true;

  config.gradient_accumulation_steps = parameters.gradient_accumulation_steps;

  config.distributed_config.world_rank = parameters.world_rank;
  config.distributed_config.world_size = parameters.world_size;
  config.distributed_config.local_rank = parameters.local_rank;
  config.distributed_config.local_size = parameters.local_size;
  config.distributed_config.data_parallel_size = parameters.data_parallel_size;
  config.distributed_config.horizontal_parallel_size = parameters.horizontal_parallel_size;

  if (parameters.use_mixed_precision) {
    training::TrainingSession::TrainingConfiguration::MixedPrecisionConfiguration mp{};
    mp.add_loss_scaling = false;
    mp.use_fp16_initializers = true;

    config.mixed_precision_config = mp;
  }

  config.loss_name =
      parameters.use_mixed_precision ? parameters.scaled_loss_output_name : parameters.loss_output_name;

  if (!parameters.training_optimizer_name.empty()) {
    training::TrainingSession::TrainingConfiguration::OptimizerConfiguration opt{};
    opt.name = parameters.training_optimizer_name;
    opt.learning_rate_input_name = parameters.lr_params_feed_name;
    opt.weight_attributes_generator = [&parameters](const std::string& weight_name) {
      const auto it = parameters.optimizer_attributes_map.find(weight_name);
      ORT_ENFORCE(
          it != parameters.optimizer_attributes_map.end(),
          "Failed to find attribute map for weight ", weight_name);
      return it->second;
    };
    opt.use_fp16_moments = parameters.use_fp16_moments;
    opt.do_all_reduce_in_fp16 = true;
    // TODO: this mapping is temporary.
    // For now, nccl allreduce kernel only implements for allreduce_post_accumulation
    // hovorod allreduce kernel only implements for not allreduce_post_accumulation.
    // eventually we will have one all reduce kernel and let opt to have
    // an allreduce_post_accumulation option and remove the use_nccl option.
    opt.use_nccl = parameters.allreduce_post_accumulation;
    opt.partition_optimizer = parameters.partition_optimizer;

    config.optimizer_config = opt;
  }

  training::TrainingSession::TrainingConfigurationResult config_result{};

  OrtPybindThrowIfError(sess->ConfigureForTraining(config, config_result));
}

void addGlobalMethods(py::module& m) {
  m.def("get_default_session_options", &GetDefaultCPUSessionOptions, "Return a default session_options instance.");
  m.def("get_session_initializer", &SessionObjectInitializer::Get, "Return a default session object initializer.");
  m.def(
      "get_device", []() -> std::string { return BACKEND_DEVICE; },
      "Return the device used to compute the prediction (CPU, MKL, ...)");
  m.def(
      "set_default_logger_severity", [](int severity) {
        ORT_ENFORCE(severity >= 0 && severity <= 4,
                    "Invalid logging severity. 0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal");
        logging::LoggingManager* default_logging_manager = SessionObjectInitializer::Get();
        default_logging_manager->SetDefaultLoggerSeverity(static_cast<logging::Severity>(severity));
      },
      "Sets the default logging severity. 0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal");
  m.def(
      "get_all_providers", []() -> const std::vector<std::string>& { return GetAllProviders(); },
      "Return list of Execution Providers that this version of Onnxruntime can support.");
  m.def(
      "get_available_providers", []() -> const std::vector<std::string>& { return GetAvailableProviders(); },
      "Return list of available Execution Providers available in this installed version of Onnxruntime.");

#ifdef USE_NUPHAR
  m.def("set_nuphar_settings", [](const std::string& str) {
    nuphar_settings = str;
  });
  m.def("get_nuphar_settings", []() -> std::string {
    return nuphar_settings;
  });
#endif

#ifdef onnxruntime_PYBIND_EXPORT_OPSCHEMA
  m.def(
      "get_all_operator_schema", []() -> const std::vector<ONNX_NAMESPACE::OpSchema> {
        return ONNX_NAMESPACE::OpSchemaRegistry::get_all_schemas_with_history();
      },
      "Return a vector of OpSchema all registed operators");
  m.def(
      "get_all_opkernel_def", []() -> const std::vector<onnxruntime::KernelDef> {
        std::vector<onnxruntime::KernelDef> result;

        // default logger is needed to create the DNNLExecutionProvider
        std::string default_logger_id{"DefaultLogger"};
        std::unique_ptr<onnxruntime::logging::LoggingManager> default_logging_manager =
            onnxruntime::make_unique<LoggingManager>(
                std::unique_ptr<onnxruntime::logging::ISink>{new onnxruntime::logging::CLogSink{}},
                onnxruntime::logging::Severity::kWARNING,
                false,
                onnxruntime::logging::LoggingManager::InstanceType::Default,
                &default_logger_id,
                /*default_max_vlog_level*/ -1);

        std::vector<std::shared_ptr<onnxruntime::IExecutionProviderFactory>> factories = {
            onnxruntime::CreateExecutionProviderFactory_CPU(0),
#ifdef USE_CUDA
            onnxruntime::CreateExecutionProviderFactory_CUDA(0),
#endif
#ifdef USE_DNNL
            onnxruntime::CreateExecutionProviderFactory_Dnnl(1),
#endif
#ifdef USE_NGRAPH
            onnxruntime::CreateExecutionProviderFactory_NGraph("CPU"),
#endif
#ifdef USE_OPENVINO
            onnxruntime::CreateExecutionProviderFactory_OpenVINO("CPU"),
#endif
#ifdef USE_TENSORRT
            onnxruntime::CreateExecutionProviderFactory_Tensorrt(0)
#endif
        };

        for (const auto& f : factories) {
          for (const auto& m : f->CreateProvider()
                                   ->GetKernelRegistry()
                                   ->GetKernelCreateMap()) {
            result.emplace_back(*(m.second.kernel_def));
          }
        }

        return result;
      },
      "Return a vector of KernelDef for all registered OpKernels");
#endif  //onnxruntime_PYBIND_EXPORT_OPSCHEMA

#ifdef USE_CUDA
  m.def("set_cuda_device_id", [](const int id) { cuda_device_id = id; });
  m.def("set_cuda_mem_limit", [](const int64_t limit) {
    cuda_mem_limit = static_cast<size_t>(limit);
  });
  m.def("set_arena_extend_strategy", [](const onnxruntime::ArenaExtendStrategy strategy) { arena_extend_strategy = strategy; });
#endif
}

#ifdef onnxruntime_PYBIND_EXPORT_OPSCHEMA

void addOpKernelSubmodule(py::module& m) {
  auto opkernel = m.def_submodule("opkernel");
  opkernel.doc() = "OpKernel submodule";
  py::class_<onnxruntime::KernelDef> kernel_def(opkernel, "KernelDef");
  kernel_def.def_property_readonly("op_name", &onnxruntime::KernelDef::OpName)
      .def_property_readonly("domain", &onnxruntime::KernelDef::Domain)
      .def_property_readonly("provider", &onnxruntime::KernelDef::Provider)
      .def_property_readonly("version_range",
                             [](const onnxruntime::KernelDef& kernelDef) -> std::pair<int, int> {
                               return kernelDef.onnxruntime::KernelDef::SinceVersion();
                             })
      .def_property_readonly("type_constraints",
                             [](const onnxruntime::KernelDef& kernelDef) -> std::unordered_map<std::string, std::vector<std::string>> {
                               std::unordered_map<std::string, std::vector<std::string>> result;
                               const auto& tempResult = kernelDef.TypeConstraints();
                               for (const auto& tc : tempResult) {
                                 result[tc.first] = std::vector<std::string>();
                                 for (const auto& dt : tc.second) {
                                   result[tc.first].emplace_back(onnxruntime::DataTypeImpl::ToString(dt));
                                 }
                               }
                               return result;
                             });
}

void addOpSchemaSubmodule(py::module& m) {
  auto schemadef = m.def_submodule("schemadef");
  schemadef.doc() = "Schema submodule";

  py::class_<ONNX_NAMESPACE::OpSchema> op_schema(schemadef, "OpSchema");
  op_schema.def_property_readonly("file", &ONNX_NAMESPACE::OpSchema::file)
      .def_property_readonly("line", &ONNX_NAMESPACE::OpSchema::line)
      .def_property_readonly("support_level", &ONNX_NAMESPACE::OpSchema::support_level)
      .def_property_readonly(
          "doc", &ONNX_NAMESPACE::OpSchema::doc, py::return_value_policy::reference)
      .def_property_readonly("since_version", &ONNX_NAMESPACE::OpSchema::since_version)
      .def_property_readonly("deprecated", &ONNX_NAMESPACE::OpSchema::deprecated)
      .def_property_readonly("domain", &ONNX_NAMESPACE::OpSchema::domain)
      .def_property_readonly("name", &ONNX_NAMESPACE::OpSchema::Name)
      .def_property_readonly("min_input", &ONNX_NAMESPACE::OpSchema::min_input)
      .def_property_readonly("max_input", &ONNX_NAMESPACE::OpSchema::max_input)
      .def_property_readonly("min_output", &ONNX_NAMESPACE::OpSchema::min_output)
      .def_property_readonly("max_output", &ONNX_NAMESPACE::OpSchema::max_output)
      .def_property_readonly("attributes", &ONNX_NAMESPACE::OpSchema::attributes)
      .def_property_readonly("inputs", &ONNX_NAMESPACE::OpSchema::inputs)
      .def_property_readonly("outputs", &ONNX_NAMESPACE::OpSchema::outputs)
      .def_property_readonly(
          "has_type_and_shape_inference_function",
          &ONNX_NAMESPACE::OpSchema::has_type_and_shape_inference_function)
      .def_property_readonly(
          "type_constraints", &ONNX_NAMESPACE::OpSchema::typeConstraintParams)
      .def_static("is_infinite", [](int v) {
        return v == std::numeric_limits<int>::max();
      });

  py::class_<ONNX_NAMESPACE::OpSchema::Attribute>(op_schema, "Attribute")
      .def_readonly("name", &ONNX_NAMESPACE::OpSchema::Attribute::name)
      .def_readonly("description", &ONNX_NAMESPACE::OpSchema::Attribute::description)
      .def_readonly("type", &ONNX_NAMESPACE::OpSchema::Attribute::type)
      .def_property_readonly(
          "_default_value",
          [](ONNX_NAMESPACE::OpSchema::Attribute* attr) -> py::bytes {
            std::string out;
            attr->default_value.SerializeToString(&out);
            return out;
          })
      .def_readonly("required", &ONNX_NAMESPACE::OpSchema::Attribute::required);

  py::class_<ONNX_NAMESPACE::OpSchema::TypeConstraintParam>(op_schema, "TypeConstraintParam")
      .def_readonly(
          "type_param_str", &ONNX_NAMESPACE::OpSchema::TypeConstraintParam::type_param_str)
      .def_readonly("description", &ONNX_NAMESPACE::OpSchema::TypeConstraintParam::description)
      .def_readonly(
          "allowed_type_strs",
          &ONNX_NAMESPACE::OpSchema::TypeConstraintParam::allowed_type_strs);

  py::enum_<ONNX_NAMESPACE::OpSchema::FormalParameterOption>(op_schema, "FormalParameterOption")
      .value("Single", ONNX_NAMESPACE::OpSchema::Single)
      .value("Optional", ONNX_NAMESPACE::OpSchema::Optional)
      .value("Variadic", ONNX_NAMESPACE::OpSchema::Variadic);

  py::class_<ONNX_NAMESPACE::OpSchema::FormalParameter>(op_schema, "FormalParameter")
      .def_property_readonly("name", &ONNX_NAMESPACE::OpSchema::FormalParameter::GetName)
      .def_property_readonly("types", &ONNX_NAMESPACE::OpSchema::FormalParameter::GetTypes)
      .def_property_readonly("typeStr", &ONNX_NAMESPACE::OpSchema::FormalParameter::GetTypeStr)
      .def_property_readonly(
          "description", &ONNX_NAMESPACE::OpSchema::FormalParameter::GetDescription)
      .def_property_readonly("option", &ONNX_NAMESPACE::OpSchema::FormalParameter::GetOption)
      .def_property_readonly(
          "isHomogeneous", &ONNX_NAMESPACE::OpSchema::FormalParameter::GetIsHomogeneous);

  py::enum_<ONNX_NAMESPACE::AttributeProto::AttributeType>(op_schema, "AttrType")
      .value("FLOAT", ONNX_NAMESPACE::AttributeProto::FLOAT)
      .value("INT", ONNX_NAMESPACE::AttributeProto::INT)
      .value("STRING", ONNX_NAMESPACE::AttributeProto::STRING)
      .value("TENSOR", ONNX_NAMESPACE::AttributeProto::TENSOR)
      .value("GRAPH", ONNX_NAMESPACE::AttributeProto::GRAPH)
      .value("FLOATS", ONNX_NAMESPACE::AttributeProto::FLOATS)
      .value("INTS", ONNX_NAMESPACE::AttributeProto::INTS)
      .value("STRINGS", ONNX_NAMESPACE::AttributeProto::STRINGS)
      .value("TENSORS", ONNX_NAMESPACE::AttributeProto::TENSORS)
      .value("GRAPHS", ONNX_NAMESPACE::AttributeProto::GRAPHS);

  py::enum_<ONNX_NAMESPACE::OpSchema::SupportType>(op_schema, "SupportType")
      .value("COMMON", ONNX_NAMESPACE::OpSchema::SupportType::COMMON)
      .value("EXPERIMENTAL", ONNX_NAMESPACE::OpSchema::SupportType::EXPERIMENTAL);
}

#endif  //onnxruntime_PYBIND_EXPORT_OPSCHEMA

void addObjectMethods(py::module& m) {
  py::enum_<GraphOptimizationLevel>(m, "GraphOptimizationLevel")
      .value("ORT_DISABLE_ALL", GraphOptimizationLevel::ORT_DISABLE_ALL)
      .value("ORT_ENABLE_BASIC", GraphOptimizationLevel::ORT_ENABLE_BASIC)
      .value("ORT_ENABLE_EXTENDED", GraphOptimizationLevel::ORT_ENABLE_EXTENDED)
      .value("ORT_ENABLE_ALL", GraphOptimizationLevel::ORT_ENABLE_ALL);

  py::enum_<ExecutionMode>(m, "ExecutionMode")
      .value("ORT_SEQUENTIAL", ExecutionMode::ORT_SEQUENTIAL)
      .value("ORT_PARALLEL", ExecutionMode::ORT_PARALLEL);

  py::class_<OrtDevice> device(m, "OrtDevice", R"pbdoc(ONNXRuntime device informaion.)pbdoc");
  device.def(py::init<OrtDevice::DeviceType, OrtDevice::MemoryType, OrtDevice::DeviceId>())
      .def("device_id", &OrtDevice::Id, R"pbdoc(Device Id.)pbdoc")
      .def("device_type", &OrtDevice::Type, R"pbdoc(Device Type.)pbdoc")
      .def_static("cpu", []() { return OrtDevice::CPU; })
      .def_static("cuda", []() { return OrtDevice::GPU; })
      .def_static("default_memory", []() { return OrtDevice::MemType::DEFAULT; });

  py::class_<SessionIOBinding> binding(m, "SessionIOBinding");
  binding
      .def(py::init<InferenceSession*>())
      .def("bind_input", [](SessionIOBinding* io_binding, const std::string& name, const OrtDevice& device, py::object element_type, std::vector<int64_t> shape, int64_t data_ptr) -> void {
        PyArray_Descr* dtype;
        if (!PyArray_DescrConverter(element_type.ptr(), &dtype))
          throw std::runtime_error("Not a valid numpy type");
        int type_num = dtype->type_num;
        Py_DECREF(dtype);

        std::string device_name = GetDeviceName(device);

        OrtMemoryInfo info(device_name.c_str(), OrtDeviceAllocator, device);

        std::unique_ptr<Tensor> p_tensor = onnxruntime::make_unique<Tensor>(NumpyTypeToOnnxRuntimeType(type_num), shape, (void*)data_ptr, info);
        OrtValue mlvalue;
        mlvalue.Init(p_tensor.release(),
                     DataTypeImpl::GetType<Tensor>(),
                     DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
        auto status = io_binding->Get()->BindInput(name, mlvalue);
        if (!status.IsOK())
          throw std::runtime_error("Error when bind input: " + status.ErrorMessage());
      })
      .def("bind_output", [](SessionIOBinding* io_binding, const std::string& name, const OrtDevice& device, py::object element_type, std::vector<int64_t> shape, int64_t data_ptr) -> void {
        PyArray_Descr* dtype;
        if (!PyArray_DescrConverter(element_type.ptr(), &dtype))
          throw std::runtime_error("Not a valid numpy type");
        int type_num = dtype->type_num;
        Py_DECREF(dtype);

        std::string device_name = GetDeviceName(device);

        OrtMemoryInfo info(device_name.c_str(), OrtDeviceAllocator, device);

        std::unique_ptr<Tensor> p_tensor = onnxruntime::make_unique<Tensor>(NumpyTypeToOnnxRuntimeType(type_num), shape, (void*)data_ptr, info);
        OrtValue mlvalue;
        mlvalue.Init(p_tensor.release(),
                     DataTypeImpl::GetType<Tensor>(),
                     DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
        auto status = io_binding->Get()->BindOutput(name, mlvalue);
        if (!status.IsOK())
          throw std::runtime_error("Error when bind input: " + status.ErrorMessage());
      })
      .def("clear_binding_inputs", [](SessionIOBinding* io_binding) -> void {
        io_binding->Get()->ClearInputs();
      })
      .def("clear_binding_outputs", [](SessionIOBinding* io_binding) -> void {
        io_binding->Get()->ClearOutputs();
      });

  py::class_<TrainingParameters> parameters(m, "TrainingParameters", R"pbdoc(Configuration information for training.)pbdoc");
  parameters.def(py::init())
      .def_readwrite("loss_output_name", &TrainingParameters::loss_output_name)
      .def_readwrite("immutable_weights", &TrainingParameters::immutable_weights)
      .def_readwrite("weights_not_to_train", &TrainingParameters::weights_not_to_train)
      .def_readwrite("weights_to_train", &TrainingParameters::weights_to_train)
      .def_readwrite("loss_scale_input_name", &TrainingParameters::loss_scale_input_name)
      .def_readwrite("scaled_loss_output_name", &TrainingParameters::scaled_loss_output_name)
      .def_readwrite("training_optimizer_name", &TrainingParameters::training_optimizer_name)
      .def_readwrite("lr_params_feed_name", &TrainingParameters::lr_params_feed_name)
      .def_readwrite("optimizer_attributes_map", &TrainingParameters::optimizer_attributes_map)
      .def_readwrite("use_fp16_moments", &TrainingParameters::use_fp16_moments)
      .def_readwrite("use_mixed_precision", &TrainingParameters::use_mixed_precision)
      .def_readwrite("allreduce_post_accumulation", &TrainingParameters::allreduce_post_accumulation)
      .def_readwrite("loss_scale", &TrainingParameters::loss_scale)
      .def_readwrite("world_rank", &TrainingParameters::world_rank)
      .def_readwrite("world_size", &TrainingParameters::world_size)
      .def_readwrite("gradient_accumulation_steps", &TrainingParameters::gradient_accumulation_steps)
      .def_readwrite("partition_optimizer", &TrainingParameters::partition_optimizer);

  py::class_<SessionOptions>
      sess(m, "SessionOptions", R"pbdoc(Configuration information for a session.)pbdoc");
  sess
      .def(py::init())
      .def_readwrite("enable_cpu_mem_arena", &SessionOptions::enable_cpu_mem_arena,
                     R"pbdoc(Enables the memory arena on CPU. Arena may pre-allocate memory for future usage.
Set this option to false if you don't want it. Default is True.)pbdoc")
      .def_readwrite("enable_profiling", &SessionOptions::enable_profiling,
                     R"pbdoc(Enable profiling for this session. Default is false.)pbdoc")
      .def_readwrite("optimized_model_filepath", &SessionOptions::optimized_model_filepath,
                     R"pbdoc(File path to serialize optimized model. By default, optimized model is not serialized if optimized_model_filepath is not provided.)pbdoc")
      .def_readwrite("enable_mem_pattern", &SessionOptions::enable_mem_pattern,
                     R"pbdoc(Enable the memory pattern optimization. Default is true.)pbdoc")
      .def_readwrite("logid", &SessionOptions::session_logid,
                     R"pbdoc(Logger id to use for session output.)pbdoc")
      .def_readwrite("log_severity_level", &SessionOptions::session_log_severity_level,
                     R"pbdoc(Log severity level. Applies to session load, initialization, etc.
0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2.)pbdoc")
      .def_readwrite("log_verbosity_level", &SessionOptions::session_log_verbosity_level,
                     R"pbdoc(VLOG level if DEBUG build and session_log_verbosity_level is 0.
Applies to session load, initialization, etc. Default is 0.)pbdoc")
      .def_readwrite("intra_op_num_threads", &SessionOptions::intra_op_num_threads,
                     R"pbdoc(Sets the number of threads used to parallelize the execution within nodes. Default is 0 to let onnxruntime choose.)pbdoc")
      .def_readwrite("inter_op_num_threads", &SessionOptions::inter_op_num_threads,
                     R"pbdoc(Sets the number of threads used to parallelize the execution of the graph (across nodes). Default is 0 to let onnxruntime choose.)pbdoc")
      .def_readwrite("execution_mode", &SessionOptions::execution_mode,
                     R"pbdoc(Sets the execution mode. Default is sequential.)pbdoc")
      .def_property(
          "graph_optimization_level",
          [](const SessionOptions* options) -> GraphOptimizationLevel {
            GraphOptimizationLevel retval = ORT_ENABLE_ALL;
            switch (options->graph_optimization_level) {
              case onnxruntime::TransformerLevel::Default:
                retval = ORT_DISABLE_ALL;
                break;
              case onnxruntime::TransformerLevel::Level1:
                retval = ORT_ENABLE_BASIC;
                break;
              case onnxruntime::TransformerLevel::Level2:
                retval = ORT_ENABLE_EXTENDED;
                break;
              case onnxruntime::TransformerLevel::Level3:
                retval = ORT_ENABLE_ALL;
                break;
              default:
                retval = ORT_ENABLE_ALL;
                LOGS_DEFAULT(WARNING) << "Got invalid graph optimization level; defaulting to ORT_ENABLE_ALL";
                break;
            }
            return retval;
          },

          [](SessionOptions* options, GraphOptimizationLevel level) -> void {
            switch (level) {
              case ORT_DISABLE_ALL:
                options->graph_optimization_level = onnxruntime::TransformerLevel::Default;
                break;
              case ORT_ENABLE_BASIC:
                options->graph_optimization_level = onnxruntime::TransformerLevel::Level1;
                break;
              case ORT_ENABLE_EXTENDED:
                options->graph_optimization_level = onnxruntime::TransformerLevel::Level2;
                break;
              case ORT_ENABLE_ALL:
                options->graph_optimization_level = onnxruntime::TransformerLevel::Level3;
                break;
            }
          },
          R"pbdoc(Graph optimization level for this session.)pbdoc");

  py::class_<RunOptions>(m, "RunOptions", R"pbdoc(Configuration information for a single Run.)pbdoc")
      .def(py::init())
      .def_readwrite("log_severity_level", &RunOptions::run_log_severity_level,
                     R"pbdoc(Log severity level for a particular Run() invocation. 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2.)pbdoc")
      .def_readwrite("log_verbosity_level", &RunOptions::run_log_verbosity_level,
                     R"pbdoc(VLOG level if DEBUG build and run_log_severity_level is 0.
Applies to a particular Run() invocation. Default is 0.)pbdoc")
      .def_readwrite("logid", &RunOptions::run_tag,
                     "To identify logs generated by a particular Run() invocation.")
      .def_readwrite("terminate", &RunOptions::terminate,
                     R"pbdoc(Set to True to terminate any currently executing calls that are using this
RunOptions instance. The individual calls will exit gracefully and return an error status.)pbdoc")
      .def_readwrite("only_execute_path_to_fetches", &RunOptions::only_execute_path_to_fetches,
                     R"pbdoc(Only execute the nodes needed by fetch list)pbdoc");

  py::class_<ModelMetadata>(m, "ModelMetadata", R"pbdoc(Pre-defined and custom metadata about the model.
It is usually used to identify the model used to run the prediction and
facilitate the comparison.)pbdoc")
      .def_readwrite("producer_name", &ModelMetadata::producer_name, "producer name")
      .def_readwrite("graph_name", &ModelMetadata::graph_name, "graph name")
      .def_readwrite("domain", &ModelMetadata::domain, "ONNX domain")
      .def_readwrite("description", &ModelMetadata::description, "description of the model")
      .def_readwrite("version", &ModelMetadata::version, "version of the model")
      .def_readwrite("custom_metadata_map", &ModelMetadata::custom_metadata_map, "additional metadata");

  py::class_<onnxruntime::NodeArg>(m, "NodeArg", R"pbdoc(Node argument definition, for both input and output,
including arg name, arg type (contains both type and shape).)pbdoc")
      .def_property_readonly("name", &onnxruntime::NodeArg::Name, "node name")
      .def_property_readonly(
          "type", [](const onnxruntime::NodeArg& na) -> std::string {
            return *(na.Type());
          },
          "node type")
      .def("__str__", [](const onnxruntime::NodeArg& na) -> std::string {
        std::ostringstream res;
        res << "NodeArg(name='" << na.Name() << "', type='" << *(na.Type()) << "', shape=";
        auto shape = na.Shape();
        std::vector<py::object> arr;
        if (shape == nullptr || shape->dim_size() == 0) {
          res << "[]";
        } else {
          res << "[";
          for (int i = 0; i < shape->dim_size(); ++i) {
            if (utils::HasDimValue(shape->dim(i))) {
              res << shape->dim(i).dim_value();
            } else if (utils::HasDimParam(shape->dim(i))) {
              res << "'" << shape->dim(i).dim_param() << "'";
            } else {
              res << "None";
            }

            if (i < shape->dim_size() - 1) {
              res << ", ";
            }
          }
          res << "]";
        }
        res << ")";

        return std::string(res.str());
      },
           "converts the node into a readable string")
      .def_property_readonly("shape", [](const onnxruntime::NodeArg& na) -> std::vector<py::object> {
        auto shape = na.Shape();
        std::vector<py::object> arr;
        if (shape == nullptr || shape->dim_size() == 0) {
          return arr;
        }

        arr.resize(shape->dim_size());
        for (int i = 0; i < shape->dim_size(); ++i) {
          if (utils::HasDimValue(shape->dim(i))) {
            arr[i] = py::cast(shape->dim(i).dim_value());
          } else if (utils::HasDimParam(shape->dim(i))) {
            arr[i] = py::cast(shape->dim(i).dim_param());
          } else {
            arr[i] = py::none();
          }
        }
        return arr;
      },
                             "node shape (assuming the node holds a tensor)");

  py::class_<SessionObjectInitializer>(m, "SessionObjectInitializer");
  py::class_<InferenceSession>(m, "InferenceSession", R"pbdoc(This is the main class used to run a model.)pbdoc")
      // In Python3, a Python bytes object will be passed to C++ functions that accept std::string or char*
      // without any conversion. So this init method can be used for model file path (string)
      // and model content (bytes)
      .def(py::init([](const SessionOptions& so, const std::string& arg, bool is_arg_file_name) {
        // Given arg is the file path. Invoke the corresponding ctor().
        if (is_arg_file_name) {
          return onnxruntime::make_unique<InferenceSession>(so, arg, SessionObjectInitializer::Get());
        }

        // Given arg is the model content as bytes. Invoke the corresponding ctor().
        std::istringstream buffer(arg);
        return onnxruntime::make_unique<InferenceSession>(so, buffer, SessionObjectInitializer::Get());
      }))
      .def(
          "load_model", [](InferenceSession* sess, std::vector<std::string>& provider_types) {
            OrtPybindThrowIfError(sess->Load());
            InitializeSession(sess, provider_types);
          },
          R"pbdoc(Load a model saved in ONNX format.)pbdoc")
      .def("run", [](InferenceSession* sess, std::vector<std::string> output_names, std::map<std::string, py::object> pyfeeds, RunOptions* run_options = nullptr) -> std::vector<py::object> {
        NameMLValMap feeds;
        for (auto _ : pyfeeds) {
          OrtValue ml_value;
          auto px = sess->GetModelInputs();
          if (!px.first.IsOK() || !px.second) {
            throw std::runtime_error("Either failed to get model inputs from the session object or the input def list was null");
          }
          CreateGenericMLValue(px.second, GetAllocator(), _.first, _.second, &ml_value);
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
            throw std::runtime_error(sType);
          }
          feeds.insert(std::make_pair(_.first, ml_value));
        }

        std::vector<OrtValue> fetches;
        common::Status status;

        {
          // release GIL to allow multiple python threads to invoke Run() in parallel.
          py::gil_scoped_release release;
          if (run_options != nullptr) {
            OrtPybindThrowIfError(sess->Run(*run_options, feeds, output_names, &fetches));
          } else {
            OrtPybindThrowIfError(sess->Run(feeds, output_names, &fetches));
          }
        }

        std::vector<py::object> rfetch;
        rfetch.reserve(fetches.size());
        for (auto _ : fetches) {
          if (_.IsTensor()) {
            AddTensorAsPyObj(_, rfetch);
          } else {
            AddNonTensorAsPyObj(_, rfetch);
          }
        }
        return rfetch;
      })
      .def("end_profiling", [](InferenceSession* sess) -> std::string {
        return sess->EndProfiling();
      })
      .def("get_providers", [](InferenceSession* sess) -> const std::vector<std::string>& {
        return sess->GetRegisteredProviderTypes();
      })
      .def_property_readonly("session_options", [](InferenceSession* sess) -> const SessionOptions& {
        return sess->GetSessionOptions();
      })
      .def_property_readonly("inputs_meta", [](const InferenceSession* sess) -> const std::vector<const onnxruntime::NodeArg*>& {
        auto res = sess->GetModelInputs();
        OrtPybindThrowIfError(res.first);
        return *(res.second);
      })
      .def_property_readonly("outputs_meta", [](const InferenceSession* sess) -> const std::vector<const onnxruntime::NodeArg*>& {
        auto res = sess->GetModelOutputs();
        OrtPybindThrowIfError(res.first);
        return *(res.second);
      })
      .def_property_readonly("overridable_initializers", [](const InferenceSession* sess) -> const std::vector<const onnxruntime::NodeArg*>& {
        auto res = sess->GetOverridableInitializers();
        OrtPybindThrowIfError(res.first);
        return *(res.second);
      })
      .def_property_readonly("model_meta", [](const InferenceSession* sess) -> const onnxruntime::ModelMetadata& {
        auto res = sess->GetModelMetadata();
        OrtPybindThrowIfError(res.first);
        return *(res.second);
      })
      .def("run_with_iobinding", [](InferenceSession* sess, SessionIOBinding& io_binding, RunOptions* run_options = nullptr) -> void {
        Status status;
        if (!run_options)
          status = sess->Run(*io_binding.Get());
        else
          status = sess->Run(*run_options, *io_binding.Get());
        if (!status.IsOK())
          throw std::runtime_error("Error in execution: " + status.ErrorMessage());
      });

  py::class_<onnxruntime::training::TrainingSession, InferenceSession> training_session(m, "TrainingSession");
  training_session.def(py::init<SessionOptions, SessionObjectInitializer>())
      .def(py::init<SessionObjectInitializer, SessionObjectInitializer>())
      .def("finalize", [](py::object) {
#ifdef USE_HOROVOD
        shutdown_horovod();
#endif
      })
      .def("load_model", [](onnxruntime::training::TrainingSession* sess, const std::string& path, TrainingParameters& parameters) {
        OrtPybindThrowIfError(sess->Load(path));

        ConfigureSessionForTraining(sess, parameters);

        std::vector<std::string> provider_types = {};
        InitializeSession(sess, provider_types);
      })
      .def("read_bytes", [](onnxruntime::training::TrainingSession* sess, const py::bytes& serialized_model, TrainingParameters& parameters) {
        std::istringstream buffer(serialized_model);
        OrtPybindThrowIfError(sess->Load(buffer));

        ConfigureSessionForTraining(sess, parameters);

        std::vector<std::string> provider_types = {};
        InitializeSession(sess, provider_types);
      })
      .def("get_state", [](onnxruntime::training::TrainingSession* sess) {
        NameMLValMap state_tensors;
        ORT_THROW_IF_ERROR(sess->GetStateTensors(state_tensors));
        auto& data_transfer_manager = sess->GetDataTransferManager();
        //convert to numpy array
        std::map<std::string, py::object> rmap;
        for (auto& kv : state_tensors) {
          if (kv.second.IsTensor()) {
            py::object obj;
            const Tensor& rtensor = kv.second.Get<Tensor>();
            GetPyObjFromTensor(rtensor, obj, &data_transfer_manager);
            rmap.insert({kv.first, obj});
          } else {
            throw std::runtime_error("Non tensor type in session state tensors is not expected.");
          }
        }
        return rmap;
      })
      .def("load_state", [](onnxruntime::training::TrainingSession* sess, std::unordered_map<std::string, py::object>& state, bool strict) {
        NameMLValMap state_tensors;
        for (auto initializer : state) {
          OrtValue ml_value;
          auto px = sess->GetModelInputs();
          if (!px.first.IsOK() || !px.second) {
            throw std::runtime_error("Either failed to get model inputs from the session object or the input def list was null");
          }
          CreateGenericMLValue(px.second, GetAllocator(), initializer.first, initializer.second, &ml_value);
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
            throw std::runtime_error(sType);
          }
          state_tensors.insert(std::make_pair(initializer.first, ml_value));
        }
        ORT_THROW_IF_ERROR(sess->SetStateTensors(state_tensors, strict));
      });

  py::enum_<onnxruntime::ArenaExtendStrategy>(m, "ArenaExtendStrategy", py::arithmetic())
      .value("kNextPowerOfTwo", onnxruntime::ArenaExtendStrategy::kNextPowerOfTwo)
      .value("kSameAsRequested", onnxruntime::ArenaExtendStrategy::kSameAsRequested)
      .export_values();
}

#if defined(USE_MIMALLOC_ARENA_ALLOCATOR)
static struct {
  PyMemAllocatorEx mem;
  PyMemAllocatorEx raw;
  PyMemAllocatorEx obj;
} allocators;
#endif

PYBIND11_MODULE(onnxruntime_pybind11_state, m) {
  m.doc() = "pybind11 stateful interface to ONNX runtime";
  RegisterExceptions(m);

#if defined(USE_MIMALLOC_ARENA_ALLOCATOR)
  PyMemAllocatorEx alloc;
  alloc.malloc = [](void* ctx, size_t size) {
    ORT_UNUSED_PARAMETER(ctx);
    return mi_malloc(size);
  };

  alloc.calloc = [](void* ctx, size_t nelem, size_t elsize) {
    ORT_UNUSED_PARAMETER(ctx);
    return mi_calloc(nelem, elsize);
  };

  alloc.realloc = [](void* ctx, void* ptr, size_t new_size) {
    if (mi_is_in_heap_region(ptr)) {
      return mi_realloc(ptr, new_size);
    } else {
      PyMemAllocatorEx* a = (PyMemAllocatorEx*)ctx;
      return a->realloc(ctx, ptr, new_size);
    }
  };

  alloc.free = [](void* ctx, void* ptr) {
    if (mi_is_in_heap_region(ptr)) {
      mi_free(ptr);
    } else {
      PyMemAllocatorEx* a = (PyMemAllocatorEx*)ctx;
      a->free(ctx, ptr);
    }
  };

  alloc.ctx = &allocators.raw;
  PyMem_GetAllocator(PYMEM_DOMAIN_RAW, &allocators.raw);
  PyMem_SetAllocator(PYMEM_DOMAIN_RAW, &alloc);

  alloc.ctx = &allocators.mem;
  PyMem_GetAllocator(PYMEM_DOMAIN_MEM, &allocators.mem);
  PyMem_SetAllocator(PYMEM_DOMAIN_MEM, &alloc);

  alloc.ctx = &allocators.obj;
  PyMem_GetAllocator(PYMEM_DOMAIN_OBJ, &allocators.obj);
  PyMem_SetAllocator(PYMEM_DOMAIN_OBJ, &alloc);

#endif

  auto initialize = [&]() {
    // Initialization of the module
    ([]() -> void {
      // import_array1() forces a void return value.
      import_array1();
    })();

    static std::unique_ptr<Environment> env;
    OrtPybindThrowIfError(Environment::Create(env));

    static bool initialized = false;
    if (initialized) {
      return;
    }
    initialized = true;
  };
  initialize();

  addGlobalMethods(m);
  addObjectMethods(m);

#ifdef onnxruntime_PYBIND_EXPORT_OPSCHEMA
  addOpSchemaSubmodule(m);
  addOpKernelSubmodule(m);
#endif
}

}  // namespace python
}  // namespace onnxruntime
