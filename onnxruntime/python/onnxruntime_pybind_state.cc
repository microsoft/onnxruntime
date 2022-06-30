// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/onnxruntime_pybind_exceptions.h"
#include "python/onnxruntime_pybind_mlvalue.h"
#include "python/onnxruntime_pybind_state_common.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL onnxruntime_python_ARRAY_API
#include <numpy/arrayobject.h>

#include "core/common/inlined_containers.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/severity.h"
#include "core/common/optional.h"
#include "core/framework/arena_extend_strategy.h"
#include "core/framework/data_transfer_utils.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/provider_options_utils.h"
#include "core/framework/random_seed.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/TensorSeq.h"
#include "core/graph/graph_viewer.h"
#include "core/platform/env.h"
#include "core/providers/get_execution_providers.h"
#include "core/providers/tensorrt/tensorrt_provider_options.h"
#include "core/session/IOBinding.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/provider_bridge_ort.h"

#ifdef ENABLE_ATEN
#include "contrib_ops/cpu/aten_ops/aten_op_executor.h"
#endif

// Explicitly provide a definition for the static const var 'GPU' in the OrtDevice struct,
// GCC 4.x doesn't seem to define this and it breaks the pipelines based on CentOS as it uses
// GCC 4.x.
// (This static var is referenced in GetCudaToHostMemCpyFunction())
const OrtDevice::DeviceType OrtDevice::GPU;

namespace onnxruntime {

}  // namespace onnxruntime

#if defined(_MSC_VER)
#pragma warning(disable : 4267 4996 4503 4003)
#endif  // _MSC_VER

#include <iterator>

#if defined(_MSC_VER)
#pragma warning(disable : 4267 4996 4503 4003)
#endif  // _MSC_VER

namespace onnxruntime {
namespace python {

namespace py = pybind11;
using namespace onnxruntime;
using namespace onnxruntime::logging;

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
// "Global initializer calls a non-constexpr function." Therefore you can't use ORT APIs in the other global initializers.
// TODO: we may delay-init this variable
#pragma warning(disable : 26426)
#endif
static Env& platform_env = Env::Default();
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#endif

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
// Custom op section starts

CustomOpLibrary::CustomOpLibrary(const char* library_path, OrtSessionOptions& ort_so) {
  {
    OrtPybindThrowIfError(platform_env.LoadDynamicLibrary(library_path, false, &library_handle_));

    OrtStatus*(ORT_API_CALL * RegisterCustomOps)(OrtSessionOptions * options, const OrtApiBase* api);

    OrtPybindThrowIfError(platform_env.GetSymbolFromLibrary(library_handle_, "RegisterCustomOps", (void**)&RegisterCustomOps));

    auto* status_raw = RegisterCustomOps(&ort_so, OrtGetApiBase());
    // Manage the raw Status pointer using a smart pointer
    auto status = std::unique_ptr<OrtStatus>(status_raw);

    // A non-nullptr indicates status indicates some error
    if (status) {
      // TODO: How to handle unload failure ?
      // Currently we ignore the returned status assuming it is successful
      ORT_IGNORE_RETURN_VALUE(platform_env.UnloadDynamicLibrary(library_handle_));

      // Construct error message string
      std::string error_string = status->msg;

      // Throw
      throw std::runtime_error(error_string);
    }

    library_path_ = std::string(library_path);
  }
}

// Unload the library when the destructor is triggered
CustomOpLibrary::~CustomOpLibrary() {
  UnloadLibrary();
}

// Logic to unload the library
void CustomOpLibrary::UnloadLibrary() {
  auto status = platform_env.UnloadDynamicLibrary(library_handle_);

  if (!status.IsOK()) {
    const logging::Logger& default_logger = logging::LoggingManager::DefaultLogger();
    LOGS(default_logger, WARNING) << "Unable to unload the custom op shared library: " << library_path_;
  }
}

// Custom op section ends
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)

template <typename T>
static py::object AddNonTensor(const OrtValue& val,
                               const DataTransferManager* /*data_transfer_manager*/,
                               const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* /*mem_cpy_to_host_functions*/) {
  return py::cast(val.Get<T>());
}

// In all cases, we may not have access to a DataTransferManager, hence the user may specify functions that
// pretty much does what a DataTransferManager does - copy data from device(s) to the host
void GetPyObjFromTensor(const Tensor& rtensor, py::object& obj,
                        const DataTransferManager* data_transfer_manager,
                        const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* mem_cpy_to_host_functions) {
  std::vector<npy_intp> npy_dims;
  const TensorShape& shape = rtensor.Shape();

  for (size_t n = 0; n < shape.NumDimensions(); ++n) {
    npy_dims.push_back(shape[n]);
  }

  MLDataType dtype = rtensor.DataType();
  const int numpy_type = OnnxRuntimeTensorToNumpyType(dtype);
  obj = py::reinterpret_steal<py::object>(PyArray_SimpleNew(
      shape.NumDimensions(), npy_dims.data(), numpy_type));

  void* out_ptr = static_cast<void*>(
      PyArray_DATA(reinterpret_cast<PyArrayObject*>(obj.ptr())));

  if (numpy_type != NPY_OBJECT) {
    // if it is not cpu tensor, need to copy to host
    auto device_type = rtensor.Location().device.Type();
    if (device_type != OrtDevice::CPU) {
      if (!data_transfer_manager && !mem_cpy_to_host_functions)
        throw std::runtime_error(
            "GetPyObjFromTensor: Either data transfer manager or a "
            "function to copy data to the host is needed to convert non-CPU tensor to numpy array");
      static const OrtMemoryInfo cpu_alloc_info{onnxruntime::CPU, OrtDeviceAllocator};

      // Prefer DataTransferManager if available
      if (data_transfer_manager) {
        auto span = gsl::make_span<char>(reinterpret_cast<char*>(out_ptr), dtype->Size() * shape.Size());
        ORT_THROW_IF_ERROR(CopyTensorDataToByteSpan(
            *data_transfer_manager, rtensor, cpu_alloc_info, span));
      } else {
        auto mem_cpy_to_host = mem_cpy_to_host_functions->find(device_type);

        ORT_ENFORCE(mem_cpy_to_host != mem_cpy_to_host_functions->end(),
                    "Unable to locate a function that can copy data to the host from the device");

        ORT_ENFORCE(mem_cpy_to_host->second != 0,
                    "No function that can copy data to the host from the device provided");

        mem_cpy_to_host->second(out_ptr, rtensor.DataRaw(), dtype->Size() * shape.Size());
      }

    } else
      memcpy(out_ptr, rtensor.DataRaw(dtype), dtype->Size() * shape.Size());
  } else {
    // Handle string type.
    // Copying strings to cpu from device is currently not supported
    ORT_ENFORCE(rtensor.Location().device.Type() == OrtDevice::CPU,
                "Copying string tensors located on another device to the host is currently not supported");
    py::object* outObj = static_cast<py::object*>(out_ptr);
    const std::string* src = rtensor.template Data<std::string>();
    for (int i = 0; i < rtensor.Shape().Size(); i++, src++) {
      outObj[i] = py::cast(*src);
    }
  }
}

const char* GetDeviceName(const OrtDevice& device) {
  switch (device.Type()) {
    case OrtDevice::CPU:
      return CPU;
    case OrtDevice::GPU:
      return CUDA;
    case OrtDevice::FPGA:
      return "FPGA";
    default:
      ORT_THROW("Unknown device type: ", device.Type());
  }
}

py::object GetPyObjectFromSparseTensor(size_t pos, const OrtValue& ort_value, const DataTransferManager* data_transfer_manager) {
#if !defined(DISABLE_SPARSE_TENSORS)
  if (!ort_value.IsSparseTensor()) {
    ORT_THROW("Must be a sparse tensor");
  }
  auto& logger = logging::LoggingManager::DefaultLogger();
  const SparseTensor& src_sparse_tensor = ort_value.Get<SparseTensor>();
  std::unique_ptr<PySparseTensor> py_sparse_tensor;
  auto device_type = src_sparse_tensor.Location().device.Type();
  if (device_type != OrtDevice::CPU) {
    if (!data_transfer_manager) {
      LOGS(logger, WARNING) << "Returned OrtValue with sparse tensor at position: " << pos << " is on GPU but no data_transfer_manager provided."
                            << " Returned it will have its data on GPU, you can copy it using numpy_array_to_cpu()";
      py_sparse_tensor = std::make_unique<PySparseTensor>(ort_value);
    } else {
      auto dst_sparse_tensor = std::make_unique<SparseTensor>(src_sparse_tensor.DataType(), src_sparse_tensor.DenseShape(), GetAllocator());
      auto status = src_sparse_tensor.Copy(*data_transfer_manager, 0, *dst_sparse_tensor);
      OrtPybindThrowIfError(status);
      py_sparse_tensor = std::make_unique<PySparseTensor>(std::move(dst_sparse_tensor));
    }
  } else {
    py_sparse_tensor = std::make_unique<PySparseTensor>(ort_value);
  }

  py::object result = py::cast(py_sparse_tensor.get(), py::return_value_policy::take_ownership);
  py_sparse_tensor.release();
  return result;
#else
  ORT_UNUSED_PARAMETER(pos);
  ORT_UNUSED_PARAMETER(ort_value);
  ORT_UNUSED_PARAMETER(data_transfer_manager);
  ORT_THROW("SparseTensor support is disabled in this build.");
#endif  // !defined(DISABLE_SPARSE_TENSORS)
}

template <>
py::object AddNonTensor<TensorSeq>(const OrtValue& val,
                                   const DataTransferManager* data_transfer_manager,
                                   const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* mem_cpy_to_host_functions) {
  const auto& seq_tensors = val.Get<TensorSeq>();
  py::list py_list;
  for (const auto& rtensor : seq_tensors) {
    py::object obj;
    GetPyObjFromTensor(rtensor, obj, data_transfer_manager, mem_cpy_to_host_functions);
    py_list.append(obj);
  }
  // XToolChain kills the build
  // local variable 'py_list' will be copied despite being returned by name [-Werror,-Wreturn-std-move]
  // call 'std::move' explicitly to avoid copying
  // We choose to cast it to object explicitly
  return py::cast<py::object>(py_list);
}

py::object AddNonTensorAsPyObj(const OrtValue& val,
                               const DataTransferManager* data_transfer_manager,
                               const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* mem_cpy_to_host_functions) {
  // Should be in sync with core/framework/datatypes.h
  auto val_type = val.Type();
  if (val_type->IsTensorSequenceType()) {
    return AddNonTensor<TensorSeq>(val, data_transfer_manager, mem_cpy_to_host_functions);
  } else {
#if !defined(DISABLE_ML_OPS)
    utils::ContainerChecker c_checker(val_type);
    if (c_checker.IsMap()) {
      if (c_checker.IsMapOf<std::string, std::string>()) {
        return AddNonTensor<MapStringToString>(val, data_transfer_manager, mem_cpy_to_host_functions);
      } else if (c_checker.IsMapOf<std::string, int64_t>()) {
        return AddNonTensor<MapStringToInt64>(val, data_transfer_manager, mem_cpy_to_host_functions);
      } else if (c_checker.IsMapOf<std::string, float>()) {
        return AddNonTensor<MapStringToFloat>(val, data_transfer_manager, mem_cpy_to_host_functions);
      } else if (c_checker.IsMapOf<std::string, double>()) {
        return AddNonTensor<MapStringToDouble>(val, data_transfer_manager, mem_cpy_to_host_functions);
      } else if (c_checker.IsMapOf<int64_t, std::string>()) {
        return AddNonTensor<MapInt64ToString>(val, data_transfer_manager, mem_cpy_to_host_functions);
      } else if (c_checker.IsMapOf<int64_t, int64_t>()) {
        return AddNonTensor<MapInt64ToInt64>(val, data_transfer_manager, mem_cpy_to_host_functions);
      } else if (c_checker.IsMapOf<int64_t, float>()) {
        return AddNonTensor<MapInt64ToFloat>(val, data_transfer_manager, mem_cpy_to_host_functions);
      } else if (c_checker.IsMapOf<int64_t, double>()) {
        return AddNonTensor<MapInt64ToDouble>(val, data_transfer_manager, mem_cpy_to_host_functions);
      }

    } else {
      if (c_checker.IsSequenceOf<std::map<std::string, float>>()) {
        return AddNonTensor<VectorMapStringToFloat>(val, data_transfer_manager, mem_cpy_to_host_functions);
      } else if (c_checker.IsSequenceOf<std::map<int64_t, float>>()) {
        return AddNonTensor<VectorMapInt64ToFloat>(val, data_transfer_manager, mem_cpy_to_host_functions);
      }
    }
#endif
  }
  ORT_THROW("Non-tensor type is not supported in this build: ", val_type);
}

py::object AddTensorAsPyObj(const OrtValue& val, const DataTransferManager* data_transfer_manager,
                            const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* mem_cpy_to_host_functions) {
  const Tensor& rtensor = val.Get<Tensor>();
  py::object obj;
  GetPyObjFromTensor(rtensor, obj, data_transfer_manager, mem_cpy_to_host_functions);
  return obj;
}

static std::unique_ptr<onnxruntime::IExecutionProvider> LoadExecutionProvider(
    const std::string& ep_shared_lib_path,
    const ProviderOptions& provider_options = {},
    const std::string& entry_symbol_name = "GetProvider") {
  void* handle;
  auto error = Env::Default().LoadDynamicLibrary(ep_shared_lib_path, false, &handle);
  if (!error.IsOK()) {
    throw std::runtime_error(error.ErrorMessage());
  }

  Provider* (*PGetProvider)();
  OrtPybindThrowIfError(Env::Default().GetSymbolFromLibrary(handle, entry_symbol_name, (void**)&PGetProvider));

  Provider* provider = PGetProvider();
  std::shared_ptr<IExecutionProviderFactory> ep_factory = provider->CreateExecutionProviderFactory(&provider_options);
  return ep_factory->CreateProvider();
}

#ifdef USE_CUDA
const CUDAExecutionProviderInfo GetCudaExecutionProviderInfo(ProviderInfo_CUDA* cuda_provider_info,
                                                             const ProviderOptionsMap& provider_options_map) {
  ORT_ENFORCE(cuda_provider_info);
  const auto it = provider_options_map.find(kCudaExecutionProvider);
  CUDAExecutionProviderInfo info;
  if (it != provider_options_map.end())
    cuda_provider_info->CUDAExecutionProviderInfo__FromProviderOptions(it->second, info);
  else {
    info.device_id = cuda_device_id;
    info.gpu_mem_limit = gpu_mem_limit;
    info.arena_extend_strategy = arena_extend_strategy;
    info.cudnn_conv_algo_search = cudnn_conv_algo_search;
    info.do_copy_in_default_stream = do_copy_in_default_stream;
    info.external_allocator_info = external_allocator_info;
  }
  return info;
}
#endif

#ifdef USE_ROCM
const ROCMExecutionProviderInfo GetRocmExecutionProviderInfo(ProviderInfo_ROCM* rocm_provider_info,
                                                             const ProviderOptionsMap& provider_options_map) {
  ORT_ENFORCE(rocm_provider_info);
  const auto it = provider_options_map.find(kRocmExecutionProvider);
  ROCMExecutionProviderInfo info;
  if (it != provider_options_map.end())
    rocm_provider_info->ROCMExecutionProviderInfo__FromProviderOptions(it->second, info);
  else {
    info.device_id = cuda_device_id;
    info.gpu_mem_limit = gpu_mem_limit;
    info.arena_extend_strategy = arena_extend_strategy;
    info.miopen_conv_exhaustive_search = miopen_conv_exhaustive_search;
    info.do_copy_in_default_stream = do_copy_in_default_stream;
    info.external_allocator_info = external_allocator_info;
  }
  return info;
}
#endif

std::unique_ptr<IExecutionProvider> CreateExecutionProviderInstance(
    const SessionOptions& session_options,
    const std::string& type,
    const ProviderOptionsMap& provider_options_map) {
  if (type == kCpuExecutionProvider) {
    return onnxruntime::CPUProviderFactoryCreator::Create(
               session_options.enable_cpu_mem_arena)
        ->CreateProvider();
  } else if (type == kTensorrtExecutionProvider) {
#ifdef USE_TENSORRT
    // If the environment variable 'ORT_TENSORRT_UNAVAILABLE' exists, then we do not load TensorRT. This is set by _ld_preload for the manylinux case
    // as in that case, trying to load the library itself will result in a crash due to the way that auditwheel strips dependencies.
    if (Env::Default().GetEnvironmentVar("ORT_TENSORRT_UNAVAILABLE").empty()) {
      std::string calibration_table, cache_path, lib_path;
      auto it = provider_options_map.find(type);
      if (it != provider_options_map.end()) {
        OrtTensorRTProviderOptionsV2 params{
            0,
            0,
            nullptr,
            1000,
            1,
            1 << 30,
            0,
            0,
            nullptr,
            0,
            0,
            0,
            0,
            0,
            nullptr,
            0,
            nullptr,
            0,
            0};
        for (auto option : it->second) {
          if (option.first == "device_id") {
            if (!option.second.empty()) {
              params.device_id = std::stoi(option.second);
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'device_id' should be a number i.e. '0'.\n");
            }
          } else if (option.first == "trt_max_partition_iterations") {
            if (!option.second.empty()) {
              params.trt_max_partition_iterations = std::stoi(option.second);
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_max_partition_iterations' should be a positive integer number i.e. '1000'.\n");
            }
          } else if (option.first == "trt_min_subgraph_size") {
            if (!option.second.empty()) {
              params.trt_min_subgraph_size = std::stoi(option.second);
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_min_subgraph_size' should be a positive integer number i.e. '1'.\n");
            }
          } else if (option.first == "trt_max_workspace_size") {
            if (!option.second.empty()) {
              params.trt_max_workspace_size = std::stoull(option.second);
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_max_workspace_size' should be a number in byte i.e. '1073741824'.\n");
            }
          } else if (option.first == "trt_fp16_enable") {
            if (option.second == "True" || option.second == "true") {
              params.trt_fp16_enable = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_fp16_enable = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_fp16_enable' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_int8_enable") {
            if (option.second == "True" || option.second == "true") {
              params.trt_int8_enable = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_int8_enable = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_int8_enable' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_int8_calibration_table_name") {
            if (!option.second.empty()) {
              calibration_table = option.second;
              params.trt_int8_calibration_table_name = calibration_table.c_str();
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_int8_calibration_table_name' should be a file name i.e. 'cal_table'.\n");
            }
          } else if (option.first == "trt_int8_use_native_calibration_table") {
            if (option.second == "True" || option.second == "true") {
              params.trt_int8_use_native_calibration_table = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_int8_use_native_calibration_table = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_int8_use_native_calibration_table' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_dla_enable") {
            if (option.second == "True" || option.second == "true") {
              params.trt_dla_enable = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_dla_enable = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_dla_enable' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_dla_core") {
            if (!option.second.empty()) {
              params.trt_dla_core = std::stoi(option.second);
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_dla_core' should be a positive integer number i.e. '0'.\n");
            }
          } else if (option.first == "trt_dump_subgraphs") {
            if (option.second == "True" || option.second == "true") {
              params.trt_dump_subgraphs = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_dump_subgraphs = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_dump_subgraphs' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_engine_cache_enable") {
            if (option.second == "True" || option.second == "true") {
              params.trt_engine_cache_enable = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_engine_cache_enable = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_engine_cache_enable' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_engine_cache_path") {
            if (!option.second.empty()) {
              cache_path = option.second;
              params.trt_engine_cache_path = cache_path.c_str();
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_engine_cache_path' should be a path string i.e. 'engine_cache'.\n");
            }
          } else if (option.first == "trt_engine_decryption_enable") {
            if (option.second == "True" || option.second == "true") {
              params.trt_engine_decryption_enable = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_engine_decryption_enable = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_engine_decryption_enable' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_engine_decryption_lib_path") {
            if (!option.second.empty()) {
              lib_path = option.second;
              params.trt_engine_decryption_lib_path = lib_path.c_str();
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_engine_decryption_lib_path' should be a path string i.e. 'decryption_lib'.\n");
            }
          } else if (option.first == "trt_force_sequential_engine_build") {
            if (option.second == "True" || option.second == "true") {
              params.trt_force_sequential_engine_build = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_force_sequential_engine_build = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_force_sequential_engine_build' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_context_memory_sharing_enable") {
            if (option.second == "True" || option.second == "true") {
              params.trt_context_memory_sharing_enable = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_context_memory_sharing_enable = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_context_memory_sharing_enable' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else {
            ORT_THROW("Invalid TensorRT EP option: ", option.first);
          }
        }
        if (std::shared_ptr<IExecutionProviderFactory> tensorrt_provider_factory = onnxruntime::TensorrtProviderFactoryCreator::Create(&params)) {
          return tensorrt_provider_factory->CreateProvider();
        }
      } else {
        if (std::shared_ptr<IExecutionProviderFactory> tensorrt_provider_factory = onnxruntime::TensorrtProviderFactoryCreator::Create(cuda_device_id)) {
          return tensorrt_provider_factory->CreateProvider();
        }
      }
    }
    LOGS_DEFAULT(WARNING) << "Failed to create " << type << ". Please reference https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements to ensure all dependencies are met.";
#endif
  } else if (type == kMIGraphXExecutionProvider) {
#ifdef USE_MIGRAPHX
    return onnxruntime::MIGraphXProviderFactoryCreator::Create(0)->CreateProvider();
#endif
  } else if (type == kCudaExecutionProvider) {
#ifdef USE_CUDA
    // If the environment variable 'CUDA_UNAVAILABLE' exists, then we do not load cuda. This is set by _ld_preload for the manylinux case
    // as in that case, trying to load the library itself will result in a crash due to the way that auditwheel strips dependencies.
    if (Env::Default().GetEnvironmentVar("ORT_CUDA_UNAVAILABLE").empty()) {
      if (auto* cuda_provider_info = TryGetProviderInfo_CUDA()) {
        const CUDAExecutionProviderInfo info = GetCudaExecutionProviderInfo(cuda_provider_info,
                                                                            provider_options_map);

        // This variable is never initialized because the APIs by which it should be initialized are deprecated, however they still
        // exist are are in-use. Neverthless, it is used to return CUDAAllocator, hence we must try to initialize it here if we can
        // since FromProviderOptions might contain external CUDA allocator.
        external_allocator_info = info.external_allocator_info;
        return cuda_provider_info->CreateExecutionProviderFactory(info)->CreateProvider();
      } else {
        if (!Env::Default().GetEnvironmentVar("CUDA_PATH").empty()) {
          ORT_THROW("CUDA_PATH is set but CUDA wasn't able to be loaded. Please install the correct version of CUDA and cuDNN as mentioned in the GPU requirements page (https://onnxruntime.ai/docs/reference/execution-providers/CUDA-ExecutionProvider.html#requirements), make sure they're in the PATH, and that your GPU is supported.");
        }
      }
    }
    LOGS_DEFAULT(WARNING) << "Failed to create " << type << ". Please reference https://onnxruntime.ai/docs/reference/execution-providers/CUDA-ExecutionProvider.html#requirements to ensure all dependencies are met.";
#endif
  } else if (type == kRocmExecutionProvider) {
#ifdef USE_ROCM
    if (auto* rocm_provider_info = TryGetProviderInfo_ROCM()) {
      const ROCMExecutionProviderInfo info = GetRocmExecutionProviderInfo(rocm_provider_info,
                                                                          provider_options_map);

      // This variable is never initialized because the APIs by which is it should be initialized are deprecated, however they still
      // exist are are in-use. Neverthless, it is used to return ROCMAllocator, hence we must try to initialize it here if we can
      // since FromProviderOptions might contain external ROCM allocator.
      external_allocator_info = info.external_allocator_info;
      return rocm_provider_info->CreateExecutionProviderFactory(info)->CreateProvider();
    } else {
      if (!Env::Default().GetEnvironmentVar("ROCM_PATH").empty()) {
        ORT_THROW("ROCM_PATH is set but ROCM wasn't able to be loaded. Please install the correct version of ROCM and MIOpen as mentioned in the GPU requirements page, make sure they're in the PATH, and that your GPU is supported.");
      }
    }
#endif
  } else if (type == kDnnlExecutionProvider) {
#ifdef USE_DNNL
    return onnxruntime::DnnlProviderFactoryCreator::Create(session_options.enable_cpu_mem_arena)->CreateProvider();
#endif
  } else if (type == kOpenVINOExecutionProvider) {
#ifdef USE_OPENVINO
    OrtOpenVINOProviderOptions params;
    params.device_type = openvino_device_type.c_str();
    std::string blob_dump_path;

    auto it = provider_options_map.find(type);
    if (it != provider_options_map.end()) {
      for (auto option : it->second) {
        if (option.first == "device_type") {
          openvino_device_type = option.second;
          params.device_type = openvino_device_type.c_str();
        } else if (option.first == "enable_vpu_fast_compile") {
          if (option.second == "True") {
            params.enable_vpu_fast_compile = true;
          } else if (option.second == "False") {
            params.enable_vpu_fast_compile = false;
          } else {
            ORT_THROW("Invalid value passed for enable_vpu_fast_compile: ", option.second);
          }

        } else if (option.first == "use_compiled_network") {
          if (option.second == "True") {
            params.use_compiled_network = true;
          } else if (option.second == "False") {
            params.use_compiled_network = false;
          } else {
            ORT_THROW("Invalid value passed for use_compiled_network: ", option.second);
          }

        } else if (option.first == "enable_opencl_throttling") {
          if (option.second == "True") {
            params.enable_opencl_throttling = true;
          } else if (option.second == "False") {
            params.enable_opencl_throttling = false;
          } else {
            ORT_THROW("Invalid value passed for enable_opencl_throttling: ", option.second);
          }
        } else if (option.first == "enable_dynamic_shapes") {
          if (option.second == "True") {
            params.enable_dynamic_shapes = true;
          } else if (option.second == "False") {
            params.enable_dynamic_shapes = false;
          } else {
            ORT_THROW("Invalid value passed for enable_dynamic_shapes: ", option.second);
          }
        } else if (option.first == "device_id") {
          params.device_id = option.second.c_str();
        } else if (option.first == "num_of_threads") {
          params.num_of_threads = std::stoi(option.second);
        } else if (option.first == "blob_dump_path") {
          blob_dump_path = option.second;
          params.blob_dump_path = blob_dump_path.c_str();
        } else if (option.first == "context") {
          params.context = (void*)(option.second.c_str());
        } else {
          ORT_THROW("Invalid OpenVINO EP option: ", option.first);
        }
      }
    }
    if (std::shared_ptr<IExecutionProviderFactory> openvino_provider_factory = onnxruntime::OpenVINOProviderFactoryCreator::Create(&params)) {
      auto p = openvino_provider_factory->CreateProvider();
      // Reset global variables config to avoid it being accidentally passed on to the next session
      openvino_device_type.clear();
      return p;
    } else {
      if (!Env::Default().GetEnvironmentVar("INTEL_OPENVINO_DIR").empty()) {
        ORT_THROW("INTEL_OPENVINO_DIR is set but OpenVINO library wasn't able to be loaded. Please install a supported version of OpenVINO as mentioned in the requirements page (https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#requirements), ensure dependency libraries are in the PATH and your hardware is supported.");
      } else {
        LOGS_DEFAULT(WARNING) << "Failed to create " << type << ". Please reference https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#requirements to ensure all dependencies are met.";
      }
    }
#endif
  } else if (type == kNupharExecutionProvider) {
#if USE_NUPHAR
    const auto it = provider_options_map.find(type);
    if (it != provider_options_map.end()) {
      ORT_THROW_IF_ERROR(
          ProviderOptionsParser{}
              .AddAssignmentToReference("nuphar_settings", nuphar_settings)
              .Parse(it->second));
    }

    auto p = onnxruntime::NupharProviderFactoryCreator::Create(true, nuphar_settings.c_str())->CreateProvider();

    // clear nuphar_settings after use to avoid it being accidentally passed on to next session
    nuphar_settings.clear();
    return p;
#endif
  } else if (type == kTvmExecutionProvider) {
#if USE_TVM
    onnxruntime::tvm::TvmEPOptions info{};
    const auto it = provider_options_map.find(type);
    if (it != provider_options_map.end()) {
      info = onnxruntime::tvm::TvmEPOptionsHelper::FromProviderOptions(it->second);
    }

    return onnxruntime::TVMProviderFactoryCreator::Create(info)->CreateProvider();
#endif
  } else if (type == kVitisAIExecutionProvider) {
#if USE_VITISAI
    // Retrieve Vitis AI provider options
    // `target`: The name of the DPU target (default is DPUCADX8G for backward compatibility).
    // `export_runtime_module`: export a Vitis AI PyXIR runtime module to the specified file.
    //    This can be used for cross compilation or saving state.
    // `load_runtime_module`: Load an exported runtime module from disk.
    std::string target = "DPUCADX8G";
    std::string export_runtime_module = "";
    std::string load_runtime_module = "";
    auto it = provider_options_map.find(type);
    if (it != provider_options_map.end()) {
      auto vitis_ai_provider_options = it->second;
      auto vai_options_it = vitis_ai_provider_options.find("target");
      if (vai_options_it != vitis_ai_provider_options.end()) {
        target = vai_options_it->second;
      }
      vai_options_it = vitis_ai_provider_options.find("export_runtime_module");
      if (vai_options_it != vitis_ai_provider_options.end()) {
        export_runtime_module = vai_options_it->second;
      }
      vai_options_it = vitis_ai_provider_options.find("load_runtime_module");
      if (vai_options_it != vitis_ai_provider_options.end()) {
        load_runtime_module = vai_options_it->second;
      }
    }
    return onnxruntime::VitisAIProviderFactoryCreator::Create(target.c_str(), 0,
                                                              export_runtime_module.c_str(),
                                                              load_runtime_module.c_str())
        ->CreateProvider();
#endif
  } else if (type == kAclExecutionProvider) {
#ifdef USE_ACL
    return onnxruntime::ACLProviderFactoryCreator::Create(
               session_options.enable_cpu_mem_arena)
        ->CreateProvider();
#endif
  } else if (type == kArmNNExecutionProvider) {
#ifdef USE_ARMNN
    return onnxruntime::ArmNNProviderFactoryCreator::Create(
               session_options.enable_cpu_mem_arena)
        ->CreateProvider();
#endif
  } else if (type == kDmlExecutionProvider) {
#ifdef USE_DML
    int device_id = 0;
    auto it = provider_options_map.find(type);
    if (it != provider_options_map.end()) {
      for (auto option : it->second) {
        if (option.first == "device_id") {
          if (!option.second.empty()) {
            device_id = std::stoi(option.second);
          }
        }
      }
    }
    return onnxruntime::DMLProviderFactoryCreator::Create(device_id)->CreateProvider();
#endif
  } else if (type == kNnapiExecutionProvider) {
#if defined(USE_NNAPI)
#if !defined(__ANDROID__)
    LOGS_DEFAULT(WARNING) << "NNAPI execution provider can only be used to generate ORT format model in this build.";
#endif
    const auto partitioning_stop_ops_list = session_options.config_options.GetConfigEntry(
        kOrtSessionOptionsConfigNnapiEpPartitioningStopOps);
    return onnxruntime::NnapiProviderFactoryCreator::Create(0, partitioning_stop_ops_list)->CreateProvider();
#endif
  } else if (type == kRknpuExecutionProvider) {
#ifdef USE_RKNPU
    return onnxruntime::RknpuProviderFactoryCreator::Create()->CreateProvider();
#endif
  } else if (type == kCoreMLExecutionProvider) {
#if defined(USE_COREML)
#if !defined(__APPLE__)
    LOGS_DEFAULT(WARNING) << "CoreML execution provider can only be used to generate ORT format model in this build.";
#endif
    return onnxruntime::CoreMLProviderFactoryCreator::Create(0)->CreateProvider();
#endif
  } else if (type == kXnnpackExecutionProvider) {
#if defined(USE_XNNPACK)
    return onnxruntime::XnnpackProviderFactoryCreator::Create(ProviderOptions{})->CreateProvider();
#endif
  } else {
    // check whether it is a dynamic load EP:
    const auto it = provider_options_map.find(type);
    if (it != provider_options_map.end()) {
      auto shared_lib_path_it = it->second.find(kExecutionProviderSharedLibraryPath);
      if (shared_lib_path_it != it->second.end()) {
        // this is an EP with dynamic loading
        // construct the provider option
        ProviderOptions provider_options;
        std::string entry_symbol = kDefaultExecutionProviderEntry;
        for (auto option : it->second) {
          if (option.first == kExecutionProviderSharedLibraryEntry) {
            entry_symbol = option.second;
          } else if (option.first != kExecutionProviderSharedLibraryPath) {
            provider_options.insert(option);
          }
        }
        return LoadExecutionProvider(shared_lib_path_it->second, provider_options, entry_symbol);
      }
    }
    // unknown provider
    throw std::runtime_error("Unknown Provider Type: " + type);
  }
  return nullptr;
}

/*
 * Register execution provider with options.
 */
static void RegisterExecutionProviders(InferenceSession* sess, const std::vector<std::string>& provider_types,
                                       const ProviderOptionsMap& provider_options_map) {
  ORT_UNUSED_PARAMETER(provider_options_map);

  for (const std::string& type : provider_types) {
    auto ep = CreateExecutionProviderInstance(sess->GetSessionOptions(), type, provider_options_map);
    if (ep)
      OrtPybindThrowIfError(sess->RegisterExecutionProvider(std::move(ep)));
  }
}

/**
 * Generate a map for mapping execution provider to excution provider options.
 *
 * @param providers vector of excution providers. [ep1, ep2, ...]
 * @param provider_options_vector vector of excution provider options. [option1, option2 ...]
 * @param provider_options_map an unordered map for mapping excution provider to excution provider options.
 *        {'ep1' -> option1, 'ep2' -> option2 ...}
 *
 */
static void GenerateProviderOptionsMap(const std::vector<std::string>& providers,
                                       const ProviderOptionsVector& provider_options_vector,
                                       ProviderOptionsMap& provider_options_map) {
  if (provider_options_vector.empty() || providers.empty()) {
    return;
  }

  std::size_t j = 0;  // index for provider_options_vector

  for (const std::string& type : providers) {
    if (j < provider_options_vector.size() && !provider_options_vector[j].empty()) {
      provider_options_map[type] = provider_options_vector[j];
    }

    j += 1;
  }
}

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
static void RegisterCustomOpDomainsAndLibraries(PyInferenceSession* sess, const PySessionOptions& so) {
  if (!so.custom_op_domains_.empty()) {
    // Register all custom op domains that will be needed for the session
    std::vector<OrtCustomOpDomain*> custom_op_domains;
    custom_op_domains.reserve(so.custom_op_domains_.size());
    for (size_t i = 0; i < so.custom_op_domains_.size(); ++i) {
      custom_op_domains.emplace_back(so.custom_op_domains_[i]);
    }
    OrtPybindThrowIfError(sess->GetSessionHandle()->AddCustomOpDomains(custom_op_domains));

    // Register all custom op libraries that will be needed for the session
    sess->AddCustomOpLibraries(so.custom_op_libraries_);
  }
}
#endif

void InitializeSession(InferenceSession* sess,
                       ExecutionProviderRegistrationFn ep_registration_fn,
                       const std::vector<std::string>& provider_types,
                       const ProviderOptionsVector& provider_options,
                       const std::unordered_set<std::string>& disabled_optimizer_names) {
  ProviderOptionsMap provider_options_map;
  GenerateProviderOptionsMap(provider_types, provider_options, provider_options_map);

  ep_registration_fn(sess, provider_types, provider_options_map);

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  if (!disabled_optimizer_names.empty()) {
    OrtPybindThrowIfError(sess->FilterEnabledOptimizers({disabled_optimizer_names.cbegin(), disabled_optimizer_names.cend()}));
  }
#else
  ORT_UNUSED_PARAMETER(disabled_optimizer_names);
#endif

  OrtPybindThrowIfError(sess->Initialize());
}

bool CheckIfTensor(const std::vector<const NodeArg*>& def_list,
                   const std::string& name,
                   /*out*/ onnx::TypeProto& type_proto) {
  auto ret_it = std::find_if(std::begin(def_list), std::end(def_list),
                             [&name](const NodeArg* node_arg) { return name == node_arg->Name(); });
  if (ret_it == std::end(def_list)) {
    throw std::runtime_error("Failed to find NodeArg with name: " + name + " in the def list");
  }

  const auto* temp = (*ret_it)->TypeAsProto();
  if (!temp) {
    throw std::runtime_error("Corresponding type_proto is null");
  } else {
    type_proto = *temp;
  }

  return type_proto.has_tensor_type();
}

#if defined(USE_NUPHAR) ||   \
    defined(USE_OPENVINO) || \
    defined(USE_CUDA) ||     \
    defined(USE_ROCM)
static void LogDeprecationWarning(
    const std::string& deprecated, const optional<std::string>& alternative = nullopt) {
  LOGS_DEFAULT(WARNING) << "This is DEPRECATED and will be removed in the future: " << deprecated;
  LOGS_DEFAULT_IF(alternative.has_value(), WARNING) << "As an alternative, use: " << *alternative;
}
#endif

void addGlobalMethods(py::module& m, Environment& env) {
  m.def("get_default_session_options", &GetDefaultCPUSessionOptions, "Return a default session_options instance.");
  m.def("get_session_initializer", &SessionObjectInitializer::Get, "Return a default session object initializer.");
  m.def(
      "get_device", []() -> std::string { return BACKEND_DEVICE; },
      "Return the device used to compute the prediction (CPU, MKL, ...)");
  m.def(
      "set_seed", [](const int64_t seed) { utils::SetRandomSeed(seed); },
      "Sets the seed used for random number generation in Onnxruntime.");
  m.def(
      "set_default_logger_severity", [&env](int severity) {
        ORT_ENFORCE(severity >= 0 && severity <= 4,
                    "Invalid logging severity. 0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal");
        logging::LoggingManager* default_logging_manager = env.GetLoggingManager();
        default_logging_manager->SetDefaultLoggerSeverity(static_cast<logging::Severity>(severity));
      },
      "Sets the default logging severity. 0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal");
  m.def(
      "set_default_logger_verbosity", [&env](int vlog_level) {
        logging::LoggingManager* default_logging_manager = env.GetLoggingManager();
        default_logging_manager->SetDefaultLoggerVerbosity(vlog_level);
      },
      "Sets the default logging verbosity level. To activate the verbose log, "
      "you need to set the default logging severity to 0:Verbose level.");
  m.def(
      "get_all_providers", []() -> const std::vector<std::string>& { return GetAllExecutionProviderNames(); },
      "Return list of Execution Providers that this version of Onnxruntime can support. "
      "The order of elements represents the default priority order of Execution Providers "
      "from highest to lowest.");
  m.def(
      "enable_telemetry_events", []() -> void { platform_env.GetTelemetryProvider().EnableTelemetryEvents(); },
      "Enables platform-specific telemetry collection where applicable.");
  m.def(
      "disable_telemetry_events", []() -> void { platform_env.GetTelemetryProvider().DisableTelemetryEvents(); },
      "Disables platform-specific telemetry collection.");
  m.def(
      "create_and_register_allocator", [&env](const OrtMemoryInfo& mem_info, const OrtArenaCfg* arena_cfg = nullptr) -> void {
        auto st = env.CreateAndRegisterAllocator(mem_info, arena_cfg);
        if (!st.IsOK()) {
          throw std::runtime_error("Error when creating and registering allocator: " + st.ErrorMessage());
        }
      });

#ifdef USE_NUPHAR
  // TODO remove deprecated global config
  m.def("set_nuphar_settings", [](const std::string& str) {
    LogDeprecationWarning("set_nuphar_settings", "Nuphar execution provider option \"nuphar_settings\"");
    nuphar_settings = str;
  });
  // TODO remove deprecated global config
  m.def("get_nuphar_settings", []() -> std::string {
    LogDeprecationWarning("get_nuphar_settings");
    return nuphar_settings;
  });
#endif

#ifdef USE_OPENVINO
  m.def(
      "get_available_openvino_device_ids", []() -> std::vector<std::string> {
        if (auto* info = GetProviderInfo_OpenVINO()) {
          return info->GetAvailableDevices();
        }
        return {};
      },
      "Lists all OpenVINO device ids available.");
  /*
   * The following APIs to set config options are deprecated. Use Session.set_providers() instead.
   */
  // TODO remove deprecated global config
  m.def(
      "set_openvino_device", [](const std::string& device_type) {
        LogDeprecationWarning("set_openvino_device", "OpenVINO execution provider option \"device_type\"");
        openvino_device_type = device_type;
      },
      "Set the prefered OpenVINO device type to be used. If left unset, the device type selected during build time will be used.");
  // TODO remove deprecated global config
  m.def(
      "get_openvino_device", []() -> std::string {
        LogDeprecationWarning("get_openvino_device");
        return openvino_device_type;
      },
      "Gets the dynamically selected OpenVINO device type for inference.");
#endif

#if defined(USE_CUDA) || defined(USE_ROCM)
  /*
   * The following set_* methods are deprecated.
   *
   * To achieve same result, please use the following python api:
   * InferenceSession.set_providers(list_of_providers, list_of_provider_option_dicts)
   *
   */
  // TODO remove deprecated global config
  m.def("set_cuda_device_id", [](const int id) {
    LogDeprecationWarning("set_cuda_device_id", "CUDA/ROCM execution provider option \"device_id\"");
    cuda_device_id = static_cast<OrtDevice::DeviceId>(id);
  });
  // TODO remove deprecated global config
  m.def("set_cudnn_conv_algo_search", [](const OrtCudnnConvAlgoSearch algo) {
    LogDeprecationWarning("set_cudnn_conv_algo_search", "CUDA execution provider option \"cudnn_conv_algo_search\"");
#ifdef USE_ROCM
    ORT_UNUSED_PARAMETER(algo);
    ORT_THROW("set_cudnn_conv_algo_search is not supported in ROCM");
#else
        cudnn_conv_algo_search = algo;
#endif
  });
  // TODO remove deprecated global config
  m.def("set_do_copy_in_default_stream", [](const bool use_single_stream) {
    LogDeprecationWarning(
        "set_do_copy_in_default_stream", "CUDA execution provider option \"do_copy_in_default_stream\"");
#ifdef USE_ROCM
    ORT_UNUSED_PARAMETER(use_single_stream);
    ORT_THROW("set_do_copy_in_default_stream is not supported in ROCM");
#else
        do_copy_in_default_stream = use_single_stream;
#endif
  });
  // TODO remove deprecated global config
  m.def("set_gpu_mem_limit", [](const int64_t limit) {
    LogDeprecationWarning(
        "set_gpu_mem_limit",
        "CUDA execution provider option \"gpu_mem_limit\", ROCM execution provider option \"gpu_mem_limit\"");
    gpu_mem_limit = gsl::narrow<size_t>(limit);
  });
  // TODO remove deprecated global config
  m.def("set_arena_extend_strategy", [](const onnxruntime::ArenaExtendStrategy strategy) {
    LogDeprecationWarning("set_arena_extend_strategy", "CUDA/ROCM execution provider option \"arena_extend_strategy\"");
    arena_extend_strategy = strategy;
  });
#endif

#ifdef ENABLE_ATEN
  m.def("register_aten_op_executor",
        [](const std::string& is_tensor_argument_address_str, const std::string& aten_op_executor_address_str) -> void {
          size_t is_tensor_argument_address_int, aten_op_executor_address_int;
          ORT_THROW_IF_ERROR(
              ParseStringWithClassicLocale(is_tensor_argument_address_str, is_tensor_argument_address_int));
          ORT_THROW_IF_ERROR(ParseStringWithClassicLocale(aten_op_executor_address_str, aten_op_executor_address_int));
          void* p_is_tensor_argument = reinterpret_cast<void*>(is_tensor_argument_address_int);
          void* p_aten_op_executor = reinterpret_cast<void*>(aten_op_executor_address_int);
          contrib::aten_ops::ATenOperatorExecutor::Instance().Initialize(p_is_tensor_argument, p_aten_op_executor);
        });
#endif
}

void addObjectMethods(py::module& m, Environment& env, ExecutionProviderRegistrationFn ep_registration_fn) {
  py::enum_<GraphOptimizationLevel>(m, "GraphOptimizationLevel")
      .value("ORT_DISABLE_ALL", GraphOptimizationLevel::ORT_DISABLE_ALL)
      .value("ORT_ENABLE_BASIC", GraphOptimizationLevel::ORT_ENABLE_BASIC)
      .value("ORT_ENABLE_EXTENDED", GraphOptimizationLevel::ORT_ENABLE_EXTENDED)
      .value("ORT_ENABLE_ALL", GraphOptimizationLevel::ORT_ENABLE_ALL);

  py::enum_<ExecutionMode>(m, "ExecutionMode")
      .value("ORT_SEQUENTIAL", ExecutionMode::ORT_SEQUENTIAL)
      .value("ORT_PARALLEL", ExecutionMode::ORT_PARALLEL);

  py::enum_<ExecutionOrder>(m, "ExecutionOrder")
      .value("DEFAULT", ExecutionOrder::DEFAULT)
      .value("PRIORITY_BASED", ExecutionOrder::PRIORITY_BASED);

  py::enum_<OrtAllocatorType>(m, "OrtAllocatorType")
      .value("INVALID", OrtInvalidAllocator)
      .value("ORT_DEVICE_ALLOCATOR", OrtDeviceAllocator)
      .value("ORT_ARENA_ALLOCATOR", OrtArenaAllocator);

  py::enum_<OrtMemType>(m, "OrtMemType")
      .value("CPU_INPUT", OrtMemTypeCPUInput)
      .value("CPU_OUTPUT", OrtMemTypeCPUOutput)
      .value("CPU", OrtMemTypeCPU)
      .value("DEFAULT", OrtMemTypeDefault);

  py::class_<OrtDevice> device(m, "OrtDevice", R"pbdoc(ONNXRuntime device informaion.)pbdoc");
  device.def(py::init<OrtDevice::DeviceType, OrtDevice::MemoryType, OrtDevice::DeviceId>())
      .def("device_id", &OrtDevice::Id, R"pbdoc(Device Id.)pbdoc")
      .def("device_type", &OrtDevice::Type, R"pbdoc(Device Type.)pbdoc")
      .def_static("cpu", []() { return OrtDevice::CPU; })
      .def_static("cuda", []() { return OrtDevice::GPU; })
      .def_static("default_memory", []() { return OrtDevice::MemType::DEFAULT; });

  py::class_<OrtArenaCfg> ort_arena_cfg_binding(m, "OrtArenaCfg");
  // There is a global var: arena_extend_strategy, which means we can't use that var name here
  // See docs/C_API.md for details on what the following parameters mean and how to choose these values
  ort_arena_cfg_binding.def(py::init([](size_t max_mem, int arena_extend_strategy_local,
                                        int initial_chunk_size_bytes, int max_dead_bytes_per_chunk) {
    auto ort_arena_cfg = std::make_unique<OrtArenaCfg>();
    ort_arena_cfg->max_mem = max_mem;
    ort_arena_cfg->arena_extend_strategy = arena_extend_strategy_local;
    ort_arena_cfg->initial_chunk_size_bytes = initial_chunk_size_bytes;
    ort_arena_cfg->max_dead_bytes_per_chunk = max_dead_bytes_per_chunk;
    return ort_arena_cfg;
  }));

  py::class_<OrtMemoryInfo> ort_memory_info_binding(m, "OrtMemoryInfo");
  ort_memory_info_binding.def(py::init([](const char* name, OrtAllocatorType type, int id, OrtMemType mem_type) {
    if (strcmp(name, onnxruntime::CPU) == 0) {
      return std::make_unique<OrtMemoryInfo>(onnxruntime::CPU, type, OrtDevice(), id, mem_type);
    } else if (strcmp(name, onnxruntime::CUDA) == 0) {
      return std::make_unique<OrtMemoryInfo>(
          onnxruntime::CUDA, type, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, static_cast<OrtDevice::DeviceId>(id)), id,
          mem_type);
    } else if (strcmp(name, onnxruntime::CUDA_PINNED) == 0) {
      return std::make_unique<OrtMemoryInfo>(
          onnxruntime::CUDA_PINNED, type, OrtDevice(OrtDevice::CPU, OrtDevice::MemType::CUDA_PINNED, static_cast<OrtDevice::DeviceId>(id)),
          id, mem_type);
    } else {
      throw std::runtime_error("Specified device is not supported.");
    }
  }));

  py::class_<PySessionOptions>
      sess(m, "SessionOptions", R"pbdoc(Configuration information for a session.)pbdoc");
  sess
      .def(py::init())
      .def_readwrite("enable_cpu_mem_arena", &PySessionOptions::enable_cpu_mem_arena,
                     R"pbdoc(Enables the memory arena on CPU. Arena may pre-allocate memory for future usage.
Set this option to false if you don't want it. Default is True.)pbdoc")
      .def_readwrite("enable_profiling", &PySessionOptions::enable_profiling,
                     R"pbdoc(Enable profiling for this session. Default is false.)pbdoc")
      .def_readwrite("profile_file_prefix", &PySessionOptions::profile_file_prefix,
                     R"pbdoc(The prefix of the profile file. The current time will be appended to the file name.)pbdoc")
      .def_readwrite("optimized_model_filepath", &PySessionOptions::optimized_model_filepath,
                     R"pbdoc(
File path to serialize optimized model to.
Optimized model is not serialized unless optimized_model_filepath is set.
Serialized model format will default to ONNX unless:
- add_session_config_entry is used to set 'session.save_model_format' to 'ORT', or
- there is no 'session.save_model_format' config entry and optimized_model_filepath ends in '.ort' (case insensitive)

)pbdoc")
      .def_readwrite("enable_mem_pattern", &PySessionOptions::enable_mem_pattern,
                     R"pbdoc(Enable the memory pattern optimization. Default is true.)pbdoc")
      .def_readwrite("enable_mem_reuse", &PySessionOptions::enable_mem_reuse,
                     R"pbdoc(Enable the memory reuse optimization. Default is true.)pbdoc")
      .def_readwrite("logid", &PySessionOptions::session_logid,
                     R"pbdoc(Logger id to use for session output.)pbdoc")
      .def_readwrite("log_severity_level", &PySessionOptions::session_log_severity_level,
                     R"pbdoc(Log severity level. Applies to session load, initialization, etc.
0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2.)pbdoc")
      .def_readwrite("log_verbosity_level", &PySessionOptions::session_log_verbosity_level,
                     R"pbdoc(VLOG level if DEBUG build and session_log_severity_level is 0.
Applies to session load, initialization, etc. Default is 0.)pbdoc")
      .def_property(
          "intra_op_num_threads",
          [](const PySessionOptions* options) -> int { return options->intra_op_param.thread_pool_size; },
          [](PySessionOptions* options, int value) -> void { options->intra_op_param.thread_pool_size = value; },
          R"pbdoc(Sets the number of threads used to parallelize the execution within nodes. Default is 0 to let onnxruntime choose.)pbdoc")
      .def_property(
          "inter_op_num_threads",
          [](const PySessionOptions* options) -> int { return options->inter_op_param.thread_pool_size; },
          [](PySessionOptions* options, int value) -> void { options->inter_op_param.thread_pool_size = value; },
          R"pbdoc(Sets the number of threads used to parallelize the execution of the graph (across nodes). Default is 0 to let onnxruntime choose.)pbdoc")
      .def_readwrite("execution_mode", &PySessionOptions::execution_mode,
                     R"pbdoc(Sets the execution mode. Default is sequential.)pbdoc")
      .def_readwrite("execution_order", &PySessionOptions::execution_order,
                     R"pbdoc(Sets the execution order. Default is basic topological order.)pbdoc")
      .def_property(
          "graph_optimization_level",
          [](const PySessionOptions* options) -> GraphOptimizationLevel {
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

          [](PySessionOptions* options, GraphOptimizationLevel level) -> void {
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
          R"pbdoc(Graph optimization level for this session.)pbdoc")
      .def_readwrite("use_deterministic_compute", &PySessionOptions::use_deterministic_compute,
                     R"pbdoc(Whether to use deterministic compute. Default is false.)pbdoc")
      .def(
          "add_free_dimension_override_by_denotation",
          [](PySessionOptions* options, const char* dim_name, int64_t dim_value)
              -> void { options->free_dimension_overrides.push_back(
                            onnxruntime::FreeDimensionOverride{
                                dim_name,
                                onnxruntime::FreeDimensionOverrideType::Denotation,
                                dim_value}); },
          R"pbdoc(Specify the dimension size for each denotation associated with an input's free dimension.)pbdoc")
      .def(
          "add_free_dimension_override_by_name",
          [](PySessionOptions* options, const char* dim_name, int64_t dim_value)
              -> void { options->free_dimension_overrides.push_back(
                            onnxruntime::FreeDimensionOverride{
                                dim_name,
                                onnxruntime::FreeDimensionOverrideType::Name,
                                dim_value}); },
          R"pbdoc(Specify values of named dimensions within model inputs.)pbdoc")
      .def(
          "add_session_config_entry",
          [](PySessionOptions* options, const char* config_key, const char* config_value) -> void {
            // config_key and config_value will be copied
            const Status status = options->config_options.AddConfigEntry(config_key, config_value);
            if (!status.IsOK())
              throw std::runtime_error(status.ErrorMessage());
          },
          R"pbdoc(Set a single session configuration entry as a pair of strings.)pbdoc")
      .def(
          "get_session_config_entry",
          [](const PySessionOptions* options, const char* config_key) -> std::string {
            const std::string key(config_key);
            std::string value;
            if (!options->config_options.TryGetConfigEntry(key, value))
              throw std::runtime_error("SessionOptions does not have configuration with key: " + key);

            return value;
          },
          R"pbdoc(Get a single session configuration value using the given configuration key.)pbdoc")
      .def(
          "register_custom_ops_library",
          [](PySessionOptions* options, const char* library_path) -> void {
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
            // We need to pass in an `OrtSessionOptions` instance because the exported method in the shared library expects that
            // Once we have access to the `OrtCustomOpDomains` within the passed in `OrtSessionOptions` instance, we place it
            // into the container we are maintaining for that very purpose and the `ortSessionoptions` instance can go out of scope.
            OrtSessionOptions s;

            options->custom_op_libraries_.emplace_back(std::make_shared<CustomOpLibrary>(library_path, s));

            // reserve enough memory to hold current contents and the new incoming contents
            options->custom_op_domains_.reserve(options->custom_op_domains_.size() + s.custom_op_domains_.size());
            for (size_t i = 0; i < s.custom_op_domains_.size(); ++i) {
              options->custom_op_domains_.emplace_back(s.custom_op_domains_[i]);
            }
#else
            ORT_UNUSED_PARAMETER(options);
            ORT_UNUSED_PARAMETER(library_path);
            ORT_THROW("Custom Ops are not supported in this build.");
#endif
          },
          R"pbdoc(Specify the path to the shared library containing the custom op kernels required to run a model.)pbdoc")
      .def(
          "add_initializer", [](PySessionOptions* options, const char* name, py::object& ml_value_pyobject) -> void {
            ORT_ENFORCE(strcmp(Py_TYPE(ml_value_pyobject.ptr())->tp_name, PYTHON_ORTVALUE_OBJECT_NAME) == 0, "The provided Python object must be an OrtValue");
            // The user needs to ensure that the python OrtValue being provided as an overriding initializer
            // is not destructed as long as any session that uses the provided OrtValue initializer is still in scope
            // This is no different than the native APIs
            const OrtValue* ml_value = ml_value_pyobject.attr(PYTHON_ORTVALUE_NATIVE_OBJECT_ATTR).cast<OrtValue*>();
            ORT_THROW_IF_ERROR(options->AddInitializer(name, ml_value));
          })
      .def("add_external_initializers", [](PySessionOptions* options, py::list& names, const py::list& ort_values) -> void {
#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)
        const auto init_num = ort_values.size();
        ORT_ENFORCE(init_num == names.size(), "Expecting names and ort_values lists to have equal length");
        InlinedVector<std::string> names_ptrs;
        InlinedVector<OrtValue> values_ptrs;
        names_ptrs.reserve(init_num);
        values_ptrs.reserve(init_num);
        for (size_t i = 0; i < init_num; ++i) {
          names_ptrs.emplace_back(py::str(names[i]));
          values_ptrs.emplace_back(*ort_values[i].attr(PYTHON_ORTVALUE_NATIVE_OBJECT_ATTR).cast<const OrtValue*>());
        }
        ORT_THROW_IF_ERROR(options->AddExternalInitializers(names_ptrs, values_ptrs));
#else
            ORT_UNUSED_PARAMETER(options);
            ORT_UNUSED_PARAMETER(names);
            ORT_UNUSED_PARAMETER(ort_values);
            ORT_THROW("External initializers are not supported in this build.");
#endif
      });

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
#ifdef ENABLE_TRAINING
      .def_readwrite("training_mode", &RunOptions::training_mode,
                     R"pbdoc(Choose to run in training or inferencing mode)pbdoc")
#endif
      .def_readwrite("only_execute_path_to_fetches", &RunOptions::only_execute_path_to_fetches,
                     R"pbdoc(Only execute the nodes needed by fetch list)pbdoc")
      .def(
          "add_run_config_entry",
          [](RunOptions* options, const char* config_key, const char* config_value) -> void {
            // config_key and config_value will be copied
            const Status status = options->config_options.AddConfigEntry(config_key, config_value);
            if (!status.IsOK())
              throw std::runtime_error(status.ErrorMessage());
          },
          R"pbdoc(Set a single run configuration entry as a pair of strings.)pbdoc")
      .def(
          "get_run_config_entry",
          [](const RunOptions* options, const char* config_key) -> std::string {
            const std::string key(config_key);
            std::string value;
            if (!options->config_options.TryGetConfigEntry(key, value))
              throw std::runtime_error("RunOptions does not have configuration with key: " + key);

            return value;
          },
          R"pbdoc(Get a single run configuration value using the given configuration key.)pbdoc");

  py::class_<ModelMetadata>(m, "ModelMetadata", R"pbdoc(Pre-defined and custom metadata about the model.
It is usually used to identify the model used to run the prediction and
facilitate the comparison.)pbdoc")
      .def_readwrite("producer_name", &ModelMetadata::producer_name, "producer name")
      .def_readwrite("graph_name", &ModelMetadata::graph_name, "graph name")
      .def_readwrite("domain", &ModelMetadata::domain, "ONNX domain")
      .def_readwrite("description", &ModelMetadata::description, "description of the model")
      .def_readwrite("graph_description", &ModelMetadata::graph_description, "description of the graph hosted in the model")
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
      .def(
          "__str__", [](const onnxruntime::NodeArg& na) -> std::string {
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
      .def_property_readonly(
          "shape", [](const onnxruntime::NodeArg& na) -> std::vector<py::object> {
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

  py::class_<SessionObjectInitializer> sessionObjectInitializer(m, "SessionObjectInitializer");
  py::class_<PyInferenceSession>(m, "InferenceSession", R"pbdoc(This is the main class used to run a model.)pbdoc")
      // In Python3, a Python bytes object will be passed to C++ functions that accept std::string or char*
      // without any conversion. So this init method can be used for model file path (string) and model content (bytes)
      .def(py::init([&env](const PySessionOptions& so, const std::string arg, bool is_arg_file_name,
                           bool load_config_from_model = false) {
        std::unique_ptr<PyInferenceSession> sess;

        // separate creation of the session from model loading unless we have to read the config from the model.
        // in a minimal build we only support load via Load(...) and not at session creation time
        if (load_config_from_model) {
#if !defined(ORT_MINIMAL_BUILD)
          sess = std::make_unique<PyInferenceSession>(env, so, arg, is_arg_file_name);

          RegisterCustomOpDomainsAndLibraries(sess.get(), so);

          OrtPybindThrowIfError(sess->GetSessionHandle()->Load());
#else
          ORT_THROW("Loading configuration from an ONNX model is not supported in this build.");
#endif
        } else {
          sess = std::make_unique<PyInferenceSession>(env, so);
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
          RegisterCustomOpDomainsAndLibraries(sess.get(), so);
#endif

          if (is_arg_file_name) {
            OrtPybindThrowIfError(sess->GetSessionHandle()->Load(arg));
          } else {
            OrtPybindThrowIfError(sess->GetSessionHandle()->Load(arg.data(), arg.size()));
          }
        }

        return sess;
      }))
      .def(
          "initialize_session",
          [ep_registration_fn](PyInferenceSession* sess,
                               const std::vector<std::string>& provider_types = {},
                               const ProviderOptionsVector& provider_options = {},
                               const std::unordered_set<std::string>& disabled_optimizer_names = {}) {
            InitializeSession(sess->GetSessionHandle(),
                              ep_registration_fn,
                              provider_types,
                              provider_options,
                              disabled_optimizer_names);
          },
          R"pbdoc(Load a model saved in ONNX or ORT format.)pbdoc")
      .def("run",
           [](PyInferenceSession* sess, std::vector<std::string> output_names,
              std::map<std::string, py::object> pyfeeds, RunOptions* run_options = nullptr)
               -> std::vector<py::object> {
             NameMLValMap feeds;
             for (auto feed : pyfeeds) {
               // No need to process 'None's sent in by the user
               // to feed Optional inputs in the graph.
               // We just won't include anything in the feed and ORT
               // will handle such implicit 'None's internally.
               if (!feed.second.is(py::none())) {
                 OrtValue ml_value;
                 auto px = sess->GetSessionHandle()->GetModelInputs();
                 if (!px.first.IsOK() || !px.second) {
                   throw std::runtime_error("Either failed to get model inputs from the session object or the input def list was null");
                 }
                 CreateGenericMLValue(px.second, GetAllocator(), feed.first, feed.second, &ml_value);
                 ThrowIfPyErrOccured();
                 feeds.insert(std::make_pair(feed.first, ml_value));
               }
             }

             std::vector<OrtValue> fetches;
             common::Status status;

             {
               // release GIL to allow multiple python threads to invoke Run() in parallel.
               py::gil_scoped_release release;
               if (run_options != nullptr) {
                 OrtPybindThrowIfError(sess->GetSessionHandle()->Run(*run_options, feeds, output_names, &fetches));
               } else {
                 OrtPybindThrowIfError(sess->GetSessionHandle()->Run(feeds, output_names, &fetches));
               }
             }

             std::vector<py::object> rfetch;
             rfetch.reserve(fetches.size());
             size_t pos = 0;
             for (auto fet : fetches) {
               if (fet.IsAllocated()) {
                 if (fet.IsTensor()) {
                   rfetch.push_back(AddTensorAsPyObj(fet, nullptr, nullptr));
                 } else if (fet.IsSparseTensor()) {
                   rfetch.push_back(GetPyObjectFromSparseTensor(pos, fet, nullptr));
                 } else {
                   rfetch.push_back(AddNonTensorAsPyObj(fet, nullptr, nullptr));
                 }
               } else {  // Send back None because the corresponding OrtValue was empty
                 rfetch.push_back(py::none());
               }
               ++pos;
             }
             return rfetch;
           })
      /// This method accepts a dictionary of feeds (name -> OrtValue) and the list of output_names
      /// and returns a list of python objects representing OrtValues. Each name may represent either
      /// a Tensor, SparseTensor or a TensorSequence.
      .def("run_with_ort_values", [](PyInferenceSession* sess, const py::dict& feeds, const std::vector<std::string>& output_names, RunOptions* run_options = nullptr) -> std::vector<OrtValue> {
        NameMLValMap ort_feeds;
        // item is always a copy since dict returns a value and not a ref
        // and Apple XToolChain barks
        for (const auto item : feeds) {
          auto name = item.first.cast<std::string>();
          const OrtValue* ort_value = item.second.cast<const OrtValue*>();
          ort_feeds.emplace(name, *ort_value);
        }

        std::vector<OrtValue> fetches;
        {
          // release GIL to allow multiple python threads to invoke Run() in parallel.
          py::gil_scoped_release release;
          if (run_options != nullptr) {
            OrtPybindThrowIfError(sess->GetSessionHandle()->Run(*run_options, ort_feeds, output_names, &fetches));
          } else {
            OrtPybindThrowIfError(sess->GetSessionHandle()->Run(ort_feeds, output_names, &fetches));
          }
        }
        return fetches;
      })
      .def("end_profiling", [](const PyInferenceSession* sess) -> std::string {
        return sess->GetSessionHandle()->EndProfiling();
      })
      .def_property_readonly("get_profiling_start_time_ns", [](const PyInferenceSession* sess) -> uint64_t {
        return sess->GetSessionHandle()->GetProfiling().GetStartTimeNs();
      })
      .def(
          "get_providers", [](const PyInferenceSession* sess) -> const std::vector<std::string>& {
            return sess->GetSessionHandle()->GetRegisteredProviderTypes();
          },
          py::return_value_policy::reference_internal)
      .def(
          "get_provider_options", [](const PyInferenceSession* sess) -> const ProviderOptionsMap& {
            return sess->GetSessionHandle()->GetAllProviderOptions();
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "session_options", [](const PyInferenceSession* sess) -> const PySessionOptions& {
            const auto& session_options = sess->GetSessionHandle()->GetSessionOptions();
            return static_cast<const PySessionOptions&>(session_options);
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "inputs_meta", [](const PyInferenceSession* sess) -> const std::vector<const onnxruntime::NodeArg*>& {
            auto res = sess->GetSessionHandle()->GetModelInputs();
            OrtPybindThrowIfError(res.first);
            return *(res.second);
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "outputs_meta", [](const PyInferenceSession* sess) -> const std::vector<const onnxruntime::NodeArg*>& {
            auto res = sess->GetSessionHandle()->GetModelOutputs();
            OrtPybindThrowIfError(res.first);
            return *(res.second);
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "overridable_initializers", [](const PyInferenceSession* sess) -> const std::vector<const onnxruntime::NodeArg*>& {
            auto res = sess->GetSessionHandle()->GetOverridableInitializers();
            OrtPybindThrowIfError(res.first);
            return *(res.second);
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "model_meta", [](const PyInferenceSession* sess) -> const onnxruntime::ModelMetadata& {
            auto res = sess->GetSessionHandle()->GetModelMetadata();
            OrtPybindThrowIfError(res.first);
            return *(res.second);
          },
          py::return_value_policy::reference_internal)
      .def("run_with_iobinding", [](PyInferenceSession* sess, SessionIOBinding& io_binding, RunOptions* run_options = nullptr) -> void {
        Status status;
        // release GIL to allow multiple python threads to invoke Run() in parallel.
        py::gil_scoped_release release;
        if (!run_options)
          status = sess->GetSessionHandle()->Run(*io_binding.Get());
        else
          status = sess->GetSessionHandle()->Run(*run_options, *io_binding.Get());
        if (!status.IsOK())
          throw std::runtime_error("Error in execution: " + status.ErrorMessage());
      });

  py::enum_<onnxruntime::ArenaExtendStrategy>(m, "ArenaExtendStrategy", py::arithmetic())
      .value("kNextPowerOfTwo", onnxruntime::ArenaExtendStrategy::kNextPowerOfTwo)
      .value("kSameAsRequested", onnxruntime::ArenaExtendStrategy::kSameAsRequested)
      .export_values();
}

void CreateInferencePybindStateModule(py::module& m) {
  m.doc() = "pybind11 stateful interface to ONNX runtime";
  RegisterExceptions(m);

  // Initialization of the module
  ([]() -> void {
    // import_array1() forces a void return value.
    import_array1();
  })();

  Environment& env = GetEnv();

  addGlobalMethods(m, env);
  addObjectMethods(m, env, RegisterExecutionProviders);
  addOrtValueMethods(m);
  addSparseTensorMethods(m);
  addIoBindingMethods(m);

#if !defined(__APPLE__) && !defined(ORT_MINIMAL_BUILD)
  if (!InitProvidersSharedLibrary()) {
    const logging::Logger& default_logger = logging::LoggingManager::DefaultLogger();
    LOGS(default_logger, WARNING) << "Init provider bridge failed.";
  }
#endif

#ifdef onnxruntime_PYBIND_EXPORT_OPSCHEMA
  addGlobalSchemaFunctions(m);
  addOpSchemaSubmodule(m);
  addOpKernelSubmodule(m);
#endif
}

void InitArray() {
  ([]() -> void {
    // import_array1() forces a void return value.
    import_array1();
  })();
}

// static variable used to create inference session and training session.
static std::unique_ptr<Environment> session_env;

void InitializeEnv() {
  auto initialize = [&]() {
    // Initialization of the module
    InitArray();
    Env::Default().GetTelemetryProvider().SetLanguageProjection(OrtLanguageProjection::ORT_PROJECTION_PYTHON);
    OrtPybindThrowIfError(Environment::Create(std::make_unique<LoggingManager>(
                                                  std::make_unique<CLogSink>(),
                                                  Severity::kWARNING, false, LoggingManager::InstanceType::Default,
                                                  &SessionObjectInitializer::default_logger_id),
                                              session_env));

    static bool initialized = false;
    if (initialized) {
      return;
    }
    initialized = true;
  };
  initialize();
}

onnxruntime::Environment& GetEnv() {
  if (!session_env) {
    InitializeEnv();
  }
  return *session_env;
}

}  // namespace python
}  // namespace onnxruntime
