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
#include "core/common/narrow.h"
#include "core/common/optional.h"
#include "core/common/path_string.h"
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

#include <pybind11/functional.h>

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

using PyCallback = std::function<void(std::vector<py::object>, py::object user_data, std::string)>;

struct AsyncResource {
  std::vector<OrtValue> feeds;
  std::vector<const OrtValue*> feeds_raw;

  std::vector<std::string> feed_names;
  std::vector<const char*> feed_names_raw;

  std::vector<OrtValue*> fetches_raw;

  std::vector<std::string> fetch_names;
  std::vector<const char*> fetch_names_raw;

  RunOptions default_run_option;
  PyCallback callback;
  py::object user_data;

  void ReserveFeeds(size_t sz) {
    feeds.reserve(sz);
    feeds_raw.reserve(sz);
    feed_names.reserve(sz);
    feed_names_raw.reserve(sz);
  }

  void ReserveFetches(size_t sz) {
    fetches_raw.reserve(sz);
    fetch_names.reserve(sz);
    fetch_names_raw.reserve(sz);
  }
};

void AsyncCallback(void* user_data, OrtValue** outputs, size_t num_outputs, OrtStatusPtr ort_status) {
  ORT_ENFORCE(user_data, "user data must not be NULL for callback in python");

  auto invoke_callback = [&]() {
    std::unique_ptr<AsyncResource> async_resource{reinterpret_cast<AsyncResource*>(user_data)};
    Ort::Status status(ort_status);

    // return on error
    if (!status.IsOK()) {
      async_resource->callback({}, async_resource->user_data, status.GetErrorMessage());
      return;
    }

    std::vector<py::object> rfetch;
    rfetch.reserve(num_outputs);
    size_t pos = 0;
    for (size_t ith = 0; ith < num_outputs; ++ith) {
      const auto& fet = *outputs[ith];
      if (fet.IsAllocated()) {
        if (fet.IsTensor()) {
          rfetch.push_back(AddTensorAsPyObj(fet, nullptr, nullptr));
        } else if (fet.IsSparseTensor()) {
          rfetch.push_back(GetPyObjectFromSparseTensor(pos, fet, nullptr));
        } else {
          rfetch.push_back(AddNonTensorAsPyObj(fet, nullptr, nullptr));
        }
      } else {
        rfetch.push_back(py::none());
      }
      ++pos;
    }
    async_resource->callback(rfetch, async_resource->user_data, "");
  };

  if (PyGILState_Check()) {
    invoke_callback();
  } else {
    // acquire GIL to safely:
    // 1) invoke python callback
    // 2) create, manipulate, and destory python objects
    py::gil_scoped_acquire acquire;
    invoke_callback();
  }
}

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
      narrow<int>(shape.NumDimensions()), npy_dims.data(), numpy_type));

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
    const std::string* src = rtensor.Data<std::string>();
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
    case OrtDevice::NPU:
#ifdef USE_CANN
      return CANN;
#else
      return "NPU";
#endif
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
      auto status = src_sparse_tensor.Copy(*data_transfer_manager, *dst_sparse_tensor);
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
    GetPyObjFromTensor(rtensor.Get<Tensor>(), obj, data_transfer_manager, mem_cpy_to_host_functions);
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
  const auto path_str = ToPathString(ep_shared_lib_path);
  auto error = Env::Default().LoadDynamicLibrary(path_str, false, &handle);
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
    info.tunable_op = tunable_op;
  }
  return info;
}
#endif

#ifdef USE_CANN
const CANNExecutionProviderInfo GetCannExecutionProviderInfo(ProviderInfo_CANN* cann_provider_info,
                                                             const ProviderOptionsMap& provider_options_map) {
  ORT_ENFORCE(cann_provider_info);
  const auto it = provider_options_map.find(kCannExecutionProvider);
  CANNExecutionProviderInfo info;
  if (it != provider_options_map.end())
    cann_provider_info->CANNExecutionProviderInfo__FromProviderOptions(it->second, info);
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
    info.tunable_op = tunable_op;
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
      std::string calibration_table, cache_path, lib_path, min_profile, max_profile, opt_profile;
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
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            -1,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
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
          } else if (option.first == "trt_layer_norm_fp32_fallback") {
            if (option.second == "True" || option.second == "true") {
              params.trt_layer_norm_fp32_fallback = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_layer_norm_fp32_fallback = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_layer_norm_fp32_fallback' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_timing_cache_enable") {
            if (option.second == "True" || option.second == "true") {
              params.trt_timing_cache_enable = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_timing_cache_enable = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_timing_cache_enable' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_force_timing_cache") {
            if (option.second == "True" || option.second == "true") {
              params.trt_force_timing_cache = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_force_timing_cache = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_force_timing_cache' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_detailed_build_log") {
            if (option.second == "True" || option.second == "true") {
              params.trt_detailed_build_log = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_detailed_build_log = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_detailed_build_log' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_build_heuristics_enable") {
            if (option.second == "True" || option.second == "true") {
              params.trt_build_heuristics_enable = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_build_heuristics_enable = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_build_heuristics_enable' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_sparsity_enable") {
            if (option.second == "True" || option.second == "true") {
              params.trt_sparsity_enable = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_sparsity_enable = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_sparsity_enable' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_builder_optimization_level") {
            if (!option.second.empty()) {
              params.trt_builder_optimization_level = std::stoi(option.second);
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_builder_optimization_level' should be a number i.e. '0'.\n");
            }
          } else if (option.first == "trt_auxiliary_streams") {
            if (!option.second.empty()) {
              params.trt_auxiliary_streams = std::stoi(option.second);
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_auxiliary_streams' should be a number i.e. '0'.\n");
            }
          } else if (option.first == "trt_tactic_sources") {
            if (!option.second.empty()) {
              params.trt_tactic_sources = option.second.c_str();
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_tactic_sources' should be a string. e.g. \"-CUDNN,+CUBLAS\" available keys: \"CUBLAS\"|\"CUBLAS_LT\"|\"CUDNN\"|\"EDGE_MASK_CONVOLUTIONS\".\n");
            }
          } else if (option.first == "trt_extra_plugin_lib_paths") {
            if (!option.second.empty()) {
              params.trt_extra_plugin_lib_paths = option.second.c_str();
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_extra_plugin_lib_paths' should be a path string.\n");
            }
          } else if (option.first == "trt_profile_min_shapes") {
            if (!option.second.empty()) {
              min_profile = option.second;
              params.trt_profile_min_shapes = min_profile.c_str();
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_profile_min_shapes' should be a string of 'input1:dim1xdimd2...,input2:dim1xdim2...,...'.\n");
            }
          } else if (option.first == "trt_profile_max_shapes") {
            if (!option.second.empty()) {
              max_profile = option.second;
              params.trt_profile_max_shapes = max_profile.c_str();
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_profile_max_shapes' should be a string of 'input1:dim1xdimd2...,input2:dim1xdim2...,...'.\n");
            }
          } else if (option.first == "trt_profile_opt_shapes") {
            if (!option.second.empty()) {
              opt_profile = option.second;
              params.trt_profile_opt_shapes = opt_profile.c_str();
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_profile_opt_shapes' should be a string of 'input1:dim1xdimd2...,input2:dim1xdim2...,...'.\n");
            }
          } else if (option.first == "trt_cuda_graph_enable") {
            if (option.second == "True" || option.second == "true") {
              params.trt_cuda_graph_enable = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_cuda_graph_enable = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_cuda_graph_enable' should be 'True' or 'False'. Default value is 'False'.\n");
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
          ORT_THROW("CUDA_PATH is set but CUDA wasn't able to be loaded. Please install the correct version of CUDA and cuDNN as mentioned in the GPU requirements page (https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements), make sure they're in the PATH, and that your GPU is supported.");
        }
      }
    }
    LOGS_DEFAULT(WARNING) << "Failed to create " << type << ". Please reference https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements to ensure all dependencies are met.";
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
    // Generate dnnl_options
    OrtDnnlProviderOptions dnnl_options;
// For Eigen and OpenMP
#if defined(DNNL_OPENMP)
    int num_threads = 0;
    auto it = provider_options_map.find(type);
    if (it != provider_options_map.end()) {
      for (auto option : it->second) {
        if (option.first == "num_of_threads") {
          num_threads = std::stoi(option.second);
          if (num_threads < 0) {
            ORT_THROW(
                "[ERROR] [OneDNN] Invalid entry for the key 'num_of_threads',"
                " set number of threads or use '0' for default\n");
            // If the user doesnt define num_threads, auto detect threads later
          }
        } else {
          ORT_THROW("Invalid OneDNN EP option: ", option.first);
        }
      }
    }
    dnnl_options.threadpool_args = static_cast<void*>(&num_threads);
#endif  // !defined(DNNL_ORT_THREAD)
    dnnl_options.use_arena = session_options.enable_cpu_mem_arena;

    return onnxruntime::DnnlProviderFactoryCreator::Create(&dnnl_options)->CreateProvider();
#endif
  } else if (type == kOpenVINOExecutionProvider) {
#ifdef USE_OPENVINO
    ProviderOptions OV_provider_options_map;
    auto it = provider_options_map.find(type);
    if (it != provider_options_map.end()) {
      for (auto option : it->second) {
        if (option.first == "device_type") {
          OV_provider_options_map[option.first] = option.second;
          continue;
        } else if (option.first == "enable_vpu_fast_compile") {
          if (!(option.second == "True" || option.second == "true" ||
                option.second == "False" || option.second == "false")) {
            ORT_THROW("Invalid value passed for enable_vpu_fast_compile: ", option.second);
          }
          OV_provider_options_map[option.first] = option.second;
        } else if (option.first == "enable_opencl_throttling") {
          if (!(option.second == "True" || option.second == "true" ||
                option.second == "False" || option.second == "false")) {
            ORT_THROW("Invalid value passed for enable_opencl_throttling: ", option.second);
          }
          OV_provider_options_map[option.first] = option.second;
        } else if (option.first == "enable_dynamic_shapes") {
          if (!(option.second == "True" || option.second == "true" ||
                option.second == "False" || option.second == "false")) {
            ORT_THROW("Invalid value passed for enable_dynamic_shapes: ", option.second);
          }
          OV_provider_options_map[option.first] = option.second;
        } else if (option.first == "device_id") {
          OV_provider_options_map[option.first] = option.second;
          continue;
        } else if (option.first == "num_of_threads") {
          OV_provider_options_map[option.first] = option.second;
          continue;
        } else if (option.first == "num_streams") {
          OV_provider_options_map[option.first] = option.second;
          continue;
        } else if (option.first == "cache_dir") {
          OV_provider_options_map[option.first] = option.second;
          continue;
        } else if (option.first == "context") {
          OV_provider_options_map[option.first] = option.second;
          continue;
        } else {
          ORT_THROW("Invalid OpenVINO EP option: ", option.first);
        }
      }
    }
    if (std::shared_ptr<IExecutionProviderFactory> openvino_provider_factory = onnxruntime::OpenVINOProviderFactoryCreator::Create(
            &OV_provider_options_map)) {
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
    const auto it = provider_options_map.find(type);
    if (it == provider_options_map.end()) {
      LOGS_DEFAULT(FATAL) << "cannot find provider options for VitisAIExecutionProvider";
    }
    const auto& vitis_option_map = it->second;
    return onnxruntime::VitisAIProviderFactoryCreator::Create(vitis_option_map)
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
    auto cit = provider_options_map.find(type);
    return onnxruntime::DMLProviderFactoryCreator::CreateFromProviderOptions(
               cit == provider_options_map.end() ? ProviderOptions{} : cit->second)
        ->CreateProvider();
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
    auto cit = provider_options_map.find(type);
    return onnxruntime::XnnpackProviderFactoryCreator::Create(
               cit == provider_options_map.end() ? ProviderOptions{} : cit->second, &session_options)
        ->CreateProvider();
#endif
  } else if (type == kCannExecutionProvider) {
#ifdef USE_CANN
    if (auto* cann_provider_info = TryGetProviderInfo_CANN()) {
      const CANNExecutionProviderInfo info = GetCannExecutionProviderInfo(cann_provider_info,
                                                                          provider_options_map);
      return cann_provider_info->CreateExecutionProviderFactory(info)->CreateProvider();
    } else {
      ORT_THROW("create CANN ExecutionProvider fail");
    }
#endif
  } else if (type == kAzureExecutionProvider) {
#ifdef USE_AZURE
    return onnxruntime::AzureProviderFactoryCreator::Create({})->CreateProvider();
#endif
  } else if (type == kQnnExecutionProvider) {
#ifdef USE_QNN
    auto cit = provider_options_map.find(type);
    return onnxruntime::QNNProviderFactoryCreator::Create(
               cit == provider_options_map.end() ? ProviderOptions{} : cit->second, &session_options)
        ->CreateProvider();
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
static void RegisterCustomOpDomains(PyInferenceSession* sess, const PySessionOptions& so) {
  if (!so.custom_op_domains_.empty()) {
    // Register all custom op domains that will be needed for the session
    std::vector<OrtCustomOpDomain*> custom_op_domains;
    custom_op_domains.reserve(so.custom_op_domains_.size());
    for (size_t i = 0; i < so.custom_op_domains_.size(); ++i) {
      custom_op_domains.emplace_back(so.custom_op_domains_[i]);
    }
    OrtPybindThrowIfError(sess->GetSessionHandle()->AddCustomOpDomains(custom_op_domains));
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

#if defined(USE_OPENVINO) || \
    defined(USE_CUDA) ||     \
    defined(USE_ROCM)
static void LogDeprecationWarning(
    const std::string& deprecated, const optional<std::string>& alternative = nullopt) {
  LOGS_DEFAULT(WARNING) << "This is DEPRECATED and will be removed in the future: " << deprecated;
  LOGS_DEFAULT_IF(alternative.has_value(), WARNING) << "As an alternative, use: " << *alternative;
}
#endif

void addGlobalMethods(py::module& m) {
  m.def("get_default_session_options", &GetDefaultCPUSessionOptions, "Return a default session_options instance.");
  m.def("get_session_initializer", &SessionObjectInitializer::Get, "Return a default session object initializer.");
  m.def(
      "get_device", []() -> std::string { return BACKEND_DEVICE; },
      "Return the device used to compute the prediction (CPU, MKL, ...)");
  m.def(
      "set_seed", [](const int64_t seed) { utils::SetRandomSeed(seed); },
      "Sets the seed used for random number generation in Onnxruntime.");
  m.def(
      "set_default_logger_severity", [](int severity) {
        ORT_ENFORCE(severity >= 0 && severity <= 4,
                    "Invalid logging severity. 0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal");
        auto env = GetEnv();
        logging::LoggingManager* default_logging_manager = env->GetLoggingManager();
        default_logging_manager->SetDefaultLoggerSeverity(static_cast<logging::Severity>(severity));
      },
      "Sets the default logging severity. 0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal");
  m.def(
      "set_default_logger_verbosity", [](int vlog_level) {
        auto env = GetEnv();
        logging::LoggingManager* default_logging_manager = env->GetLoggingManager();
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
      "create_and_register_allocator", [](const OrtMemoryInfo& mem_info, const OrtArenaCfg* arena_cfg = nullptr) -> void {
        auto env = GetEnv();
        auto st = env->CreateAndRegisterAllocator(mem_info, arena_cfg);
        if (!st.IsOK()) {
          throw std::runtime_error("Error when creating and registering allocator: " + st.ErrorMessage());
        }
      });
  m.def(
      "create_and_register_allocator_v2", [](const std::string& provider_type, const OrtMemoryInfo& mem_info, const ProviderOptions& options, const OrtArenaCfg* arena_cfg = nullptr) -> void {
        auto env = GetEnv();
        auto st = env->CreateAndRegisterAllocatorV2(provider_type, mem_info, options, arena_cfg);
        if (!st.IsOK()) {
          throw std::runtime_error("Error when creating and registering allocator in create_and_register_allocator_v2: " + st.ErrorMessage());
        }
      });

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

void addObjectMethods(py::module& m, ExecutionProviderRegistrationFn ep_registration_fn) {
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
      .def_static("cann", []() { return OrtDevice::NPU; })
      .def_static("fpga", []() { return OrtDevice::FPGA; })
      .def_static("npu", []() { return OrtDevice::NPU; })
      .def_static("dml", []() { return OrtDevice::GPU; })
      .def_static("default_memory", []() { return OrtDevice::MemType::DEFAULT; });

  py::class_<OrtArenaCfg> ort_arena_cfg_binding(m, "OrtArenaCfg");
  // Note: Doesn't expose initial_growth_chunk_sizes_bytes/max_power_of_two_extend_bytes option.
  // This constructor kept for backwards compatibility, key-value pair constructor overload exposes all options
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
                       }))
      .def(py::init([](const py::dict& feeds) {
        auto ort_arena_cfg = std::make_unique<OrtArenaCfg>();
        for (const auto kvp : feeds) {
          std::string key = kvp.first.cast<std::string>();
          if (key == "max_mem") {
            ort_arena_cfg->max_mem = kvp.second.cast<size_t>();
          } else if (key == "arena_extend_strategy") {
            ort_arena_cfg->arena_extend_strategy = kvp.second.cast<int>();
          } else if (key == "initial_chunk_size_bytes") {
            ort_arena_cfg->initial_chunk_size_bytes = kvp.second.cast<int>();
          } else if (key == "max_dead_bytes_per_chunk") {
            ort_arena_cfg->max_dead_bytes_per_chunk = kvp.second.cast<int>();
          } else if (key == "initial_growth_chunk_size_bytes") {
            ort_arena_cfg->initial_growth_chunk_size_bytes = kvp.second.cast<int>();
          } else if (key == "max_power_of_two_extend_bytes") {
            ort_arena_cfg->max_power_of_two_extend_bytes = kvp.second.cast<int>();
          } else {
            ORT_THROW("Invalid OrtArenaCfg option: ", key);
          }
        }
        return ort_arena_cfg;
      }))
      .def_readwrite("max_mem", &OrtArenaCfg::max_mem)
      .def_readwrite("arena_extend_strategy", &OrtArenaCfg::arena_extend_strategy)
      .def_readwrite("initial_chunk_size_bytes", &OrtArenaCfg::initial_chunk_size_bytes)
      .def_readwrite("max_dead_bytes_per_chunk", &OrtArenaCfg::max_dead_bytes_per_chunk)
      .def_readwrite("initial_growth_chunk_size_bytes", &OrtArenaCfg::initial_growth_chunk_size_bytes)
      .def_readwrite("max_power_of_two_extend_bytes", &OrtArenaCfg::max_power_of_two_extend_bytes);

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
      .def_property(
          "enable_cpu_mem_arena",
          [](const PySessionOptions* options) -> bool { return options->value.enable_cpu_mem_arena; },
          [](PySessionOptions* options, bool enable_cpu_mem_arena) -> void {
            options->value.enable_cpu_mem_arena = enable_cpu_mem_arena;
          },
          R"pbdoc(Enables the memory arena on CPU. Arena may pre-allocate memory for future usage.
Set this option to false if you don't want it. Default is True.)pbdoc")
      .def_property(
          "enable_profiling",
          [](const PySessionOptions* options) -> bool { return options->value.enable_profiling; },
          [](PySessionOptions* options, bool enable_profiling) -> void {
            options->value.enable_profiling = enable_profiling;
          },
          R"pbdoc(Enable profiling for this session. Default is false.)pbdoc")
      .def_property(
          "profile_file_prefix",
          [](const PySessionOptions* options) -> std::basic_string<ORTCHAR_T> {
            return options->value.profile_file_prefix;
          },
          [](PySessionOptions* options, std::basic_string<ORTCHAR_T> profile_file_prefix) -> void {
            options->value.profile_file_prefix = std::move(profile_file_prefix);
          },
          R"pbdoc(The prefix of the profile file. The current time will be appended to the file name.)pbdoc")
      .def_property(
          "optimized_model_filepath",
          [](const PySessionOptions* options) -> std::basic_string<ORTCHAR_T> {
            return options->value.optimized_model_filepath;
          },
          [](PySessionOptions* options, std::basic_string<ORTCHAR_T> optimized_model_filepath) -> void {
            options->value.optimized_model_filepath = std::move(optimized_model_filepath);
          },
          R"pbdoc(
File path to serialize optimized model to.
Optimized model is not serialized unless optimized_model_filepath is set.
Serialized model format will default to ONNX unless:
- add_session_config_entry is used to set 'session.save_model_format' to 'ORT', or
- there is no 'session.save_model_format' config entry and optimized_model_filepath ends in '.ort' (case insensitive)

)pbdoc")
      .def_property(
          "enable_mem_pattern",
          [](const PySessionOptions* options) -> bool { return options->value.enable_mem_pattern; },
          [](PySessionOptions* options, bool enable_mem_pattern) -> void {
            options->value.enable_mem_pattern = enable_mem_pattern;
          },
          R"pbdoc(Enable the memory pattern optimization. Default is true.)pbdoc")
      .def_property(
          "enable_mem_reuse",
          [](const PySessionOptions* options) -> bool { return options->value.enable_mem_reuse; },
          [](PySessionOptions* options, bool enable_mem_reuse) -> void {
            options->value.enable_mem_reuse = enable_mem_reuse;
          },
          R"pbdoc(Enable the memory reuse optimization. Default is true.)pbdoc")
      .def_property(
          "logid",
          [](const PySessionOptions* options) -> std::string {
            return options->value.session_logid;
          },
          [](PySessionOptions* options, std::string logid) -> void {
            options->value.session_logid = std::move(logid);
          },
          R"pbdoc(Logger id to use for session output.)pbdoc")
      .def_property(
          "log_severity_level",
          [](const PySessionOptions* options) -> int { return options->value.session_log_severity_level; },
          [](PySessionOptions* options, int log_severity_level) -> void {
            options->value.session_log_severity_level = log_severity_level;
          },
          R"pbdoc(Log severity level. Applies to session load, initialization, etc.
0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2.)pbdoc")
      .def_property(
          "log_verbosity_level",
          [](const PySessionOptions* options) -> int { return options->value.session_log_verbosity_level; },
          [](PySessionOptions* options, int log_verbosity_level) -> void {
            options->value.session_log_verbosity_level = log_verbosity_level;
          },
          R"pbdoc(VLOG level if DEBUG build and session_log_severity_level is 0.
Applies to session load, initialization, etc. Default is 0.)pbdoc")
      .def_property(
          "intra_op_num_threads",
          [](const PySessionOptions* options) -> int { return options->value.intra_op_param.thread_pool_size; },
          [](PySessionOptions* options, int value) -> void { options->value.intra_op_param.thread_pool_size = value; },
          R"pbdoc(Sets the number of threads used to parallelize the execution within nodes. Default is 0 to let onnxruntime choose.)pbdoc")
      .def_property(
          "inter_op_num_threads",
          [](const PySessionOptions* options) -> int { return options->value.inter_op_param.thread_pool_size; },
          [](PySessionOptions* options, int value) -> void { options->value.inter_op_param.thread_pool_size = value; },
          R"pbdoc(Sets the number of threads used to parallelize the execution of the graph (across nodes). Default is 0 to let onnxruntime choose.)pbdoc")
      .def_property(
          "execution_mode",
          [](const PySessionOptions* options) -> ExecutionMode { return options->value.execution_mode; },
          [](PySessionOptions* options, ExecutionMode execution_mode) -> void {
            options->value.execution_mode = execution_mode;
          },
          R"pbdoc(Sets the execution mode. Default is sequential.)pbdoc")
      .def_property(
          "execution_order",
          [](const PySessionOptions* options) -> ExecutionOrder { return options->value.execution_order; },
          [](PySessionOptions* options, ExecutionOrder execution_order) -> void {
            options->value.execution_order = execution_order;
          },
          R"pbdoc(Sets the execution order. Default is basic topological order.)pbdoc")
      .def_property(
          "graph_optimization_level",
          [](const PySessionOptions* options) -> GraphOptimizationLevel {
            GraphOptimizationLevel retval = ORT_ENABLE_ALL;
            switch (options->value.graph_optimization_level) {
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
                options->value.graph_optimization_level = onnxruntime::TransformerLevel::Default;
                break;
              case ORT_ENABLE_BASIC:
                options->value.graph_optimization_level = onnxruntime::TransformerLevel::Level1;
                break;
              case ORT_ENABLE_EXTENDED:
                options->value.graph_optimization_level = onnxruntime::TransformerLevel::Level2;
                break;
              case ORT_ENABLE_ALL:
                options->value.graph_optimization_level = onnxruntime::TransformerLevel::Level3;
                break;
            }
          },
          R"pbdoc(Graph optimization level for this session.)pbdoc")
      .def_property(
          "use_deterministic_compute",
          [](const PySessionOptions* options) -> bool { return options->value.use_deterministic_compute; },
          [](PySessionOptions* options, bool use_deterministic_compute) -> void {
            options->value.use_deterministic_compute = use_deterministic_compute;
          },
          R"pbdoc(Whether to use deterministic compute. Default is false.)pbdoc")
      .def(
          "add_free_dimension_override_by_denotation",
          [](PySessionOptions* options, const char* dim_name, int64_t dim_value)
              -> void { options->value.free_dimension_overrides.push_back(
                            onnxruntime::FreeDimensionOverride{
                                dim_name,
                                onnxruntime::FreeDimensionOverrideType::Denotation,
                                dim_value}); },
          R"pbdoc(Specify the dimension size for each denotation associated with an input's free dimension.)pbdoc")
      .def(
          "add_free_dimension_override_by_name",
          [](PySessionOptions* options, const char* dim_name, int64_t dim_value)
              -> void { options->value.free_dimension_overrides.push_back(
                            onnxruntime::FreeDimensionOverride{
                                dim_name,
                                onnxruntime::FreeDimensionOverrideType::Name,
                                dim_value}); },
          R"pbdoc(Specify values of named dimensions within model inputs.)pbdoc")
      .def(
          "add_session_config_entry",
          [](PySessionOptions* options, const char* config_key, const char* config_value) -> void {
            // config_key and config_value will be copied
            const Status status = options->value.config_options.AddConfigEntry(config_key, config_value);
            if (!status.IsOK())
              throw std::runtime_error(status.ErrorMessage());
          },
          R"pbdoc(Set a single session configuration entry as a pair of strings.)pbdoc")
      .def(
          "get_session_config_entry",
          [](const PySessionOptions* options, const char* config_key) -> std::string {
            const std::string key(config_key);
            std::string value;
            if (!options->value.config_options.TryGetConfigEntry(key, value))
              throw std::runtime_error("SessionOptions does not have configuration with key: " + key);

            return value;
          },
          R"pbdoc(Get a single session configuration value using the given configuration key.)pbdoc")
      .def(
          "register_custom_ops_library",
          [](PySessionOptions* options, const char* library_name) -> void {
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
            OrtPybindThrowIfError(options->RegisterCustomOpsLibrary(ToPathString(library_name)));
#else
            ORT_UNUSED_PARAMETER(options);
            ORT_UNUSED_PARAMETER(library_name);
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
            ORT_THROW_IF_ERROR(options->value.AddInitializer(name, ml_value));
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
        ORT_THROW_IF_ERROR(options->value.AddExternalInitializers(names_ptrs, values_ptrs));
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
      .def(py::init([](const PySessionOptions& so, const std::string arg, bool is_arg_file_name,
                       bool load_config_from_model = false) {
        auto env = GetEnv();
        std::unique_ptr<PyInferenceSession> sess;

        // separate creation of the session from model loading unless we have to read the config from the model.
        // in a minimal build we only support load via Load(...) and not at session creation time
        if (load_config_from_model) {
#if !defined(ORT_MINIMAL_BUILD)
          sess = std::make_unique<PyInferenceSession>(std::move(env), so, arg, is_arg_file_name);

          RegisterCustomOpDomains(sess.get(), so);

          OrtPybindThrowIfError(sess->GetSessionHandle()->Load());
#else
          ORT_THROW("Loading configuration from an ONNX model is not supported in this build.");
#endif
        } else {
          sess = std::make_unique<PyInferenceSession>(std::move(env), so);
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
          RegisterCustomOpDomains(sess.get(), so);
#endif

          if (is_arg_file_name) {
            OrtPybindThrowIfError(sess->GetSessionHandle()->Load(arg));
          } else {
            OrtPybindThrowIfError(sess->GetSessionHandle()->Load(arg.data(), narrow<int>(arg.size())));
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
      .def("run_async",
           [](PyInferenceSession* sess,
              std::vector<std::string> output_names,
              std::map<std::string, py::object> pyfeeds,
              PyCallback callback, py::object user_data = {},
              RunOptions* run_options = nullptr)
               -> void {
             std::unique_ptr<AsyncResource> async_resource = std::make_unique<AsyncResource>();
             async_resource->callback = callback;
             async_resource->user_data = user_data;
             // prepare feeds
             async_resource->ReserveFeeds(pyfeeds.size());
             for (auto feed : pyfeeds) {
               if (!feed.second.is(py::none())) {
                 OrtValue ml_value;
                 auto px = sess->GetSessionHandle()->GetModelInputs();
                 if (!px.first.IsOK() || !px.second) {
                   throw std::runtime_error("Either failed to get model inputs from the session object or the input def list was null");
                 }
                 CreateGenericMLValue(px.second, GetAllocator(), feed.first, feed.second, &ml_value);
                 ThrowIfPyErrOccured();
                 async_resource->feeds.push_back(ml_value);
                 async_resource->feeds_raw.push_back(&async_resource->feeds.back());
                 async_resource->feed_names.push_back(feed.first);
                 async_resource->feed_names_raw.push_back(async_resource->feed_names.back().c_str());
               }
             }
             // prepare fetches
             async_resource->ReserveFetches(output_names.size());
             for (auto& output_name : output_names) {
               async_resource->fetch_names.push_back(output_name);
               async_resource->fetch_names_raw.push_back(async_resource->fetch_names.back().c_str());
               async_resource->fetches_raw.push_back({});
             }
             const RunOptions* run_async_option = run_options ? run_options : &async_resource->default_run_option;
             common::Status status = sess->GetSessionHandle()->RunAsync(run_async_option,
                                                                        gsl::span(async_resource->feed_names_raw.data(), async_resource->feed_names_raw.size()),
                                                                        gsl::span(async_resource->feeds_raw.data(), async_resource->feeds_raw.size()),
                                                                        gsl::span(async_resource->fetch_names_raw.data(), async_resource->fetch_names_raw.size()),
                                                                        gsl::span(async_resource->fetches_raw.data(), async_resource->fetches_raw.size()),
                                                                        AsyncCallback,
                                                                        async_resource.get());
             if (status.IsOK()) {
               async_resource.release();
             }
             OrtPybindThrowIfError(status);
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
      .def("run_with_ortvaluevector", [](PyInferenceSession* sess, RunOptions run_options, const std::vector<std::string>& feed_names, const std::vector<OrtValue>& feeds, const std::vector<std::string>& fetch_names, std::vector<OrtValue>& fetches, const std::vector<OrtDevice>& fetch_devices) -> void {
        // release GIL to allow multiple python threads to invoke Run() in parallel.
        py::gil_scoped_release release;
        OrtPybindThrowIfError(sess->GetSessionHandle()->Run(run_options, feed_names, feeds, fetch_names, &fetches, &fetch_devices));
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
          "session_options", [](const PyInferenceSession* sess) -> PySessionOptions* {
            auto session_options = std::make_unique<PySessionOptions>();
            session_options->value = sess->GetSessionHandle()->GetSessionOptions();
            return session_options.release();
          },
          py::return_value_policy::take_ownership)
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
      })
      .def("get_tuning_results", [](PyInferenceSession* sess) -> py::list {
#if !defined(ORT_MINIMAL_BUILD)
        py::list ret;
        for (const auto& trs : sess->GetSessionHandle()->GetTuningResults()) {
          py::dict py_trs;
          py_trs["ep"] = trs.ep;
          py_trs["results"] = trs.results;
          py_trs["validators"] = trs.validators;
          ret.append(std::move(py_trs));
        }

        return ret;
#else
            ORT_UNUSED_PARAMETER(sess);
            ORT_THROW("TunableOp and get_tuning_results are not supported in this build.");
#endif
      })
      .def("set_tuning_results", [](PyInferenceSession* sess, py::list results, bool error_on_invalid) -> void {
#if !defined(ORT_MINIMAL_BUILD)
        std::vector<TuningResults> tuning_results;
        for (auto handle : results) {
          auto py_trs = handle.cast<py::dict>();
          TuningResults trs;
          trs.ep = py_trs["ep"].cast<py::str>();

          for (const auto [py_op_sig, py_kernel_map] : py_trs["results"].cast<py::dict>()) {
            KernelMap kernel_map;
            for (const auto [py_params_sig, py_kernel_id] : py_kernel_map.cast<py::dict>()) {
              kernel_map[py_params_sig.cast<py::str>()] = py_kernel_id.cast<py::int_>();
            }
            trs.results[py_op_sig.cast<py::str>()] = kernel_map;
          }

          for (const auto [k, v] : py_trs["validators"].cast<py::dict>()) {
            trs.validators[k.cast<py::str>()] = v.cast<py::str>();
          }

          tuning_results.emplace_back(std::move(trs));
        }

        Status status = sess->GetSessionHandle()->SetTuningResults(tuning_results, error_on_invalid);
        if (!status.IsOK()) {
          throw std::runtime_error("Error in execution: " + status.ErrorMessage());
        }
#else
            ORT_UNUSED_PARAMETER(sess);
            ORT_UNUSED_PARAMETER(results);
            ORT_UNUSED_PARAMETER(error_on_invalid);
            ORT_THROW("TunableOp and set_tuning_results are not supported in this build.");
#endif
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

  auto env = GetEnv();

  addGlobalMethods(m);
  addObjectMethods(m, RegisterExecutionProviders);
  addOrtValueMethods(m);
  addSparseTensorMethods(m);
  addIoBindingMethods(m);

#if !defined(__APPLE__) && !defined(ORT_MINIMAL_BUILD)
  if (!InitProvidersSharedLibrary()) {
    const logging::Logger& default_logger = logging::LoggingManager::DefaultLogger();
    LOGS(default_logger, WARNING) << "Init provider bridge failed.";
  }
#endif

  addGlobalSchemaFunctions(m);
  addOpSchemaSubmodule(m);
  addOpKernelSubmodule(m);
}

void InitArray() {
  ([]() -> void {
    // import_array1() forces a void return value.
    import_array1();
  })();
}

namespace {
// This class provides a static shell for on-demand and thread-safe construction
// of Environment object for both Inference and Training python layers.
// Environment class contains objects such as default logger, that must be available
// for the entire duration of a program that makes use of onnxruntime library.
// Because Python is a garbage collected language and the order of destruction of objects
// is not guaranteed we design this class with the following important features.

// 1) we make this class a singleton that is a function local static. The function local statics
//    are constructed when the function is called the very first time. This fact has several important
//    properties.
//    - First, it is constructed before it is first needed possibly by another static object
//      and destroyed after that object is destroyed.
//    - Second, it is constructed in a thread safe manner.
//    - Last, this order of construction/destruction is enforced across the compilation units, as opposed
//      to the static objects that are simply declared in order in a single unit, but their lifespan is
//      unconnected to that of in other compilation units. This is achieved automatically by run-time
//      by execution atexit() to build a chain.
//  2) We make Environment owned by a shared_ptr. This is done because python objects such as Inference and Training
//    sessions depend on this global. We acquire a shared_ptr instance when those objects are instantiated
//    and release it automatically when they are garbage collected. Although with this change all of the
//    globals seem to have been destroyed after module is unloaded and GC runs before that, it is cheap and gives
//    a piece of mind as there were situations when GC was still running in the past after Env was gone.
//    TrainingEnv global also holds shared reference to this global.
// 3) We guard against singleton resurrection attempts to detect code runs that when it should
//    not and make necessary adjustments.
//    For all the related details and why it is needed see "Modern C++ design" by A. Alexandrescu Chapter 6.
class EnvInitializer {
 public:
  static std::shared_ptr<onnxruntime::Environment> SharedInstance() {
    // Guard against attempts to resurrect the singleton
    if (EnvInitializer::destroyed) {
      ORT_THROW("Detected an attempt to resurrect destroyed Environment");
    }
    static EnvInitializer env_holder;
    return env_holder.Get();
  }

 private:
  EnvInitializer() {
    // Initialization of the module
    InitArray();
    std::unique_ptr<Environment> env_ptr;
    Env::Default().GetTelemetryProvider().SetLanguageProjection(OrtLanguageProjection::ORT_PROJECTION_PYTHON);
    OrtPybindThrowIfError(Environment::Create(std::make_unique<LoggingManager>(
                                                  std::make_unique<CLogSink>(),
                                                  Severity::kWARNING, false, LoggingManager::InstanceType::Default,
                                                  &SessionObjectInitializer::default_logger_id),
                                              env_ptr));
    session_env_ = std::shared_ptr<Environment>(env_ptr.release());
    destroyed = false;
  }

  ~EnvInitializer() {
    destroyed = true;
  }

  std::shared_ptr<Environment> Get() const {
    return session_env_;
  }

  std::shared_ptr<Environment> session_env_;

  static bool destroyed;
};

bool EnvInitializer::destroyed = false;
}  // namespace

std::shared_ptr<onnxruntime::Environment> GetEnv() {
  return EnvInitializer::SharedInstance();
}

}  // namespace python
}  // namespace onnxruntime
