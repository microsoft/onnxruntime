// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include <hip/hip_version.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/providers/shared_library/provider_api.h"
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "core/common/safeint.h"
#include "core/common/logging/severity.h"
#include "core/providers/migraphx/migraphx_execution_provider.h"
#include "core/providers/migraphx/migraphx_execution_provider_info.h"
#include "core/providers/migraphx/migraphx_execution_provider_utils.h"
#include "core/providers/migraphx/migraphx_allocator.h"
#include "core/providers/migraphx/gpu_data_transfer.h"
#include "core/providers/migraphx/migraphx_call.h"
#include "core/providers/migraphx/migraphx_stream_handle.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#if defined(_MSC_VER)
#pragma warning(default : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic pop
#endif

#define MEMCPY_S(dest, src, destsz, srcsz) memcpy(dest, src, std::min(destsz, srcsz))

namespace onnxruntime {

class Memcpy final : public OpKernel {
 public:
  Memcpy(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    const auto* X = ctx->Input<Tensor>(0);
    ORT_ENFORCE(X != nullptr, "Memcpy: Input tensor is nullptr.");
    Tensor* Y = ctx->Output(0, X->Shape());
    ORT_ENFORCE(Y != nullptr, "Memcpy: Failed to allocate output tensor.");
    const IDataTransfer* gpu_data_transfer = Info().GetDataTransferManager().GetDataTransfer(X->Location().device, Y->Location().device);
    if (!gpu_data_transfer)
      return Status(common::ONNXRUNTIME, common::EP_FAIL, "gpu data transfer is missing in Migraphx EP.");
    // CopyTensorAsync could handle both pinned memory and non-pinned CPU memory.
    // For non-pinned CPU memory, the copy is synchronous.
    return gpu_data_transfer->CopyTensorAsync(*X, *Y, *(ctx->GetComputeStream()));
  }
};

template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kMIGraphXExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kMIGraphXExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMIGraphXExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMIGraphXExecutionProvider, kOnnxDomain, 1, MemcpyToHost);

static std::shared_ptr<KernelRegistry> s_kernel_registry;

void InitializeRegistry() {
  s_kernel_registry = KernelRegistry::Create();

  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMIGraphXExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMIGraphXExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
  };

  for (auto& function_table_entry : function_table) {
    ORT_THROW_IF_ERROR(s_kernel_registry->Register(function_table_entry()));
  }
}

void DeleteRegistry() {
  s_kernel_registry.reset();
}

std::shared_ptr<KernelRegistry> MIGraphXExecutionProvider::GetKernelRegistry() const {
  return s_kernel_registry;
}

static std::string_view GetArenaExtendStrategyName(ArenaExtendStrategy strategy) {
  switch (strategy) {
    case ArenaExtendStrategy::kNextPowerOfTwo:
      return "kNextPowerOfTwo";
    case ArenaExtendStrategy::kSameAsRequested:
      return "kSameAsRequested";
    default:
      return "Unknown";
  }
}

#define GET_ENV(variable, value, ...)                              \
  const auto value##env{GetEnvironmentVar(variable)};              \
  if (!value##env.empty()) {                                       \
    __VA_ARGS__;                                                   \
    LOGS_DEFAULT(INFO) << "\n " << variable << ": " << value##env; \
  }

#define GET_ENV_BOOL(variable, value) \
  GET_ENV(variable, value, value = std::stoi(value##env) != 0)

#define GET_ENV_STRING(variable, value) \
  GET_ENV(variable, value, value = value##env)

MIGraphXExecutionProvider::MIGraphXExecutionProvider(const MIGraphXExecutionProviderInfo& info)
    : IExecutionProvider{kMIGraphXExecutionProvider, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::AMD, info.device_id)},
      device_id_{info.device_id},
      fp16_enable_{info.fp16_enable},
#if HIP_VERSION_MAJOR > 6 || (HIP_VERSION_MAJOR == 6 && (HIP_VERSION_MINOR > 4 || (HIP_VERSION_MINOR == 4 && HIP_VERSION_PATCH >= 2)))
      bf16_enable_{info.bf16_enable},
#endif
#if HIP_VERSION_MAJOR > 6 || (HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR >= 4)
      fp8_enable_{info.fp8_enable},
#endif
      int8_enable_{info.int8_enable},
      model_cache_path_{info.model_cache_dir},
      t_{info.target_device.c_str()},
      exhaustive_tune_{info.exhaustive_tune},
      metadef_id_generator_{ModelMetadefIdGenerator::Create()},
      external_alloc_{info.external_alloc},
      external_free_{info.external_free},
      external_empty_cache_{info.external_empty_cache},
      max_dynamic_batch_{info.max_dynamic_batch},
      max_compiled_models_{info.max_compiled_models} {
  InitProviderOrtApi();

  // Set GPU device to be used and read device properties for feature usage.

  HIP_CALL_THROW(hipSetDevice(device_id_));
  HIP_CALL_THROW(hipGetDeviceProperties(&device_prop_, device_id_));

  // Overwrite initialized values with values from environment variables.

  LOGS_DEFAULT(INFO) << "[MIGraphX EP] MIGraphX ENV Override Variables Set:";
  GET_ENV_BOOL(migraphx_env_vars::kFP16Enable, fp16_enable_);
#if HIP_VERSION_MAJOR > 6 || (HIP_VERSION_MAJOR == 6 && (HIP_VERSION_MINOR > 4 || (HIP_VERSION_MINOR == 4 && HIP_VERSION_PATCH >= 2)))
  GET_ENV_BOOL(migraphx_env_vars::kBF16Enable, bf16_enable_);
#endif
#if HIP_VERSION_MAJOR > 6 || (HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR >= 4)
  GET_ENV_BOOL(migraphx_env_vars::kFP8Enable, fp8_enable_);
#endif
  GET_ENV_BOOL(migraphx_env_vars::kINT8Enable, int8_enable_);
  GET_ENV(migraphx_env_vars::kINT8CalibrationTableName, int8_calibration_cache_name_);
  GET_ENV(migraphx_env_vars::kINT8UseNativeMIGraphXCalibrationTable, int8_use_native_migraphx_calibration_table_);
  GET_ENV_STRING(migraphx_env_vars::kCachePath, calibration_cache_path_);
  GET_ENV_STRING(migraphx_env_vars::kModelCachePath, model_cache_path_);
  GET_ENV_BOOL(migraphx_env_vars::kDumpModelOps, dump_model_ops_);
  GET_ENV_BOOL(migraphx_env_vars::kExhaustiveTune, exhaustive_tune_);
  GET_ENV(migraphx_env_vars::kMaxCompiledModels, max_compiled_models_, max_compiled_models_ = std::stoul(max_compiled_models_env));

  // Verify configuration correctness and adjust accordingly.

#if HIP_VERSION_MAJOR < 6 || (HIP_VERSION_MAJOR == 6 && (HIP_VERSION_MINOR < 4 || (HIP_VERSION_MINOR == 4 && HIP_VERSION_PATCH < 2)))
  LOGS_DEFAULT(VERBOSE) << "MIGraphX: BF16 Quantization requires ROCm 6.4.2 or greater";
  bf16_enable_ = false;
#endif

  if (bf16_enable_ && fp16_enable_) {
    bf16_enable_ = false;
    fp16_enable_ = false;
    LOGS_DEFAULT(FATAL) << "MIGraphX: BF16 and FP16 Quantization Mutually exclusive. Ignoring both Quantization flags";
  }

#if HIP_VERSION_MAJOR < 6 || (HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR < 4)
  LOGS_DEFAULT(VERBOSE) << "MIGraphX: FP8 Quantization requires ROCm 6.4 or greater";
  fp8_enable_ = false;
#endif

  if (int8_enable_ && fp8_enable_) {
    LOGS_DEFAULT(FATAL) << "MIGraphX: FP8 and INT8 Quantization Mutually exclusive. Ignoring both Quantization flags";
  }

  if (int8_enable_ ^ fp8_enable_) {
    int8_calibration_table_name_ =
        int8_calibration_cache_name_env.empty() ? info.int8_calibration_table_name : int8_calibration_cache_name_env;
    int8_use_native_calibration_table_ =
        int8_use_native_migraphx_calibration_table_env.empty() ? info.int8_use_native_calibration_table : std::stoi(int8_use_native_migraphx_calibration_table_env) != 0;
  }

  int8_calibration_cache_available_ =
    (int8_enable_ || fp8_enable_) && !int8_calibration_table_name_.empty();


  // Load INT8 calibration table
  if (int8_calibration_cache_available_) {
    std::unordered_map<std::string, float> dynamic_range_map;
    auto calibration_cache_path = GetCachePath(calibration_cache_path_, int8_calibration_table_name_);
    if (!ReadDynamicRange(calibration_cache_path, int8_use_native_calibration_table_, dynamic_range_map)) {
      throw std::runtime_error("Session Failed to read INT8 calibration table " + calibration_cache_path.string());
    }
  }

  // Print configured options for the session.

  LOGS_DEFAULT(VERBOSE) << "[MIGraphX EP] MIGraphX provider Session Options:"
                        << "\n " << migraphx_provider_option::kDeviceId << ": " << device_id_
                        << "\n " << migraphx_provider_option::kFp16Enable << ": " << fp16_enable_
                        << "\n " << migraphx_provider_option::kBf16Enable << ": " << bf16_enable_
                        << "\n " << migraphx_provider_option::kFp8Enable << ": " << fp8_enable_
                        << "\n " << migraphx_provider_option::kInt8Enable << ": " << int8_enable_
                        << "\n " << migraphx_provider_option::kMemLimit << ": " << mem_limit_
                        << "\n " << migraphx_provider_option::kArenaExtendStrategy << ": " << GetArenaExtendStrategyName(arena_extend_strategy_)
                        << "\n dump_model_ops: " << dump_model_ops_
                        << "\n " << migraphx_provider_option::kExhaustiveTune << ": " << exhaustive_tune_
                        << "\n " << migraphx_provider_option::kInt8CalibTable << ": " << int8_calibration_table_name_
                        << "\n int8_calibration_cache_available: " << int8_calibration_cache_available_
                        << "\n " << migraphx_provider_option::kInt8UseNativeCalibTable << ": " << int8_use_native_calibration_table_
                        << "\n " << migraphx_provider_option::kModelCacheDir << ": " << model_cache_path_
                        << "\n " << migraphx_provider_option::kModelMaxDynamicBatch << ": " << max_dynamic_batch_;
}

std::vector<AllocatorPtr> MIGraphXExecutionProvider::CreatePreferredAllocators() {
  AllocatorCreationInfo default_memory_info(
      [](OrtDevice::DeviceId device_id) {
        return std::make_unique<MIGraphXAllocator>(device_id, onnxruntime::CUDA);
      },
      device_id_);
  AllocatorCreationInfo pinned_allocator_info(
      [](OrtDevice::DeviceId device_id) {
        return std::make_unique<MIGraphXPinnedAllocator>(device_id, CUDA_PINNED);
      },
      device_id_);
  return std::vector<AllocatorPtr>{CreateAllocator(default_memory_info), CreateAllocator(pinned_allocator_info)};
}

std::unique_ptr<onnxruntime::IDataTransfer> MIGraphXExecutionProvider::GetDataTransfer() const {
  return std::make_unique<onnxruntime::GPUDataTransfer>();
}

static bool IsTypeSupported(const NodeArg* node_arg) {
  const auto* type_proto = node_arg->TypeAsProto();
  if (!type_proto) {
    return false;
  }

  switch (type_proto->tensor_type().elem_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BFLOAT16:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT4E2M1:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FN:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FNUZ:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2FNUZ:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT4:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT4:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT16:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT64:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL:
      return true;
    default:
      return false;
  }
}

static bool getMIGraphXType(ONNXTensorElementDataType type,
                            migraphx_shape_datatype_t& mgx_type) {
  mgx_type = migraphx_shape_float_type;
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      mgx_type = migraphx_shape_half_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      mgx_type = migraphx_shape_bf16_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      mgx_type = migraphx_shape_float_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      mgx_type = migraphx_shape_double_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
      mgx_type = migraphx_shape_fp8e4m3fnuz_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
      mgx_type = migraphx_shape_fp8e4m3fn_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
      mgx_type = migraphx_shape_fp8e5m2_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
      mgx_type = migraphx_shape_fp8e5m2fnuz_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1:
      mgx_type = migraphx_shape_fp4x2_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
      mgx_type = migraphx_shape_int8_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      mgx_type = migraphx_shape_int8_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      mgx_type = migraphx_shape_int16_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      mgx_type = migraphx_shape_int32_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      mgx_type = migraphx_shape_int64_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:
      mgx_type = migraphx_shape_uint8_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      mgx_type = migraphx_shape_uint8_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      mgx_type = migraphx_shape_uint16_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      mgx_type = migraphx_shape_uint32_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      mgx_type = migraphx_shape_uint64_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      mgx_type = migraphx_shape_bool_type;
      break;
    default:
      LOGS_DEFAULT(VERBOSE) << "MiGraphx: unsupported data type " << type << ", fallback to CPU";
      LOGS_DEFAULT(VERBOSE) << "implementation";
      return false;
  }

  return true;
}

std::vector<int64_t> toVector(const ONNX_NAMESPACE::int64s& nums) {
  std::vector<int64_t> result;
  size_t num = nums.size();
  for (size_t i = 0; i < num; ++i) {
    result.push_back(nums[i]);
  }

  return result;
}

static bool IsUnsupportedOpMode(const onnxruntime::GraphViewer& graph_viewer, const Node* node) {
  std::vector<NodeIndex> input_nodes;
  const auto& optype = node->OpType();
  if (optype == "ArgMax" || optype == "ArgMin") {
    const auto& attributes = node->GetAttributes();
    // we do not support select_last_index = 1 for now
    auto sli_attr = attributes.find("select_last_index");
    if (sli_attr != attributes.end() && (*sli_attr).second.i() != 0) {
      return true;
    }
  } else if (optype == "ConstantOfShape") {
    if (!canEvalNodeArgument(graph_viewer, node, {0}, input_nodes)) {
      return true;
    }
  } else if (optype == "ConvInteger") {
    // only support int8 and uint8 type
    const auto& input_type = node->InputDefs()[0]->TypeAsProto();
    if (input_type == nullptr) {
      return true;
    }

    if ((input_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) &&
        (input_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8)) {
      return true;
    }
  } else if (optype == "Expand") {
    // MIGraphX only supports constant shape input values
    if (!canEvalNodeArgument(graph_viewer, node, {1}, input_nodes)) {
      return true;
    }
  } else if (optype == "MaxPool") {
    // MaxPool "indices" output is not currently supported.
    if (node->OutputDefs().size() > 1) {
      return true;
    }

    // ceil_mode and dilations attrs are not supported in MIGraphX
    const auto& attributes = node->GetAttributes();
    auto dila_attr = attributes.find("dilations");
    if (dila_attr != attributes.end()) {
      auto dilas = toVector((*dila_attr).second.ints());
      bool ret = std::all_of(dilas.begin(), dilas.end(), [](auto i) { return i == 1; });
      if (ret == false) {
        return true;
      }
    }

    // storage order 1 (column major format) is not supported
    auto storage_order_attr = attributes.find("storage_order");
    if (storage_order_attr != attributes.end() && (*storage_order_attr).second.i() != 0) {
      return true;
    }

    // do not support int8 and uint8 type
    const auto& input_type = node->InputDefs()[0]->TypeAsProto();
    if (input_type == nullptr) {
      return true;
    }
    auto data_type = input_type->tensor_type().elem_type();
    if (data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8 ||
        data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8) {
      return true;
    }
  } else if (optype == "MatMulInteger") {
    // only support int8 and uint8 type
    const auto& input_type = node->InputDefs()[0]->TypeAsProto();
    if (input_type == nullptr) {
      return true;
    }

    if ((input_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) &&
        (input_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8)) {
      return true;
    }
  } else if (optype == "NonZero") {
    if (!canEvalNodeArgument(graph_viewer, node, {0}, input_nodes)) {
      return true;
    }
  } else if (optype == "OneHot") {
    if (!canEvalNodeArgument(graph_viewer, node, {1}, input_nodes)) {
      return true;
    }
  } else if (optype == "Pad") {
    const auto& args = node->InputDefs();
    // if pad size is not constant, migraphx cannot support
    if (args.size() >= 2) {
      if (!canEvalNodeArgument(graph_viewer, node, {1}, input_nodes)) {
        return true;
      }
    }

    const auto& attributes = node->GetAttributes();
    // Pad only support reflect, constant and edge mode currently
    auto mode_attr = attributes.find("mode");
    std::string mode = "constant";
    if (mode_attr != attributes.end()) {
      mode = (*mode_attr).second.s();
    }
    static const std::set<std::string> allowed_modes = {"constant", "reflect", "edge"};
    if (allowed_modes.count(mode) == 0) {
      return true;
    }

  } else if (optype == "Range") {
    auto arg_num = node->InputDefs().size();
    std::vector<std::size_t> vec(arg_num);
    std::iota(vec.begin(), vec.end(), 0);
    if (!canEvalNodeArgument(graph_viewer, node, vec, input_nodes)) {
      return true;
    }
  } else if (optype == "Reshape") {
    const auto& args = node->InputDefs();
    if (args.size() == 2) {
      if (canEvalNodeArgument(graph_viewer, node, {1}, input_nodes)) {
        return false;
      }
      return true;
    }
  } else if (optype == "Resize" || optype == "Upsample") {
    const auto& attributes = node->GetAttributes();
    auto ct_attr = attributes.find("coordinate_transformation_mode");
    if (ct_attr != attributes.end()) {
      auto ct = (*ct_attr).second.s();
      if (ct == "tf_crop_and_resize") {
        return true;
      }
    }

    auto mode_attr = attributes.find("mode");
    if (mode_attr != attributes.end()) {
      auto mode = (*mode_attr).second.s();
      if (mode == "cubic") {
        return true;
      }
    }
  } else if (optype == "ReduceSum") {
    const auto& args = node->InputDefs();
    if (args.size() == 2) {
      if (canEvalNodeArgument(graph_viewer, node, {1}, input_nodes)) {
        return false;
      }
      return true;
    }
  } else if (optype == "Slice") {
    // MIGraphX does not properly handle the situation where any
    // value of the "starts" attribute is higher than a corresponding
    // value in the "ends"
    auto arg_num = node->InputDefs().size();
    std::vector<std::size_t> vec(arg_num);
    std::iota(vec.begin(), vec.end(), 0);
    vec.erase(vec.begin());
    if (!canEvalNodeArgument(graph_viewer, node, vec, input_nodes)) {
      return true;
    }

    const auto& attributes = node->GetAttributes();
    if (attributes.count("starts") > 0 && attributes.count("ends") > 0) {
      auto starts = toVector((*attributes.find("starts")).second.ints());
      auto ends = toVector((*attributes.find("ends")).second.ints());

      for (std::size_t i = 0; i < starts.size(); ++i) {
        if (starts.at(i) > ends.at(i)) {
          return true;
        }
      }
    }
  } else if (optype == "Split") {
    // cannot process input dim of 0 size
    const auto arg_s = node->InputDefs()[0]->Shape();
    if (arg_s != nullptr) {
      const auto& tensor_dims = arg_s->dim();
      std::vector<std::size_t> dims;
      for (auto&& dim : tensor_dims) {
        dims.emplace_back(dim.has_dim_value() ? dim.dim_value() : 0);
      }
      if (dims == std::vector<std::size_t>{0}) {
        return true;
      }
    }

    const auto& args = node->InputDefs();
    if (args.size() == 2) {
      if (canEvalNodeArgument(graph_viewer, node, {1}, input_nodes)) {
        return false;
      }
      return true;
    }
  } else if (optype == "Tile") {
    if (!canEvalNodeArgument(graph_viewer, node, {1}, input_nodes)) {
      return true;
    }
  } else if (optype == "TopK") {
    if (!canEvalNodeArgument(graph_viewer, node, {1}, input_nodes)) {
      return true;
    }
  } else if (optype == "Unsqueeze" || optype == "Squeeze") {
    const auto& args = node->InputDefs();
    if (args.size() == 2) {
      if (canEvalNodeArgument(graph_viewer, node, {1}, input_nodes)) {
        return false;
      }
      return true;
    }
  }

  // Op doesn't fall into known any of unsupported modes.
  return false;
}

void SubgraphPostProcessing(const onnxruntime::GraphViewer& graph_viewer, std::vector<std::vector<NodeIndex>>& clusters,
                            [[maybe_unused]] const logging::Logger& logger) {
  // Then check whether a subgraph should fall back to CPU
  // 1. Check whether a subgraph contains a RNN operator
  std::unordered_set<std::string> rnn_names = {"RNN", "GRU", "LSTM"};
  std::unordered_set<std::string> op_names = {"AveragePool", "Conv", "Gemm", "LRN", "MatMul", "MaxPool"};

  auto it = std::remove_if(clusters.begin(), clusters.end(), [&](auto git) {
    for (auto index : git) {
      auto node = graph_viewer.GetNode(index);
      if (node->OpType() == "Reshape") {
        const auto& args = node->InputDefs();
        if (args.size() == 2) {
          std::vector<NodeIndex> node_inputs;
          if (canEvalNodeArgument(graph_viewer, node, {1}, node_inputs)) {
            return !std::all_of(node_inputs.begin(), node_inputs.end(), [&](auto i) {
              return std::find(git.begin(), git.end(), i) != git.end();
            });
          } else {
            return true;
          }
        }
      }
    }

    // rnn operators, run on GPU
    if (std::any_of(git.begin(), git.end(), [&](auto nid) {
          const auto& node = graph_viewer.GetNode(nid);
          const auto& op_type = node->OpType();
          return (rnn_names.count(op_type) > 0);
        })) {
      return false;
    }

    // check operators gemm, matmul, convolution, lrn.
    if (std::any_of(git.begin(), git.end(), [&](auto nid) {
          const auto& node = graph_viewer.GetNode(nid);
          const auto& op_type = node->OpType();
          if (op_names.count(op_type) > 0) {
            // check number of elements in input
            auto inputs = node->InputDefs();
            if (std::any_of(inputs.begin(), inputs.end(), [&](auto& arg) {
                  const auto& arg_s = arg->Shape();
                  if (arg_s == nullptr) return false;
                  const auto& tensor_dims = arg_s->dim();
                  std::vector<std::size_t> dims;
                  for (auto&& dim : tensor_dims) {
                    dims.emplace_back(dim.has_dim_value() ? dim.dim_value() : 1);
                  }
                  return (std::accumulate(dims.begin(), dims.end(), 1ULL, std::multiplies<std::size_t>{}) > 300);
                })) {
              return false;
            }

            return true;
          }

          return false;
        })) {
      return false;
    }

    return true;
  });

  clusters.erase(it, clusters.end());
}

static bool IsNodeSupported(const std::set<std::string>& op_set,
                            const onnxruntime::GraphViewer& graph_viewer,
                            const NodeIndex node_idx,
                            [[maybe_unused]] const logging::Logger& logger) {
  const auto& node = graph_viewer.GetNode(node_idx);
  const auto& optype = node->OpType();
  const auto& domain = node->Domain();

  // Three types of checking:
  // 1. Check input and output data types are supported.
  // 2. Check op_type is implemented in migraphx
  // 3. Check the mode is implemented in migraphx
  // if 3. failed, call the constant folding capability in migraphx
  // to see whether some input parameters can be calculated statically
  // check data type
  bool are_types_supported = true;

  node->ForEachDef([&are_types_supported](const onnxruntime::NodeArg& node_arg, bool /*is_input*/) {
    are_types_supported &= IsTypeSupported(&node_arg);
  });

  if (!are_types_supported) {
    return false;
  }

  // whether an operator implemented in migraphx
  if (op_set.count(optype) == 0) {
    return false;
  }

  // check that some modes might not be supported in migraphx for some operators
  if (domain == kOnnxDomain && IsUnsupportedOpMode(graph_viewer, node)) {
    // not supported, then check the constant folding capability of migraphx
    // to see whether it is supported
    return false;
  }

  return true;
}

std::unique_ptr<IndexedSubGraph> MIGraphXExecutionProvider::GetSubGraph(const std::vector<std::size_t>& graph_nodes_index, const GraphViewer& graph, bool is_graph_split) const {
  std::unordered_set<size_t> node_set;
  node_set.reserve(graph_nodes_index.size());
  for (const auto& index : graph_nodes_index) {
    node_set.insert(index);
  }

  // Get parent graph output names
  std::vector<std::string> graph_output_names;
  for (const auto* output_arg : graph.GetOutputs()) {
    graph_output_names.push_back(output_arg->Name());
  }

  // Find inputs and outputs of the subgraph
  std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::IndexedSubGraph::Create();
  std::unordered_map<const NodeArg*, int> fused_inputs, fused_outputs, fused_outputs_to_add, graph_outputs_to_add;
  std::unordered_set<const NodeArg*> erased;
  int input_order = 0;
  int output_order = 0;

  for (const auto& index : graph_nodes_index) {
    sub_graph->Nodes().push_back(index);
    const auto& node = graph.GetNode(index);
    for (const auto& input : node->InputDefs()) {
      const auto& it = fused_outputs.find(input);
      if (it != fused_outputs.end()) {
        fused_outputs.erase(it);
        erased.insert(input);
      } else if (erased.find(input) == erased.end()) {
        // Only when input is neither in output list nor erased list, add the input to input list
        fused_inputs[input] = input_order++;
      }
    }

    for (const auto& input : node->ImplicitInputDefs()) {
      const auto& it = fused_outputs.find(input);
      if (it != fused_outputs.end()) {
        fused_outputs.erase(it);
        erased.insert(input);
      } else if (erased.find(input) == erased.end()) {
        // Only when input is neither in output list nor erased list, add the input to input list
        fused_inputs[input] = input_order++;
      }
    }

    // For output searching, there are two special cases,
    // One is, if node's OutputEdges are more than its outputs, meaning certain output is used more than once,
    // if the output is connected to nodes that don't belong to the subgraph, the output need to be added
    // to the output list
    // The other one is, if subgraph's node output is parent graph's output. the node output should
    // be also added to the subgraph's output list
    if (node->GetOutputEdgesCount() > node->OutputDefs().size()) {
      for (auto it = node->OutputEdgesBegin(), end = node->OutputEdgesEnd(); it != end; ++it) {
        const auto& target_node = it->GetNode();
        const auto& target_op_type = target_node.OpType();

        if (target_op_type == "If" || target_op_type == "Loop" || target_op_type == "Scan") {
          const auto& src_output_idx = it->GetSrcArgIndex();

          // Do this to avoid signed to unsigned comparrison here
          // if src_output_index is invalid (-1 or less) signal that to be larger than size + 1
          // This ensures the check below fails
          size_t output_index = 0;
          if(src_output_idx < 0)
            output_index = node->OutputDefs().size() + 1;

          if (output_index < node->OutputDefs().size()) {
            const auto* output_def = node->OutputDefs()[src_output_idx];
            if (output_def && fused_outputs.find(output_def) == fused_outputs.end() && erased.find(output_def) == erased.end()) {
              fused_outputs_to_add[output_def] = output_order++;
            }
          }
          continue;
        }
        const auto& node_idx = target_node.Index();
        const auto& output = (it->GetNode()).InputDefs()[it->GetDstArgIndex()];
        if (node_set.find(node_idx) != node_set.end()) {
          const auto& iter = fused_inputs.find(output);
          if (iter != fused_inputs.end()) {
            fused_inputs.erase(iter);
            erased.insert(output);
          } else if (erased.find(output) == erased.end()) {
            if (std::find(graph_output_names.begin(),
                          graph_output_names.end(), output->Name()) != graph_output_names.end()) {
              graph_outputs_to_add[output] = output_order;
            }
            fused_outputs[output] = output_order++;
          }
        } else {
          fused_outputs_to_add[output] = output_order++;
        }
      }
    } else {
      for (const auto& output : node->OutputDefs()) {
        const auto& it = fused_inputs.find(output);
        if (it != fused_inputs.end()) {
          fused_inputs.erase(it);
          erased.insert(output);
        }
        // Only when output is neither in input list nor erased list, add the output to output list
        else {
          if (erased.find(output) == erased.end()) {
            if (std::find(graph_output_names.begin(),
                          graph_output_names.end(), output->Name()) != graph_output_names.end()) {
              graph_outputs_to_add[output] = output_order;
            }
            fused_outputs[output] = output_order++;
          }
        }
      }
    }
  }

  fused_outputs.insert(fused_outputs_to_add.begin(), fused_outputs_to_add.end());
  fused_outputs.insert(graph_outputs_to_add.begin(), graph_outputs_to_add.end());

  // Sort inputs and outputs by the order they were added
  std::multimap<int, const NodeArg*> inputs, outputs;
  for (auto it = fused_inputs.begin(), end = fused_inputs.end(); it != end; ++it) {
    inputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
  }

  for (auto it = fused_outputs.begin(), end = fused_outputs.end(); it != end; ++it) {
    outputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
  }

  // It is possible that an output of an node is put bebind the output of an later
  // node in the graph output list. So we should sort the output name according
  // to the graph output names
  std::vector<std::string> output_names;
  std::unordered_set<std::string> graph_out_names;
  for (const auto& output : outputs) {
    if (output.second->Exists()) {
      auto name = output.second->Name();
      if (std::find(graph_output_names.begin(), graph_output_names.end(), name) == graph_output_names.end()) {
        // if graph is split we dont know if output is used so we need this, otherwise if the graph isn't split
        // then we can safely assume this output is a dangling output from a node and to discard it as part of the
        // final graph output
        if (is_graph_split) {
          output_names.push_back(name);
        }
      } else {
        graph_out_names.insert(name);
      }
    }
  }

  for (auto& name : graph_output_names) {
    if (std::find(graph_out_names.begin(), graph_out_names.end(), name) != graph_out_names.end())
      output_names.push_back(name);
  }

  // Generate unique kernel name for MIGraphX subgraph
  uint64_t model_hash = 0;
  int id = metadef_id_generator_->GenerateId(graph, model_hash);
  std::string subgraph_id = std::to_string(model_hash) + "_" + std::to_string(id);
  auto meta_def = IndexedSubGraph_MetaDef::Create();
  const std::string graph_type = graph.IsSubgraph() ? "subgraph" : "graph";
  meta_def->name() = "MGXKernel_" + graph_type + "_" + graph.Name() + "_" + subgraph_id;

  // Assign inputs and outputs to subgraph's meta_def
  for (const auto& input : inputs) {
    if (input.second->Exists()) {
      meta_def->inputs().push_back(input.second->Name());
    }
  }

  for (const auto& output : output_names) {
    meta_def->outputs().push_back(output);
  }

  meta_def->domain() = kMSDomain;
  meta_def->since_version() = 1;
  sub_graph->SetMetaDef(std::move(meta_def));

  return sub_graph;
}

static std::vector<NodeIndex>
GetUnsupportedNodeIndices(const GraphViewer& graph_viewer,
                          /*out*/ std::unordered_set<std::string>& mgx_required_initializers,
                          const logging::Logger& logger) {

#ifdef HAVE_MIGRAPHX_API_GET_ONNX_OPERATORS
  // In ROCm 7.2 onward we'll query the MIGraphX API to get the supported op list
  static std::set<std::string> mgx_supported_ops{};
  auto list = migraphx::get_onnx_operators();
  for(const auto& name : list)
  {
    mgx_supported_ops.emplace(name);
  }
#else
  static std::set<std::string> mgx_supported_ops = {"Abs",
                                                    "Acos",
                                                    "Acosh",
                                                    "Add",
                                                    "And",
                                                    "ArgMax",
                                                    "ArgMin",
                                                    "Asin",
                                                    "Asinh",
                                                    "Atan",
                                                    "Atanh",
                                                    "ATen",
                                                    "Attention",
                                                    "AveragePool",
                                                    "BatchNormalization",
                                                    "BiasGelu",
                                                    "Cast",
                                                    "Ceil",
                                                    "Celu",
                                                    "Clip",
                                                    "Concat",
                                                    "Constant",
                                                    "ConstantFill",
                                                    "ConstantOfShape",
                                                    "Conv",
                                                    "ConvInteger",
                                                    "ConvTranspose",
                                                    "Cos",
                                                    "Cosh",
                                                    "CumSum",
                                                    "DepthToSpace",
                                                    "DequantizeLinear",
                                                    "Div",
                                                    "Dropout",
                                                    "Einsum",
                                                    "Elu",
                                                    "Equal",
                                                    "Erf",
                                                    "Exp",
                                                    "Expand",
                                                    "EyeLike",
                                                    "FastGelu",
                                                    "Flatten",
                                                    "Floor",
                                                    "GRU",
                                                    "Gather",
                                                    "GatherElements",
                                                    "GatherND",
                                                    "Gelu",
                                                    "Gemm",
                                                    "GlobalAveragePool",
                                                    "GlobalMaxPool",
                                                    "Greater",
                                                    "GreaterOrEqual",
                                                    "GroupNormalization",
                                                    "GroupNorm",
                                                    "GroupQueryAttention",
                                                    "HardSigmoid",
                                                    "HardSwish",
                                                    "Identity",
                                                    "If",
                                                    "ImageScaler",
                                                    "InstanceNormalization",
                                                    "IsNan",
                                                    "LayerNormalization",
                                                    "LeakyRelu",
                                                    "Less",
                                                    "LessOrEqual",
                                                    "Log",
                                                    "LogSoftmax",
                                                    "Loop",
                                                    "LpNormalization",
                                                    "LRN",
                                                    "LSTM",
                                                    "MatMul",
                                                    "MatMulInteger",
                                                    "MatMulNBits",
                                                    "Max",
                                                    "MaxPool",
                                                    "Mean",
                                                    "Min",
                                                    "Mod",
                                                    "Mul",
                                                    "Multinomial",
                                                    "MultiHeadAttention",
                                                    "Neg",
                                                    "NegativeLogLikelihoodLoss",
                                                    "NhwcConv",
                                                    "NonMaxSuppression",
                                                    "NonZero",
                                                    "Not",
                                                    "OneHot",
                                                    "Or",
                                                    "Pad",
                                                    "Pow",
                                                    "PRelu",
                                                    "QLinearAdd",
                                                    "QLinearConv",
                                                    "QLinearMatMul",
                                                    "QuantizeLinear",
                                                    "QuickGelu",
                                                    "DynamicQuantizeLinear",
                                                    "RandomNormal",
                                                    "RandomNormalLike",
                                                    "RandomUniform",
                                                    "RandomUniformLike",
                                                    "Range",
                                                    "Reciprocal",
                                                    "ReduceL1",
                                                    "ReduceL2",
                                                    "ReduceLogSum",
                                                    "ReduceLogSumExp",
                                                    "ReduceMax",
                                                    "ReduceMean",
                                                    "ReduceMin",
                                                    "ReduceProd",
                                                    "ReduceSum",
                                                    "ReduceSumSquare",
                                                    "Relu",
                                                    "Reshape",
                                                    "Resize",
                                                    "ReverseSequence",
                                                    "RNN",
                                                    "Roialign",
                                                    "RotaryEmbedding",
                                                    "Round",
                                                    "Scatter",
                                                    "ScatterElements",
                                                    "ScatterND",
                                                    "Selu",
                                                    "Shape",
                                                    "Sigmoid",
                                                    "Sign",
                                                    "SimplifiedLayerNormalization",
                                                    "Sin",
                                                    "Sinh",
                                                    "Size",
                                                    "SkipLayerNormalization",
                                                    "SkipSimplifiedLayerNormalization",
                                                    "Slice",
                                                    "Softmax",
                                                    "SoftmaxCrossEntropyLoss",
                                                    "Softplus",
                                                    "Softsign",
                                                    "SpaceToDepth",
                                                    "Split",
                                                    "Sqrt",
                                                    "Squeeze",
                                                    "Sub",
                                                    "Sum",
                                                    "Tan",
                                                    "Tanh",
                                                    "ThresholdedRelu",
                                                    "Tile",
                                                    "TopK",
                                                    "Transpose",
                                                    "Trilu",
                                                    "Unsqueeze",
                                                    "Upsample",
                                                    "Where",
                                                    "Xor"};
#endif

  std::vector<NodeIndex> unsupported_nodes_idx;
  for (const auto& node_idx : graph_viewer.GetNodesInTopologicalOrder()) {
    if (IsNodeSupported(mgx_supported_ops, graph_viewer, node_idx, logger)) {
      // Collect inputs that are initializers
      graph_viewer.GetNode(node_idx)->ForEachDef([&mgx_required_initializers,
                                                  &graph_viewer](const onnxruntime::NodeArg& node_arg, bool is_input) {
              if(is_input && graph_viewer.GetAllInitializedTensors().count(node_arg.Name())) {
                mgx_required_initializers.insert(node_arg.Name());
              } },
                                                 true);
    } else {
      unsupported_nodes_idx.push_back(node_idx);
    }
  }

  return unsupported_nodes_idx;
}

// Returns a vector clusters(or node_idx). For each unsupported node, the graph
// is split into 3 parts. supported_cluster + (UNsupported_node + rest_of_the_graph).
// This functions returns vector of all supported_subgraphx by amdmigraphx
static std::vector<std::vector<NodeIndex>>
GetPartitionedSubgraphs(const std::vector<NodeIndex>& topological_order,
                        const std::vector<NodeIndex>& unsupported_nodes) {
  std::vector<std::vector<NodeIndex>> mgx_subgraphx;

  auto prev = topological_order.begin();

  for (const auto& unsup_node : unsupported_nodes) {
    auto it = std::find(prev, topological_order.end(), unsup_node);
    // Create a cluster vector[supported_node_idx, unsupported_node_idx)
    // and append it to return list.
    std::vector<NodeIndex> this_subgraph{prev, it};
    if (!this_subgraph.empty()) {
      mgx_subgraphx.push_back(std::move(this_subgraph));
    }
    // Point prev to node idx past this unsuported node.
    prev = ++it;
  }

  // Tail
  std::vector<NodeIndex> this_subgraph{prev, topological_order.end()};
  if (!this_subgraph.empty()) {
    mgx_subgraphx.push_back(std::move(this_subgraph));
  }

  return mgx_subgraphx;
}

void MIGraphXExecutionProvider::dump_model_as_onnx(const std::string& onnx_buffer,
                                                   const std::string& model_name) const {
  // dump onnx file if environment var is set
  if (dump_model_ops_) {
    std::ofstream ofs(model_name, std::ios::binary);
    if (!ofs.is_open()) {
      ORT_THROW("Failed to open file to dump ONNX model: " + model_name);
    }
    ofs.write(onnx_buffer.c_str(), onnx_buffer.size());
    ofs.close();
    LOGS_DEFAULT(INFO) << "ONNX model dumped to " << model_name;
  }
}

std::vector<std::unique_ptr<ComputeCapability>>
MIGraphXExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                         const IKernelLookup& /*kernel_lookup*/,
                                         const GraphOptimizerRegistry& /* graph_optimizer_registry */,
                                         IResourceAccountant* /* resource_accountant */) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  if (graph_viewer.IsSubgraph()) {
    const auto* parent_node = graph_viewer.ParentNode();
    if (parent_node) {
      const auto& parent_op_type = parent_node->OpType();
      if (parent_op_type == "If" || parent_op_type == "Loop" || parent_op_type == "Scan") {
        return result;
      }
    }
  }

  auto model = graph_viewer.CreateModel(*GetLogger());
  auto model_proto = model->ToProto();
  graph_viewer.ToProto(*model_proto->mutable_graph(), true, true);
  model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string onnx_string_buffer;
  model_proto->SerializeToString(onnx_string_buffer);
  model_path_ = graph_viewer.ModelPath();

  dump_model_as_onnx(onnx_string_buffer, graph_viewer.Name() + ".onnx");

  // This is a list of initializers that migraphx considers as constants.
  // Example weights, reshape shape etc.
  std::unordered_set<std::string> mgx_required_initializers;
  const auto unsupported_nodes = GetUnsupportedNodeIndices(graph_viewer, mgx_required_initializers, *GetLogger());

  if (unsupported_nodes.size() > 0) {
    LOGS_DEFAULT(VERBOSE) << "============= Unsupported nodes ====================";
    for (auto idx : unsupported_nodes) {
      LOGS_DEFAULT(VERBOSE) << graph_viewer.GetNode(idx)->OpType();
    }
    LOGS_DEFAULT(VERBOSE) << "************* Unsupported nodes ********************";
  }

  if (unsupported_nodes.size() > 10) {
    return result;
  }

  bool is_graph_not_split = unsupported_nodes.empty();

  // If all ops are supported, no partitioning is required. Short-circuit and avoid splitting.
  if (is_graph_not_split) {
    auto node_indices = graph_viewer.GetNodesInTopologicalOrder();
    auto sub_graph = GetSubGraph(node_indices, graph_viewer, !is_graph_not_split);
    result.push_back(ComputeCapability::Create(std::move(sub_graph)));
  } else {
    auto mgx_clusters = GetPartitionedSubgraphs(graph_viewer.GetNodesInTopologicalOrder(), unsupported_nodes);

    // check whether a subgrap should fallback to CPU
    SubgraphPostProcessing(graph_viewer, mgx_clusters, *GetLogger());

    for (const auto& this_cluster : mgx_clusters) {
      auto sub_graph = GetSubGraph(this_cluster, graph_viewer, !is_graph_not_split);
      result.push_back(ComputeCapability::Create(std::move(sub_graph)));
    }
  }

  return result;
}

// Get input and output names from the graph
static std::pair<std::vector<std::string>, std::vector<std::string>> get_io_names(const GraphViewer& graph) {
  const auto& input_args = graph.GetInputs();
  std::vector<std::string> input_names;
  input_names.reserve(input_args.size());
  for (const auto& arg : input_args) {
    if (arg != nullptr) {
      input_names.push_back(arg->Name());
    }
  }

  const auto& out_args = graph.GetOutputs();
  std::vector<std::string> output_names;
  output_names.reserve(out_args.size());
  for (const auto& arg : out_args) {
    if (arg != nullptr) {
      output_names.push_back(arg->Name());
    }
  }

  return {std::move(input_names), std::move(output_names)};
}

// Attempt to load a model and catch any exceptions on load fail.
// Useful to default to EP to trigger the compile if file doesn't exist or loading fails.
bool load_precompiled_model(migraphx::program& prog, const std::filesystem::path& path) try {
  if (!path.empty() && exists(path)) {
    LOGS_DEFAULT(VERBOSE) << "[load_precompiled_model] Attempting to load model from disk: " << path.string();
    migraphx::file_options fo;
    fo.set_file_format("msgpack");
    prog = migraphx::load(path.string().c_str(), fo);
    LOGS_DEFAULT(VERBOSE) << "[load_precompiled_model] ✓ Successfully loaded model from disk";
    return true;
  }
  LOGS_DEFAULT(VERBOSE) << "[load_precompiled_model] Cache file does not exist: " 
                        << (path.empty() ? "(no path specified)" : path.string());
  return false;
} catch (const std::exception& e) {
  LOGS_DEFAULT(VERBOSE) << "[load_precompiled_model] ✗ Failed to load model from disk: " << e.what();
  return false;
  } catch (...) {
  LOGS_DEFAULT(VERBOSE) << "[load_precompiled_model] ✗ Failed to load model from disk (unknown exception)";
  return false;
}

void save_compiled_model(const migraphx::program& prog, const std::filesystem::path& path) {
  if (!path.empty()) {
    LOGS_DEFAULT(VERBOSE) << "[save_compiled_model] Saving compiled model to disk: " << path.string();
    migraphx::file_options fo;
    fo.set_file_format("msgpack");
    save(prog, path.string().c_str(), fo);
    LOGS_DEFAULT(VERBOSE) << "[save_compiled_model] ✓ Model saved successfully";
  }
}

// Generate a vector of batch sizes to compile based on max_compiled_models setting
// Batch sizes are evenly spaced between 1 and max_batch_size.
// max_compiled_models == 1: {max}
// max_compiled_models == 2: {1, max}
// max_compiled_models == 3: {1, max/2, max}
// max_compiled_models == N: N evenly spaced values from 1 to max
static std::vector<std::size_t> generate_compiled_batch_sizes(std::size_t max_batch_size, std::size_t max_compiled_models) {
  std::vector<std::size_t> batch_sizes;
  if (max_batch_size == 0) {
    return batch_sizes;
  }

  if (max_compiled_models == 0) {
    LOGS_DEFAULT(WARNING) << "max_compiled_models is 0. Defaulting to 1 (compile max batch size only).";
    max_compiled_models = 1;
  } else if (max_compiled_models > max_batch_size) {
    LOGS_DEFAULT(WARNING) << "max_compiled_models (" << max_compiled_models
                          << ") exceeds max_batch_size (" << max_batch_size
                          << "). Setting max_compiled_models to " << max_batch_size << ".";
    max_compiled_models = max_batch_size;
  }

  if (max_compiled_models == 1) {
    batch_sizes.push_back(max_batch_size);
  } else if (max_compiled_models >= 2) {
    // Evenly divide the range [1, max_batch_size] into max_compiled_models points
    std::size_t n = max_compiled_models;
    for (std::size_t i = 0; i < n; ++i) {
      std::size_t bs = 1 + (max_batch_size - 1) * i / (n - 1);
      // Avoid duplicates (can happen when max_batch_size is small relative to n)
      if (batch_sizes.empty() || bs > batch_sizes.back()) {
        batch_sizes.push_back(bs);
      }
    }
  }

  std::ostringstream oss;
  oss << "[MIGraphX] max_batch_size=" << max_batch_size
      << ", max_compiled_models=" << max_compiled_models
      << ", batch_sizes_to_compile=[";
  for (std::size_t i = 0; i < batch_sizes.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << batch_sizes[i];
  }
  oss << "] (count=" << batch_sizes.size() << ")";
  LOGS_DEFAULT(VERBOSE) << oss.str();

  return batch_sizes;
}

// Find the smallest compiled batch size >= requested_batch
// Uses the same evenly-spaced scheme as generate_compiled_batch_sizes
static std::size_t find_nearest_compiled_batch_size(std::size_t requested_batch,
                                                           std::size_t max_batch_size,
                                                           std::size_t max_compiled_models) {
  if (max_batch_size == 0) {
    return 0;
  }

  if (max_compiled_models == 1) {
    return max_batch_size;
  }

  // Walk the evenly-spaced batch sizes and return the first one >= requested_batch
  std::size_t n = max_compiled_models;
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t bs = 1 + (max_batch_size - 1) * i / (n - 1);
    if (bs >= requested_batch) {
      return bs;
    }
  }
  return max_batch_size;
}

// Pad input tensor data to a larger batch size
// Copies the original data and replicates the last batch element to fill the padding
static void pad_input_tensor(const void* src_data, void* dst_data,
                             std::size_t original_batch, std::size_t padded_batch,
                             std::size_t element_size_bytes, std::size_t elements_per_batch,
                             hipStream_t stream) {
  std::size_t bytes_per_batch = element_size_bytes * elements_per_batch;
  
  // Copy original data
  HIP_CALL_THROW(hipMemcpyAsync(dst_data, src_data, 
                                original_batch * bytes_per_batch,
                                hipMemcpyDeviceToDevice, stream));
  
  // Pad with last batch element replicated
  if (original_batch > 0 && padded_batch > original_batch) {
    const char* last_batch = static_cast<const char*>(src_data) + (original_batch - 1) * bytes_per_batch;
    char* pad_start = static_cast<char*>(dst_data) + original_batch * bytes_per_batch;
    
    for (std::size_t i = original_batch; i < padded_batch; ++i) {
      HIP_CALL_THROW(hipMemcpyAsync(pad_start, last_batch, bytes_per_batch,
                                    hipMemcpyDeviceToDevice, stream));
      pad_start += bytes_per_batch;
    }
  }
}

// Allocate padded input buffers and pad the data for dynamic batching
// Returns true if padding was applied, false otherwise
// OPTIMIZATION: Reuses existing buffers if padded batch size matches
static bool allocate_and_pad_inputs(
    MIGraphXFuncState* mgx_state,
    Ort::KernelContext& ctx,
    std::size_t original_batch_size,
    std::size_t padded_batch_size,
    hipStream_t stream) {
  
  if (padded_batch_size <= original_batch_size || mgx_state->cached_inputs.empty()) {
    return false;  // No padding needed
  }
  
  // ═══════════════════════════════════════════════════════════════════════════
  // OPTIMIZATION: Check if we can reuse existing padded buffers
  // ═══════════════════════════════════════════════════════════════════════════
  bool can_reuse_buffers = (
      mgx_state->last_padded_batch_size == padded_batch_size &&
      !mgx_state->padded_input_buffers.empty() &&
      mgx_state->padded_input_buffers.size() == mgx_state->cached_inputs.size()
  );
  
  if (can_reuse_buffers) {
    LOGS_DEFAULT(VERBOSE) << "[allocate_and_pad_inputs] ✓✓✓ REUSING existing padded buffers "
                          << "(original=" << original_batch_size << ", padded=" << padded_batch_size << ")";
    
    // Just copy new data into existing buffers - skip allocation
    for (size_t i = 0; i < mgx_state->cached_inputs.size(); ++i) {
      const auto& cached_inp = mgx_state->cached_inputs[i];
      auto input_tensor = ctx.GetInput(cached_inp.ort_index);
      auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
      const auto tensor_shape = tensor_info.GetShape();
      
      if (tensor_shape.empty()) continue;
      
      auto& padded_buf = mgx_state->padded_input_buffers[i];
      
      // Calculate elements per batch
      std::size_t elements_per_batch = 1;
      for (std::size_t j = 1; j < tensor_shape.size(); ++j) {
        elements_per_batch *= tensor_shape[j];
      }
      
      // Calculate element size from tensor type
      std::size_t element_size_bytes;
      switch (tensor_info.GetElementType()) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
          element_size_bytes = sizeof(float);
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
          element_size_bytes = sizeof(uint16_t);
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
          element_size_bytes = sizeof(int64_t);
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
          element_size_bytes = sizeof(int32_t);
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
          element_size_bytes = sizeof(int16_t);
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
          element_size_bytes = sizeof(int8_t);
          break;
        default:
          element_size_bytes = sizeof(float);  // Fallback to float
          break;
      }
      
      // Reuse existing buffer - just pad with new data
      const void* original_data = input_tensor.GetTensorRawData();
      pad_input_tensor(original_data, padded_buf.data, original_batch_size, padded_batch_size,
                      element_size_bytes, elements_per_batch, stream);
    }
    
    // Update original batch tracking (padded batch is already correct)
    mgx_state->last_original_batch_size = original_batch_size;
    
    return true;
  }
  
  // ═══════════════════════════════════════════════════════════════════════════
  // Normal path: Allocate new buffers (batch size changed or first run)
  // ═══════════════════════════════════════════════════════════════════════════
  LOGS_DEFAULT(VERBOSE) << "[allocate_and_pad_inputs] Allocating NEW padded buffers: original=" 
                        << original_batch_size << ", padded=" << padded_batch_size;
  
  // Free old buffers if they exist
  for (auto& buf : mgx_state->padded_input_buffers) {
    if (buf.data != nullptr) {
      HIP_CALL_THROW(hipFree(buf.data));
      buf.data = nullptr;
    }
  }
  mgx_state->padded_input_buffers.clear();
  
  // Allocate and pad each input
  mgx_state->padded_input_buffers.reserve(mgx_state->cached_inputs.size());
  
  for (const auto& cached_inp : mgx_state->cached_inputs) {
    auto input_tensor = ctx.GetInput(cached_inp.ort_index);
    auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
    const auto tensor_shape = tensor_info.GetShape();
    
    if (tensor_shape.empty()) {
      LOGS_DEFAULT(VERBOSE) << "[allocate_and_pad_inputs] Skipping empty shape input";
      continue;
    }
    
    // Calculate padded shape
    std::vector<std::size_t> padded_lens(tensor_shape.begin(), tensor_shape.end());
    padded_lens[0] = padded_batch_size;  // Replace batch dimension
    
    // Create padded MIGraphX shape
    migraphx::shape padded_mgx_shape{cached_inp.mgx_shape.type(), padded_lens};
    std::size_t padded_bytes = padded_mgx_shape.bytes();
    
    // Allocate GPU buffer for padded data
    void* padded_data = nullptr;
    HIP_CALL_THROW(hipMalloc(&padded_data, padded_bytes));
    
    // Calculate elements per batch
    std::size_t elements_per_batch = 1;
    for (std::size_t i = 1; i < tensor_shape.size(); ++i) {
      elements_per_batch *= tensor_shape[i];
    }
    
    // Calculate element size from tensor type
    std::size_t element_size_bytes;
    switch (tensor_info.GetElementType()) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        element_size_bytes = sizeof(float);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        element_size_bytes = sizeof(uint16_t);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        element_size_bytes = sizeof(int64_t);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        element_size_bytes = sizeof(int32_t);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        element_size_bytes = sizeof(int16_t);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        element_size_bytes = sizeof(int8_t);
        break;
      default:
        element_size_bytes = sizeof(float);  // Fallback to float
        LOGS_DEFAULT(VERBOSE) << "[allocate_and_pad_inputs] Unknown element type, assuming float";
        break;
    }
    
    // Pad the data
    const void* original_data = input_tensor.GetTensorRawData();
    pad_input_tensor(original_data, padded_data, original_batch_size, padded_batch_size,
                    element_size_bytes, elements_per_batch, stream);
    
    // Store padded buffer info
    MIGraphXFuncState::PaddedBuffer buf;
    buf.data = padded_data;
    buf.size_bytes = padded_bytes;
    buf.mgx_shape = padded_mgx_shape;
    mgx_state->padded_input_buffers.push_back(buf);
    
    LOGS_DEFAULT(VERBOSE) << "[allocate_and_pad_inputs] Padded input '" << cached_inp.name 
                         << "': " << padded_bytes << " bytes";
  }
  
  // Update batch tracking
  mgx_state->last_original_batch_size = original_batch_size;
  mgx_state->last_padded_batch_size = padded_batch_size;
  
  LOGS_DEFAULT(VERBOSE) << "[allocate_and_pad_inputs] Allocated " 
                        << mgx_state->padded_input_buffers.size() << " padded buffers";
  return true;
}

// Helper: Extract output index from MIGraphX output parameter name
// MIGraphX names outputs as "#output_0", "#output_1", etc.
static int compute_output_index(const std::string_view sv) {
  constexpr std::string_view out_name_prefix = "#output_";
  const auto pos = sv.find(out_name_prefix);
  if (pos == std::string_view::npos) {
    return -1;
  }
  const auto index_str = sv.substr(pos + out_name_prefix.length());
  return ToInteger(Trim(index_str, std::isdigit));
}

// Free temporary output buffers
static void free_temp_output_buffers(MIGraphXFuncState* mgx_state) {
  for (auto& buf : mgx_state->temp_output_buffers) {
    if (buf.data != nullptr) {
      (void)hipFree(buf.data);  // Don't throw on cleanup
      buf.data = nullptr;
    }
  }
  mgx_state->temp_output_buffers.clear();
  mgx_state->temp_output_padded_batch_size = 0;
}

// Clear cached MIGraphX shapes (call when program changes)
static void clear_cached_mgx_shapes(MIGraphXFuncState* mgx_state) {
  mgx_state->cached_mgx_param_shapes.reset();
  mgx_state->cached_mgx_output_shapes.reset();
  mgx_state->ultra_fast_caches_populated = false;
  mgx_state->cached_program_hash.clear();
}

// Allocate or reuse temporary output buffers for slicing mode
// Returns vector of raw pointers for use with handle_program_input_outputs
static std::vector<void*> get_or_allocate_temp_output_buffers(
    MIGraphXFuncState* mgx_state,
    const migraphx::program_parameter_shapes& param_shapes,
    const migraphx::shapes& output_shapes,
    const std::unordered_map<std::string, std::size_t>& map_input_name_index,
    std::size_t padded_batch_size)
{
  // Check if we can reuse existing buffers
  bool can_reuse = (
      mgx_state->temp_output_padded_batch_size == padded_batch_size &&
      !mgx_state->temp_output_buffers.empty()
  );
  
  if (can_reuse) {
    LOGS_DEFAULT(VERBOSE) << "[get_or_allocate_temp_output_buffers] ✓✓✓ REUSING " 
                          << mgx_state->temp_output_buffers.size() << " temp output buffers";
    // Return raw pointers from existing buffers
    std::vector<void*> ptrs;
    ptrs.reserve(mgx_state->temp_output_buffers.size());
    for (const auto& buf : mgx_state->temp_output_buffers) {
      ptrs.push_back(buf.data);
    }
    return ptrs;
  }
  
  // Free old buffers if they exist
  free_temp_output_buffers(mgx_state);
  
  LOGS_DEFAULT(VERBOSE) << "[get_or_allocate_temp_output_buffers] Allocating NEW temp output buffers";
  
  // Count outputs and allocate
  std::vector<void*> ptrs;
  for (const auto& name : param_shapes.names()) {
    // Skip inputs
    if (map_input_name_index.find(name) != map_input_name_index.end()) {
      continue;
    }
    
    // This is an output
    const auto output_index = compute_output_index(name);
    if (output_index != -1) {
      const auto& mgx_shape = param_shapes[name];
      std::size_t size_bytes = mgx_shape.bytes();
      
      void* buffer = nullptr;
      auto hip_status = hipMalloc(&buffer, size_bytes);
      if (hip_status != hipSuccess) {
        // Clean up any allocated buffers on failure
        for (auto& buf : mgx_state->temp_output_buffers) {
          if (buf.data) (void)hipFree(buf.data);
        }
        mgx_state->temp_output_buffers.clear();
        ORT_THROW("hipMalloc failed for temporary output buffer");
      }
      
      MIGraphXFuncState::TempOutputBuffer temp_buf;
      temp_buf.data = buffer;
      temp_buf.size_bytes = size_bytes;
      temp_buf.mgx_shape = mgx_shape;
      mgx_state->temp_output_buffers.push_back(temp_buf);
      ptrs.push_back(buffer);
      
      LOGS_DEFAULT(VERBOSE) << "[get_or_allocate_temp_output_buffers] Allocated output " 
                            << output_index << ": " << size_bytes << " bytes";
    }
  }
  
  mgx_state->temp_output_padded_batch_size = padded_batch_size;
  LOGS_DEFAULT(VERBOSE) << "[get_or_allocate_temp_output_buffers] Allocated " 
                        << mgx_state->temp_output_buffers.size() << " temp output buffers";
  
  return ptrs;
}

// Order matters here especially if the program uses mixed quantization
// Calibrate on full precision for int8/fp8 and then quantize down to fp16
void calibrate_and_quantize(migraphx::program& prog,
                            const migraphx::target& t,
                            const migraphx::program_parameters quant_params,
                            bool fp16_enable,
                            bool bf16_enable,
                            bool int8_enable,
                            bool fp8_enable,
                            bool int8_calibration_cache_available,
                            std::unordered_map<std::string, float>& dynamic_range_map) {
  // Read in the calibration data and map it to an migraphx paramater map for the calibration ops
  if ((int8_enable ^ fp8_enable) && int8_calibration_cache_available) {
    LOGS_DEFAULT(VERBOSE) << "Quantizing input program";

    auto param_shapes = prog.get_parameter_shapes();

    // Add all calibration data read in from int8 table
    for (auto& [cal_key, cal_val] : dynamic_range_map) {
      auto cal_val_shape = migraphx::shape(migraphx_shape_float_type);
      quant_params.add(cal_key.c_str(), migraphx::argument(cal_val_shape, static_cast<void*>(std::move(&cal_val))));
    }

    // perform static quantization on the programs
    if (int8_enable) {
      LOGS_DEFAULT(VERBOSE) << "Quantizing input program to int8";
      migraphx::quantize_int8_options quant_opts;
      quant_opts.add_calibration_data(quant_params);
      // specify thing we want to int8 quantize
      quant_opts.add_op_name("convolution");
      quant_opts.add_op_name("dot");
      migraphx::quantize_int8(prog, t, quant_opts);
      LOGS_DEFAULT(VERBOSE) << "Quantizing int8: Complete";
    } else if (fp8_enable) {
#if HIP_VERSION_MAJOR > 6 || (HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR >= 4)
      LOGS_DEFAULT(VERBOSE) << "Quantizing input program to fp8";
      migraphx::quantize_fp8_options quant_opts;
      quant_opts.add_calibration_data(quant_params);
      migraphx::quantize_fp8(prog, t, quant_opts);
      LOGS_DEFAULT(VERBOSE) << "Quantizing fp8: Complete";
#endif
    }
  }

  if (fp16_enable) {
    LOGS_DEFAULT(VERBOSE) << "Quantizing input program to fp16";
    migraphx::quantize_fp16(prog);
    LOGS_DEFAULT(VERBOSE) << "Quantizing fp16: Complete";
  }

#if HIP_VERSION_MAJOR > 6 || (HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR >= 4 && HIP_VERSION_PATCH >= 2)
  if (bf16_enable) {
    LOGS_DEFAULT(VERBOSE) << "Quantizing input program to bf16";
    migraphx::quantize_bf16(prog);
    LOGS_DEFAULT(VERBOSE) << "Quantizing bf16: Complete";
  }
#endif
}

void compile_program(migraphx::program& prog,
                     const migraphx::target& t,
                     bool exhaustive_tune) {
  LOGS_DEFAULT(VERBOSE) << "Model Compile: Begin";
  migraphx::compile_options co;
  co.set_fast_math(false);
  co.set_exhaustive_tune_flag(exhaustive_tune);
  prog.compile(t, co);
  LOGS_DEFAULT(VERBOSE) << "Model Compile: Complete";
}

std::string to_hex(const uint64_t v) {
  std::array<char, sizeof v << 1> s{};
  auto [ptr, _] = std::to_chars(s.data(), s.data() + s.size(), v, 16);
  return std::string{s.data(), ptr};
}

template <typename T>
std::string make_hash(T v) {
  std::array<std::uint32_t, 4> temp{};
  MurmurHash3::x86_128(v.data(), gsl::narrow_cast<int32_t>(v.size()), temp[0], temp.data());
  return to_hex(temp[0] | static_cast<uint64_t>(temp[1]) << 32);
}

template <>
std::string make_hash(const char* v) {
  return make_hash(std::string_view{v});
}

// Helper: Compile a MIGraphX program from ONNX buffer
// If input_names and all_input_base_shapes are provided, sets batch-specific shapes.
// Otherwise, uses shapes already configured in options.
// If ctx and map_input_name_index are provided, populates quant_params for int8/fp8 calibration.
migraphx::program CompileProgramWithBatch(
    const std::string& onnx_string,
    migraphx::onnx_options& options,
    const migraphx::target& t,
    bool fp16_enable,
    bool bf16_enable,
    bool int8_enable,
    bool fp8_enable,
    bool int8_calibration_cache_available,
    std::unordered_map<std::string, float>& dynamic_range_map,
    bool exhaustive_tune,
    const std::filesystem::path& model_path,
    Ort::KernelContext* ctx = nullptr,
    const std::unordered_map<std::string, std::size_t>* map_input_name_index = nullptr,
    const std::vector<std::string>& input_names = {},
    const std::vector<std::vector<std::int64_t>>& all_input_base_shapes = {},
    size_t batch_size = 0)
{
  LOGS_DEFAULT(VERBOSE) << "[CompileBatch] Starting compilation";

  // Set input shapes with the specified batch size for ALL inputs (if provided)
  if (!input_names.empty() && !all_input_base_shapes.empty() && batch_size > 0) {
    LOGS_DEFAULT(VERBOSE) << "[CompileBatch] Setting batch size " << batch_size << " for " << input_names.size() << " inputs";
    for (size_t i = 0; i < input_names.size() && i < all_input_base_shapes.size(); ++i) {
      std::vector<std::size_t> shape_with_batch;
      shape_with_batch.push_back(batch_size);
      for (auto dim : all_input_base_shapes[i]) {
        shape_with_batch.push_back(static_cast<std::size_t>(dim));
      }
      options.set_input_parameter_shape(input_names[i], shape_with_batch);

      std::ostringstream ss;
      ss << "[";
      for (size_t j = 0; j < shape_with_batch.size(); ++j) {
        if (j > 0) ss << ", ";
        ss << shape_with_batch[j];
      }
      ss << "]";
      LOGS_DEFAULT(VERBOSE) << "[CompileBatch] Input '" << input_names[i] << "' shape: " << ss.str();
    }
  } else {
    LOGS_DEFAULT(VERBOSE) << "[CompileBatch] Using shapes already configured in options";
  }

#ifndef ENABLE_TRAINING_CORE
#ifdef HAVE_MIGRAPHX_API_ONNX_OPTIONS_SET_EXTERNAL_DATA_PATH
  if (!model_path.empty()) {
    options.set_external_data_path(model_path.parent_path().string());
  }
#endif
#endif

  LOGS_DEFAULT(VERBOSE) << "[CompileBatch] Parsing ONNX buffer";
  migraphx::program prog = migraphx::parse_onnx_buffer(onnx_string, options);
  LOGS_DEFAULT(VERBOSE) << "[CompileBatch] ONNX parsing complete";

  // Populate quant_params if int8/fp8 calibration is needed and runtime context is available
  migraphx::program_parameters quant_params;
  if ((int8_enable ^ fp8_enable) && int8_calibration_cache_available && ctx != nullptr && map_input_name_index != nullptr) {
    LOGS_DEFAULT(VERBOSE) << "[CompileBatch] Setting up quantization parameters from runtime tensors";
    auto local_param_shapes = prog.get_parameter_shapes();
    for (auto&& name : local_param_shapes.names()) {
      if (map_input_name_index->count(name) > 0) {
        auto input_tensor = ctx->GetInput(map_input_name_index->at(name));
        auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
        const auto tensor_shape = tensor_info.GetShape();
        const auto tensor_type = tensor_info.GetElementType();

        migraphx_shape_datatype_t mgx_type;
        getMIGraphXType(tensor_type, mgx_type);
        auto mgx_s = local_param_shapes[name];

        if (mgx_type != mgx_s.type()) {
          LOGS_DEFAULT(FATAL) << "MIGraphX: param type mismatch";
        }
        quant_params.add(name, migraphx::argument(local_param_shapes[name], const_cast<void*>(input_tensor.GetTensorRawData())));
      }
    }
  }

  calibrate_and_quantize(prog, t, quant_params, fp16_enable, bf16_enable, int8_enable,
                         fp8_enable, int8_calibration_cache_available, dynamic_range_map);
  compile_program(prog, t, exhaustive_tune);

  LOGS_DEFAULT(VERBOSE) << "[CompileBatch] Compilation complete";
  return prog;
}

// Helper: Load a precompiled model from cache or compile and save it
// This function encapsulates the common pattern of:
// 1. Try to load from cache
// 2. If cache miss, compile using CompileProgramWithBatch
// 3. Save the compiled model to cache
// Returns the loaded or compiled program
// Optional ctx and map_input_name_index can be provided for int8/fp8 calibration during compilation
static migraphx::program load_or_compile_model(
    const std::filesystem::path& cache_file,
    const std::string& onnx_string,
    migraphx::onnx_options& options,
    const migraphx::target& t,
    bool fp16_enable,
    bool bf16_enable,
    bool int8_enable,
    bool fp8_enable,
    bool int8_calibration_cache_available,
    std::unordered_map<std::string, float>& dynamic_range_map,
    bool exhaustive_tune,
    const std::filesystem::path& model_path,
    Ort::KernelContext* ctx = nullptr,
    const std::unordered_map<std::string, std::size_t>* map_input_name_index = nullptr,
    const std::vector<std::string>& input_names = {},
    const std::vector<std::vector<std::int64_t>>& all_input_base_shapes = {},
    size_t batch_size = 0)
{
  migraphx::program prog;

  LOGS_DEFAULT(VERBOSE) << "[load_or_compile_model] ==== ENTERING ====";
  LOGS_DEFAULT(VERBOSE) << "[load_or_compile_model] Cache file: " << (cache_file.empty() ? "(none)" : cache_file.string());
  LOGS_DEFAULT(VERBOSE) << "[load_or_compile_model] Batch size: " << (batch_size > 0 ? std::to_string(batch_size) : "(default)");

  if (!load_precompiled_model(prog, cache_file)) {
    // Cache miss - need to compile
    if (batch_size > 0) {
      LOGS_DEFAULT(VERBOSE) << "[load_or_compile_model] ✗ CACHE MISS for batch size " << batch_size << " - COMPILING...";
    } else {
      LOGS_DEFAULT(VERBOSE) << "[load_or_compile_model] ✗ CACHE MISS - COMPILING...";
    }
    LOGS_DEFAULT(VERBOSE) << "[load_or_compile_model] Compilation started (this may take a while)...";

    prog = CompileProgramWithBatch(
        onnx_string,
        options,
        t,
        fp16_enable,
        bf16_enable,
        int8_enable,
        fp8_enable,
        int8_calibration_cache_available,
        dynamic_range_map,
        exhaustive_tune,
        model_path,
        ctx,
        map_input_name_index,
        input_names,
        all_input_base_shapes,
        batch_size);

    LOGS_DEFAULT(VERBOSE) << "[load_or_compile_model] Compilation finished";
    
    save_compiled_model(prog, cache_file);
    if (!cache_file.empty()) {
      LOGS_DEFAULT(VERBOSE) << "[load_or_compile_model] Saved compiled model to disk: " << cache_file.string();
    }
  } else {
    // Cache hit - loaded from disk
    if (batch_size > 0) {
      LOGS_DEFAULT(VERBOSE) << "[load_or_compile_model] ✓ CACHE HIT - LOADING FROM DISK for batch size " << batch_size;
    } else {
      LOGS_DEFAULT(VERBOSE) << "[load_or_compile_model] ✓ CACHE HIT - LOADING FROM DISK";
    }
    LOGS_DEFAULT(VERBOSE) << "[load_or_compile_model] Loaded precompiled model from: " << cache_file.string();
  }

  LOGS_DEFAULT(VERBOSE) << "[load_or_compile_model] ==== EXITING ====";
  return prog;
}

// Helper: Run the MIGraphX program and handle outputs
// This function executes the compiled MIGraphX program and copies outputs that
// were not pre-allocated (input parameters reused as outputs) to the ORT output tensors
// If original_batch_size is provided and < padded batch size, slices the output to remove padding
static void run_migraphx_program(
    std::mutex* mgx_mu_ptr,
    const OrtApi* api,
    OrtKernelContext* context,
    Ort::KernelContext& ctx,
    migraphx::program& prog,
    migraphx::program_parameters& m,
    const std::vector<std::size_t>& prog_output_indices,
    std::size_t original_batch_size = 0,
    std::size_t padded_batch_size = 0)
{
  void* rocm_stream;
  Ort::ThrowOnError(api->KernelContext_GetGPUComputeStream(context, &rocm_stream));

  std::optional<migraphx::arguments> prog_outputs;
  {  // Scoped lock for thread safety
    std::lock_guard<std::mutex> lock(*mgx_mu_ptr);
    prog_outputs = prog.run_async(m, static_cast<hipStream_t>(rocm_stream));
  }

  bool needs_slicing = (original_batch_size > 0 && padded_batch_size > 0 && 
                        original_batch_size < padded_batch_size);

  // Process ALL outputs for proper slicing when needed
  auto output_num = prog_outputs->size();
  
  std::unordered_set<std::size_t> prog_output_indices_set(prog_output_indices.begin(), prog_output_indices.end());
  std::vector<std::size_t> outputs_to_copy;  // Outputs that need memcpy (not pre-allocated)
  
  for (std::size_t i = 0; i < output_num; ++i) {
    if (prog_output_indices_set.count(i) == 0) {
      outputs_to_copy.push_back(i);
    }
  }
  
  // First, handle pre-allocated outputs (need slicing but were already bound)
  // NOTE: This is a defensive path - pre-allocated outputs should NOT exist when slicing is needed.
  if (needs_slicing && !prog_output_indices_set.empty()) {
    for (std::size_t i = 0; i < output_num; ++i) {
      if (prog_output_indices_set.count(i) > 0) {
        // This output was pre-allocated with padded shape - need to copy sliced data
        auto gpu_res = (*prog_outputs)[i];
        migraphx::shape res_shape = gpu_res.get_shape();
        auto res_lens = res_shape.lengths();
        
        // Create sliced shape for ORT output
        std::vector<int64_t> ort_shape{res_lens.begin(), res_lens.end()};
        if (!ort_shape.empty() && static_cast<std::size_t>(ort_shape[0]) != original_batch_size) {
          ort_shape[0] = static_cast<int64_t>(original_batch_size);
          
          // Calculate bytes to copy (sliced portion only)
          std::size_t bytes_per_batch = res_shape.bytes() / padded_batch_size;
          std::size_t bytes_to_copy = bytes_per_batch * original_batch_size;
          
          // Allocate temp buffer for sliced data on GPU
          void* temp_sliced_buffer = nullptr;
          auto hip_status = hipMalloc(&temp_sliced_buffer, bytes_to_copy);
          if (hip_status != hipSuccess) {
            ORT_THROW("hipMalloc failed for sliced output buffer");
          }
          
          // Copy sliced data from MIGraphX output to temp buffer
          HIP_CALL_THROW(hipMemcpyWithStream(temp_sliced_buffer,
                                             gpu_res.data(),
                                             bytes_to_copy,
                                             hipMemcpyDeviceToDevice,
                                             static_cast<hipStream_t>(rocm_stream)));
          
          // Synchronize to ensure copy is complete before allocating ORT output
          HIP_CALL_THROW(hipStreamSynchronize(static_cast<hipStream_t>(rocm_stream)));
          
          // Now allocate the ORT output tensor with the SLICED shape
          auto output_tensor = ctx.GetOutput(i, ort_shape.data(), ort_shape.size());
          void* output_data = output_tensor.GetTensorMutableRawData();
          
          // Copy from temp buffer to ORT output
          HIP_CALL_THROW(hipMemcpyWithStream(output_data,
                                             temp_sliced_buffer,
                                             bytes_to_copy,
                                             hipMemcpyDeviceToDevice,
                                             static_cast<hipStream_t>(rocm_stream)));
          
          // Free temporary buffer
          (void)hipFree(temp_sliced_buffer);
        }
      }
    }
  }
  
  // Now handle outputs that need memcpy (not pre-allocated)
  if (prog_output_indices.size() < output_num) {
    for (std::size_t i : outputs_to_copy) {
      auto gpu_res = (*prog_outputs)[i];
      migraphx::shape res_shape = gpu_res.get_shape();
      auto res_lens = res_shape.lengths();
      
      // Adjust output shape if slicing is needed
      std::vector<int64_t> ort_shape{res_lens.begin(), res_lens.end()};
      if (needs_slicing && !ort_shape.empty()) {
        ort_shape[0] = original_batch_size;  // Slice batch dimension
      }
      
      auto output_tensor = ctx.GetOutput(i, ort_shape.data(), ort_shape.size());
      void* output_data = output_tensor.GetTensorMutableRawData();

      // Calculate bytes to copy (slice if needed)
      std::size_t bytes_to_copy = res_shape.bytes();
      if (needs_slicing && res_lens.size() > 0) {
        bytes_to_copy = (res_shape.bytes() / padded_batch_size) * original_batch_size;
      }

      HIP_CALL_THROW(hipMemcpyWithStream(output_data,
                                         gpu_res.data(),
                                         bytes_to_copy,
                                         hipMemcpyDeviceToDevice,
                                         static_cast<hipStream_t>(rocm_stream)));
    }
  }

}

// Helper: Handle input shape mismatch by recompiling the model with new input shapes
// This function is called when runtime input shapes differ from compiled shapes
static void handle_input_shape_mismatch(
    MIGraphXFuncState* mgx_state,
    const std::filesystem::path& model_cache_path,
    const std::filesystem::path& model_path,
    const std::string& mxr_filename_prefix,
    Ort::KernelContext& ctx,
    migraphx::program_parameter_shapes& param_shapes,
    std::vector<std::int64_t>& input_shapes)
{
  // Extract references from mgx_state for convenience
  auto& prog = mgx_state->prog;
  auto& cmp_options = mgx_state->options;
  const auto& map_input_name_index = mgx_state->input_name_indexes;

  // Build cache key from all inputs in map_input_name_index (already filtered to model inputs only)
  std::vector<std::int64_t> all_input_shapes;
  for (const auto& it : map_input_name_index) {
    auto input_tensor = ctx.GetInput(it.second);
    auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
    const auto tensor_shape = tensor_info.GetShape();
    all_input_shapes.insert(all_input_shapes.end(), tensor_shape.begin(), tensor_shape.end());
  }
  auto cache_hash = make_hash(all_input_shapes);

  // Check in-memory cached_programs first (before disk cache)
  if (mgx_state->cached_programs_ref.has_value()) {
    auto& cached_progs = mgx_state->cached_programs_ref.value().get();
    auto it = cached_progs.find(cache_hash);
    if (it != cached_progs.end()) {
      prog = it->second;
      param_shapes = prog.get_parameter_shapes();
      return;  // Early exit - no need to load from disk or compile
    }
  }

  std::filesystem::path model_cache_file;
  // empty cache path means the MXR caching is disabled - always compile
  if (!model_cache_path.empty()) {
    model_cache_file = mgx_state->model_cache_dir / (mxr_filename_prefix + cache_hash + ".mxr");
  }

  // Set input parameter shapes from runtime tensors before compilation
  LOGS_DEFAULT(VERBOSE) << "[Compute] Setting " << map_input_name_index.size()
                        << " input parameter shapes as static in MIGraphX options (excluding constants)";

  for (const auto& it : map_input_name_index) {
    const auto& name = it.first;
    const auto& index = it.second;
    auto input_tensor = ctx.GetInput(index);
    auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
    const auto tensor_shape = tensor_info.GetShape();
    std::vector<std::size_t> ort_lens(tensor_shape.begin(), tensor_shape.end());
    cmp_options.set_input_parameter_shape(name, ort_lens);

    LOGS_DEFAULT(VERBOSE) << "[Compute] Set static shape for input parameter '" << name << "': ["
                          << [&]() {
                              std::ostringstream ss;
                              for (size_t i = 0; i < ort_lens.size(); ++i) {
                                if (i > 0) ss << ", ";
                                ss << ort_lens[i];
                              }
                              return ss.str();
                            }() << "]";
  }

  // Use load_or_compile_model helper - handles cache loading, compilation, and saving
  prog = load_or_compile_model(
      model_cache_file,
      mgx_state->onnx_string,
      cmp_options,
      mgx_state->t,
      mgx_state->fp16_enable,
      mgx_state->bf16_enable,
      mgx_state->int8_enable,
      mgx_state->fp8_enable,
      mgx_state->int8_calibration_cache_available,
      mgx_state->dynamic_range_map,
      mgx_state->exhaustive_tune,
      mgx_state->model_cache_dir,
      &ctx,
      &map_input_name_index);

  // Store the compiled/loaded program in the in-memory cached_programs cache
  if (mgx_state->cached_programs_ref.has_value()) {
    mgx_state->cached_programs_ref.value().get()[cache_hash] = prog;
  }

  // Invalidate ultra-fast path caches (will be repopulated on next run)
  mgx_state->caches_valid = false;
  mgx_state->cached_inputs.clear();
  mgx_state->cached_outputs.clear();
  mgx_state->cached_output_ort_shapes.clear();
  mgx_state->cached_prog_params = std::nullopt;
  mgx_state->cached_prog_output_indices.clear();
  mgx_state->last_input_shapes_raw.clear();
  mgx_state->last_input_shape_hash.clear();

  param_shapes = prog.get_parameter_shapes();
  mgx_state->defer_compilation = false;
}



// Overload: Handle program inputs and outputs binding with pre-cached output shapes
// This avoids calling prog.get_output_shapes() when shapes are already cached
// When needs_slicing is true, allocates temporary GPU buffers for outputs instead of binding directly
static
std::pair<migraphx::program_parameters, std::vector<std::size_t>> handle_program_input_outputs(
    const migraphx::program_parameter_shapes& param_shapes,
    const migraphx::shapes& output_shapes,
    const std::unordered_map<std::string, std::size_t>& map_input_name_index,
    const Ort::KernelContext& ctx,
    bool needs_slicing = false,
    std::vector<void*>* temp_output_buffers = nullptr)
{
  LOGS_DEFAULT(VERBOSE) << "[handle_program_input_outputs] ==== ENTERING ====";
  LOGS_DEFAULT(VERBOSE) << "[handle_program_input_outputs] needs_slicing=" << needs_slicing 
                        << ", temp_output_buffers=" << (temp_output_buffers != nullptr ? "valid" : "nullptr");
  LOGS_DEFAULT(VERBOSE) << "[handle_program_input_outputs] param_shapes.size()=" << param_shapes.size()
                        << ", output_shapes.size()=" << output_shapes.size();
  
  migraphx::program_parameters m;
  std::vector<std::size_t> prog_output_indices;
  prog_output_indices.reserve(output_shapes.size());

  std::size_t input_count = 0;
  std::size_t output_count = 0;
  std::size_t temp_buffer_count = 0;

  if (param_shapes.size() > 0) {
    for (const auto& name : param_shapes.names()) {
      auto it = map_input_name_index.find(name);
      if (it != map_input_name_index.end()) {
        // Input parameter
        input_count++;
        const auto& input_tensor = ctx.GetInput(it->second);
        const auto& mgx_s = param_shapes[name];
        m.add(name, migraphx::argument(mgx_s,
                                       const_cast<void*>(input_tensor.GetTensorRawData())));
      } else {
        // Output parameter
        const auto output_index = compute_output_index(name);
        if (output_index != -1) {
          output_count++;
          const auto& mgx_arg_shape = param_shapes[name];
          
          LOGS_DEFAULT(VERBOSE) << "[handle_program_input_outputs] Processing output '" << name 
                                << "' (index=" << output_index << ")"
                                << ", needs_slicing=" << needs_slicing 
                                << ", temp_output_buffers=" << (temp_output_buffers != nullptr ? "valid" : "nullptr");
          
          if (needs_slicing && temp_output_buffers != nullptr) {
            // When slicing, use pre-allocated temp buffer or allocate new one
            // Don't add to prog_output_indices since these aren't pre-allocated ORT outputs
            std::size_t output_size_bytes = mgx_arg_shape.bytes();
            void* temp_buffer = nullptr;
            
            // OPTIMIZATION: Check if buffer is already pre-allocated
            if (temp_buffer_count < temp_output_buffers->size()) {
              // Use pre-allocated buffer from previous run
              temp_buffer = (*temp_output_buffers)[temp_buffer_count];
              LOGS_DEFAULT(VERBOSE) << "[handle_program_input_outputs] ✓ REUSING pre-allocated TEMP BUFFER for output " 
                                    << output_index << " '" << name << "' (" << output_size_bytes << " bytes)";
            } else {
              // Allocate new buffer (first run or buffer list is empty)
              auto hip_status = hipMalloc(&temp_buffer, output_size_bytes);
              if (hip_status != hipSuccess) {
                ORT_THROW("hipMalloc failed for temporary output buffer");
              }
              temp_output_buffers->push_back(temp_buffer);
              LOGS_DEFAULT(VERBOSE) << "[handle_program_input_outputs] ✓ Allocated NEW TEMP BUFFER for output " 
                                    << output_index << " '" << name << "' (" << output_size_bytes << " bytes)";
            }
            temp_buffer_count++;
            m.add(name, migraphx::argument(mgx_arg_shape, temp_buffer));
            LOGS_DEFAULT(VERBOSE) << "[handle_program_input_outputs] - NOT adding to prog_output_indices";
          } else {
            // Normal path: bind directly to ORT output tensor
            LOGS_DEFAULT(VERBOSE) << "[handle_program_input_outputs] Using NORMAL PATH for output " 
                                  << output_index << " '" << name << "'"
                                  << " - adding to prog_output_indices";
            prog_output_indices.push_back(static_cast<std::size_t>(output_index));
            const auto& lens = output_shapes[output_index].lengths();
            const std::vector<int64_t> ort_output_shape(lens.begin(), lens.end());
            auto output_tensor = ctx.GetOutput(output_index, ort_output_shape.data(), ort_output_shape.size());
            m.add(name, migraphx::argument(mgx_arg_shape, output_tensor.GetTensorMutableRawData()));
          }
        }
      }
    }
  }
  
  LOGS_DEFAULT(VERBOSE) << "[handle_program_input_outputs] ==== SUMMARY ====";
  LOGS_DEFAULT(VERBOSE) << "[handle_program_input_outputs] Processed " << input_count << " inputs, " 
                        << output_count << " outputs";
  LOGS_DEFAULT(VERBOSE) << "[handle_program_input_outputs] Temp buffers allocated: " << temp_buffer_count;
  LOGS_DEFAULT(VERBOSE) << "[handle_program_input_outputs] prog_output_indices.size()=" << prog_output_indices.size();
  LOGS_DEFAULT(VERBOSE) << "[handle_program_input_outputs] ==== EXITING ====";
  
  return {m, prog_output_indices};
}

// Helper: Populate optimized caches for ultra-fast path
// This separates inputs from outputs, pre-computes indices, and pre-allocates output shapes
// When slicing is needed, stores sliced output shapes instead of padded shapes
static void populate_ultra_fast_caches(
    MIGraphXFuncState* mgx_state,
    const migraphx::program_parameter_shapes& param_shapes,
    const migraphx::shapes& output_shapes,
    const std::unordered_map<std::string, std::size_t>& map_input_name_index,
    std::size_t original_batch_size = 0,
    std::size_t padded_batch_size = 0)
{
  bool needs_slicing = (original_batch_size > 0 && padded_batch_size > 0 && 
                        original_batch_size < padded_batch_size);
  
  // Clear existing caches
  mgx_state->cached_inputs.clear();
  mgx_state->cached_outputs.clear();
  mgx_state->cached_output_ort_shapes.clear();

  // Reserve space for outputs
  mgx_state->cached_outputs.reserve(output_shapes.size());
  mgx_state->cached_output_ort_shapes.reserve(output_shapes.size());

  // Separate inputs from outputs
  if (param_shapes.size() > 0) {
    for (const auto& name : param_shapes.names()) {
      auto it = map_input_name_index.find(name);
      if (it != map_input_name_index.end()) {
        // This is an input parameter
        MIGraphXFuncState::CachedInputParam inp;
        inp.name = name;
        inp.ort_index = it->second;
        inp.mgx_shape = param_shapes[name];
        mgx_state->cached_inputs.push_back(std::move(inp));
      } else {
        // This is an output parameter
        const int output_index = compute_output_index(name);
        if (output_index != -1) {
          // When slicing, don't cache outputs (ultra-fast path won't be used)
          if (!needs_slicing) {
            MIGraphXFuncState::CachedOutputParam out;
            out.name = name;
            out.output_index = output_index;
            out.mgx_shape = param_shapes[name];
            mgx_state->cached_outputs.push_back(std::move(out));

            // Pre-allocate ORT-format output shape vector
            const auto& lens = output_shapes[output_index].lengths();
            mgx_state->cached_output_ort_shapes.emplace_back(lens.begin(), lens.end());
          }
        }
      }
    }
  }
}

// Helper: Build input shapes vector in cached_inputs order (MIGraphX parameter order)
// This ensures consistency between how shapes are stored and how they're compared in ultra-fast path
static std::vector<std::int64_t> build_input_shapes_in_cached_order(
    MIGraphXFuncState* mgx_state,
    Ort::KernelContext& ctx,
    std::size_t padded_batch_size = 0)
{
  std::vector<std::int64_t> shapes;
  shapes.reserve(mgx_state->cached_inputs.size() * 4);  // Estimate average 4 dims per input
  
  for (const auto& cached_inp : mgx_state->cached_inputs) {
    auto input_tensor = ctx.GetInput(cached_inp.ort_index);
    auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
    const auto tensor_shape = tensor_info.GetShape();
    
    if (!tensor_shape.empty()) {
      if (padded_batch_size > 0) {
        // Use padded batch size for first dimension
        shapes.push_back(static_cast<std::int64_t>(padded_batch_size));
        shapes.insert(shapes.end(), tensor_shape.begin() + 1, tensor_shape.end());
      } else {
        // Use original shape
        shapes.insert(shapes.end(), tensor_shape.begin(), tensor_shape.end());
      }
    }
  }
  
  return shapes;
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXECUTION PATH FUNCTIONS - Encapsulated paths for cleaner compute_func
// ═══════════════════════════════════════════════════════════════════════════════

// Ultra-fast path: Shapes unchanged from last run - just rebind pointers and execute
// Returns true if executed successfully, false if shapes don't match
static bool execute_ultra_fast_path(
    MIGraphXFuncState* mgx_state,
    const OrtApi* api,
    OrtKernelContext* context,
    Ort::KernelContext& ctx)
{
  if (!mgx_state->caches_valid || mgx_state->last_input_shapes_raw.empty()) {
    return false;
  }
  
  // Ultra-fast path not supported when outputs need dynamic slicing
  if (mgx_state->cached_outputs.empty()) {
    return false;
  }

  // Quick shape comparison
  bool shapes_match = true;
  std::size_t offset = 0;
  const auto& last_shapes = mgx_state->last_input_shapes_raw;
  
  std::size_t original_batch_size = 0;
  std::size_t padded_batch_size = 0;
  bool is_first = true;

  for (const auto& inp : mgx_state->cached_inputs) {
    const auto& shape = ctx.GetInput(inp.ort_index).GetTensorTypeAndShapeInfo().GetShape();
    
    if (offset + shape.size() > last_shapes.size()) {
      shapes_match = false;
      break;
    }
    
    // For dynamic batch, we check if the current batch needs padding
    if (mgx_state->has_dynamic_batch && !mgx_state->compiled_batch_sizes.empty()) {
      // Get batch sizes from first input
      if (is_first) {
        original_batch_size = static_cast<std::size_t>(shape[0]);
        padded_batch_size = static_cast<std::size_t>(last_shapes[offset]);
        is_first = false;
        
        // Check if the batch size matches (original or padded)
        if (shape[0] != last_shapes[offset]) {
          // Batch size changed - check if we can use padding
          std::size_t required_padded = find_nearest_compiled_batch_size(
              original_batch_size, mgx_state->max_dynamic_batch, mgx_state->max_compiled_models);
          
          if (required_padded != padded_batch_size) {
            shapes_match = false;
            break;
          }
        }
      }
      
      // All current inputs should have the same batch size (original_batch_size)
      if (static_cast<std::size_t>(shape[0]) != original_batch_size) {
        shapes_match = false;
        break;
      }
      
      // Cached shape should have padded batch size in dimension 0
      if (last_shapes[offset] != static_cast<std::int64_t>(padded_batch_size)) {
        shapes_match = false;
        break;
      }
      
      // Check non-batch dimensions (current vs cached)
      bool rest_matches = true;
      for (std::size_t i = 1; i < shape.size(); ++i) {
        if (last_shapes[offset + i] != shape[i]) {
          rest_matches = false;
          break;
        }
      }
      if (!rest_matches) {
        shapes_match = false;
        break;
      }
    } else {
      // No dynamic batching - strict comparison
      for (std::size_t i = 0; i < shape.size(); ++i) {
        if (last_shapes[offset + i] != shape[i]) {
          shapes_match = false;
          break;
        }
      }
    }
    
    if (!shapes_match) break;
    offset += shape.size();
  }

  if (!shapes_match || offset != last_shapes.size()) {
    return false;
  }

  // Ultra-fast path doesn't support output slicing because cached_output_ort_shapes
  // contains padded shapes, not sliced shapes. Fall back to fast path which handles
  // slicing properly via temp output buffers.
  if (padded_batch_size > 0 && original_batch_size > 0 && padded_batch_size > original_batch_size) {
    return false;
  }

  // Shapes unchanged (or compatible with padding) - rebind pointers and run directly
  auto& m = mgx_state->cached_prog_params.value();
  auto& prog = mgx_state->prog;
  
  // Allocate and pad inputs if needed for dynamic batching
  bool using_padded_inputs = false;
  if (padded_batch_size > original_batch_size) {
    void* rocm_stream_ptr;
    Ort::ThrowOnError(api->KernelContext_GetGPUComputeStream(context, &rocm_stream_ptr));
    auto rocm_stream = static_cast<hipStream_t>(rocm_stream_ptr);
    using_padded_inputs = allocate_and_pad_inputs(mgx_state, ctx, original_batch_size, 
                                                  padded_batch_size, rocm_stream);
  }

  // Rebind inputs - use padded buffers if available, otherwise use original inputs
  if (using_padded_inputs && mgx_state->padded_input_buffers.size() == mgx_state->cached_inputs.size()) {
    for (size_t i = 0; i < mgx_state->cached_inputs.size(); ++i) {
      const auto& inp = mgx_state->cached_inputs[i];
      const auto& padded_buf = mgx_state->padded_input_buffers[i];
      m.add(inp.name.c_str(), migraphx::argument(padded_buf.mgx_shape, padded_buf.data));
    }
  } else {
    for (const auto& inp : mgx_state->cached_inputs) {
      const auto& input_tensor = ctx.GetInput(inp.ort_index);
      m.add(inp.name.c_str(), migraphx::argument(inp.mgx_shape,
                                         const_cast<void*>(input_tensor.GetTensorRawData())));
    }
  }

  // Rebind outputs - direct iteration, uses pre-allocated shape vectors
  for (std::size_t i = 0; i < mgx_state->cached_outputs.size(); ++i) {
    const auto& out = mgx_state->cached_outputs[i];
    const auto& ort_shape = mgx_state->cached_output_ort_shapes[i];
    auto output_tensor = ctx.GetOutput(out.output_index, ort_shape.data(), ort_shape.size());
    m.add(out.name.c_str(), migraphx::argument(out.mgx_shape,
                                       output_tensor.GetTensorMutableRawData()));
  }

  // Run directly - minimal overhead path
  run_migraphx_program(mgx_state->mgx_mu_ptr, api, context, ctx, prog, m,
                       mgx_state->cached_prog_output_indices,
                       original_batch_size, padded_batch_size);
  
  return true;
}

// Fast path: Found cached program for this shape hash - populate caches and execute
// Returns true if a cached program was found and executed
// Note: all_input_shapes is only consumed (moved) if the function returns true
static bool execute_fast_path(
    MIGraphXFuncState* mgx_state,
    const OrtApi* api,
    OrtKernelContext* context,
    Ort::KernelContext& ctx,
    const std::string& current_hash,
    std::vector<std::int64_t>& all_input_shapes)
{
  LOGS_DEFAULT(VERBOSE) << "[FAST_PATH] Checking for cached program with hash: " << current_hash;
  
  if (mgx_state->defer_compilation || !mgx_state->cached_programs_ref.has_value()) {
    LOGS_DEFAULT(VERBOSE) << "[FAST_PATH] Skipping - defer_compilation=" << mgx_state->defer_compilation
                          << ", has cache=" << mgx_state->cached_programs_ref.has_value();
    return false;
  }

  auto& cached_programs = mgx_state->cached_programs_ref.value().get();
  LOGS_DEFAULT(VERBOSE) << "[FAST_PATH] Memory cache size: " << cached_programs.size();
  auto prog_it = cached_programs.find(current_hash);
  
  // If not found directly, check if we need to use a padded batch size
  std::size_t original_batch_size = 0;
  std::size_t padded_batch_size = 0;
  bool needs_padding = false;
  
  if (prog_it == cached_programs.end() && mgx_state->has_dynamic_batch && 
      !mgx_state->compiled_batch_sizes.empty()) {
    LOGS_DEFAULT(VERBOSE) << "[FAST_PATH][DynamicBatch] Direct hash miss - checking for padded batch";
    // Try to find a padded batch size
    const auto& map_input_name_index = mgx_state->input_name_indexes;
    
    for (const auto& [name, index] : map_input_name_index) {
      auto input_tensor = ctx.GetInput(index);
      auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
      const auto tensor_shape = tensor_info.GetShape();
      if (!tensor_shape.empty()) {
        original_batch_size = static_cast<std::size_t>(tensor_shape[0]);
        padded_batch_size = find_nearest_compiled_batch_size(original_batch_size,
                                                                    mgx_state->max_dynamic_batch,
                                                                    mgx_state->max_compiled_models);
        needs_padding = (padded_batch_size > original_batch_size);
        LOGS_DEFAULT(VERBOSE) << "[FAST_PATH][DynamicBatch] Original batch: " << original_batch_size
                              << ", padded: " << padded_batch_size << ", needs_padding: " << needs_padding;
        break;
      }
    }
    
    if (needs_padding && padded_batch_size > 0) {
      LOGS_DEFAULT(VERBOSE) << "[FAST_PATH][DynamicBatch] Building padded shape key for hash lookup";
      // Build padded shapes in alphabetical order (map order) for hash calculation
      // This matches the order used during compilation in compile_dynamic_batch_models
      std::vector<std::int64_t> padded_shapes_for_hash;
      padded_shapes_for_hash.reserve(all_input_shapes.size());
      
      for (const auto& [name, index] : map_input_name_index) {
        auto input_tensor = ctx.GetInput(index);
        auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
        const auto tensor_shape = tensor_info.GetShape();
        
        if (!tensor_shape.empty()) {
          padded_shapes_for_hash.push_back(static_cast<std::int64_t>(padded_batch_size));
          padded_shapes_for_hash.insert(padded_shapes_for_hash.end(), tensor_shape.begin() + 1, tensor_shape.end());
        }
      }
      
      auto padded_hash = make_hash(padded_shapes_for_hash);
      LOGS_DEFAULT(VERBOSE) << "[FAST_PATH][DynamicBatch] Padded hash (map order): " << padded_hash;
      prog_it = cached_programs.find(padded_hash);
      
      if (prog_it != cached_programs.end()) {
        LOGS_DEFAULT(VERBOSE) << "[FAST_PATH][DynamicBatch] ✓✓✓ CACHE HIT using padded batch size " 
                           << padded_batch_size << " for original batch " << original_batch_size;
        
        // Now rebuild padded_shapes in cached_inputs order for saving to last_input_shapes_raw
        // This ensures ultra-fast path shape comparison works correctly
        if (!mgx_state->cached_inputs.empty()) {
          LOGS_DEFAULT(VERBOSE) << "[FAST_PATH][DynamicBatch] Rebuilding in cached_inputs order for saving";
          std::vector<std::int64_t> padded_shapes_for_cache;
          padded_shapes_for_cache.reserve(mgx_state->cached_inputs.size() * 2);
          
          for (const auto& cached_inp : mgx_state->cached_inputs) {
            auto input_tensor = ctx.GetInput(cached_inp.ort_index);
            auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
            const auto tensor_shape = tensor_info.GetShape();
            
            if (!tensor_shape.empty()) {
              padded_shapes_for_cache.push_back(static_cast<std::int64_t>(padded_batch_size));
              padded_shapes_for_cache.insert(padded_shapes_for_cache.end(), tensor_shape.begin() + 1, tensor_shape.end());
            }
          }
          all_input_shapes = std::move(padded_shapes_for_cache);
        } else {
          // Fallback: use map order (shouldn't happen if caches are populated)
          all_input_shapes = std::move(padded_shapes_for_hash);
        }
      } else {
        LOGS_DEFAULT(VERBOSE) << "[FAST_PATH][DynamicBatch] Cache miss for padded hash too";
      }
    }
  } else if (prog_it != cached_programs.end()) {
    LOGS_DEFAULT(VERBOSE) << "[FAST_PATH] ✓ CACHE HIT for exact hash";
  }
  
  if (prog_it == cached_programs.end()) {
    LOGS_DEFAULT(VERBOSE) << "[FAST_PATH] No cached program found";
    return false;
  }

  // Determine which hash was used to find the program
  // This is needed to detect program changes and invalidate caches
  std::string effective_program_hash = current_hash;
  if (needs_padding && padded_batch_size > 0) {
    // If we used padded hash, compute it for tracking
    std::vector<std::int64_t> padded_shapes_for_hash_tracking;
    for (const auto& [name, index] : mgx_state->input_name_indexes) {
      auto input_tensor = ctx.GetInput(index);
      auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
      const auto tensor_shape = tensor_info.GetShape();
      if (!tensor_shape.empty()) {
        padded_shapes_for_hash_tracking.push_back(static_cast<std::int64_t>(padded_batch_size));
        padded_shapes_for_hash_tracking.insert(padded_shapes_for_hash_tracking.end(), 
                                               tensor_shape.begin() + 1, tensor_shape.end());
      }
    }
    effective_program_hash = make_hash(padded_shapes_for_hash_tracking);
  }

  // Found cached program - use it and populate caches
  LOGS_DEFAULT(VERBOSE) << "[FAST_PATH] Using cached program (hash: " << effective_program_hash << ")";
  auto& prog = mgx_state->prog;
  prog = prog_it->second;

  const auto& map_input_name_index = mgx_state->input_name_indexes;

  // ═══════════════════════════════════════════════════════════════════════════
  // OPTIMIZATION 1: Cache MIGraphX API results (avoid redundant API calls)
  // Check if program changed - if so, invalidate caches
  // ═══════════════════════════════════════════════════════════════════════════
  bool program_changed = (mgx_state->cached_program_hash != effective_program_hash);
  
  if (program_changed) {
    LOGS_DEFAULT(VERBOSE) << "[FAST_PATH] Program changed (old: " << mgx_state->cached_program_hash 
                          << ", new: " << effective_program_hash << ") - clearing caches";
    clear_cached_mgx_shapes(mgx_state);
    free_temp_output_buffers(mgx_state);
    mgx_state->cached_program_hash = effective_program_hash;
  }
  
  if (!mgx_state->cached_mgx_param_shapes.has_value()) {
    LOGS_DEFAULT(VERBOSE) << "[FAST_PATH] Caching MIGraphX shapes (first time or after program change)";
    mgx_state->cached_mgx_param_shapes = prog.get_parameter_shapes();
    mgx_state->cached_mgx_output_shapes = prog.get_output_shapes();
  } else {
    LOGS_DEFAULT(VERBOSE) << "[FAST_PATH] ✓ Using cached MIGraphX shapes";
  }
  const auto& param_shapes = mgx_state->cached_mgx_param_shapes.value();
  const auto& output_shapes = mgx_state->cached_mgx_output_shapes.value();

  bool needs_slicing = (original_batch_size > 0 && padded_batch_size > 0 && 
                        original_batch_size < padded_batch_size);
  
  // ═══════════════════════════════════════════════════════════════════════════
  // OPTIMIZATION 2: Skip populate_ultra_fast_caches when already populated
  // ═══════════════════════════════════════════════════════════════════════════
  if (!mgx_state->ultra_fast_caches_populated) {
    LOGS_DEFAULT(VERBOSE) << "[FAST_PATH] Populating ultra-fast caches (first time)";
    populate_ultra_fast_caches(mgx_state, param_shapes, output_shapes, map_input_name_index,
                              original_batch_size, padded_batch_size);
    mgx_state->ultra_fast_caches_populated = true;
  } else {
    LOGS_DEFAULT(VERBOSE) << "[FAST_PATH] ✓ Ultra-fast caches already populated";
  }

  // Allocate and pad inputs if needed for dynamic batching
  bool using_padded_inputs = false;
  if (padded_batch_size > original_batch_size) {
    void* rocm_stream_ptr;
    Ort::ThrowOnError(api->KernelContext_GetGPUComputeStream(context, &rocm_stream_ptr));
    auto rocm_stream = static_cast<hipStream_t>(rocm_stream_ptr);
    using_padded_inputs = allocate_and_pad_inputs(mgx_state, ctx, original_batch_size, 
                                                  padded_batch_size, rocm_stream);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // OPTIMIZATION 3: Reuse temp output buffers when slicing
  // ═══════════════════════════════════════════════════════════════════════════
  std::vector<void*> temp_output_buffer_ptrs;
  if (needs_slicing) {
    temp_output_buffer_ptrs = get_or_allocate_temp_output_buffers(
        mgx_state, param_shapes, output_shapes, map_input_name_index, padded_batch_size);
  }

  // Bind inputs/outputs (use temp buffers for outputs when slicing)
  LOGS_DEFAULT(VERBOSE) << "[FAST_PATH] Calling handle_program_input_outputs with needs_slicing=" << needs_slicing;
  auto [m, prog_output_indices] = handle_program_input_outputs(
      param_shapes, output_shapes, map_input_name_index, ctx, needs_slicing, 
      needs_slicing ? &temp_output_buffer_ptrs : nullptr);

  LOGS_DEFAULT(VERBOSE) << "[FAST_PATH] handle_program_input_outputs returned:";
  LOGS_DEFAULT(VERBOSE) << "[FAST_PATH]   prog_output_indices.size()=" << prog_output_indices.size();
  LOGS_DEFAULT(VERBOSE) << "[FAST_PATH]   temp_output_buffer_ptrs.size()=" << temp_output_buffer_ptrs.size();
  if (needs_slicing && !prog_output_indices.empty()) {
    LOGS_DEFAULT(VERBOSE) << "[FAST_PATH]   ⚠️  WARNING: prog_output_indices is NOT empty with slicing enabled!";
  }

  mgx_state->cached_prog_params = std::move(m);
  mgx_state->cached_prog_output_indices = std::move(prog_output_indices);
  
  // IMPORTANT: Build last_input_shapes_raw in cached_inputs order (MIGraphX parameter order)
  // This ensures ultra-fast path shape comparison uses consistent ordering
  mgx_state->last_input_shapes_raw = build_input_shapes_in_cached_order(
      mgx_state, ctx, using_padded_inputs ? padded_batch_size : 0);
  LOGS_DEFAULT(VERBOSE) << "[FAST_PATH] Built last_input_shapes_raw in cached_inputs order, size=" 
                        << mgx_state->last_input_shapes_raw.size();
  
  mgx_state->last_input_shape_hash = current_hash;
  mgx_state->caches_valid = true;

  // Rebind padded inputs to program parameters
  if (using_padded_inputs && mgx_state->padded_input_buffers.size() == mgx_state->cached_inputs.size()) {
    LOGS_DEFAULT(VERBOSE) << "[FAST_PATH] Rebinding padded input buffers";
    auto& prog_params = mgx_state->cached_prog_params.value();
    for (size_t i = 0; i < mgx_state->cached_inputs.size(); ++i) {
      const auto& inp = mgx_state->cached_inputs[i];
      const auto& padded_buf = mgx_state->padded_input_buffers[i];
      prog_params.add(inp.name.c_str(), migraphx::argument(padded_buf.mgx_shape, padded_buf.data));
    }
  }

  LOGS_DEFAULT(VERBOSE) << "[FAST_PATH] Running program (original_batch=" << original_batch_size 
                     << ", padded=" << padded_batch_size << ")";
  run_migraphx_program(mgx_state->mgx_mu_ptr, api, context, ctx, prog,
                       mgx_state->cached_prog_params.value(),
                       mgx_state->cached_prog_output_indices,
                       original_batch_size, padded_batch_size);
  
  // NOTE: Temp output buffers are kept for reuse - they will be freed when batch size changes
  // NOTE: Padded input buffers are also kept for reuse
  
  LOGS_DEFAULT(VERBOSE) << "[FAST_PATH] Complete";
  return true;
}

// Result structure for handle_input_shape function
struct InputShapeResult {
  bool input_shape_match;
  migraphx::program_parameter_shapes param_shapes;
  std::vector<std::int64_t> input_shapes;
};

// Helper: Handle input shape processing for both dynamic and static cases
// This function processes runtime input shapes and determines if recompilation is needed
// Compares all input dimensions of the compiled program against runtime input dimensions
static InputShapeResult handle_input_shape(
    bool defer_compilation,
    const std::unordered_map<std::string, std::size_t>& map_input_name_index,
    Ort::KernelContext& ctx,
    migraphx::onnx_options& cmp_options,
    const migraphx::program& prog)
{
  bool input_shape_match = true;
  migraphx::program_parameter_shapes param_shapes;
  std::vector<std::int64_t> input_shapes;

  if (defer_compilation) {
    LOGS_DEFAULT(VERBOSE) << "[Compute] No static input shapes available, setting from runtime inputs";
    // NOTE: map_input_name_index only contains actual model inputs, not constants/initializers
    // Constants and initializers are embedded in the graph and MIGraphX infers their shapes
    LOGS_DEFAULT(VERBOSE) << "[Compute] Setting shapes for " << map_input_name_index.size()
                          << " model input parameters (excluding constants)";

    for (const auto& it : map_input_name_index) {
      const auto& name = it.first;
      const auto& index = it.second;
      auto input_tensor = ctx.GetInput(index);
      auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
      const auto tensor_shape = tensor_info.GetShape();
      std::vector<std::size_t> ort_lens(tensor_shape.begin(), tensor_shape.end());

      // Override default batch size with incoming batch size and treat as static
      cmp_options.set_input_parameter_shape(name, ort_lens);
      input_shape_match = false;

      // Include all inputs in cache key (map_input_name_index already filtered to model inputs only)
      input_shapes.insert(input_shapes.end(), tensor_shape.begin(), tensor_shape.end());
    }
    LOGS_DEFAULT(VERBOSE) << "[Compute] All runtime input shapes set as static parameters in MIGraphX options";
    LOGS_DEFAULT(VERBOSE) << "[Compute] MIGraphX will infer shapes for constants and intermediate tensors";
  } else {
    LOGS_DEFAULT(VERBOSE) << "[Compute] Checking if compiled program shapes match runtime inputs";
    param_shapes = prog.get_parameter_shapes();

    // Check if all input shapes match the compiled program's shapes
    if (param_shapes.size() > 0) {
      for (auto&& name : param_shapes.names()) {
        if (map_input_name_index.count(name) > 0) {
          auto input_tensor = ctx.GetInput(map_input_name_index.at(name));
          auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
          const auto tensor_shape = tensor_info.GetShape();
          std::vector<std::size_t> ort_lens(tensor_shape.begin(), tensor_shape.end());

          auto mgx_s = param_shapes[name];
          auto mgx_lens = mgx_s.lengths();
          auto mgx_strides = mgx_s.strides();

          // Handle scalar tensors (rank-0 tensors)
          if (mgx_lens.size() == 1 && mgx_lens[0] == 1 &&
              mgx_strides.size() == 1 && mgx_strides[0] == 0) {
            mgx_lens.clear();
          }

          // Check if shapes match
          if (mgx_lens != ort_lens) {
            LOGS_DEFAULT(VERBOSE) << "[Compute] Shape mismatch for input '" << name << "': "
                                  << "compiled shape vs runtime shape differ";
            cmp_options.set_input_parameter_shape(name, ort_lens);
            input_shape_match = false;
          }

          // Include all inputs in cache key (map_input_name_index already filtered to model inputs only)
          input_shapes.insert(input_shapes.end(), tensor_shape.begin(), tensor_shape.end());
        }
      }
    }
  }

  return {input_shape_match, param_shapes, input_shapes};
}

// Helper: Compile models for all configured batch sizes and cache them
static void compile_dynamic_batch_models(
    MIGraphXFuncState* mgx_state,
    const std::filesystem::path& model_cache_path,
    const std::filesystem::path& model_path,
    const std::string& mxr_filename_prefix,
    const Ort::KernelContext& ctx) {
  
  LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE] ==== ENTERING compile_dynamic_batch_models ====";
  LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE] has_dynamic_batch = " << mgx_state->has_dynamic_batch;
  LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE] compiled_batch_sizes.size() = "
                     << mgx_state->compiled_batch_sizes.size();
  LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE] max_dynamic_batch = " << mgx_state->max_dynamic_batch;
  
  if (!mgx_state->has_dynamic_batch || mgx_state->compiled_batch_sizes.empty()) {
    LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE] Skipping - dynamic batch disabled or no batch sizes";
    return;
  }
  
  LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE] Compiling models for " 
                     << mgx_state->compiled_batch_sizes.size() << " batch sizes";
  LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE] Batch sizes: ";
  for (const auto& bs : mgx_state->compiled_batch_sizes) {
    LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE]   - " << bs;
  }
  
  // Get input names and base shapes (without batch dimension)
  const auto& map_input_name_index = mgx_state->input_name_indexes;
  std::vector<std::string> input_names;
  std::vector<std::vector<std::int64_t>> all_input_base_shapes;
  
  LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE] Processing " << map_input_name_index.size() << " input parameters";
  for (const auto& [name, index] : map_input_name_index) {
    input_names.push_back(name);
    auto input_tensor = ctx.GetInput(index);
    auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
    const auto tensor_shape = tensor_info.GetShape();
    
    LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE] Input '" << name << "' (index " << index 
                       << ") runtime shape: [" << [&]() {
                         std::ostringstream ss;
                         for (size_t i = 0; i < tensor_shape.size(); ++i) {
                           if (i > 0) ss << ", ";
                           ss << tensor_shape[i];
                         }
                         return ss.str();
                       }() << "]";
    
    // Store shape without batch dimension
    std::vector<std::int64_t> base_shape;
    if (tensor_shape.size() > 1) {
      base_shape.assign(tensor_shape.begin() + 1, tensor_shape.end());
    }
    all_input_base_shapes.push_back(base_shape);
    
    LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE]   Base shape (no batch): [" << [&]() {
                         std::ostringstream ss;
                         for (size_t i = 0; i < base_shape.size(); ++i) {
                           if (i > 0) ss << ", ";
                           ss << base_shape[i];
                         }
                         return ss.str();
                       }() << "]";
  }
  
  // Compile a model for each configured batch size
  for (const auto& batch_size : mgx_state->compiled_batch_sizes) {
    LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE] ---- Processing batch size: " << batch_size << " ----";
    
    // Build cache key for this batch size
    std::vector<std::int64_t> batch_shape_key;
    for (size_t i = 0; i < input_names.size(); ++i) {
      batch_shape_key.push_back(batch_size);
      batch_shape_key.insert(batch_shape_key.end(), 
                            all_input_base_shapes[i].begin(), 
                            all_input_base_shapes[i].end());
    }
    auto cache_hash = make_hash(batch_shape_key);
    
    LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE] Shape key for batch " << batch_size << ": [" << [&]() {
                         std::ostringstream ss;
                         for (size_t i = 0; i < batch_shape_key.size(); ++i) {
                           if (i > 0) ss << ", ";
                           ss << batch_shape_key[i];
                         }
                         return ss.str();
                       }() << "]";
    LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE] Cache hash: " << cache_hash;
    
    // Check if already cached
    if (mgx_state->cached_programs_ref.has_value()) {
      auto& cached_progs = mgx_state->cached_programs_ref.value().get();
      LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE] Checking in-memory cache (size: " 
                         << cached_progs.size() << ")";
      if (cached_progs.find(cache_hash) != cached_progs.end()) {
        LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE] ✓ Batch size " << batch_size 
                          << " already in memory cache, skipping";
        continue;
      }
      LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE] Cache miss - need to compile/load";
    }
    
    // Build cache file path
    std::filesystem::path batch_cache_file;
    if (!model_cache_path.empty()) {
      batch_cache_file = model_cache_path / (mxr_filename_prefix + cache_hash + ".mxr");
      LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE] Disk cache file: " << batch_cache_file.string();
    } else {
      LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE] No disk cache path configured";
    }
    
    // Compile or load the model for this batch size
    LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE] Calling load_or_compile_model for batch " << batch_size;
    migraphx::program batch_prog = load_or_compile_model(
        batch_cache_file,
        mgx_state->onnx_string,
        mgx_state->options,
        mgx_state->t,
        mgx_state->fp16_enable,
        mgx_state->bf16_enable,
        mgx_state->int8_enable,
        mgx_state->fp8_enable,
        mgx_state->int8_calibration_cache_available,
        mgx_state->dynamic_range_map,
        mgx_state->exhaustive_tune,
        model_path,
        nullptr,  // ctx not needed for compilation
        nullptr,  // map_input_name_index not needed
        input_names,
        all_input_base_shapes,
        batch_size);
    
    // Store in cache
    if (mgx_state->cached_programs_ref.has_value()) {
      mgx_state->cached_programs_ref.value().get()[cache_hash] = batch_prog;
      LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE] ✓ Stored program for batch size " << batch_size 
                         << " in memory cache with hash " << cache_hash;
      LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE] Memory cache now contains " 
                         << mgx_state->cached_programs_ref.value().get().size() << " programs";
    }
  }
  
  LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE] ==== All batch models compiled and cached ====";
  LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE] Setting max_dynamic_batch to 0 to disable future compilation";
  
  // Disable dynamic batch compilation for subsequent runs (set max_dynamic_batch to 0)
  mgx_state->max_dynamic_batch = 0;
  
  // Also disable defer_compilation since we've now compiled
  mgx_state->defer_compilation = false;
  LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE] Set defer_compilation = false";
  
  LOGS_DEFAULT(VERBOSE) << "[DynamicBatch][COMPILE] ==== EXITING compile_dynamic_batch_models ====";
}

// Standard path: Shape checking, potential recompilation, and execution
static void execute_standard_path(
    MIGraphXFuncState* mgx_state,
    const OrtApi* api,
    OrtKernelContext* context,
    Ort::KernelContext& ctx,
    const std::string& current_hash,
    std::vector<std::int64_t>&& all_input_shapes,
    const std::filesystem::path& model_cache_path,
    const std::filesystem::path& model_path,
    const std::string& mxr_filename_prefix)
{
  LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH] ==== ENTERING execute_standard_path ====";
  LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH] current_hash = " << current_hash;
  LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH] has_dynamic_batch = " << mgx_state->has_dynamic_batch;
  LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH] max_dynamic_batch = " << mgx_state->max_dynamic_batch;
  
  auto& prog = mgx_state->prog;
  auto& cmp_options = mgx_state->options;
  const auto& map_input_name_index = mgx_state->input_name_indexes;

  // Check if this is the first run with dynamic batch enabled
  // NOTE: max_dynamic_batch > 0 means compilation was deferred to runtime (not precompiled)
  // If precompilation happened during Compile(), max_dynamic_batch will be > 0 but defer_compilation = false
  // In that case, the programs are already in cache and we can skip runtime compilation
  if (mgx_state->has_dynamic_batch && mgx_state->max_dynamic_batch > 0 && mgx_state->defer_compilation) {
    // Runtime compilation path - used when precompilation was not possible (e.g., non-pure dynamic batch)
    LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] *** RUNTIME COMPILATION REQUIRED ***";
    LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] max_dynamic_batch=" 
                       << mgx_state->max_dynamic_batch;
    LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] Initiating runtime compilation of batch models";
    
    // Compile all batch models at runtime
    compile_dynamic_batch_models(mgx_state, model_cache_path, model_path, mxr_filename_prefix, ctx);
    
    LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] Runtime compilation complete, max_dynamic_batch now = " 
                       << mgx_state->max_dynamic_batch;
    LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] Proceeding with execution using closest compiled batch";
  } else if (mgx_state->has_dynamic_batch) {
    LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] Dynamic batch enabled, models already precompiled";
  }

  // Extract current batch size from first input
  std::size_t original_batch_size = 0;
  std::size_t padded_batch_size = 0;
  bool needs_padding = false;
  
  if (mgx_state->has_dynamic_batch && !mgx_state->compiled_batch_sizes.empty()) {
    LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] Checking for batch padding requirements";
    // Get the batch size from the first input
    for (const auto& [name, index] : map_input_name_index) {
      auto input_tensor = ctx.GetInput(index);
      auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
      const auto tensor_shape = tensor_info.GetShape();
      if (!tensor_shape.empty()) {
        original_batch_size = static_cast<std::size_t>(tensor_shape[0]);
        padded_batch_size = find_nearest_compiled_batch_size(original_batch_size,
                                                                    mgx_state->max_dynamic_batch,
                                                                    mgx_state->max_compiled_models);
        needs_padding = (padded_batch_size > original_batch_size);
        
        LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] Original batch size: " << original_batch_size;
        LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] Padded batch size: " << padded_batch_size;
        LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] Needs padding: " << (needs_padding ? "YES" : "NO");
        break;  // Only need batch size from first input
      }
    }
    
    // We need to fetch from cache whether padding is needed or not
    // Even when batch size matches exactly, we still use cached compiled programs
    if (padded_batch_size > 0) {
      if (needs_padding) {
        LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] Building padded shape key for cache lookup (needs padding)";
      } else {
        LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] Building shape key for cache lookup (exact match, no padding)";
      }
      // Update the shape hash and all_input_shapes to use the padded batch size
      std::vector<std::int64_t> padded_shapes;
      padded_shapes.reserve(all_input_shapes.size());
      
      for (const auto& [name, index] : map_input_name_index) {
        auto input_tensor = ctx.GetInput(index);
        auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
        const auto tensor_shape = tensor_info.GetShape();
        
        // Replace batch dimension with padded size
        if (!tensor_shape.empty()) {
          padded_shapes.push_back(static_cast<std::int64_t>(padded_batch_size));
          padded_shapes.insert(padded_shapes.end(), tensor_shape.begin() + 1, tensor_shape.end());
        }
      }
      
      // Look up the cached program for the padded batch size
      auto padded_hash = make_hash(padded_shapes);
      LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] Padded shape hash: " << padded_hash;
      
      if (mgx_state->cached_programs_ref.has_value()) {
        auto& cached_progs = mgx_state->cached_programs_ref.value().get();
        LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] Searching in memory cache (size: " 
                          << cached_progs.size() << ")";
        auto prog_it = cached_progs.find(padded_hash);
        if (prog_it != cached_progs.end()) {
          LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] ✓✓✓ CACHE HIT for batch size " 
                             << padded_batch_size << (needs_padding ? " (will pad)" : " (exact match, no padding)");
          prog = prog_it->second;
          
          // Get shapes for the cached program
          auto param_shapes = prog.get_parameter_shapes();
          auto output_shapes = prog.get_output_shapes();
          
          LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] Cached program param_shapes.size()=" 
                               << param_shapes.size() << ", output_shapes.size()=" << output_shapes.size();
          
          if (needs_padding) {
            // ============ PADDING PATH: Batch size needs to be padded ============
            LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] Taking PADDING path";
            
            // Populate caches (with slicing info so ultra-fast path is disabled)
            populate_ultra_fast_caches(mgx_state, param_shapes, output_shapes, map_input_name_index,
                                      original_batch_size, padded_batch_size);
            
            // Rebuild padded_shapes in cached_inputs order (MIGraphX parameter order)
            // This ensures consistency with ultra-fast path shape comparison
            padded_shapes.clear();
            padded_shapes.reserve(mgx_state->cached_inputs.size() * 2);
            for (const auto& cached_inp : mgx_state->cached_inputs) {
              auto input_tensor = ctx.GetInput(cached_inp.ort_index);
              auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
              const auto tensor_shape = tensor_info.GetShape();
              
              if (!tensor_shape.empty()) {
                padded_shapes.push_back(static_cast<std::int64_t>(padded_batch_size));
                padded_shapes.insert(padded_shapes.end(), tensor_shape.begin() + 1, tensor_shape.end());
              }
            }
            LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] Rebuilt padded_shapes in cached_inputs order";
            
            // Allocate and pad inputs for dynamic batching
            void* rocm_stream_ptr;
            Ort::ThrowOnError(api->KernelContext_GetGPUComputeStream(context, &rocm_stream_ptr));
            auto rocm_stream = static_cast<hipStream_t>(rocm_stream_ptr);
            bool using_padded_inputs = allocate_and_pad_inputs(mgx_state, ctx, original_batch_size, 
                                                               padded_batch_size, rocm_stream);
            
            // Bind inputs and outputs with temporary output buffers (for slicing)
            LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] Calling handle_program_input_outputs with needs_slicing=true";
            LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] param_shapes.size()=" << param_shapes.size()
                                 << ", output_shapes.size()=" << output_shapes.size();
            std::vector<void*> temp_output_buffers;
            auto [m, prog_output_indices] = handle_program_input_outputs(
                param_shapes, output_shapes, map_input_name_index, ctx, true, &temp_output_buffers);
            
            LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] handle_program_input_outputs returned:";
            LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch]   prog_output_indices.size()=" << prog_output_indices.size();
            LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch]   temp_output_buffers.size()=" << temp_output_buffers.size();
            if (!prog_output_indices.empty()) {
              LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch]   ⚠️  WARNING: prog_output_indices is NOT empty!";
              LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch]   ⚠️  This means outputs were pre-allocated instead of using temp buffers!";
            }
            
            mgx_state->cached_prog_params = m;
            mgx_state->cached_prog_output_indices = prog_output_indices;
            mgx_state->last_input_shapes_raw = std::move(padded_shapes);
            mgx_state->last_input_shape_hash = padded_hash;
            mgx_state->caches_valid = true;
            
            // Rebind padded inputs to program parameters
            if (using_padded_inputs && mgx_state->padded_input_buffers.size() == mgx_state->cached_inputs.size()) {
              LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] Rebinding padded input buffers";
              for (size_t i = 0; i < mgx_state->cached_inputs.size(); ++i) {
                const auto& inp = mgx_state->cached_inputs[i];
                const auto& padded_buf = mgx_state->padded_input_buffers[i];
                m.add(inp.name.c_str(), migraphx::argument(padded_buf.mgx_shape, padded_buf.data));
              }
            }
            
            LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] Running with output slicing enabled";
            // Run with slicing enabled
            run_migraphx_program(mgx_state->mgx_mu_ptr, api, context, ctx, prog, m, 
                                prog_output_indices, original_batch_size, padded_batch_size);
            
            // Free temporary output buffers
            for (void* buf : temp_output_buffers) {
              if (buf != nullptr) {
                (void)hipFree(buf);
              }
            }
            
            // NOTE: Padded buffers are kept for reuse - they will be freed when batch size changes
            // or when the state is destroyed
            
            LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] ==== EXITING execute_standard_path (padded path) ====";
            return;
          } else {
            // ============ EXACT MATCH PATH: Batch size matches exactly, no padding needed ============
            LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] Taking EXACT MATCH path (no padding/slicing)";
            
            // Populate caches for ultra-fast path (no slicing needed)
            populate_ultra_fast_caches(mgx_state, param_shapes, output_shapes, map_input_name_index);
            
            // Bind inputs and allocate outputs (no slicing)
            auto [m, prog_output_indices] = handle_program_input_outputs(
                param_shapes, output_shapes, map_input_name_index, ctx);
            
            LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] handle_program_input_outputs returned:";
            LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch]   prog_output_indices.size()=" << prog_output_indices.size();
            
            // Complete cache population
            mgx_state->cached_prog_params = m;
            mgx_state->cached_prog_output_indices = prog_output_indices;
            
            // IMPORTANT: Build last_input_shapes_raw in cached_inputs order (MIGraphX parameter order)
            // This ensures ultra-fast path shape comparison uses consistent ordering
            mgx_state->last_input_shapes_raw = build_input_shapes_in_cached_order(mgx_state, ctx, 0);
            LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] Built last_input_shapes_raw in cached_inputs order, size=" 
                                  << mgx_state->last_input_shapes_raw.size();
            
            mgx_state->last_input_shape_hash = current_hash;
            mgx_state->caches_valid = true;
            
            LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] Running program (exact batch match, no padding/slicing)";
            run_migraphx_program(mgx_state->mgx_mu_ptr, api, context, ctx, prog, m, prog_output_indices, 
                                0, 0);  // Pass 0,0 for batch sizes to indicate no slicing needed
            
            LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] ==== EXITING execute_standard_path (exact match path) ====";
            return;
          }
        } else {
          LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] ✗✗✗ CACHE MISS for hash " 
                                << padded_hash;
          LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH][DynamicBatch] This shouldn't happen - expected batch "
                                << padded_batch_size << " to be pre-compiled";
        }
      }
    }
  }

  LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH] Proceeding with normal shape checking";
  auto [input_shape_match, param_shapes, input_shapes] = handle_input_shape(
      mgx_state->defer_compilation, map_input_name_index, ctx, cmp_options, prog);

  LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH] input_shape_match = " << input_shape_match;
  
  if (!input_shape_match) {
    LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH] Shape mismatch detected - recompiling";
    // Invalidate caches before recompilation
    mgx_state->caches_valid = false;

    handle_input_shape_mismatch(
        mgx_state,
        model_cache_path,
        model_path,
        mxr_filename_prefix,
        ctx,
        param_shapes,
        input_shapes);

    // Re-fetch param_shapes after recompilation
    param_shapes = prog.get_parameter_shapes();
  }

  // Fetch output shapes once
  auto output_shapes = prog.get_output_shapes();

  // Populate optimized caches for ultra-fast path
  populate_ultra_fast_caches(mgx_state, param_shapes, output_shapes, map_input_name_index);

  // Bind inputs and allocate outputs
  auto [m, prog_output_indices] = handle_program_input_outputs(
      param_shapes, output_shapes, map_input_name_index, ctx);

  // Complete cache population
  mgx_state->cached_prog_params = m;
  mgx_state->cached_prog_output_indices = prog_output_indices;
  
  // IMPORTANT: Build last_input_shapes_raw in cached_inputs order (MIGraphX parameter order)
  // This ensures ultra-fast path shape comparison uses consistent ordering
  mgx_state->last_input_shapes_raw = build_input_shapes_in_cached_order(mgx_state, ctx, 0);
  LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH] Built last_input_shapes_raw in cached_inputs order, size=" 
                        << mgx_state->last_input_shapes_raw.size();
  
  mgx_state->last_input_shape_hash = current_hash;
  mgx_state->caches_valid = true;

  LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH] Running program (no padding/slicing)";
  run_migraphx_program(mgx_state->mgx_mu_ptr, api, context, ctx, prog, m, prog_output_indices, 
                      original_batch_size, padded_batch_size);
  LOGS_DEFAULT(VERBOSE) << "[STANDARD_PATH] ==== EXITING execute_standard_path (normal path) ====";
}

// Build MIGraphX ONNX options with default shapes for symbolic dimensions
// Sets default batch size of 1 for symbolic batch dimensions, 1 for other symbolic dimensions
static migraphx::onnx_options get_program_parameter_options(
    const std::vector<std::string>& input_names,
    const std::vector<const NodeArg*>& input_tensor,
    const InitializedTensorSet& initializers) {
  migraphx::onnx_options options;
  constexpr std::size_t default_batch_size = 1;

  for (std::size_t i = 0; i < input_names.size(); ++i) {
    // Skip if this is an initializer/constant - let MIGraphX infer its shape
    if (initializers.count(input_names[i]) > 0) {
      LOGS_DEFAULT(VERBOSE) << "[Compile] Skipping '" << input_names[i] << "' (initializer/constant)";
      continue;
    }

    if (i < input_tensor.size()) {
      auto tensor_shape = input_tensor[i]->Shape();
      if (tensor_shape != nullptr && tensor_shape->dim_size() > 0) {
        std::vector<std::size_t> default_shape;
        bool has_symbolic = false;

        for (int j = 0; j < tensor_shape->dim_size(); ++j) {
          const auto& dim = tensor_shape->dim(j);
          if (dim.has_dim_value()) {
            default_shape.push_back(static_cast<std::size_t>(dim.dim_value()));
          } else if (dim.has_dim_param() || !dim.has_dim_value()) {
            // Symbolic or unknown dimension - use default batch size for dim 0, 1 for others
            has_symbolic = true;
            default_shape.push_back(j == 0 ? default_batch_size : 1);
            LOGS_DEFAULT(VERBOSE) << "[Compile] Input parameter '" << input_names[i]
                                  << "' dimension " << j << " is symbolic, using default";
          }
        }

        if (has_symbolic && !default_shape.empty()) {
          options.set_input_parameter_shape(input_names[i], default_shape);
          LOGS_DEFAULT(VERBOSE) << "[Compile] Set default shape for input parameter '" << input_names[i] << "'";
        }
      }
    }
  }
  LOGS_DEFAULT(VERBOSE) << "[Compile] Constants and initializers will have shapes inferred by MIGraphX";

  return options;
}

// Build a map from input parameter name to index
// If model_input_names is provided, only includes inputs that are in that set (excludes weights/constants)
template <typename Container>
static std::unordered_map<std::string, std::size_t> get_input_name_map(
    const Container& input_defs,
    const std::set<std::string>* model_input_names = nullptr) {
  std::unordered_map<std::string, std::size_t> input_name_index;
  input_name_index.reserve(input_defs.size());
  std::size_t i = 0;
  for (const auto& def : input_defs) {
    const auto& name = def->Name();
    // Only include if it's a model input parameter (skip weights/constants)
    if (model_input_names == nullptr || model_input_names->count(name) > 0) {
      input_name_index[name] = i;
    }
    ++i;  // Always increment index to maintain correct ORT input indices
  }
  return input_name_index;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PRECOMPILATION HELPER FUNCTIONS - Move compilation from compute_func to Compile()
// ═══════════════════════════════════════════════════════════════════════════════

// Check if model has only dynamic batch dimension (all other dimensions are static)
// Returns true if ONLY the batch dimension (dim 0) is symbolic/dynamic for all inputs
static inline bool has_only_dynamic_batch_dimension(
    const std::vector<std::string>& input_names,
    const std::vector<const NodeArg*>& input_tensor,
    const InitializedTensorSet& initializers)
{
  // Build a map from input name to NodeArg* for correct name-based lookup
  // This is necessary because input_tensor (from main_graph.GetInputs()) may have
  // different ordering than input_names (from graph_body_viewer)
  std::unordered_map<std::string, const NodeArg*> name_to_nodearg;
  for (const auto* nodearg : input_tensor) {
    if (nodearg != nullptr) {
      name_to_nodearg[nodearg->Name()] = nodearg;
    }
  }
  
  for (const auto& name : input_names) {
    // Skip initializers/constants
    if (initializers.count(name) > 0) {
      continue;
    }
    
    // Find the NodeArg by NAME (not position!)
    auto it = name_to_nodearg.find(name);
    if (it != name_to_nodearg.end()) {
      auto tensor_shape = it->second->Shape();
      if (tensor_shape != nullptr && tensor_shape->dim_size() > 0) {
        for (int j = 0; j < tensor_shape->dim_size(); ++j) {
          const auto& dim = tensor_shape->dim(j);
          bool is_symbolic = !dim.has_dim_value();
          
          if (j == 0) {
            // Batch dimension - should be symbolic for dynamic batch
            // It's OK if it's static too (we'll just precompile for that shape)
            continue;
          } else {
            // Non-batch dimension - should be static
            if (is_symbolic) {
              LOGS_DEFAULT(VERBOSE) << "[has_only_dynamic_batch_dimension] Input '" << name
                                    << "' has symbolic non-batch dimension " << j << " - NOT a pure dynamic batch model";
              return false;
            }
          }
        }
      }
    }
  }
  return true;
}

// Extract base shapes (non-batch dimensions) from graph definition
// Returns a tuple of:
//   - bool: true if extraction was successful (all non-batch dims are concrete), false if symbolic dims found
//   - vector of input names
//   - vector of corresponding base shapes (non-batch dimensions)
// IMPORTANT: Does NOT default symbolic dimensions to any value - returns failure instead
static inline std::tuple<bool, std::vector<std::string>, std::vector<std::vector<std::int64_t>>>
extract_base_shapes_from_graph(
    const std::vector<std::string>& input_names,
    const std::vector<const NodeArg*>& input_tensor,
    const InitializedTensorSet& initializers,
    const std::unordered_map<std::string, std::size_t>& input_name_index)
{
  std::vector<std::string> ordered_names;
  std::vector<std::vector<std::int64_t>> base_shapes;
  bool all_concrete = true;
  
  LOGS_DEFAULT(VERBOSE) << "[extract_base_shapes_from_graph] input_names size: " << input_names.size()
                        << ", input_tensor size: " << input_tensor.size()
                        << ", input_name_index size: " << input_name_index.size();
  
  // Build a map from input name to NodeArg* for O(1) lookup
  // This is necessary because input_tensor comes from main_graph.GetInputs() which may have
  // different ordering than input_names (from graph_body_viewer)
  std::unordered_map<std::string, const NodeArg*> name_to_nodearg;
  for (const auto* nodearg : input_tensor) {
    if (nodearg != nullptr) {
      name_to_nodearg[nodearg->Name()] = nodearg;
      LOGS_DEFAULT(VERBOSE) << "[extract_base_shapes_from_graph] Indexed NodeArg: '" << nodearg->Name() << "'";
    }
  }
  
  // Process inputs in the order they appear in input_name_index (map order for hash consistency)
  for (const auto& [name, idx] : input_name_index) {
    // Skip initializers/constants
    if (initializers.count(name) > 0) {
      LOGS_DEFAULT(VERBOSE) << "[extract_base_shapes_from_graph] Skipping initializer: '" << name << "'";
      continue;
    }
    
    ordered_names.push_back(name);
    
    // Find the corresponding NodeArg by NAME (not position!)
    std::vector<std::int64_t> base_shape;
    auto it = name_to_nodearg.find(name);
    if (it != name_to_nodearg.end()) {
      const NodeArg* nodearg = it->second;
      auto tensor_shape = nodearg->Shape();
      if (tensor_shape != nullptr && tensor_shape->dim_size() > 1) {
        LOGS_DEFAULT(VERBOSE) << "[extract_base_shapes_from_graph] Processing input '" << name 
                              << "' with " << tensor_shape->dim_size() << " dimensions";
        // Extract non-batch dimensions (skip dim 0)
        for (int j = 1; j < tensor_shape->dim_size(); ++j) {
          const auto& dim = tensor_shape->dim(j);
          if (dim.has_dim_value()) {
            base_shape.push_back(dim.dim_value());
            LOGS_DEFAULT(VERBOSE) << "[extract_base_shapes_from_graph]   dim[" << j << "] = " << dim.dim_value();
          } else {
            // Symbolic non-batch dimension found - cannot precompile
            // Do NOT default to any value - mark as failure
            LOGS_DEFAULT(WARNING) << "[extract_base_shapes_from_graph] Input '" << name
                                  << "' dim " << j << " is symbolic - cannot extract concrete base shapes";
            all_concrete = false;
          }
        }
      } else if (tensor_shape != nullptr && tensor_shape->dim_size() == 1) {
        // Single dimension input (just batch) - no base shape needed
        LOGS_DEFAULT(VERBOSE) << "[extract_base_shapes_from_graph] Input '" << name 
                              << "' has only 1 dimension (batch only) - empty base shape";
      } else {
        LOGS_DEFAULT(WARNING) << "[extract_base_shapes_from_graph] Input '" << name 
                              << "' has null or empty shape";
      }
    } else {
      LOGS_DEFAULT(WARNING) << "[extract_base_shapes_from_graph] Input '" << name 
                            << "' not found in NodeArg map - may be subgraph-only input";
    }
    base_shapes.push_back(base_shape);
    
    // Log the extracted base shape
    std::ostringstream ss;
    ss << "[";
    for (std::size_t k = 0; k < base_shape.size(); ++k) {
      if (k > 0) ss << ", ";
      ss << base_shape[k];
    }
    ss << "]";
    LOGS_DEFAULT(VERBOSE) << "[extract_base_shapes_from_graph] Input '" << name << "' base_shape: " << ss.str();
  }
  
  if (all_concrete) {
    LOGS_DEFAULT(VERBOSE) << "[extract_base_shapes_from_graph] Successfully extracted " << ordered_names.size() 
                          << " input base shapes (all concrete)";
  } else {
    LOGS_DEFAULT(WARNING) << "[extract_base_shapes_from_graph] Failed - found symbolic non-batch dimensions";
  }
  return {all_concrete, ordered_names, base_shapes};
}

// Compile a single model for a specific batch size and cache it
// Returns the cache hash for the compiled program
static inline std::string precompile_model_for_batch(
    std::size_t batch_size,
    const std::vector<std::string>& input_names,
    const std::vector<std::vector<std::int64_t>>& all_input_base_shapes,
    const std::string& onnx_string,
    migraphx::onnx_options& options,
    const migraphx::target& t,
    bool fp16_enable,
    bool bf16_enable,
    bool int8_enable,
    bool fp8_enable,
    bool int8_calibration_cache_available,
    std::unordered_map<std::string, float>& dynamic_range_map,
    bool exhaustive_tune,
    const std::filesystem::path& model_path,
    const std::filesystem::path& model_cache_path,
    const std::string& mxr_filename_prefix,
    std::unordered_map<std::string, migraphx::program>& cached_programs)
{
  // Build cache key for this batch size
  std::vector<std::int64_t> batch_shape_key;
  for (std::size_t i = 0; i < input_names.size(); ++i) {
    batch_shape_key.push_back(static_cast<std::int64_t>(batch_size));
    batch_shape_key.insert(batch_shape_key.end(), 
                          all_input_base_shapes[i].begin(), 
                          all_input_base_shapes[i].end());
  }
  auto cache_hash = make_hash(batch_shape_key);
  
  LOGS_DEFAULT(VERBOSE) << "[precompile_model_for_batch] Batch " << batch_size << " -> hash: " << cache_hash;
  
  // Check if already cached in memory
  if (cached_programs.find(cache_hash) != cached_programs.end()) {
    LOGS_DEFAULT(VERBOSE) << "[precompile_model_for_batch] ✓ Batch " << batch_size << " already cached";
    return cache_hash;
  }
  
  // Build disk cache file path
  std::filesystem::path batch_cache_file;
  if (!model_cache_path.empty()) {
    batch_cache_file = model_cache_path / (mxr_filename_prefix + cache_hash + ".mxr");
  }
  
  LOGS_DEFAULT(VERBOSE) << "[precompile_model_for_batch] Compiling/loading batch " << batch_size << "...";
  
  // Load or compile the model
  migraphx::program batch_prog = load_or_compile_model(
      batch_cache_file,
      onnx_string,
      options,
      t,
      fp16_enable,
      bf16_enable,
      int8_enable,
      fp8_enable,
      int8_calibration_cache_available,
      dynamic_range_map,
      exhaustive_tune,
      model_path,
      nullptr,  // ctx not needed for precompilation
      nullptr,  // map_input_name_index not needed
      input_names,
      all_input_base_shapes,
      batch_size);
  
  // Store in memory cache
  cached_programs[cache_hash] = std::move(batch_prog);
  LOGS_DEFAULT(VERBOSE) << "[precompile_model_for_batch] ✓ Stored batch " << batch_size << " in cache";
  
  return cache_hash;
}

// Precompile all batch models during Compile() phase
// This moves compilation from compute_func() to initialization time
// Uses parallel loading to speed up cache loading, but serializes compilation
// to avoid thread-safety issues in MIGraphX compile()
static inline void precompile_all_dynamic_batch_models(
    const std::vector<std::size_t>& compiled_batch_sizes,
    const std::vector<std::string>& input_names,
    const std::vector<std::vector<std::int64_t>>& all_input_base_shapes,
    const std::string& onnx_string,
    migraphx::onnx_options& options,
    const migraphx::target& t,
    bool fp16_enable,
    bool bf16_enable,
    bool int8_enable,
    bool fp8_enable,
    bool int8_calibration_cache_available,
    std::unordered_map<std::string, float>& dynamic_range_map,
    bool exhaustive_tune,
    const std::filesystem::path& model_path,
    const std::filesystem::path& model_cache_path,
    const std::string& mxr_filename_prefix,
    std::unordered_map<std::string, migraphx::program>& cached_programs)
{
  LOGS_DEFAULT(INFO) << "[precompile_all_dynamic_batch_models] Processing " 
                     << compiled_batch_sizes.size() << " batch models...";
  
  // Structure to hold batch info for loading/compiling
  struct BatchInfo {
    std::size_t batch_size;
    std::string cache_hash;
    std::filesystem::path cache_file;
  };
  
  // Build batch info for all batch sizes
  std::vector<BatchInfo> batch_infos;
  for (const auto& batch_size : compiled_batch_sizes) {
    BatchInfo info;
    info.batch_size = batch_size;
    
    // Build cache key for this batch size
    std::vector<std::int64_t> batch_shape_key;
    for (std::size_t i = 0; i < input_names.size(); ++i) {
      batch_shape_key.push_back(static_cast<std::int64_t>(batch_size));
      batch_shape_key.insert(batch_shape_key.end(), 
                            all_input_base_shapes[i].begin(), 
                            all_input_base_shapes[i].end());
    }
    info.cache_hash = make_hash(batch_shape_key);
    
    // Build disk cache file path
    if (!model_cache_path.empty()) {
      info.cache_file = model_cache_path / (mxr_filename_prefix + info.cache_hash + ".mxr");
    }
    
    // Skip if already in memory cache
    if (cached_programs.find(info.cache_hash) != cached_programs.end()) {
      LOGS_DEFAULT(VERBOSE) << "[precompile_all_dynamic_batch_models] Batch " << batch_size 
                            << " already in memory cache, skipping";
      continue;
    }
    
    batch_infos.push_back(info);
  }
  
  if (batch_infos.empty()) {
    LOGS_DEFAULT(INFO) << "[precompile_all_dynamic_batch_models] All models already cached in memory";
    return;
  }
  
  // ============================================================================
  // PHASE 1: Parallel loading from disk cache
  // ============================================================================
  LOGS_DEFAULT(INFO) << "[precompile_all_dynamic_batch_models] Phase 1: Attempting parallel load from disk cache...";
  
  // Mutex to protect shared state
  std::mutex cache_mutex;
  
  // Track which batch sizes need compilation (cache misses)
  std::vector<BatchInfo> needs_compilation;
  std::mutex compile_list_mutex;
  
  // Launch async tasks for parallel loading
  std::vector<std::future<void>> load_futures;
  
  for (const auto& info : batch_infos) {
    load_futures.push_back(std::async(std::launch::async, 
      [&, info]() {
        LOGS_DEFAULT(VERBOSE) << "[precompile_all_dynamic_batch_models] Trying to load batch " 
                              << info.batch_size << " from disk...";
        
        migraphx::program prog;
        bool loaded = load_precompiled_model(prog, info.cache_file);
        
        if (loaded) {
          // Cache hit - store in memory cache
          std::lock_guard<std::mutex> lock(cache_mutex);
          cached_programs[info.cache_hash] = std::move(prog);
          LOGS_DEFAULT(VERBOSE) << "[precompile_all_dynamic_batch_models] ✓ Loaded batch " 
                                << info.batch_size << " from disk cache";
        } else {
          // Cache miss - add to compilation list
          std::lock_guard<std::mutex> lock(compile_list_mutex);
          needs_compilation.push_back(info);
          LOGS_DEFAULT(VERBOSE) << "[precompile_all_dynamic_batch_models] ✗ Batch " 
                                << info.batch_size << " not in disk cache, needs compilation";
        }
      }
    ));
  }
  
  // Wait for all loading tasks to complete
  for (auto& future : load_futures) {
    future.get();
  }
  
  std::size_t loaded_count = batch_infos.size() - needs_compilation.size();
  LOGS_DEFAULT(INFO) << "[precompile_all_dynamic_batch_models] Phase 1 complete: " 
                     << loaded_count << " loaded from cache, " 
                     << needs_compilation.size() << " need compilation";
  
  // ============================================================================
  // PHASE 2: Sequential compilation for cache misses
  // ============================================================================
  if (!needs_compilation.empty()) {
    LOGS_DEFAULT(INFO) << "[precompile_all_dynamic_batch_models] Phase 2: Compiling " 
                       << needs_compilation.size() << " models sequentially...";
    
    // Sort by batch size for consistent ordering
    std::sort(needs_compilation.begin(), needs_compilation.end(),
              [](const BatchInfo& a, const BatchInfo& b) { return a.batch_size < b.batch_size; });
    
    for (const auto& info : needs_compilation) {
      LOGS_DEFAULT(INFO) << "[precompile_all_dynamic_batch_models] Compiling batch size " 
                         << info.batch_size << "...";
      
      // Compile the model (this is the thread-unsafe part that must be serialized)
      migraphx::program batch_prog = CompileProgramWithBatch(
          onnx_string,
          options,
          t,
          fp16_enable,
          bf16_enable,
          int8_enable,
          fp8_enable,
          int8_calibration_cache_available,
          dynamic_range_map,
          exhaustive_tune,
          model_path,
          nullptr,  // ctx not needed for precompilation
          nullptr,  // map_input_name_index not needed
          input_names,
          all_input_base_shapes,
          info.batch_size);
      
      LOGS_DEFAULT(INFO) << "[precompile_all_dynamic_batch_models] ✓ Compiled batch size " 
                         << info.batch_size;
      
      // Save to disk cache
      save_compiled_model(batch_prog, info.cache_file);
      if (!info.cache_file.empty()) {
        LOGS_DEFAULT(VERBOSE) << "[precompile_all_dynamic_batch_models] Saved to disk: " 
                              << info.cache_file.string();
      }
      
      // Store in memory cache
      cached_programs[info.cache_hash] = std::move(batch_prog);
    }
    
    LOGS_DEFAULT(INFO) << "[precompile_all_dynamic_batch_models] Phase 2 complete: " 
                       << needs_compilation.size() << " models compiled";
  }
  
  LOGS_DEFAULT(INFO) << "[precompile_all_dynamic_batch_models] ✓ All " 
                     << cached_programs.size() << " models ready";
}

// Precompile static model (no dynamic batching) during Compile() phase
// IMPORTANT: This function should ONLY be called when all dimensions are concrete.
// The caller must verify this before calling - symbolic dimensions are NOT allowed.
static inline void precompile_static_model(
    const std::vector<std::string>& input_names,
    const std::vector<const NodeArg*>& input_tensor,
    const InitializedTensorSet& initializers,
    const std::unordered_map<std::string, std::size_t>& input_name_index,
    const std::string& onnx_string,
    migraphx::onnx_options& options,
    const migraphx::target& t,
    bool fp16_enable,
    bool bf16_enable,
    bool int8_enable,
    bool fp8_enable,
    bool int8_calibration_cache_available,
    std::unordered_map<std::string, float>& dynamic_range_map,
    bool exhaustive_tune,
    const std::filesystem::path& model_path,
    const std::filesystem::path& model_cache_path,
    const std::string& mxr_filename_prefix,
    std::unordered_map<std::string, migraphx::program>& cached_programs)
{
  LOGS_DEFAULT(INFO) << "[precompile_static_model] Precompiling static model...";
  
  // Build a map from input name to NodeArg* for correct name-based lookup
  // This is necessary because input_tensor (from main_graph.GetInputs()) may have
  // different ordering than input_names (from graph_body_viewer)
  std::unordered_map<std::string, const NodeArg*> name_to_nodearg;
  for (const auto* nodearg : input_tensor) {
    if (nodearg != nullptr) {
      name_to_nodearg[nodearg->Name()] = nodearg;
    }
  }
  
  // Build full shapes (including batch dimension) from graph definition
  // All dimensions must be concrete - no defaulting of symbolic dims
  std::vector<std::int64_t> shape_key;
  std::vector<std::string> ordered_names;
  std::vector<std::vector<std::int64_t>> full_shapes;
  
  for (const auto& [name, idx] : input_name_index) {
    // Skip initializers/constants
    if (initializers.count(name) > 0) {
      continue;
    }
    
    ordered_names.push_back(name);
    
    // Find the corresponding NodeArg by NAME (not position!)
    std::vector<std::int64_t> shape;
    auto it = name_to_nodearg.find(name);
    if (it != name_to_nodearg.end()) {
      auto tensor_shape = it->second->Shape();
      if (tensor_shape != nullptr && tensor_shape->dim_size() > 0) {
        for (int j = 0; j < tensor_shape->dim_size(); ++j) {
          const auto& dim = tensor_shape->dim(j);
          if (dim.has_dim_value()) {
            shape.push_back(dim.dim_value());
            shape_key.push_back(dim.dim_value());
          } else {
            // Symbolic dimension found - this should NOT happen!
            // The caller should have verified all dims are concrete before calling.
            LOGS_DEFAULT(ERROR) << "[precompile_static_model] Unexpected symbolic dimension in input '"
                                << name << "' dim " << j << " - aborting precompilation";
            return;  // Abort precompilation
          }
        }
      }
    } else {
      LOGS_DEFAULT(WARNING) << "[precompile_static_model] Input '" << name 
                            << "' not found in NodeArg map - skipping";
    }
    full_shapes.push_back(shape);
  }
  
  if (shape_key.empty()) {
    LOGS_DEFAULT(VERBOSE) << "[precompile_static_model] No model inputs to compile";
    return;
  }
  
  auto cache_hash = make_hash(shape_key);
  
  // Check if already cached
  if (cached_programs.find(cache_hash) != cached_programs.end()) {
    LOGS_DEFAULT(VERBOSE) << "[precompile_static_model] ✓ Model already cached with hash: " << cache_hash;
    return;
  }
  
  // Build disk cache file path
  std::filesystem::path cache_file;
  if (!model_cache_path.empty()) {
    cache_file = model_cache_path / (mxr_filename_prefix + cache_hash + ".mxr");
  }
  
  LOGS_DEFAULT(INFO) << "[precompile_static_model] Loading/compiling model (hash: " << cache_hash << ")...";
  
  // Extract base shapes (without batch) for compilation
  std::vector<std::vector<std::int64_t>> base_shapes;
  std::int64_t batch_size = 1;
  for (const auto& shape : full_shapes) {
    if (!shape.empty()) {
      batch_size = shape[0];
      std::vector<std::int64_t> base(shape.begin() + 1, shape.end());
      base_shapes.push_back(base);
    } else {
      base_shapes.push_back({});
    }
  }
  
  // Load or compile
  migraphx::program prog = load_or_compile_model(
      cache_file,
      onnx_string,
      options,
      t,
      fp16_enable,
      bf16_enable,
      int8_enable,
      fp8_enable,
      int8_calibration_cache_available,
      dynamic_range_map,
      exhaustive_tune,
      model_path,
      nullptr,
      nullptr,
      ordered_names,
      base_shapes,
      static_cast<std::size_t>(batch_size));
  
  // Store in cache
  cached_programs[cache_hash] = std::move(prog);
  LOGS_DEFAULT(INFO) << "[precompile_static_model] ✓ Static model precompiled and cached";
}

// Encapsulates precompilation decision logic from Compile()
// Returns true if compilation should be deferred to runtime, false if precompilation succeeded
static inline bool handle_precompilation_decision(
    const std::string& node_name,
    const std::vector<std::string>& input_names,
    const std::vector<const NodeArg*>& input_tensor,
    const InitializedTensorSet& initializers,
    const std::unordered_map<std::string, std::size_t>& input_name_index,
    const std::string& onnx_string_buffer,
    migraphx::onnx_options& options,
    const migraphx::target& t,
    bool fp16_enable,
    bool bf16_enable,
    bool int8_enable,
    bool fp8_enable,
    bool int8_calibration_cache_available,
    std::unordered_map<std::string, float>& dynamic_range_map,
    bool exhaustive_tune,
    const std::filesystem::path& model_path,
    const std::filesystem::path& model_cache_path,
    const std::string& mxr_filename_prefix,
    std::unordered_map<std::string, migraphx::program>& cached_programs,
    std::size_t max_dynamic_batch,
    std::size_t max_compiled_models)
{
  // ═══════════════════════════════════════════════════════════════════════════
  // PRECOMPILATION: Compile models during Compile() phase instead of compute_func()
  // ═══════════════════════════════════════════════════════════════════════════
  // 
  // Precompilation rules:
  // 1. max_dynamic_batch > 0 AND all non-batch dims are concrete:
  //    -> Precompile all batch models (symbolic batch dim is OK)
  // 2. max_dynamic_batch > 0 AND some non-batch dims are symbolic:
  //    -> Defer to runtime (cannot precompile without concrete non-batch shapes)
  // 3. max_dynamic_batch == 0 AND all dims are concrete:
  //    -> Precompile static model with concrete shapes
  // 4. max_dynamic_batch == 0 AND some dims are symbolic:
  //    -> Defer to runtime (cannot precompile with symbolic dimensions)
  //
  // IMPORTANT: We do NOT default symbolic dimensions to any value.
  // Precompilation only happens when we have concrete shapes from the graph.
  // ═══════════════════════════════════════════════════════════════════════════
  
  LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] ════════════════════════════════════════════════════";
  LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] Starting precompilation decision for node '" << node_name << "'";
  LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] max_dynamic_batch = " << max_dynamic_batch;
  LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] Number of inputs: " << input_names.size();
  LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] Number of input tensors: " << input_tensor.size();
  LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] Number of initializers: " << initializers.size();
  
  // Check if model has only dynamic batch dimension (other dims are static)
  bool only_dynamic_batch = has_only_dynamic_batch_dimension(input_names, input_tensor, initializers);
  LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] only_dynamic_batch = " << (only_dynamic_batch ? "true" : "false");
  
  if (max_dynamic_batch > 0) {
    LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] Mode: DYNAMIC BATCH (max_dynamic_batch=" << max_dynamic_batch << ")";
    
    // Dynamic batch mode - try to precompile if all non-batch dimensions are concrete
    if (only_dynamic_batch) {
      LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] Model has only dynamic batch dimension - attempting to extract base shapes";
      
      // Extract base shapes - this will FAIL if any non-batch dim is symbolic
      auto [shapes_valid, ordered_names, base_shapes] = extract_base_shapes_from_graph(
          input_names, input_tensor, initializers, input_name_index);
      
      LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] extract_base_shapes_from_graph result: shapes_valid=" 
                            << (shapes_valid ? "true" : "false");
      
      if (shapes_valid) {
        
        // All non-batch dimensions are concrete - precompile all batch models
        auto compiled_batch_sizes = generate_compiled_batch_sizes(max_dynamic_batch, max_compiled_models);
        
        std::ostringstream batch_ss;
        batch_ss << "[";
        for (std::size_t i = 0; i < compiled_batch_sizes.size(); ++i) {
          if (i > 0) batch_ss << ", ";
          batch_ss << compiled_batch_sizes[i];
        }
        batch_ss << "]";
        LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] Batch sizes to compile: " << batch_ss.str();
        LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] >>> STARTING DYNAMIC BATCH PRECOMPILATION <<<";
        
        precompile_all_dynamic_batch_models(
            compiled_batch_sizes,
            ordered_names,
            base_shapes,
            onnx_string_buffer,
            options,
            t,
            fp16_enable,
            bf16_enable,
            int8_enable,
            fp8_enable,
            int8_calibration_cache_available,
            dynamic_range_map,
            exhaustive_tune,
            model_path,
            model_cache_path,
            mxr_filename_prefix,
            cached_programs);
        
        // Precompilation complete - disable deferred compilation
        LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] ✓✓✓ Dynamic batch precompilation COMPLETE for node '" 
                              << node_name << "'";
        LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] defer_compilation set to FALSE";
        LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] cached_programs size: " << cached_programs.size();
        LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] ════════════════════════════════════════════════════";
        return false;  // No need to defer
      } else {
        // Non-batch dimensions contain symbolic values - cannot precompile
        LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] ✗ CANNOT PRECOMPILE: Non-batch dimensions contain symbolic values";
        LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] Deferring compilation to runtime for node '" << node_name << "'";
        LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] defer_compilation set to TRUE";
        LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] ════════════════════════════════════════════════════";
        return true;  // Defer to runtime
      }
    } else {
      // Model has multiple dynamic dimensions (not just batch) - defer to runtime
      LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] ✗ CANNOT PRECOMPILE: Model has non-batch dynamic dimensions";
      LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] Deferring compilation to runtime for node '" << node_name << "'";
      LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] defer_compilation set to TRUE";
      LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] ════════════════════════════════════════════════════";
      return true;  // Defer to runtime
    }
  } else {
    LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] Mode: STATIC (max_dynamic_batch=0)";
    
    // Static model (max_dynamic_batch == 0) - only precompile if ALL dimensions are concrete
    // Check if any dimension is symbolic
    bool has_symbolic_dims = false;
    std::string symbolic_info;
    for (std::size_t i = 0; i < input_names.size(); ++i) {
      if (initializers.count(input_names[i]) > 0) continue;  // Skip initializers
      if (i < input_tensor.size()) {
        auto tensor_shape = input_tensor[i]->Shape();
        if (tensor_shape != nullptr) {
          for (int j = 0; j < tensor_shape->dim_size(); ++j) {
            if (!tensor_shape->dim(j).has_dim_value()) {
              has_symbolic_dims = true;
              symbolic_info = "Input '" + input_names[i] + "' dim " + std::to_string(j);
              LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] Found symbolic dimension: " << symbolic_info;
              break;
            }
          }
        }
      }
      if (has_symbolic_dims) break;
    }
    
    LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] has_symbolic_dims = " << (has_symbolic_dims ? "true" : "false");
    
    if (!has_symbolic_dims) {
      // All dimensions are concrete - precompile static model
      LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] All dimensions are concrete - precompiling static model";
      LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] >>> STARTING STATIC MODEL PRECOMPILATION <<<";
      
      precompile_static_model(
          input_names,
          input_tensor,
          initializers,
          input_name_index,
          onnx_string_buffer,
          options,
          t,
          fp16_enable,
          bf16_enable,
          int8_enable,
          fp8_enable,
          int8_calibration_cache_available,
          dynamic_range_map,
          exhaustive_tune,
          model_path,
          model_cache_path,
          mxr_filename_prefix,
          cached_programs);
      
      // Precompilation complete - disable deferred compilation
      LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] ✓✓✓ Static model precompilation COMPLETE for node '" 
                            << node_name << "'";
      LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] defer_compilation set to FALSE";
      LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] cached_programs size: " << cached_programs.size();
      LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] ════════════════════════════════════════════════════";
      return false;  // No need to defer
    } else {
      // Has symbolic dimensions and max_dynamic_batch == 0 - defer to runtime
      LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] ✗ CANNOT PRECOMPILE: Has symbolic dimensions but max_dynamic_batch=0";
      LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] Symbolic dimension found: " << symbolic_info;
      LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] Deferring compilation to runtime for node '" << node_name << "'";
      LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] defer_compilation set to TRUE";
      LOGS_DEFAULT(VERBOSE) << "[Compile][PRECOMPILE] ════════════════════════════════════════════════════";
      return true;  // Defer to runtime
    }
  }
}

constexpr std::uint64_t MIGraphX_Version =
    ((MIGRAPHX_VERSION_MAJOR << 16) | (MIGRAPHX_VERSION_MINOR << 8) | MIGRAPHX_VERSION_PATCH);

Status MIGraphXExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                                          std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node_graph : fused_nodes) {
    const GraphViewer& graph_body_viewer = fused_node_graph.filtered_graph;
    const Node& fused_node = fused_node_graph.fused_node;

    std::filesystem::path model_cache_file;
    auto mxr_filename_prefix = to_hex(MIGraphX_Version) + "-" + GenerateGraphId(graph_body_viewer) + "-" + make_hash(std::string_view(device_prop_.gcnArchName)) + "-";

    // Get model input names (only first layer) - these are actual model inputs, not weights/constants
    const Graph* cur_graph = &graph_body_viewer.GetGraph();
    while (cur_graph->IsSubgraph()) {
      cur_graph = cur_graph->ParentGraph();
    }
    const Graph& main_graph = *cur_graph;
    const auto& input_tensor = main_graph.GetInputs();
    std::set<std::string>& node_session_input_names = map_session_input_names_[fused_node.Name()];
    for (auto i : input_tensor) {
      node_session_input_names.insert(i->Name());
    }
    LOGS_DEFAULT(VERBOSE) << "[Compile] Node '" << fused_node.Name() << "' has "
                          << node_session_input_names.size() << " model input parameters (excluding weights/constants)";

    // Build input name to index map, only for model input parameters (excludes weights/constants)
    auto input_name_index = get_input_name_map(fused_node.InputDefs(), &node_session_input_names);
    LOGS_DEFAULT(VERBOSE) << "[Compile] input_name_index has " << input_name_index.size()
                          << " entries (model inputs only)";

    auto model = graph_body_viewer.CreateModel(*GetLogger());
    auto model_proto = model->ToProto();
    graph_body_viewer.ToProto(*model_proto->mutable_graph(), true, true);
    model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    std::string onnx_string_buffer;
    model_proto->SerializeToString(onnx_string_buffer);

    dump_model_as_onnx(onnx_string_buffer, std::string{fused_node.Name() + ".onnx"});

    // map parameter input name to index
    auto [input_names, output_names] = get_io_names(graph_body_viewer);

    // Get initializers and build ONNX options with default shapes for symbolic dimensions
    const auto& initializers = graph_body_viewer.GetAllInitializedTensors();
    migraphx::onnx_options options = get_program_parameter_options(input_names, input_tensor, initializers);

    // Initialize the cached_programs map for this node if not already done
    if (cached_programs_.find(fused_node.Name()) == cached_programs_.end()) {
      cached_programs_[fused_node.Name()] = std::unordered_map<std::string, migraphx::program>();
    }
    
    // Perform precompilation decision and execution
    map_defer_compilation_[fused_node.Name()] = handle_precompilation_decision(
        fused_node.Name(),
        input_names,
        input_tensor,
        initializers,
        input_name_index,
        onnx_string_buffer,
        options,
        t_,
        fp16_enable_,
        bf16_enable_,
        int8_enable_,
        fp8_enable_,
        int8_calibration_cache_available_,
        dynamic_range_map_,
        exhaustive_tune_,
        model_path_,
        model_cache_path_,
        mxr_filename_prefix,
        cached_programs_[fused_node.Name()],
        max_dynamic_batch_,
        max_compiled_models_);

    // Create program object (may be empty if precompiled programs are in cache)
    migraphx::program prog;
    map_progs_[fused_node.Name()] = prog;

    map_onnx_string_[fused_node.Name()] = onnx_string_buffer;
    map_input_index_[fused_node.Name()] = input_name_index;

    // NOTE: cached_programs_ was initialized earlier before precompilation

    NodeComputeInfo compute_info;
    compute_info.create_state_func = [=](ComputeContext* context, FunctionState* state) {
      std::unique_ptr<MIGraphXFuncState> p = std::make_unique<MIGraphXFuncState>();
      *p = {context->allocate_func, context->release_func, context->allocator_handle, map_progs_[context->node_name],
            map_onnx_string_[context->node_name], options, t_, map_input_index_[context->node_name], &mgx_mu_,
            map_defer_compilation_[context->node_name], fp16_enable_, bf16_enable_, fp8_enable_, int8_enable_,
            int8_calibration_cache_available_, dynamic_range_map_,
            model_cache_path_, dump_model_ops_, exhaustive_tune_, max_dynamic_batch_,
            max_compiled_models_, std::ref(cached_programs_[context->node_name])};
      
      // Initialize dynamic batch support if max_dynamic_batch > 0
      if (max_dynamic_batch_ > 0) {
        p->has_dynamic_batch = true;
        p->compiled_batch_sizes = generate_compiled_batch_sizes(max_dynamic_batch_, max_compiled_models_);
        LOGS_DEFAULT(VERBOSE) << "[Compile][CREATE_STATE] Dynamic batch enabled for node '" << context->node_name 
                              << "' with max_dynamic_batch=" << max_dynamic_batch_
                              << ", max_compiled_models=" << max_compiled_models_
                              << ", generated " << p->compiled_batch_sizes.size() << " batch sizes to compile";
        LOGS_DEFAULT(VERBOSE) << "[Compile][CREATE_STATE] defer_compilation=" << p->defer_compilation;
      } else {
        LOGS_DEFAULT(VERBOSE) << "[Compile][CREATE_STATE] Static model mode for node '" << context->node_name << "'";
        LOGS_DEFAULT(VERBOSE) << "[Compile][CREATE_STATE] defer_compilation=" << p->defer_compilation;
      }
      
      *state = p.release();
      return 0;
    };

    compute_info.release_state_func = [](FunctionState state) {
      if (state)
        delete static_cast<MIGraphXFuncState*>(state);
    };

    compute_info.compute_func = [this, mxr_filename_prefix](FunctionState state, const OrtApi* api, OrtKernelContext* context) {
      Ort::KernelContext ctx(context);
      MIGraphXFuncState* mgx_state = reinterpret_cast<MIGraphXFuncState*>(state);

      const auto& map_input_name_index = mgx_state->input_name_indexes;

      // ═══════════════════════════════════════════════════════════════════════
      // ULTRA-FAST PATH: Shapes unchanged from last run
      // ═══════════════════════════════════════════════════════════════════════
      if (execute_ultra_fast_path(mgx_state, api, context, ctx)) {
        return Status::OK();
      }

      // ═══════════════════════════════════════════════════════════════════════
      // Build input shape hash - only computed when shapes change
      // ═══════════════════════════════════════════════════════════════════════
      std::vector<std::int64_t> all_input_shapes;
      all_input_shapes.reserve(map_input_name_index.size() * 4);
      for (const auto& [name, index] : map_input_name_index) {
        const auto& shape = ctx.GetInput(index).GetTensorTypeAndShapeInfo().GetShape();
        all_input_shapes.insert(all_input_shapes.end(), shape.begin(), shape.end());
      }
      const auto current_hash = make_hash(all_input_shapes);

      // ═══════════════════════════════════════════════════════════════════════
      // FAST PATH: Check cached programs for this shape hash
      // ═══════════════════════════════════════════════════════════════════════
      if (execute_fast_path(mgx_state, api, context, ctx, current_hash, all_input_shapes)) {
        return Status::OK();
      }

      // ═══════════════════════════════════════════════════════════════════════
      // STANDARD PATH: Shape checking and potential recompilation
      // ═══════════════════════════════════════════════════════════════════════
      execute_standard_path(mgx_state, api, context, ctx, current_hash, std::move(all_input_shapes),
                            model_cache_path_, model_path_, mxr_filename_prefix);

      return Status::OK();
    };
    node_compute_funcs.push_back(compute_info);
  }

  return Status::OK();
}

void MIGraphXExecutionProvider::RegisterStreamHandlers(IStreamCommandHandleRegistry& stream_handle_registry,
                                                       AllocatorMap& allocators) const {
  auto allocator = allocators[GetOrtDeviceByMemType(OrtMemTypeCPU)];
  RegisterMIGraphXStreamHandles(stream_handle_registry, OrtDevice::GPU, allocator, true, stream_, false /*TODO:external_stream_*/);
}

OrtDevice MIGraphXExecutionProvider::GetOrtDeviceByMemType(OrtMemType mem_type) const {
  if (mem_type == OrtMemTypeCPUInput)
    return OrtDevice();
  if (mem_type == OrtMemTypeCPUOutput)
    return OrtDevice(OrtDevice::GPU, OrtDevice::MemType::HOST_ACCESSIBLE, OrtDevice::VendorIds::AMD,
                     default_device_.Id());
  return default_device_;
}

Status MIGraphXExecutionProvider::Sync() const {
  HIP_CALL_THROW(hipStreamSynchronize(static_cast<hipStream_t>(nullptr)));

  auto status = hipStreamQuery(stream_);
  if (status != hipSuccess) {
    return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::EP_FAIL);
  }
  return Status::OK();
}

Status MIGraphXExecutionProvider::OnRunStart(const onnxruntime::RunOptions& /*run_options*/) {
  return Status::OK();
}

Status MIGraphXExecutionProvider::OnRunEnd(bool /*sync_stream*/, const onnxruntime::RunOptions& /*run_options*/) {
  auto status = hipStreamQuery(stream_);

  if (status != hipSuccess) {
    HIP_CALL_THROW(hipStreamSynchronize(stream_));
  }
  return Status::OK();
}

}  // namespace onnxruntime
