// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "opencl_execution_provider.h"
#include "opencl_allocator.h"
#include "opencl_program_manager.h"
#include "opencl_data_transfer.h"
#include "opencl_tunning_utils.h"
#include "core/common/logging/logging.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"

#include <array>
#include <utility>

// Add includes of kernel implementations
#include "memcpy_kernel.h"
#include "core/providers/opencl/math/elementwise.h"
#include "core/providers/opencl/math/matmul.h"
#include "core/providers/opencl/math/gemm.h"
#include "core/providers/opencl/math/softmax.h"
#include "core/providers/opencl/tensor/tensor.h"
#include "core/providers/opencl/activation/activations.h"
#include "core/providers/opencl/reduction/reduction.h"
#include "core/providers/opencl/generator/generator.h"
#include "core/providers/opencl/math/trigonometric.h"
#include "core/providers/opencl/math/Neg.h"

namespace onnxruntime {
namespace opencl {

Status RegisterOpenCLKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 1, 12, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, 14, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 15, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, 20, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, 17, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, Sqrt)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, 20, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 11, Range)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Cos)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Sin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 14, Trilu)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, 13, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 14, 18, Reshape)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 14, float,
                                                                Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 14, double,
                                                                Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 14, int64_t,
                                                                Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 14, int32_t,
                                                                Add)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 14, float,
                                                                Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 14, double,
                                                                Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 14, int64_t,
                                                                Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 14, int32_t,
                                                                Mul)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 14, float,
                                                                Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 14, double,
                                                                Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 14, int64_t,
                                                                Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 14, int32_t,
                                                                Sub)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 14, float,
                                                                Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 14, double,
                                                                Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 14, int64_t,
                                                                Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 14, int32_t,
                                                                Div)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, 14, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, float,
                                                                Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, double,
                                                                Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, int32_t,
                                                                Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, int64_t,
                                                                Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, 18,
                                                                          bool, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, 18,
                                                                          int32_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, 18,
                                                                          int64_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, 18,
                                                                          float, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, 18,
                                                                          double, Equal)>,

      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, int32_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, int64_t, Slice)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 9, 15,
                                                                            float, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 9, 15,
                                                                            double, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 9, 15,
                                                                            int32_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 9, 15,
                                                                            int64_t, Where)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, float, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, double,
                                                                  Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, int32_t,
                                                                  Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, int64_t,
                                                                  Expand)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, uint32_t,
                                                                  Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, uint64_t,
                                                                  Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, bool, Expand)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, 20,
                                                                    Squeeze)>,

  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      VLOGS_DEFAULT(V_GENERIC) << "Register kernel name: " << info.kernel_def->OpName()
                               << ", domain: " << info.kernel_def->Domain();
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}

std::shared_ptr<KernelRegistry> GetOpenCLKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  ORT_THROW_IF_ERROR(RegisterOpenCLKernels(*kernel_registry));
  return kernel_registry;
}

namespace {

std::string GetDeviceInfo(cl_device_id dev, cl_device_info info_name) {
  size_t ret_size;
  ORT_THROW_IF_CL_ERROR(clGetDeviceInfo(dev, info_name, 0, nullptr, &ret_size));
  std::string ret(ret_size, '\0');
  ORT_THROW_IF_CL_ERROR(clGetDeviceInfo(dev, info_name, ret.size(), ret.data(), nullptr));
  ret.resize(ret.size() - 1);  // get rid of the ending '\0'
  return ret;
};

bool ShouldFlushAfterLaunch(const std::string& device_name) {
  return device_name.find("Mali") != std::string::npos;
}

}  // namespace
}  // namespace opencl

OpenCLExecutionProvider::OpenCLExecutionProvider(const OpenCLExecutionProviderInfo& info)
    : IExecutionProvider{kOpenCLExecutionProvider, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0)}, auto_runing_level_(info.auto_runing_level), use_fp16_(info.use_fp16) {
  Status status;
#ifdef CL3W_ENABLE
  if (cl3wInit() != CL3W_OK) {
    ORT_THROW("cl3w initialization failure.");
  }
#endif
  ORT_THROW_IF_ERROR(InitOpenCLContext());
  ORT_THROW_IF_ERROR(InitCompileOptions());
  program_manager_ = std::make_unique<opencl::OpenCLProgramManager>(*this);
  InitCopyKernels();

#ifdef TRACY_ENABLE
  tracy_cl_ctx_ = TracyCLContext(ctx_, dev_);
#endif
}

OpenCLExecutionProvider::~OpenCLExecutionProvider() {
  // FIXME: kernel manager should release all managed kernels and programs

#ifdef TRACY_ENABLE
  TracyCLCollect(tracy_cl_ctx_);
  TracyCLDestroy(tracy_cl_ctx_);
#endif

  clReleaseCommandQueue(cmd_queue_);
  clReleaseCommandQueue(cmd_tune_queue_);
  clReleaseDevice(dev_);
  clReleaseContext(ctx_);
}

std::shared_ptr<KernelRegistry> OpenCLExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = opencl::GetOpenCLKernelRegistry();
  return kernel_registry;
}
template <class T>
static void GetCLDevInfo(cl_device_id  device ,
                  cl_device_info param_name, T* param) {
  static_assert(std::is_same<T, size_t>::value || std::is_same<T, uint64_t>::value
      || std::is_same<T, uint32_t>::value || std::is_same<T, size_t[3]>::value);
  ORT_THROW_IF_CL_ERROR(clGetDeviceInfo(device, param_name, sizeof(T), param, nullptr));
}
// Gpu SubGroup, referenced to TNN, GPU-model->a exprienced magic number, as CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE is not implemented util opencl 2.1
static std::map<int, int> AdrenoSubGroup{
    {640, 128},
    {630, 128},
    {616, 128},
    {612, 64},
    {610, 64},
    {540, 32},
    {530, 32},
    {512, 32},
    {510, 32},
    {509, 32},
    {506, 32},
    {505, 32},
    {405, 32},
    {330, 16},
};
Status OpenCLExecutionProvider::InitOpenCLContext() {
  cl_uint num_platforms;
  ORT_RETURN_IF_CL_ERROR(clGetPlatformIDs(0, nullptr, &num_platforms));
  // NOTE: the EP is in construction, the logger_ is not registered
  LOGS_DEFAULT(VERBOSE) << "[CL] num platforms: " << num_platforms;
  ORT_RETURN_IF_NOT(num_platforms > 0, "Cannot find OpenCL platform.");

  std::vector<cl_platform_id> platforms(num_platforms);
  ORT_RETURN_IF_CL_ERROR(clGetPlatformIDs(static_cast<cl_uint>(platforms.size()), platforms.data(), nullptr));
  int selected_platform_idx = -1;
  // FIXME: add platform selection logic
  for (auto& platform : platforms) {
    size_t ret_size;
    ORT_RETURN_IF_CL_ERROR(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &ret_size));
    std::string vendor(ret_size, '\0');
    ORT_RETURN_IF_CL_ERROR(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendor.size(), vendor.data(), nullptr));
    LOGS_DEFAULT(VERBOSE) << "[CL] platform vendor: " << vendor;
    if (vendor == "Oclgrind") {
      LOGS_DEFAULT(INFO) << "[CL] platform " << vendor << " selected";
      selected_platform_idx = 1;
      break;
    }
  }
  if (selected_platform_idx == -1) {
    LOGS_DEFAULT(INFO) << "[CL] default platform selected";
    selected_platform_idx = 0;
  }
  auto* selected_platform = platforms[selected_platform_idx];

  cl_int err{};
  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)selected_platform, 0};
  ctx_ = clCreateContextFromType(properties, CL_DEVICE_TYPE_GPU, /*pfn_notify=*/nullptr, /*user_data=*/nullptr, &err);
  ORT_RETURN_IF_CL_ERROR(err);

  size_t ret_size;
  ORT_RETURN_IF_CL_ERROR(clGetContextInfo(ctx_, CL_CONTEXT_DEVICES, 0, nullptr, &ret_size));
  std::vector<cl_device_id> devices(ret_size);
  ORT_RETURN_IF_CL_ERROR(clGetContextInfo(ctx_, CL_CONTEXT_DEVICES, devices.size(), devices.data(), nullptr));
  LOGS_DEFAULT(VERBOSE) << "[CL] num devices: " << devices.size();
  ORT_RETURN_IF(devices.empty(), "Cannot find OpenCL device.");
  dev_ = devices[0];

  dev_info_.device_name = opencl::GetDeviceInfo(dev_, CL_DEVICE_NAME);
  auto device_version = opencl::GetDeviceInfo(dev_, CL_DEVICE_VERSION);
  LOGS_DEFAULT(INFO) << "[CL] device name: " << dev_info_.device_name;
  LOGS_DEFAULT(VERBOSE) << "[CL] device vendor: " << device_version;
  LOGS_DEFAULT(VERBOSE) << "[CL] device version: " << opencl::GetDeviceInfo(dev_, CL_DEVICE_VERSION);
  auto exts = opencl::GetDeviceInfo(dev_, CL_DEVICE_EXTENSIONS);
  LOGS_DEFAULT(VERBOSE) << "[CL] device extensions: " << exts << std::endl;
  dev_info_.has_fp16 = exts.find("cl_khr_fp16") != std::string::npos;
  if (!dev_info_.has_fp16 && UseFp16()) {
    LOGS_DEFAULT(WARNING) << "[CL] FP16 is requested, but is not supported by the device!";
    DisableFp16();
  }
  flush_after_launch_ = opencl::ShouldFlushAfterLaunch(dev_info_.device_name);
  LOGS_DEFAULT(INFO) << "[CL] FP16: " << UseFp16();
  LOGS_DEFAULT(INFO) << "[CL] clFlush after launch: " << flush_after_launch_;
  if (dev_info_.device_name == "QUALCOMM Adreno(TM)") {
    dev_info_.gpu_type = opencl::GpuType::ADRENO;
    //windows will report a warning os sscanf_s
#if !(defined(WIN32) || defined(_WIN32) || defined(_WIN32_) || \
    defined(WIN64) || defined(_WIN64) || defined(_WIN64_))
    sscanf(device_version.c_str(), "%*s%*f%*s%d", &dev_info_.gpu_model);
#endif
#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_TARGET_OPENCL_VERSION >= 210 && defined(CL_HPP_USE_CL_SUB_GROUPS_KHR)
    cl_int cl_ret;
    sub_group_size = kernel.getSubGroupInfo<CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE>(*device_, range, &cl_ret);
    if (cl_ret != CL_SUCCESS) {
      CHECK_CL_SUCCESS(cl_ret)
      sub_group_size = 0;
    }
#else
    dev_info_.sub_group_size = AdrenoSubGroup.count(dev_info_.gpu_model) ?
        AdrenoSubGroup[dev_info_.gpu_model] : 0;
#endif
  }
  GetCLDevInfo(dev_, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &dev_info_.global_memery_cachesize_);
  GetCLDevInfo(dev_, CL_DEVICE_MAX_COMPUTE_UNITS, &dev_info_.compute_units_);
  GetCLDevInfo(dev_, CL_DEVICE_MAX_CLOCK_FREQUENCY, &dev_info_.max_freq_);
  GetCLDevInfo(dev_, CL_DEVICE_LOCAL_MEM_SIZE, &dev_info_.local_memory_size_);
  GetCLDevInfo(dev_, CL_DEVICE_MAX_WORK_GROUP_SIZE, &dev_info_.max_work_group_size);
  GetCLDevInfo(dev_, CL_DEVICE_MAX_WORK_ITEM_SIZES, &(dev_info_.max_work_item_size));
  GetCLDevInfo(dev_, CL_DEVICE_IMAGE2D_MAX_WIDTH, &dev_info_.image_2d_max_size[0]);
  GetCLDevInfo(dev_, CL_DEVICE_IMAGE2D_MAX_HEIGHT, &dev_info_.image_2d_max_size[1]);

  cmd_tune_queue_ = clCreateCommandQueue(ctx_, dev_, /*properties=*/CL_QUEUE_PROFILING_ENABLE, &err);
#ifdef TRACY_ENABLE
  cmd_queue_ = clCreateCommandQueue(ctx_, dev_, CL_QUEUE_PROFILING_ENABLE, &err);
#else
  cmd_queue_ = clCreateCommandQueue(ctx_, dev_, /*properties=*/0, &err);
#endif
  ORT_RETURN_IF_CL_ERROR(err);

  return Status::OK();
}

Status OpenCLExecutionProvider::InitCompileOptions() {
  compile_options_.reserve(2048);
  if (dev_info_.device_name == "Mali-G51") {
    compile_options_.append(" -D CONFORMANCE_could_not_emit_constant_value_abstractly");
  }
  return Status::OK();
}

opencl::NDRange OpenCLExecutionProvider::DefaultLocalWG2DOrTune(const opencl::NDRange& gws, const opencl::TuneKernelWithTimeFunc& func) const {
  /*
  //read well-tuned local_size from cache
  auto& tunedLws = runtime->tunedLwsMap();
  std::pair<std::string, std::vector<uint32_t>> info = std::make_pair(kernelName, gws);
  if (tunedLws.find(info) != tunedLws.end()) {
    return tunedLws[info];
  }
  */
  if (TuneEnabledLevel()) {
    return RunTuneLWS2D(gws, dev_info_, func, TuneEnabledLevel());
  }

  if (dev_info_.gpu_type != opencl::GpuType::ADRENO) {
    return opencl::NDRange();
  }
  std::vector<size_t> lwgs(2);
  if (dev_info_.max_work_group_size == 0) {
    lwgs[0] = lwgs[1] = 1;
  } else {
    lwgs = AdrenoLocalSize2D(gws, dev_info_);
  }
  if (lwgs.size() == 0) {
    return opencl::NDRange();
  }
  return opencl::NDRange(lwgs[0], lwgs[1]);
}

std::vector<AllocatorPtr> OpenCLExecutionProvider::CreatePreferredAllocators() {
  std::vector<AllocatorPtr> allocators;

  cl_context context = GetOpenCLContext();

  // OpenCL Allocator
  AllocatorPtr opencl_allocator = std::make_shared<onnxruntime::opencl::OpenCLAllocator>(context);
  allocators.push_back(opencl_allocator);

  // CPU Output Allocator
  AllocatorPtr cpu_output_allocator = CreateAllocator(AllocatorCreationInfo{
      [](int) {
        return std::make_unique<CPUAllocator>(
            OrtMemoryInfo(opencl::CPUAllocatorName, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput));
      }});
  allocators.push_back(cpu_output_allocator);

  // CPU Input Allocator
  AllocatorPtr cpu_input_allocator = CreateAllocator(AllocatorCreationInfo{
      [](int) {
        return std::make_unique<CPUAllocator>(
            OrtMemoryInfo(opencl::CPUInputAllocatorName, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUInput));
      }});
  allocators.push_back(cpu_input_allocator);

  return allocators;
}

Status OpenCLExecutionProvider::Sync() const {
  ORT_RETURN_IF_CL_ERROR(clFinish(cmd_queue_));
  return Status::OK();
}


#ifdef TRACY_ENABLE
constexpr const char* kFrameMarkName = "OpenCL EP Run";
#endif

// Status OpenCLExecutionProvider::OnRunStart() {
// #ifdef TRACY_ENABLE
//   FrameMarkStart(kFrameMarkName);
// #endif

//   return Status::OK();
// }

// Status OpenCLExecutionProvider::OnRunEnd(bool sync_stream) {
//   ORT_RETURN_IF_CL_ERROR(clFlush(cmd_queue_));
//   if (sync_stream) {
//     ORT_RETURN_IF_ERROR(Sync());
//   }

// #ifdef TRACY_ENABLE
//   FrameMarkEnd(kFrameMarkName);
//   TracyCLCollect(tracy_cl_ctx_);
// #endif

//   return Status::OK();
// }

cl_mem OpenCLExecutionProvider::GetScratchBufferTmp(size_t nbytes) const {

  if (nbytes == 0) {
      std::cerr << "GetScratchBuffer bufferSizes is null." << std::endl;
      return NULL;
  }
  int err = 0;
  auto ptr = clCreateBuffer(this->GetOpenCLContext(), CL_MEM_READ_WRITE, nbytes, NULL, &err);
  ORT_THROW_IF_CL_ERROR(err);
  VLOGF_DEFAULT(V_ALLOC, "Allocated Buffer(%p){size=%zu}", ptr, nbytes);

  return ptr;
}
void OpenCLExecutionProvider::ReadFromCLBuffer(cl_mem buffer,void* host_ptr,size_t nbytes){
      cl_int err;
    // Ensure host_ptr is not null
    if (!host_ptr) {
        std::cerr << "ReadFromCLBuffer:Host pointer is null." << std::endl;
        return;
    }
    // Enqueue the command to read the buffer
    err = clEnqueueReadBuffer(this->GetCommandQueue(), buffer, CL_TRUE, 0, nbytes, host_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error in clEnqueueReadBuffer: " << err << std::endl;
        return;
    }
    err = clFinish(this->GetCommandQueue());
    if (err != CL_SUCCESS) {
        std::cerr << "Error in clFinish: " << err << std::endl;
        return;
    }
}

void OpenCLExecutionProvider::WriteToCLBuffer(cl_mem buffer, const void* host_ptr, size_t nbytes){
    cl_int err;
    // Ensure host_ptr is not null
    if (!host_ptr) {
        std::cerr << "WriteToCLBuffer:Host pointer is null." << std::endl;
        return;
    }
    // Enqueue the command to write to the buffer
    err = clEnqueueWriteBuffer(this->GetCommandQueue(), buffer, CL_TRUE, 0, nbytes, host_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error in clEnqueueWriteBuffer: " << err << std::endl;
        return;
    }
    // Wait for the command to complete
    err = clFinish(this->GetCommandQueue());
    if (err != CL_SUCCESS) {
        std::cerr << "Error in clFinish: " << err << std::endl;
        return;
    }
}


// IAllocatorUniquePtrToClMem OpenCLExecutionProvider::GetScratchImage2D(const opencl::Image2DDesc& desc) const {
//   auto base_alloc = GetAllocator(0, (OrtMemType)opencl::CLMemType::OPENCL_IMAGE_2D);
//   auto* alloc = static_cast<opencl::OpenCLImage2DAllocator*>(base_alloc.get());
//   return IAllocatorUniquePtr<std::remove_pointer_t<cl_mem>>{
//       static_cast<cl_mem>(alloc->Alloc(desc)),
//       [=](void* ptr) {
//         alloc->Free(ptr);
//       }};
// }

Status OpenCLExecutionProvider::AfterCLLaunch() const {
  // We observe some performance boost on some devices. If you observe bubble
  // in command queue while host kernel launch is fast enough to throttle the
  // GPU during profiling, then you should consider flush the queue immediately
  // after the kernel launch.
  if (flush_after_launch_) {
    ORT_RETURN_IF_CL_ERROR(clFlush(cmd_queue_), "command queue flush failure.");
  }
  return Status::OK();
}

const opencl::OpenCLProgramManager& OpenCLExecutionProvider::GetProgramManager() const {
  return *program_manager_;
}

opencl::OpenCLProgramManager& OpenCLExecutionProvider::GetProgramManager() {
  return *program_manager_;
}

/*
#pragma region IDataTransfer related code
*/
std::unique_ptr<onnxruntime::IDataTransfer> OpenCLExecutionProvider::GetDataTransfer() const {
  return std::make_unique<opencl::OpenCLGPUDataTransfer>(this);
}

namespace {
#define CONTENT_NAME copy_tensors_src
#include "opencl_generated/kernels/copy_tensors.cl.inc"
}  // namespace

void OpenCLExecutionProvider::InitCopyKernels() {
  copy_kernels_ = std::make_unique<opencl::OpenCLKernelHolder>(GetProgramManager());
  copy_kernels_->LoadProgram(copy_tensors_src, copy_tensors_src_len);
  copy_kernels_->LoadKernel("Nop");
}
/*
#pragma endregion
*/

}  // namespace onnxruntime
