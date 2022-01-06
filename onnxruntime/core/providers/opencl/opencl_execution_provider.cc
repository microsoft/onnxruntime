// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "opencl_execution_provider.h"
#include "opencl_allocator.h"
#include "opencl_kernel_holder.h"
#include "opencl_data_transfer.h"
#include "core/common/logging/logging.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"

#include <array>
#include <utility>

// Add includes of kernel implementations
#include "memcpy_kernel.h"
#include "core/providers/opencl/math/clip.h"
#include "core/providers/opencl/math/elementwise.h"
#include "core/providers/opencl/nn/conv_image2d.h"
#include "core/providers/opencl/nn/global_average_pool_image2d.h"
#include "core/providers/opencl/tensor/shape.h"

namespace onnxruntime {
namespace opencl {

Status RegisterOpenCLKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 1, 12, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, 14, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 15, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 6, Clip)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 1, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 1, 10, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 11, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kMSDomain, 1, FusedConv)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    VLOGS_DEFAULT(1) << "[CL] RegisterOpenCLKernels...";
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      VLOGS_DEFAULT(0) << "[CL]  register kernel name: " << info.kernel_def->OpName() << ", domain: " << info.kernel_def->Domain();
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

}  // namespace opencl

OpenCLExecutionProvider::OpenCLExecutionProvider(const OpenCLExecutionProviderInfo& info)
    : IExecutionProvider(kOpenCLExecutionProvider), use_fp16_(info.use_fp16) {
  Status status;
  ORT_THROW_IF_ERROR(InitOpenCLContext());
#ifdef TRACY_ENABLE
  tracy_cl_ctx_ = TracyCLContext(ctx_, dev_);
#endif
}

OpenCLExecutionProvider::OpenCLExecutionProvider(OpenCLExecutionProvider&& provider) noexcept
    : IExecutionProvider(kOpenCLExecutionProvider), use_fp16_(provider.use_fp16_) {
  std::swap(dev_, provider.dev_);
  std::swap(ctx_, provider.ctx_);
  std::swap(cmd_queue_, provider.cmd_queue_);
}

OpenCLExecutionProvider::~OpenCLExecutionProvider() {
#ifdef TRACY_ENABLE
  TracyCLCollect(tracy_cl_ctx_);
  TracyCLDestroy(tracy_cl_ctx_);
#endif

  clReleaseCommandQueue(cmd_queue_);
  clReleaseDevice(dev_);
  clReleaseContext(ctx_);
}

std::shared_ptr<KernelRegistry> OpenCLExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = opencl::GetOpenCLKernelRegistry();
  return kernel_registry;
}

Status OpenCLExecutionProvider::InitOpenCLContext() {
  cl_uint num_platforms;
  ORT_RETURN_IF_CL_ERROR(clGetPlatformIDs(0, nullptr, &num_platforms));
  VLOGS_DEFAULT(1) << "[CL] num platforms: " << num_platforms;
  ORT_ENFORCE(num_platforms > 0);

  std::vector<cl_platform_id> platforms(num_platforms);
  ORT_RETURN_IF_CL_ERROR(clGetPlatformIDs(platforms.size(), platforms.data(), nullptr));
  int selected_platform_idx = -1;
  // FIXME: add platform selection logic
  for (int i = 0; i < platforms.size(); i++) {
    size_t ret_size;
    ORT_RETURN_IF_CL_ERROR(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 0, nullptr, &ret_size));
    std::string vendor(ret_size, '\0');
    ORT_RETURN_IF_CL_ERROR(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, vendor.size(), vendor.data(), nullptr));
    std::cout << "[CL] platform vendor: " << vendor << "\n";
    if (vendor == "Oclgrind") {
      std::cout << "[CL] platform " << vendor << " selected" << "\n";
      selected_platform_idx = 1;
      break;
    }
  }
  if (selected_platform_idx == -1) {
    std::cout << "[CL] default platform selected"
              << "\n";
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
  VLOGS_DEFAULT(1) << "[CL] num devices: " << devices.size();
  ORT_ENFORCE(!devices.empty());
  dev_ = devices[0];

  auto GetDeviceInfo = [=](cl_device_info info_name) -> std::string {
    size_t ret_size;
    ORT_THROW_IF_CL_ERROR(clGetDeviceInfo(dev_, info_name, 0, nullptr, &ret_size));
    std::string ret(ret_size, '\0');
    ORT_THROW_IF_CL_ERROR(clGetDeviceInfo(dev_, info_name, ret.size(), ret.data(), nullptr));
    return ret;
  };

  // NOTE: use stdout for mobile
  // FIXME: use logger latter
  auto device_name = GetDeviceInfo(CL_DEVICE_NAME);
  std::cout << "[CL] device name: " << device_name << "\n";
  std::cout << "[CL] device vendor: " << GetDeviceInfo(CL_DEVICE_VENDOR) << "\n";
  std::cout << "[CL] device version: " << GetDeviceInfo(CL_DEVICE_VERSION) << "\n";
  auto exts = GetDeviceInfo(CL_DEVICE_EXTENSIONS);
  std::cout << "[CL] device extensions: " << exts << std::endl;
  bool has_fp16 = exts.find("cl_khr_fp16") != std::string::npos;
  if (!has_fp16 && UseFp16()) {
    std::cout << "[CL] FP16 is requested, but is not supported by the device!";
    DisableFp16();
  }
  flush_after_launch_ = ShouldFlushAfterLaunch(device_name);
  std::cout << "[CL] FP16: " << UseFp16() << "\n";
  std::cout << "[CL] clFlush after launch: " << flush_after_launch_ << "\n";

#ifdef TRACY_ENABLE
  cmd_queue_ = clCreateCommandQueue(ctx_, dev_, CL_QUEUE_PROFILING_ENABLE, &err);
#else
  cmd_queue_ = clCreateCommandQueue(ctx_, dev_, /*properties=*/0, &err);
#endif
  ORT_RETURN_IF_CL_ERROR(err);

  InitCopyKernels();

  return Status::OK();
}

void OpenCLExecutionProvider::RegisterAllocator(std::shared_ptr<AllocatorManager> allocator_manager) {
  // FIXME: Is it possible to use arena on OpenCL? cl_mem is opaque pointer in
  // OpenCL 1.2 and Shared Virtual Memory (SVM) is only available in OpenCL
  // 2.0, which still have limited support on a wide range of devices. Without
  // SVM we are unable to slice pre-allocated buffer, thus, unable to use it as
  // an arena.
  //
  // See https://stackoverflow.com/a/40951614
  InsertAllocator(CreateAllocator(AllocatorCreationInfo{
      [=](int) {
        return std::make_unique<opencl::OpenCLBufferAllocator>(this->ctx_);
      },
      0,
      /*use_arena=*/false,
  }));

  InsertAllocator(CreateAllocator(AllocatorCreationInfo{
      [=](int) {
        return std::make_unique<opencl::OpenCLImage2DAllocator>(this->ctx_, this->UseFp16());
      },
      0,
      /*use_arena=*/false,
  }));

  InsertAllocator(CreateAllocator(AllocatorCreationInfo{
      [](int) {
        return std::make_unique<CPUAllocator>(
            OrtMemoryInfo(opencl::CPUAllocatorName, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput));
      }}));

  InsertAllocator(CreateAllocator(AllocatorCreationInfo{
      [](int) {
        return std::make_unique<CPUAllocator>(
            OrtMemoryInfo(opencl::CPUInputAllocatorName, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUInput));
      }}));
}

IAllocatorUniquePtr<std::remove_pointer_t<cl_mem>> OpenCLExecutionProvider::GetScratchBuffer(size_t nbytes) const {
  auto alloc = GetAllocator(0, (OrtMemType)opencl::CLMemType::OPENCL_BUFFER);
  return IAllocatorUniquePtr<std::remove_pointer_t<cl_mem>>{
      static_cast<cl_mem>(alloc->Alloc(nbytes)),
      [=](void* ptr) {
        alloc->Free(ptr);
      }};
}

IAllocatorUniquePtr<std::remove_pointer_t<cl_mem>> OpenCLExecutionProvider::GetScratchImage2D(opencl::Image2DDesc desc) const {
  auto alloc = GetAllocator(0, (OrtMemType)opencl::CLMemType::OPENCL_IMAGE_2D);
  return IAllocatorUniquePtr<std::remove_pointer_t<cl_mem>>{
      static_cast<cl_mem>(alloc->Alloc(desc.AsTensorShape())),
      [=](void* ptr) {
        alloc->Free(ptr);
      }};
}

Status OpenCLExecutionProvider::AfterCLLaunch() const {
  if (flush_after_launch_) {
    ORT_RETURN_IF_CL_ERROR(clFlush(cmd_queue_), "command queue flush failure.");
  }
  return Status::OK();
}

/*
#pragma region IDataTransfer related code
*/
std::unique_ptr<onnxruntime::IDataTransfer> OpenCLExecutionProvider::GetDataTransfer() const {
  return std::make_unique<opencl::OpenCLDataTransfer>(this, copy_kernels_.get());
}

namespace {
#define CONTENT_NAME copy_tensors_src
#include "opencl_generated/kernels/copy_tensors.cl.inc"
}  // namespace

void OpenCLExecutionProvider::InitCopyKernels() {
  copy_kernels_ = std::make_unique<opencl::OpenCLKernelHolder>();
  copy_kernels_->LoadProgram(this, copy_tensors_src, copy_tensors_src_len);
  copy_kernels_->LoadKernel("CopyBuffer1DToImage2D");
  copy_kernels_->LoadKernel("CopyImage2DToBuffer1D");
  copy_kernels_->LoadKernel("CopyBufferNCHWToImage2D");
  copy_kernels_->LoadKernel("CopyImage2DToBufferNCHW");
  copy_kernels_->LoadKernel("Conv2DWeightBufferToImage");
  copy_kernels_->LoadKernel("CopyDepthwiseConvWeightBufferToImage");
}
/*
#pragma endregion
*/

bool OpenCLExecutionProvider::ShouldFlushAfterLaunch(const std::string& device_name) {
  return device_name.find("Mali") != std::string::npos;
}

}  // namespace onnxruntime
