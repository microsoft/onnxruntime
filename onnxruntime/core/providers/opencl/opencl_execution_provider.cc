// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "opencl_execution_provider.h"
#include "opencl_allocator.h"
#include "opencl_data_transfer.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"

#include <array>
#include <utility>

// Add includes of kernel implementations
#include "memcpy_kernel.h"
#include "core/providers/opencl/math/elementwise.h"

namespace onnxruntime {
namespace opencl {

Status RegisterOpenCLKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Div)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    std::cout << "RegisterOpenCLKernels...\n";
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      std::cout << " register kernel name: " << info.kernel_def->OpName() << ", domain: " << info.kernel_def->Domain() << "\n";
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
    : IExecutionProvider(kOpenCLExecutionProvider) {
  Status status;
  status = InitOpenCLContext();
  if (!status.IsOK()) {
    // FIXME:
  }
}

OpenCLExecutionProvider::OpenCLExecutionProvider(OpenCLExecutionProvider&& provider) noexcept
    : IExecutionProvider(kOpenCLExecutionProvider) {
  std::swap(dev_, provider.dev_);
  std::swap(ctx_, provider.ctx_);
  std::swap(cmd_queue_, provider.cmd_queue_);
}

OpenCLExecutionProvider::~OpenCLExecutionProvider() = default;

std::shared_ptr<KernelRegistry> OpenCLExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = opencl::GetOpenCLKernelRegistry();
  return kernel_registry;
}

Status OpenCLExecutionProvider::InitOpenCLContext() {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  std::cerr << "num platforms:" << platforms.size() << "\n";
  ORT_ENFORCE(!platforms.empty());
  // FIXME: add platform selection logic
  auto selected_platform = platforms[0];
  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(selected_platform)(), 0};
  cl_int err{};
  ctx_ = cl::Context(CL_DEVICE_TYPE_GPU, properties, /*notifyFptr=*/nullptr, /*data=*/nullptr, &err);
  OPENCL_CHECK_ERROR(err);
  std::cerr << "created cl::Context(" << ctx_() << ")\n";

  std::vector<cl::Device> devices = ctx_.getInfo<CL_CONTEXT_DEVICES>();
  std::cout << "num devices:" << devices.size() << std::endl;
  ORT_ENFORCE(!devices.empty());
  // FIXME: add device selection logic
  dev_ = std::move(devices[0]);

  cmd_queue_ = cl::CommandQueue(ctx_, dev_, /*properties=*/0, &err);
  OPENCL_CHECK_ERROR(err);
  std::cerr << "created cl::CommandQueue(" << cmd_queue_() << ") in cl::Context(" << ctx_() << ")\n";

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
      [ctx = this->ctx_](int) {
        return std::make_unique<opencl::OpenCLAllocator>(ctx);
      },
      0,
      /*use_arena=*/false,
  }));

  InsertAllocator(CreateAllocator(AllocatorCreationInfo{
      [](int) {
        return std::make_unique<CPUAllocator>(
            OrtMemoryInfo(opencl::CPUAllocatorName, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput));
      }}));
}

std::unique_ptr<onnxruntime::IDataTransfer> OpenCLExecutionProvider::GetDataTransfer() const {
  return std::make_unique<opencl::OpenCLDataTransfer>(cmd_queue_);
}

}  // namespace onnxruntime
