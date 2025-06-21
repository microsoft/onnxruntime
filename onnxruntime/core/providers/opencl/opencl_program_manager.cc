// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "opencl_program_manager.h"

#include "core/framework/murmurhash3.h"

namespace onnxruntime {
namespace opencl {

OpenCLKernelHolder::~OpenCLKernelHolder() {
  for (const auto& [_, kernel] : kernels_) {
    ORT_UNUSED_PARAMETER(_);
    mgr_->ReleaseKernel(kernel);
  }
  mgr_->ReleaseProgram(program_);
}

void OpenCLKernelHolder::LoadProgram(const char* src_body, size_t src_len) {
  LoadProgram({src_body, src_len});
}

void OpenCLKernelHolder::LoadProgram(std::string_view src_body) {
  program_ = mgr_->GetProgram(src_body);
}

void OpenCLKernelHolder::LoadKernel(std::string_view name) {
  cl_kernel kernel = mgr_->GetKernel(program_, name);
  kernels_[std::string{name}] = kernel;
}

cl_kernel OpenCLKernelHolder::GetKernel(std::string_view name) const {
  auto it = kernels_.find(std::string{name});
  if (it != kernels_.end()) {
    return it->second;
  }
  ORT_THROW("Unable to find kernel ", name);
}

namespace {
#define CONTENT_NAME prelude_f16_src
#include "opencl_generated/kernels/prelude_f16.cl.inc"
#define CONTENT_NAME prelude_f32_src
#include "opencl_generated/kernels/prelude_f32.cl.inc"
}  // namespace

std::string GetFullSource(std::string_view src_body, bool use_fp16) {
  std::ostringstream oss;
  if (use_fp16) {
    oss << std::string_view{prelude_f16_src, prelude_f16_src_len} << "\n";
  } else {
    oss << std::string_view{prelude_f32_src, prelude_f32_src_len} << "\n";
  }
  oss << src_body;
  return oss.str();
}

Status CreateProgramWithSource(cl_context ctx, cl_device_id dev, std::string_view src,
                               const std::string& options, cl_program* program) {
  cl_int err{};
  const auto* data = src.data();
  const auto size = src.size();
  {
    *program = clCreateProgramWithSource(ctx, 1, &data, &size, &err);
  }
  ORT_RETURN_IF_CL_ERROR(err);

  // Specially handle this error, we need compiler error message here.
  {
    err = clBuildProgram(*program, 1, &dev, options.c_str(), nullptr, nullptr);
  }
  if (err != CL_SUCCESS) {
    size_t ret_size;
    clGetProgramBuildInfo(*program, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &ret_size);
    std::string log(ret_size + 1, '\0');
    clGetProgramBuildInfo(*program, dev, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
    LOGS_DEFAULT(ERROR) << "\nKernel Source:\n"
                        << src
                        << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^";
    LOGS_DEFAULT(ERROR) << "\nBuild Options:\n"
                        << options
                        << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^";
    LOGS_DEFAULT(ERROR) << "\nBuild Log:\n"
                        << log
                        << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^";
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "\nOpenCL Error Code  : ", static_cast<int>(err),
                           "\n       Error String: ", onnxruntime::opencl::GetErrorString(err));
  }
  return Status::OK();
}

Status LoadKernelFromProgram(cl_program program, std::string_view name, cl_kernel* kernel) {
  cl_int err{};
  {
    // TODO: try out clCreateKernelsInProgram for batch kernel creation, to see
    // if it will be faster.
    *kernel = clCreateKernel(program, std::string{name}.c_str(), &err);
  }
  ORT_RETURN_IF_CL_ERROR(err, "clCreateKernel failed to create kernel \"", name, "\" from program ", program);
  return Status::OK();
}

inline std::array<uint32_t, 4> GetProgramKeyFromFullSource(const std::string_view full_src) {
  std::array<uint32_t, 4> hash{0, 0, 0, 0};
  MurmurHash3::x86_128(full_src.data(), static_cast<int>(full_src.length()), hash[0], hash.data());
  return hash;
}

cl_program OpenCLProgramManager::GetProgram(std::string_view src_body) {
  auto full_src = GetFullSource(src_body, exec_->UseFp16());
  auto key = GetProgramKeyFromFullSource(full_src);

  const auto& it = program_registry_.find(key);
  if (it != program_registry_.cend()) {
    cl_program program = it->second;
    VLOGS_DEFAULT(V_GENERIC) << "Program " << program << " reused";
    RefProgram(program);
    return program;
  }

  // TODO: try out clCreateProgramWithBinary with disk cache to see if it  will
  // be faster.
  cl_program program;
  ORT_THROW_IF_ERROR(CreateProgramWithSource(exec_->GetOpenCLContext(), exec_->GetOpenCLDevice(), full_src,
                                             exec_->GetCompileOptions(), &program));
  VLOGS_DEFAULT(V_GENERIC) << "Program " << program << " created from source";
  TakeinProgram(key, program);
  return program;
}

void OpenCLProgramManager::ReleaseProgram(cl_program program) {
  auto rc = DerefProgram(program);
  if (rc == 0) {
    EvictProgram(program);
    ORT_THROW_IF_CL_ERROR(clReleaseProgram(program));
  }
}

cl_kernel OpenCLProgramManager::GetKernel(cl_program program, std::string_view kernel_name) {
  KernelKey key{program, kernel_name};
  const auto& it = kernel_registry_.find(key);
  if (it != kernel_registry_.cend()) {
    VLOGS_DEFAULT(V_GENERIC) << "Reusing kernel " << kernel_name << " of program " << program;
    cl_kernel kernel = it->second;
    RefKernel(kernel);
    return kernel;
  }

  VLOGS_DEFAULT(V_GENERIC) << "Loading kernel " << kernel_name << " from program " << program;
  cl_kernel kernel;
  ORT_THROW_IF_ERROR(LoadKernelFromProgram(program, kernel_name, &kernel));
  TakeinKernel(key, kernel);
  return kernel;
}

void OpenCLProgramManager::ReleaseKernel(cl_kernel kernel) {
  auto rc = DerefKernel(kernel);
  if (rc == 0) {
    EvictKernel(kernel);
    clReleaseKernel(kernel);
  }
}

void OpenCLProgramManager::SetLocalSizeToCache(const cl_kernel kernel, const NDRange& global, NDRange local) {
  local_size_cache_[kernel_meta_[kernel].key][global] = local;
}

std::optional<const NDRange> OpenCLProgramManager::GetLocalSizeFromCache(const cl_kernel kernel, const NDRange& global) {
  auto& f_key = kernel_meta_[kernel].key;
  if (!local_size_cache_.count(f_key) || !local_size_cache_[f_key].count(global)) {
    return std::nullopt;
  }
  return local_size_cache_[f_key][global];
}

void OpenCLProgramManager::TakeinProgram(ProgramKey key, cl_program program) {
  program_registry_[key] = program;
  ProgramMeta meta{};
  meta.key = key;
  meta.rc = 1;
  program_meta_[program] = meta;
}

void OpenCLProgramManager::EvictProgram(cl_program program) {
  auto& meta = program_meta_[program];
  ORT_ENFORCE(meta.rc == 0, "EvictProgram: invalid program reference counter");
  ORT_ENFORCE(meta.kernels.empty(), "kernels not evicted");
  program_registry_.erase(meta.key);
  program_meta_.erase(program);
}

void OpenCLProgramManager::RefProgram(cl_program program, ProgramMeta* meta) {
  if (meta == nullptr) {
    meta = &program_meta_[program];
  }
  meta->rc += 1;
}

int32_t OpenCLProgramManager::DerefProgram(cl_program program, ProgramMeta* meta) {
  if (meta == nullptr) {
    meta = &program_meta_[program];
  }
  meta->rc -= 1;
  ORT_ENFORCE(meta->rc >= 0, "DerefProgram: invalid program reference counter (rc=", meta->rc, ")");
  return meta->rc;
}

void OpenCLProgramManager::TakeinKernel(KernelKey key, cl_kernel kernel) {
  kernel_registry_[key] = kernel;
  KernelMeta kernel_meta{};
  kernel_meta.key = key;
  kernel_meta.rc = 1;
  kernel_meta_[kernel] = kernel_meta;
  auto* program_meta = &program_meta_[key.first];
  // ORT_ENFORCE(program_meta->kernels.find(kernel) == program_meta->kernels.end(), "kernel ", kernel, " is already managed by OpenCLProgramManager");
  program_meta->kernels.insert(kernel);
  RefProgram(key.first, program_meta);
}

void OpenCLProgramManager::EvictKernel(cl_kernel kernel) {
  auto& kernel_meta = kernel_meta_[kernel];
  ORT_ENFORCE(kernel_meta.rc == 0, "EvictKernel: invalid kernel reference counter (rc=", kernel_meta.rc, ")");
  cl_program program = kernel_meta.key.first;
  auto* program_meta = &program_meta_[program];
  program_meta->kernels.erase(kernel);
  DerefProgram(program, program_meta);
}

void OpenCLProgramManager::RefKernel(cl_kernel kernel, KernelMeta* meta) {
  if (meta == nullptr) {
    meta = &kernel_meta_[kernel];
  }
  meta->rc += 1;
}

int32_t OpenCLProgramManager::DerefKernel(cl_kernel kernel, KernelMeta* meta) {
  if (meta == nullptr) {
    meta = &kernel_meta_[kernel];
  }
  meta->rc -= 1;
  ORT_ENFORCE(meta->rc >= 0, "DerefKernel: invalid kernel reference counter (rc=", meta->rc, ")");
  return meta->rc;
}
}  // namespace opencl
}  // namespace onnxruntime
