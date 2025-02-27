// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/murmurhash3.h"

#include "opencl_forward_decl.h"
#include "opencl_utils.h"
#include "opencl_execution_provider.h"
#include "absl/hash/hash.h"

#include <string_view>

namespace onnxruntime {
namespace opencl {

using ProgramKey = std::array<uint32_t, 4>;
using KernelKey = std::pair<cl_program, std::string>;

}  // namespace opencl
}  // namespace onnxruntime

template <>
struct std::hash<onnxruntime::opencl::ProgramKey> {
  size_t operator()(const onnxruntime::opencl::ProgramKey& key) const {
    std::hash<uint32_t> h{};
    size_t v = 0;
    v = onnxruntime::opencl::HashCombine(v, h(key[0]));
    v = onnxruntime::opencl::HashCombine(v, h(key[1]));
    v = onnxruntime::opencl::HashCombine(v, h(key[2]));
    v = onnxruntime::opencl::HashCombine(v, h(key[3]));
    return v;
  }
};

template <>
struct std::hash<onnxruntime::opencl::KernelKey> {
  size_t operator()(const onnxruntime::opencl::KernelKey& kernel_key) const {
    auto a = std::hash<size_t>{}(reinterpret_cast<size_t>(kernel_key.first));
    auto b = std::hash<std::string>{}(kernel_key.second);
    return onnxruntime::opencl::HashCombine(a, b);
  }
};

namespace onnxruntime {
namespace opencl {

class OpenCLKernelHolder {
 public:
  explicit OpenCLKernelHolder(OpenCLProgramManager& mgr) : mgr_{&mgr} {}
  ~OpenCLKernelHolder();

  void LoadProgram(const char* src_body, size_t src_len);

  void LoadProgram(std::string_view src_body);

  void LoadKernel(std::string_view kernel_name);

  cl_kernel GetKernel(std::string_view kernel_name) const;

 private:
  OpenCLProgramManager* mgr_;
  cl_program program_;
  std::unordered_map<std::string, cl_kernel> kernels_;
};

/*

Why not use C++ wrapper to deal with cl::Program and cl::Kernel lifetime mgmt?

We need to cache program and kerenl instance for reusing (compiling on mobile
platform is slow), that is, we are holding at least one reference to each of
these objects. But there is no way to reliablely query the value of reference
counter[1]. We also need a central manager for identifing the program by source
and retrieving the already created program instance. So we take over the mgmt
all at once.

[1]: The reference count returned should be considered immediately stale. It is
unsuitable for general use in applications. This feature is provided for
identifying memory leaks.

Lifetime characteristics on clReleaseProgram and clReleaseKernel

- cl_program

  The program reference count is decremented. The program object is deleted
  after all kernel objects associated with program have been deleted and the
  program reference count becomes zero.

- cl_kernel

  The kernel object is deleted once the number of instances that are retained
  to kernel become zero and the kernel object is no longer needed by any
  enqueued commands that use kernel. Using this function to release a reference
  that was not obtained by creating the object or by calling clRetainKernel
  causes undefined behavior.

*/

// TODO: make it thread safe?
class OpenCLProgramManager {
 public:
  OpenCLProgramManager() = delete;
  explicit OpenCLProgramManager(OpenCLExecutionProvider& exec) : exec_{&exec} {}
  ~OpenCLProgramManager() {}  // FIXME: release all managed kernels and programs
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(OpenCLProgramManager);

  cl_program GetProgram(std::string_view src_body);
  void ReleaseProgram(cl_program program);
  cl_kernel GetKernel(cl_program program, std::string_view kernel_name);
  void ReleaseKernel(cl_kernel kernel);
  void SetLocalSizeToCache(const cl_kernel kernel, const NDRange& global, NDRange local);
  std::optional<const NDRange> GetLocalSizeFromCache(const cl_kernel kernel, const NDRange& global);

 private:
  struct ProgramMeta {
    ProgramKey key;
    int32_t rc;
    std::unordered_set<cl_kernel> kernels;
  };

  struct KernelMeta {
    KernelKey key;
    int32_t rc;
  };

  void TakeinProgram(ProgramKey key, cl_program program);
  void EvictProgram(cl_program program);
  void RefProgram(cl_program program, ProgramMeta* meta = nullptr);
  int32_t DerefProgram(cl_program program, ProgramMeta* meta = nullptr);
  void TakeinKernel(KernelKey, cl_kernel kernel);
  void EvictKernel(cl_kernel kernel);
  void RefKernel(cl_kernel kernel, KernelMeta* meta = nullptr);
  int32_t DerefKernel(cl_kernel kernel, KernelMeta* meta = nullptr);

  OpenCLExecutionProvider* exec_;
  std::unordered_map<ProgramKey, cl_program> program_registry_;
  std::unordered_map<cl_program, ProgramMeta> program_meta_;
  std::unordered_map<KernelKey, cl_kernel> kernel_registry_;
  std::unordered_map<cl_kernel, KernelMeta> kernel_meta_;

  std::unordered_map<KernelKey, std::unordered_map<NDRange, NDRange, absl::Hash<NDRange>>> local_size_cache_;
};

}  // namespace opencl
}  // namespace onnxruntime
