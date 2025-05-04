// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.
#include "python/onnxruntime_pybind_model_compiler.h"

#include <cassert>
#include "core/common/common.h"
#include "core/framework/error_code_helper.h"
#include "core/session/utils.h"

namespace onnxruntime {
namespace python {

onnxruntime::Status PyModelCompiler::Create(/*out*/ std::unique_ptr<PyModelCompiler>& out,
                                            std::shared_ptr<onnxruntime::Environment> env,
                                            const PySessionOptions& sess_options,
                                            std::string&& input_model_path_or_bytes, bool input_model_is_path,
                                            bool embed_compiled_data_into_model,
                                            const std::string& external_initializers_file_path,
                                            size_t external_initializers_size_threshold) {
  auto model_compiler = std::make_unique<PyModelCompiler>(env, sess_options, PrivateConstructorTag{});
  ModelCompilationOptions& compile_options = model_compiler->model_compile_options_;

  if (input_model_is_path) {
    compile_options.SetInputModelPath(input_model_path_or_bytes);
  } else {
    model_compiler->input_model_bytes_ = std::move(input_model_path_or_bytes);
    compile_options.SetInputModelFromBuffer(reinterpret_cast<const void*>(model_compiler->input_model_bytes_.data()),
                                            model_compiler->input_model_bytes_.size());
  }

  ORT_RETURN_IF_ERROR(compile_options.SetEpContextEmbedMode(embed_compiled_data_into_model));

  if (!external_initializers_file_path.empty()) {
    compile_options.SetOutputModelExternalInitializersFile(external_initializers_file_path,
                                                           external_initializers_size_threshold);
  }

  out = std::move(model_compiler);
  return Status::OK();
}

onnxruntime::Status PyModelCompiler::CompileToFile(const std::string& output_model_path) {
  ORT_RETURN_IF_ERROR(model_compile_options_.SetOutputModelPath(output_model_path));
  ORT_RETURN_IF_ERROR(model_compile_options_.Check());
  ORT_RETURN_IF_ERROR(onnxruntime::CompileModel(*env_, model_compile_options_));
  return Status::OK();
}

struct AllocatorOverString : public OrtAllocator {
  AllocatorOverString(std::string& backing_buffer)
      : backing_buffer_(backing_buffer), memory_info_(CPU, OrtAllocatorType::OrtDeviceAllocator) {
    OrtAllocator::version = ORT_API_VERSION;
    OrtAllocator::Alloc = [](OrtAllocator* this_, size_t size) {
      return static_cast<AllocatorOverString*>(this_)->Alloc(size);
    };
    OrtAllocator::Free = [](OrtAllocator* this_, void* p) {
      static_cast<AllocatorOverString*>(this_)->Free(p);
    };
    OrtAllocator::Info = [](const OrtAllocator* this_) -> const OrtMemoryInfo* {
      return static_cast<const AllocatorOverString*>(this_)->Info();
    };
    OrtAllocator::Reserve = [](OrtAllocator* this_, size_t size) {
      return static_cast<AllocatorOverString*>(this_)->Alloc(size);
    };
  }

  ~AllocatorOverString() {
  }

  void* Alloc(size_t size) {
    backing_buffer_.resize(size);
    return reinterpret_cast<void*>(backing_buffer_.data());
  }

  void Free(void* p) {
    ORT_UNUSED_PARAMETER(p);
    backing_buffer_.clear();
  }

  const OrtMemoryInfo* Info() const { return &memory_info_; }

 private:
  AllocatorOverString(const AllocatorOverString&) = delete;
  AllocatorOverString& operator=(const AllocatorOverString&) = delete;

  std::string& backing_buffer_;
  OrtMemoryInfo memory_info_;
};

onnxruntime::Status PyModelCompiler::CompileToBytes(std::string& output_buffer) {
  AllocatorOverString allocator(output_buffer);

  void* tmp_buffer_data = nullptr;
  size_t tmp_buffer_size = 0;
  ORT_RETURN_IF_ERROR(model_compile_options_.SetOutputModelBuffer(&allocator, &tmp_buffer_data, &tmp_buffer_size));
  ORT_RETURN_IF_ERROR(model_compile_options_.Check());
  ORT_RETURN_IF_ERROR(onnxruntime::CompileModel(*env_, model_compile_options_));
  assert(tmp_buffer_data == reinterpret_cast<void*>(output_buffer.data()));
  assert(tmp_buffer_size == output_buffer.size());
  return Status::OK();
}

PyModelCompiler::PyModelCompiler(std::shared_ptr<onnxruntime::Environment> env, const PySessionOptions& sess_options,
                                 PrivateConstructorTag)
    : env_(env), model_compile_options_(*env, sess_options) {
}
}  // namespace python
}  // namespace onnxruntime
