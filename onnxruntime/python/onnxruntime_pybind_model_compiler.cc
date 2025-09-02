// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.
#include "python/onnxruntime_pybind_model_compiler.h"

#include <algorithm>
#include <cassert>
#include <iterator>
#include "core/common/common.h"
#include "core/framework/error_code_helper.h"
#include "core/session/utils.h"

namespace onnxruntime {
namespace python {

onnxruntime::Status PyModelCompiler::Create(/*out*/ std::unique_ptr<PyModelCompiler>& out,
                                            onnxruntime::Environment& env,
                                            const PySessionOptions& sess_options,
                                            std::string&& input_model_path_or_bytes, bool input_model_is_path,
                                            bool embed_compiled_data_into_model,
                                            const std::string& external_initializers_file_path,
                                            size_t external_initializers_size_threshold,
                                            size_t flags,
                                            GraphOptimizationLevel graph_optimization_level) {
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

  if (flags != 0) {
    ORT_RETURN_IF_ERROR(compile_options.SetFlags(flags));
  }

  ORT_RETURN_IF_ERROR(compile_options.SetGraphOptimizationLevel(graph_optimization_level));

  out = std::move(model_compiler);
  return Status::OK();
}

onnxruntime::Status PyModelCompiler::CompileToFile(const std::string& output_model_path) {
  ORT_RETURN_IF_ERROR(model_compile_options_.SetOutputModelPath(output_model_path));
  ORT_RETURN_IF_ERROR(onnxruntime::CompileModel(env_, model_compile_options_));
  return Status::OK();
}

onnxruntime::Status PyModelCompiler::CompileToBytes(std::string& output_buffer) {
  if (!output_buffer.empty()) {
    // Opt to return an error if the output buffer is not empty instead of just calling output_buffer.clear()
    // because the C++ standard does not explicitly require that capacity is unchanged by a call to clear().
    // Don't want to reallocate a large buffer an extra time unnecessarily. So, we'll consider this an internal
    // ORT error.
    // Refer to: https://en.cppreference.com/w/cpp/string/basic_string/clear
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Output buffer should be empty.");
  }

  onnxruntime::AllocatorPtr allocator = std::make_shared<CPUAllocator>();

  void* buffer_data = nullptr;
  size_t buffer_size = 0;
  ORT_RETURN_IF_ERROR(model_compile_options_.SetOutputModelBuffer(allocator, &buffer_data, &buffer_size));
  ORT_RETURN_IF_ERROR(onnxruntime::CompileModel(env_, model_compile_options_));

  // Copy into output buffer.
  output_buffer.reserve(buffer_size);
  gsl::span<char> src(reinterpret_cast<char*>(buffer_data), buffer_size);
  std::copy(src.begin(), src.end(), std::back_inserter(output_buffer));
  return Status::OK();
}

PyModelCompiler::PyModelCompiler(onnxruntime::Environment& env, const PySessionOptions& sess_options,
                                 PrivateConstructorTag)
    : env_(env), model_compile_options_(env, sess_options) {
}
}  // namespace python
}  // namespace onnxruntime
