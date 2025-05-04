// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.
#pragma once

#if !defined(ORT_MINIMAL_BUILD)
#include <memory>
#include <string>
#include "core/common/status.h"
#include "core/session/model_compilation_options.h"
#include "python/onnxruntime_pybind_state_common.h"

namespace onnxruntime {
class Environment;

namespace python {
class PyModelCompiler {
 private:
  // private tag to pass to constructor to ensure that constructor cannot be directly called externally
  struct PrivateConstructorTag {};

 public:
  static onnxruntime::Status Create(/*out*/ std::unique_ptr<PyModelCompiler>& out,
                                    std::shared_ptr<onnxruntime::Environment> env,
                                    const PySessionOptions& sess_options,
                                    std::string&& input_model_path_or_bytes, bool input_model_is_path,
                                    bool embed_compiled_data_into_model = false,
                                    const std::string& external_initializers_file_path = {},
                                    size_t external_initializers_size_threshold = 1024);

  // Note: Creation should be done via Create(). This constructor is public so that it can be called from
  // std::make_shared().
  PyModelCompiler(std::shared_ptr<onnxruntime::Environment> env, const PySessionOptions& sess_options,
                  PrivateConstructorTag);

  onnxruntime::Status CompileToFile(const std::string& output_model_path = {});
  onnxruntime::Status CompileToBytes(std::string& output_buffer);

 private:
  std::shared_ptr<onnxruntime::Environment> env_;
  onnxruntime::ModelCompilationOptions model_compile_options_;
  std::string input_model_bytes_;
};
}  // namespace python
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
