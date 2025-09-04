// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <string>
#include "core/common/status.h"
#include "core/session/model_compilation_options.h"
#include "python/onnxruntime_pybind_state_common.h"

namespace onnxruntime {
class Environment;

namespace python {
// Type of the function provided by Python code that is called by ORT to write out the compiled model.
using PyOutStreamWriteFunc = std::function<void(const pybind11::bytes& buffer)>;

// Type of the function provided by Python code that is called by ORT to handle every initializer.
using PyGetInitializerLocationFunc = std::function<std::shared_ptr<const OrtExternalInitializerInfo>(
    const std::string& initializer_name,
    const OrtValue& initializer_value,
    const OrtExternalInitializerInfo* external_info)>;

/// <summary>
/// Class exposed to Python that enables compiling ONNX models.
/// Internally wraps a onnxruntime::ModelCompilationOptions that stores and validates settings.
/// </summary>
class PyModelCompiler {
#if defined(ORT_MINIMAL_BUILD)
 public:
  bool not_defined_in_this_build{};  // Prevent empty class warning.
#else
 private:
  // private tag to pass to constructor to ensure that constructor cannot be directly called externally
  struct PrivateConstructorTag {};

 public:
  /// <summary>
  /// Static class function that creates a unique_ptr<PyModelCompiler> with the given settings.
  /// </summary>
  /// <param name="out">Output parameter for the result</param>
  /// <param name="env">The Environment instance</param>
  /// <param name="sess_options">The SessionOptions from which to initialize compilation options.</param>
  /// <param name="input_model_path_or_bytes">An r-value string that could be the input model's path or bytes</param>
  /// <param name="input_model_is_path">True if 'input_model_path_or_bytes' is a path, and false if its bytes.</param>
  /// <param name="embed_compiled_data_into_model">True to embed compiled binary data into EPContext nodes.</param>
  /// <param name="external_initializers_file_path">The file into which to store initializers for non-compiled
  /// nodes.</param>
  /// <param name="external_initializers_size_threshold">Ignored if 'external_initializers_file_path' is empty.
  /// Initializers with a size greater than this threshold are dumped into the external file.</param>
  /// <param name="flags">Flags from OrtCompileApiFlags</param>
  /// <param name="graph_opt_level">Optimization level for graph transformations on the model.
  /// Defaults to ORT_DISABLE_ALL to allow EP to get the original loaded model.</param>
  /// <param name="py_get_initializer_location_func">User's function to handle saving of initializers.</param>
  /// <returns>A Status indicating error or success.</returns>
  static onnxruntime::Status Create(/*out*/ std::unique_ptr<PyModelCompiler>& out,
                                    onnxruntime::Environment& env,
                                    const PySessionOptions& sess_options,
                                    std::string&& input_model_path_or_bytes, bool input_model_is_path,
                                    bool embed_compiled_data_into_model = false,
                                    const std::string& external_initializers_file_path = {},
                                    size_t external_initializers_size_threshold = 1024,
                                    uint32_t flags = 0,
                                    GraphOptimizationLevel graph_opt_level = GraphOptimizationLevel::ORT_DISABLE_ALL,
                                    const PyGetInitializerLocationFunc& py_get_initializer_location_func = nullptr);

  // Note: Creation should be done via Create(). This constructor is public so that it can be called from
  // std::make_shared().
  PyModelCompiler(onnxruntime::Environment& env, const PySessionOptions& sess_options,
                  const PyGetInitializerLocationFunc& py_get_initializer_location_func,
                  PrivateConstructorTag);

  /// <summary>
  /// Compiles the input model and saves the result to an output file.
  /// If the 'output_model_path' is not specified,
  /// it is generated based on the input model's path by replacing '.onnx' with '_ctx.onnx'.
  /// </summary>
  /// <param name="output_model_path">The path into which to save the compiled model.</param>
  /// <returns>A Status indicating error or success.</returns>
  onnxruntime::Status CompileToFile(const std::string& output_model_path = {});

  /// <summary>
  /// Compiles the input model and stores the result into a buffer.
  /// </summary>
  /// <param name="output_buffer">A reference to the output buffer into which to store the
  /// serialized ONNX model bytes.</param>
  /// <returns>A Status indicating error or success.</returns>
  onnxruntime::Status CompileToBytes(std::string& output_buffer);

  /// <summary>
  /// Compiles the input model and writes the result into the provided output stream (write functor).
  /// </summary>
  /// <param name="write_func">Write functor that encapsulates the stream's state.</param>
  /// <returns>A Status indicating error or success.</returns>
  onnxruntime::Status CompileToOutStream(PyOutStreamWriteFunc& write_func);

 private:
  onnxruntime::Environment& env_;
  onnxruntime::ModelCompilationOptions model_compile_options_;
  std::string input_model_bytes_;
  PyGetInitializerLocationFunc py_get_initializer_location_func_;
#endif  // defined(ORT_MINIMAL_BUILD)
};
}  // namespace python
}  // namespace onnxruntime
