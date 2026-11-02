// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.
#if !defined(ORT_MINIMAL_BUILD)
#include "python/onnxruntime_pybind_model_compiler.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iterator>
#include <limits>
#include "core/common/common.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/ortmemoryinfo.h"
#include "core/graph/abi_graph_types.h"
#include "core/session/utils.h"

namespace onnxruntime {
namespace python {

namespace {

OrtStatus* ToCallbackStatus(OrtErrorCode error_code, const char* prefix, const std::exception& ex) {
  if (error_code == ORT_INVALID_ARGUMENT) {
    return ToOrtStatus(ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, prefix, ex.what()));
  }

  return ToOrtStatus(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, prefix, ex.what()));
}

OrtStatus* ORT_API_CALL PyEpContextDataWriteFuncWrapper(void* state, const char* name,
                                                        const void* buffer, size_t buffer_size) {
  auto* py_func = static_cast<PyEpContextDataWriteFunc*>(state);

  try {
    pybind11::gil_scoped_acquire acquire;
    try {
      auto data = std::make_shared<PyEpContextData>(buffer, buffer_size);
      auto invalidate_data = gsl::finally([&data]() { data->Invalidate(); });
      (*py_func)(name, data);
      return nullptr;
    } catch (const std::exception& ex) {
      return ToCallbackStatus(ORT_FAIL, "Python EPContext data write callback failed: ", ex);
    }
  } catch (const std::exception& ex) {
    return ToCallbackStatus(ORT_FAIL, "Python EPContext data write callback failed: ", ex);
  }
}

}  // namespace

pybind11::bytes PyEpContextData::Read(size_t offset, std::optional<size_t> length) const {
  ORT_ENFORCE(valid_, "EPContext data is valid only during the write callback.");
  ORT_ENFORCE(offset <= size_, "EPContext data offset exceeds the payload size.");

  const size_t read_size = length.value_or(size_ - offset);
  ORT_ENFORCE(read_size <= size_ - offset, "EPContext data range exceeds the payload size.");
  ORT_ENFORCE(read_size <= static_cast<size_t>(std::numeric_limits<Py_ssize_t>::max()),
              "EPContext data range is too large for one Python bytes object. Read it in chunks.");
  ORT_ENFORCE(read_size == 0 || data_ != nullptr,
              "EPContext data has a null buffer for a non-empty range.");

  const char* begin = read_size == 0 ? "" : static_cast<const char*>(data_) + offset;
  return pybind11::bytes(begin, static_cast<Py_ssize_t>(read_size));
}

void PyEpContextData::Invalidate() noexcept {
  valid_ = false;
  data_ = nullptr;
}

PyEpContextDataBuffer::~PyEpContextDataBuffer() {
  Invalidate();
}

PyEpContextDataBuffer::PyEpContextDataBuffer(OrtAllocator* allocator, size_t max_data_size)
    : allocator_(allocator), max_data_size_(max_data_size) {
  const OrtMemoryInfo* memory_info = allocator_ == nullptr ? nullptr : allocator_->Info(allocator_);
  if (memory_info == nullptr || !memory_info->device.UsesCpuMemory()) {
    throw std::invalid_argument(
        "Python EPContext data callbacks require a CPU or HOST_ACCESSIBLE allocator.");
  }
}

void PyEpContextDataBuffer::Invalidate() noexcept {
  if (buffer_ != nullptr && allocator_ != nullptr) {
    allocator_->Free(allocator_, buffer_);
  }
  buffer_ = nullptr;
  allocator_ = nullptr;
  valid_ = false;
}

void PyEpContextDataBuffer::ThrowIfInvalid() const {
  ORT_ENFORCE(valid_, "EPContext data output is valid only during the read callback.");
}

void PyEpContextDataBuffer::Allocate(size_t size) {
  ThrowIfInvalid();
  if (allocated_) {
    throw std::invalid_argument("EPContext data output has already been allocated.");
  }
  if (size > max_data_size_) {
    throw std::invalid_argument("EPContext data exceeds the configured maximum size.");
  }

  void* buffer = size == 0 ? nullptr : allocator_->Alloc(allocator_, size);
  if (size != 0 && buffer == nullptr) {
    ORT_THROW("The ORT allocator failed to allocate the EPContext data output buffer.");
  }

  buffer_ = buffer;
  size_ = size;
  allocated_ = true;
}

void PyEpContextDataBuffer::Write(size_t offset, const pybind11::buffer& data) {
  ThrowIfInvalid();
  if (!allocated_) {
    throw std::invalid_argument("allocate() must be called before write().");
  }

  Py_buffer view{};
  if (PyObject_GetBuffer(data.ptr(), &view, PyBUF_CONTIG_RO) != 0) {
    throw pybind11::error_already_set();
  }
  auto release_view = gsl::finally([&view]() { PyBuffer_Release(&view); });

  if (view.len < 0) {
    throw std::invalid_argument("Python buffer has a negative size.");
  }
  const size_t data_size = static_cast<size_t>(view.len);
  if (offset > size_ || data_size > size_ - offset) {
    throw std::invalid_argument("Python buffer does not fit in the allocated EPContext data output range.");
  }
  if (data_size != 0) {
    std::memcpy(static_cast<char*>(buffer_) + offset, view.buf, data_size);
  }
}

void PyEpContextDataBuffer::Detach(void*& buffer, size_t& size) {
  ThrowIfInvalid();
  if (!allocated_) {
    throw std::invalid_argument("The EPContext data read callback must allocate its output buffer.");
  }

  buffer = buffer_;
  size = size_;
  buffer_ = nullptr;
  allocator_ = nullptr;
  valid_ = false;
}

OrtStatus* ORT_API_CALL PyEpContextDataReadFuncWrapper(void* state, const char* name, OrtAllocator* allocator,
                                                       void** buffer, size_t* data_size) {
  auto* registration = static_cast<PyEpContextDataReadRegistration*>(state);
  *buffer = nullptr;
  *data_size = 0;

  try {
    pybind11::gil_scoped_acquire acquire;
    try {
      auto output = std::make_shared<PyEpContextDataBuffer>(allocator, registration->max_data_size);
      auto invalidate_output = gsl::finally([&output]() { output->Invalidate(); });
      registration->read_func(name, output);
      output->Detach(*buffer, *data_size);
      return nullptr;
    } catch (const std::invalid_argument& ex) {
      return ToCallbackStatus(ORT_INVALID_ARGUMENT, "Python EPContext data read callback failed: ", ex);
    } catch (const std::exception& ex) {
      return ToCallbackStatus(ORT_FAIL, "Python EPContext data read callback failed: ", ex);
    }
  } catch (const std::exception& ex) {
    return ToCallbackStatus(ORT_FAIL, "Python EPContext data read callback failed: ", ex);
  }
}

/// <summary>
/// This function is called by ORT to allow the user to handle where every initializer is stored
/// (i.e., externally or internally). This function wraps (and calls) the actual Python function
/// provided by the user.
/// </summary>
/// <param name="state">Opaque state that holds a pointer to the user's Python function.</param>
/// <param name="initializer_name">The name of the initializer to handle.</param>
/// <param name="initializer_value">The OrtValue with the initializer's data, type, and shape.</param>
/// <param name="external_info">The original external location of the initializer, if any. May be null.</param>
/// <param name="new_external_info">Output parameter set to the initializer's new external location. Function may
/// return NULL if the initializer should be stored within the compiled ONNX model.</param>
/// <returns>A status indicating success or an error.</returns>
static OrtStatus* ORT_API_CALL PyGetInitializerLocationFuncWrapper(
    void* state,
    const char* initializer_name,
    const OrtValue* initializer_value,
    const OrtExternalInitializerInfo* external_info,
    /*out*/ OrtExternalInitializerInfo** new_external_info) {
  PyGetInitializerLocationFunc* py_func = reinterpret_cast<PyGetInitializerLocationFunc*>(state);
  OrtStatus* status = nullptr;
  *new_external_info = nullptr;

  // Call the Python function and convert any exceptions to a status.
  ORT_TRY {
    pybind11::gil_scoped_acquire acquire;
    auto py_new_external_info = (*py_func)(initializer_name, *initializer_value, external_info);
    if (py_new_external_info) {
      // ORT expects to take ownership of the new external info, so make a copy because other Python code
      // may be holding a reference to the `py_new_external_info`.
      auto py_result_copy = std::make_unique<OrtExternalInitializerInfo>(*py_new_external_info.get());
      *new_external_info = py_result_copy.release();
    }
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ToOrtStatus(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what()));
    });
  }

  return status;
}

onnxruntime::Status PyModelCompiler::Create(/*out*/ std::unique_ptr<PyModelCompiler>& out,
                                            onnxruntime::Environment& env,
                                            const PySessionOptions& sess_options,
                                            std::string&& input_model_path_or_bytes, bool input_model_is_path,
                                            bool embed_compiled_data_into_model,
                                            const std::string& external_initializers_file_path,
                                            size_t external_initializers_size_threshold,
                                            uint32_t flags,
                                            GraphOptimizationLevel graph_optimization_level,
                                            const PyGetInitializerLocationFunc& py_get_initializer_location_func) {
  auto sess_options_snapshot = CreatePySessionOptionsSnapshot(sess_options);
  auto model_compiler = std::make_unique<PyModelCompiler>(
      env, sess_options_snapshot.options,
      std::move(sess_options_snapshot.py_ep_selection_registration),
      std::move(sess_options_snapshot.py_ep_context_data_read_registration),
      py_get_initializer_location_func, PrivateConstructorTag{});
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

  if (model_compiler->py_get_initializer_location_func_) {
    compile_options.SetOutputModelGetInitializerLocationFunc(
        PyGetInitializerLocationFuncWrapper,
        reinterpret_cast<void*>(&model_compiler->py_get_initializer_location_func_));
  }

  out = std::move(model_compiler);
  return Status::OK();
}

onnxruntime::Status PyModelCompiler::BeginCompilation() {
  std::lock_guard<std::mutex> lock{compilation_mutex_};
  ORT_RETURN_IF(compilation_in_progress_, "Compilation is already in progress for this ModelCompiler.");
  compilation_in_progress_ = true;
  return Status::OK();
}

void PyModelCompiler::EndCompilation() noexcept {
  std::lock_guard<std::mutex> lock{compilation_mutex_};
  compilation_in_progress_ = false;
}

onnxruntime::Status PyModelCompiler::CompileToFile(const std::string& output_model_path) {
  ORT_RETURN_IF_ERROR(BeginCompilation());
  auto finish_compilation = gsl::finally([this]() { EndCompilation(); });

  ORT_RETURN_IF_ERROR(model_compile_options_.SetOutputModelPath(output_model_path));
  Status status;
  {
    pybind11::gil_scoped_release release;
    status = onnxruntime::CompileModel(env_, model_compile_options_);
  }
  ORT_RETURN_IF_ERROR(status);
  return Status::OK();
}

onnxruntime::Status PyModelCompiler::CompileToBytes(std::string& output_buffer) {
  ORT_RETURN_IF_ERROR(BeginCompilation());
  auto finish_compilation = gsl::finally([this]() { EndCompilation(); });

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
  Status status;
  {
    pybind11::gil_scoped_release release;
    status = onnxruntime::CompileModel(env_, model_compile_options_);
  }
  ORT_RETURN_IF_ERROR(status);

  // Copy into output buffer.
  output_buffer.reserve(buffer_size);
  gsl::span<char> src(reinterpret_cast<char*>(buffer_data), buffer_size);
  std::copy(src.begin(), src.end(), std::back_inserter(output_buffer));
  return Status::OK();
}

/// <summary>
/// Function called by ORT to allow the user to write out the compiled ONNX model bytes to a custom output stream.
/// This function wraps (and calls) the actual Python function provided by the user.
/// </summary>
/// <param name="stream_state">Opaque state that holds a pointer to the user's Python function.</param>
/// <param name="buffer">The buffer to write out. Contains a portion of the compiled ONNX model's bytes.</param>
/// <param name="buffer_num_bytes">The number of bytes in the buffer.</param>
/// <returns>A status indicating success or an error.</returns>
static OrtStatus* ORT_API_CALL PyOutStreamWriteFuncWrapper(void* stream_state, const void* buffer,
                                                           size_t buffer_num_bytes) {
  PyOutStreamWriteFunc* py_write_func = reinterpret_cast<PyOutStreamWriteFunc*>(stream_state);
  OrtStatus* status = nullptr;

  // Call the Python write function and convert any exceptions to a status.
  ORT_TRY {
    pybind11::gil_scoped_acquire acquire;
    pybind11::bytes py_bytes(reinterpret_cast<const char*>(buffer), buffer_num_bytes);
    (*py_write_func)(py_bytes);
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ToOrtStatus(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what()));
    });
  }

  return status;
}

onnxruntime::Status PyModelCompiler::CompileToOutStream(PyOutStreamWriteFunc& write_func) {
  ORT_RETURN_IF_ERROR(BeginCompilation());
  auto finish_compilation = gsl::finally([this]() { EndCompilation(); });

  model_compile_options_.SetOutputModelWriteFunc(PyOutStreamWriteFuncWrapper,
                                                 reinterpret_cast<void*>(&write_func));
  Status status;
  {
    pybind11::gil_scoped_release release;
    status = onnxruntime::CompileModel(env_, model_compile_options_);
  }
  ORT_RETURN_IF_ERROR(status);
  return Status::OK();
}

void PyModelCompiler::SetEpContextDataWriteFunc(PyEpContextDataWriteFunc write_func) {
  ORT_ENFORCE(static_cast<bool>(write_func), "EPContext data write callback must not be None.");
  PyEpContextDataWriteFunc old_write_func;
  {
    std::lock_guard<std::mutex> lock{compilation_mutex_};
    ORT_ENFORCE(!compilation_in_progress_,
                "Cannot change the EPContext data write callback while compilation is in progress.");
    old_write_func = std::move(py_ep_context_data_write_func_);
    py_ep_context_data_write_func_ = std::move(write_func);
    model_compile_options_.SetEpContextDataWriteFunc(PyEpContextDataWriteFuncWrapper,
                                                     &py_ep_context_data_write_func_);
  }
}

void PyModelCompiler::ClearEpContextDataWriteFunc() {
  PyEpContextDataWriteFunc old_write_func;
  {
    std::lock_guard<std::mutex> lock{compilation_mutex_};
    ORT_ENFORCE(!compilation_in_progress_,
                "Cannot change the EPContext data write callback while compilation is in progress.");
    model_compile_options_.SetEpContextDataWriteFunc(nullptr, nullptr);
    old_write_func = std::move(py_ep_context_data_write_func_);
  }
}

PyModelCompiler::PyModelCompiler(onnxruntime::Environment& env, const OrtSessionOptions& sess_options,
                                 std::shared_ptr<PyEpSelectionRegistration> py_ep_selection_registration,
                                 std::shared_ptr<PyEpContextDataReadRegistration> py_ep_context_data_read_registration,
                                 const PyGetInitializerLocationFunc& py_get_initializer_location_func,
                                 PrivateConstructorTag)
    : env_(env),
      py_ep_selection_registration_(std::move(py_ep_selection_registration)),
      py_ep_context_data_read_registration_(std::move(py_ep_context_data_read_registration)),
      py_get_initializer_location_func_(py_get_initializer_location_func),
      model_compile_options_(env, sess_options) {
}
}  // namespace python
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
