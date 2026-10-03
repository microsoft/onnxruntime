// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)
#include <cstring>
#include <limits>
#include <memory>
#include <streambuf>
#include <utility>
#include "core/common/safeint.h"
#include "core/framework/ep_context_utils.h"
#include "core/framework/error_code_helper.h"
#include "core/graph/model_saving_options.h"
#include "core/platform/env.h"

#include "google/protobuf/io/zero_copy_stream_impl.h"

namespace onnxruntime {
namespace epctx {

namespace {

class CountingStreamBuf final : public std::streambuf {
 public:
  size_t Size() const noexcept { return size_; }

 protected:
  std::streamsize xsputn(const char*, std::streamsize count) override {
    if (count < 0 || static_cast<uintmax_t>(count) > std::numeric_limits<size_t>::max() - size_) {
      return 0;
    }

    size_ += static_cast<size_t>(count);
    return count;
  }

  int_type overflow(int_type value) override {
    if (traits_type::eq_int_type(value, traits_type::eof())) {
      return traits_type::not_eof(value);
    }

    if (size_ == std::numeric_limits<size_t>::max()) {
      return traits_type::eof();
    }

    ++size_;
    return value;
  }

 private:
  size_t size_ = 0;
};

class FixedBufferStreamBuf final : public std::streambuf {
 public:
  FixedBufferStreamBuf(void* buffer, size_t size) : buffer_{static_cast<char*>(buffer)}, size_{size} {}

 protected:
  std::streamsize xsputn(const char* data, std::streamsize count) override {
    if (count < 0 || static_cast<uintmax_t>(count) > size_ - position_) {
      return 0;
    }

    std::memcpy(buffer_ + position_, data, static_cast<size_t>(count));
    position_ += static_cast<size_t>(count);
    return count;
  }

  int_type overflow(int_type value) override {
    if (traits_type::eq_int_type(value, traits_type::eof())) {
      return traits_type::not_eof(value);
    }

    if (position_ == size_) {
      return traits_type::eof();
    }

    buffer_[position_++] = traits_type::to_char_type(value);
    return value;
  }

 private:
  char* buffer_;
  size_t size_;
  size_t position_ = 0;
};

void ApplyExternalInitializerAlignment(const ModelGenOptions& gen_options,
                                       ModelSavingOptions& saving_options) {
  const size_t alignment = gen_options.external_initializers_alignment;
  saving_options.align_offset = alignment != 0;
  if (alignment != 0) {
    saving_options.on_disk_alignment = SafeInt<int64_t>(alignment);
    saving_options.align_threshold = SafeInt<int64_t>(gen_options.external_initializers_alignment_threshold);
  }
}

Status SerializeExternalInitializersToBuffer(const Model& model,
                                             const ExternalInitializerBufferInfo& buffer_info,
                                             const ModelGenOptions& gen_options,
                                             ONNX_NAMESPACE::ModelProto& model_proto) {
  ModelSavingOptions saving_options{buffer_info.size_threshold};
  ApplyExternalInitializerAlignment(gen_options, saving_options);
  const std::filesystem::path logical_file_name{buffer_info.logical_file_name};

  CountingStreamBuf counting_buffer;
  std::ostream counting_stream{&counting_buffer};
  ONNX_NAMESPACE::ModelProto sizing_proto;
  ORT_RETURN_IF_ERROR(model.ToGraphProtoWithExternalInitializers(logical_file_name, saving_options,
                                                                 counting_stream, sizing_proto));
  ORT_RETURN_IF_NOT(counting_stream.good(), "Failed to compute external initializer buffer size");

  const size_t buffer_size = counting_buffer.Size();
  if (buffer_size == 0) {
    model_proto = std::move(sizing_proto);
    *buffer_info.buffer_ptr = nullptr;
    *buffer_info.buffer_size_ptr = 0;
    return Status::OK();
  }

  IAllocatorUniquePtr<void> buffer = IAllocator::MakeUniquePtr<void>(buffer_info.buffer_allocator, buffer_size);
  ORT_RETURN_IF_NOT(buffer != nullptr, "Failed to allocate external initializer buffer");

  FixedBufferStreamBuf output_buffer{buffer.get(), buffer_size};
  std::ostream output_stream{&output_buffer};
  ONNX_NAMESPACE::ModelProto output_proto;
  ORT_RETURN_IF_ERROR(model.ToGraphProtoWithExternalInitializers(logical_file_name, saving_options,
                                                                 output_stream, output_proto));
  ORT_RETURN_IF_NOT(output_stream.good(), "Failed to serialize external initializers to buffer");

  model_proto = std::move(output_proto);
  *buffer_info.buffer_ptr = buffer.release();
  *buffer_info.buffer_size_ptr = buffer_size;
  return Status::OK();
}

}  // namespace

// Serialize an EPContext model into a onnx::ModelProto.
Status EpContextModelToProto(const onnxruntime::Model& ep_context_model,
                             const std::filesystem::path& validated_model_path,
                             const epctx::ModelGenOptions& ep_context_gen_options,
                             /*out*/ ONNX_NAMESPACE::ModelProto& model_proto) {
  // Handle case where initializers are stored inline within the ONNX model.
  if (ep_context_gen_options.AreInitializersEmbeddedInOutputModel()) {
    // if no external ini file specified, set force_embed_external_ini to true to avoid intermediate file creation
    // and force all initializers embed into the ONNX file.
    ModelSavingOptions model_saving_options{/*size_threshold*/ SIZE_MAX};
    model_saving_options.force_embed_external_ini = true;

    model_proto = ep_context_model.ToGraphProtoWithExternalInitializers(std::filesystem::path{},
                                                                        validated_model_path,
                                                                        model_saving_options);
    return Status::OK();
  }

  // Handle case where initializers (with size > threshold) are stored in an external file.
  if (const epctx::ExternalInitializerFileInfo* ext_info = ep_context_gen_options.TryGetExternalInitializerFileInfo();
      ext_info != nullptr) {
    ModelSavingOptions model_saving_options{ext_info->size_threshold};
    ApplyExternalInitializerAlignment(ep_context_gen_options, model_saving_options);

    model_proto = ep_context_model.ToGraphProtoWithExternalInitializers(ext_info->file_path,
                                                                        validated_model_path,
                                                                        model_saving_options);
    return Status::OK();
  }

  if (const auto* buffer_info = ep_context_gen_options.TryGetExternalInitializerBufferInfo()) {
    return SerializeExternalInitializersToBuffer(ep_context_model, *buffer_info,
                                                 ep_context_gen_options, model_proto);
  }

  // Handle case where user specified a custom handler function that determines how each initializer is saved.
  if (const epctx::InitializerHandler* custom_handler = ep_context_gen_options.TryGetInitializerHandler();
      custom_handler != nullptr) {
    ORT_RETURN_IF_ERROR(ep_context_model.ToGraphProtoWithCustomInitializerHandling(
        custom_handler->handle_initializer_func,
        custom_handler->state,
        model_proto));
    return Status::OK();
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unexpected location for initializers while generating ",
                         validated_model_path);
}

// Validate the ep_context_path to make sure it is a file path and check whether the file exists already.
Status GetValidatedEpContextPath(const std::filesystem::path& ep_context_path,
                                 const std::filesystem::path& model_path,
                                 std::filesystem::path& context_cache_path,
                                 bool error_if_output_file_exists) {
  if (!ep_context_path.empty()) {
    context_cache_path = ep_context_path;
    if (!(context_cache_path.has_filename() && context_cache_path.extension() != "")) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "context_file_path should not point to a folder.");
    }
  } else if (!model_path.empty()) {
    auto pos = model_path.native().find_last_of(ORT_TSTR("."));
    if (pos != std::string::npos) {
      context_cache_path = model_path.native().substr(0, pos) + ORT_TSTR("_ctx.onnx");
    } else {
      context_cache_path = model_path.native() + ORT_TSTR("_ctx.onnx");
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Both ep_context_path and model_path are empty.");
  }

  if (std::filesystem::exists(context_cache_path) && error_if_output_file_exists) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to generate EP context model since the file '",
                           context_cache_path, "' exists already. Please remove the EP context model if you want to re-generate it.");
  }

  return Status::OK();
}

Status SaveModelProtoToLocation(ONNX_NAMESPACE::ModelProto& model_proto,
                                const epctx::ModelGenOptions& gen_options,
                                const std::filesystem::path& valid_output_model_path,
                                const logging::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);
  const epctx::BufferHolder* output_buffer_holder = gen_options.TryGetOutputModelBuffer();
  const epctx::BufferWriteFuncHolder* output_write_func_holder = gen_options.TryGetOutputModelWriteFunc();

  if (output_buffer_holder != nullptr) {
    // Write output model into a buffer ORT allocates for the user.
    size_t buffer_size = model_proto.ByteSizeLong();
    ORT_RETURN_IF(buffer_size > static_cast<size_t>(std::numeric_limits<int>::max()),
                  "Cannot serialize ONNX ModelProto larger than 2GB");

    AllocatorPtr allocator = output_buffer_holder->buffer_allocator;
    IAllocatorUniquePtr<void> buffer = IAllocator::MakeUniquePtr<void>(allocator, buffer_size);
    const bool ok = model_proto.SerializeToArray(buffer.get(), static_cast<int>(buffer_size));
    ORT_RETURN_IF(!ok, "Protobuf serialization failed when saving model to output buffer");

    *output_buffer_holder->buffer_size_ptr = buffer_size;
    *output_buffer_holder->buffer_ptr = buffer.release();
  } else if (output_write_func_holder != nullptr) {
    // Write output model to user's output stream.
    size_t buffer_size = model_proto.ByteSizeLong();
    ORT_RETURN_IF(buffer_size > static_cast<size_t>(std::numeric_limits<int>::max()),
                  "Cannot serialize ONNX ModelProto larger than 2GB");

    auto out_stream_buf = std::make_unique<epctx::OutStreamBuf>(*output_write_func_holder);
    std::ostream out_stream(out_stream_buf.get());

    model_proto.SerializeToOstream(&out_stream);
    out_stream.flush();
    ORT_RETURN_IF_ERROR(out_stream_buf->GetStatus());
  } else {
    // Write output model to a file.
    int fd = 0;
    Status status = Env::Default().FileOpenWr(valid_output_model_path, fd);
    ORT_RETURN_IF_ERROR(status);

    ORT_TRY {
      google::protobuf::io::FileOutputStream output(fd);
      bool serialize_result = model_proto.SerializeToZeroCopyStream(&output) && output.Flush();
      if (!serialize_result) {
        status = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_PROTOBUF,
                                 "Protobuf serialization failed when saving model to ",
                                 valid_output_model_path);
      }
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, ex.what());
      });
    }
    if (!status.IsOK()) {
      GSL_SUPPRESS(es .84)
      ORT_IGNORE_RETURN_VALUE(Env::Default().FileClose(fd));
      return status;
    }
    ORT_RETURN_IF_ERROR(Env::Default().FileClose(fd));
  }

  return Status::OK();
}

Status BuildAndSaveOptimizedModel(const onnxruntime::Model& model,
                                  const epctx::ModelGenOptions& gen_options,
                                  const logging::Logger& logger) {
  // Resolve the validated output path up front so any file-conflict error surfaces before serialization.
  std::filesystem::path valid_output_model_path;
  const epctx::BufferHolder* output_buffer_holder = gen_options.TryGetOutputModelBuffer();
  const epctx::BufferWriteFuncHolder* output_write_func_holder = gen_options.TryGetOutputModelWriteFunc();
  const std::filesystem::path* output_model_path_ptr = gen_options.TryGetOutputModelPath();
  const bool output_is_to_file = (output_buffer_holder == nullptr && output_write_func_holder == nullptr);
  const bool needs_path_for_external_initializers = (gen_options.TryGetExternalInitializerFileInfo() != nullptr);

  if ((output_is_to_file || needs_path_for_external_initializers) &&
      (output_model_path_ptr != nullptr || !model.MainGraph().ModelPath().empty())) {
    std::filesystem::path output_model_path = (output_model_path_ptr != nullptr)
                                                  ? *output_model_path_ptr
                                                  : std::filesystem::path("");
    ORT_RETURN_IF_ERROR(GetValidatedEpContextPath(output_model_path,
                                                  model.MainGraph().ModelPath(),
                                                  valid_output_model_path,
                                                  gen_options.error_if_output_file_exists));
  }

  // Build the ModelProto from the in-memory model (optimized to the session's configured level). The
  // initializer location is one of three mutually exclusive cases; the default (neither external file
  // nor custom handler) embeds them inline.
  ONNX_NAMESPACE::ModelProto model_proto;
  if (const epctx::ExternalInitializerFileInfo* ext_info = gen_options.TryGetExternalInitializerFileInfo();
      ext_info != nullptr) {
    ModelSavingOptions model_saving_options{ext_info->size_threshold};
    ApplyExternalInitializerAlignment(gen_options, model_saving_options);
    model_proto = model.ToGraphProtoWithExternalInitializers(ext_info->file_path,
                                                             valid_output_model_path,
                                                             model_saving_options);
  } else if (const auto* buffer_info = gen_options.TryGetExternalInitializerBufferInfo()) {
    ORT_RETURN_IF_ERROR(SerializeExternalInitializersToBuffer(model, *buffer_info, gen_options, model_proto));
  } else if (const epctx::InitializerHandler* custom_handler = gen_options.TryGetInitializerHandler();
             custom_handler != nullptr) {
    ORT_RETURN_IF_ERROR(model.ToGraphProtoWithCustomInitializerHandling(
        custom_handler->handle_initializer_func, custom_handler->state, model_proto));
  } else {
    // Default: embed all initializers inline. Passing an empty external-file path with a SIZE_MAX
    // threshold and force_embed_external_ini avoids creating an intermediate file.
    ModelSavingOptions model_saving_options{/*size_threshold*/ SIZE_MAX};
    model_saving_options.force_embed_external_ini = true;
    model_proto = model.ToGraphProtoWithExternalInitializers(std::filesystem::path{},
                                                             valid_output_model_path,
                                                             model_saving_options);
  }

  return SaveModelProtoToLocation(model_proto, gen_options, valid_output_model_path, logger);
}

//
// OutStreamBuf class:
//

OutStreamBuf::OutStreamBuf(BufferWriteFuncHolder write_func_holder)
    : write_func_holder_(write_func_holder), buffer_(65536) {
  setp(buffer_.data(), buffer_.data() + buffer_.size());
}

OutStreamBuf::~OutStreamBuf() {
  sync();
}

// Called when the buffer_ is full. Flushes the buffer_ (via sync()) and then writes the overflow character to buffer_.
std::streambuf::int_type OutStreamBuf::overflow(std::streambuf::int_type ch) {
  if (sync() == -1) {
    return traits_type::eof();
  }

  if (ch != traits_type::eof()) {
    *pptr() = static_cast<char>(ch);
    pbump(1);
  }

  return ch;
}

// Flushes the entire buffer_ to the user's write function.
int OutStreamBuf::sync() {
  if (!last_status_.IsOK()) {
    return -1;
  }

  std::ptrdiff_t num_bytes = pptr() - pbase();
  if (num_bytes == 0) {
    return 0;
  }

  // Can only call pbump() with an int, so can only write at most (2^31 - 1) bytes.
  if (num_bytes > std::numeric_limits<int>::max()) {
    num_bytes = std::numeric_limits<int>::max();
  }

  char* ptr = pbase();

  Status status = Status::OK();

  ORT_TRY {
    status = ToStatusAndRelease(write_func_holder_.write_func(write_func_holder_.stream_state,
                                                              ptr, num_bytes));
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Caught exception while calling user's OrtOutStreamWriteFunc callback: ", e.what());
    });
  }

  if (!status.IsOK()) {
    last_status_ = std::move(status);
    return -1;
  }

  pbump(-static_cast<int>(num_bytes));  // Reset internal pointer to point to the beginning of the buffer_
  return 0;
}

}  // namespace epctx
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
