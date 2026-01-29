// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)
#include <limits>
#include <filesystem>
#include <utility>
#include "core/framework/ep_context_utils.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/model_saving_options.h"

namespace onnxruntime {
namespace epctx {

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

    model_proto = ep_context_model.ToGraphProtoWithExternalInitializers(ext_info->file_path,
                                                                        validated_model_path,
                                                                        model_saving_options);
    return Status::OK();
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

static Status GetMissingExternalInitializerFilesImpl(const onnxruntime::Graph& graph,
                                                     const std::filesystem::path& model_dir,
                                                     std::unordered_set<std::filesystem::path>& checked_files,
                                                     std::unordered_set<std::filesystem::path>& invalid_files) {
  for (const auto& node : graph.Nodes()) {
    if (node.ContainsSubgraph()) {
      for (const auto& subgraph : node.GetSubgraphs()) {
        ORT_RETURN_IF_ERROR(GetMissingExternalInitializerFilesImpl(*subgraph, model_dir,
                                                                   checked_files, invalid_files));
      }
    }
  }

  for (const auto& [initializer_name, initializer_proto] : graph.GetAllInitializedTensors()) {
    if (utils::HasExternalDataInFile(*initializer_proto)) {
      std::unique_ptr<ExternalDataInfo> external_data_info;
      ORT_RETURN_IF_ERROR(ExternalDataInfo::Create(initializer_proto->external_data(), external_data_info));

      std::filesystem::path initializer_file_path = model_dir / external_data_info->GetRelPath();

      if (checked_files.find(initializer_file_path) != checked_files.end()) {
        continue;  // Already checked this file
      }

      if (!std::filesystem::exists(initializer_file_path)) {
        invalid_files.insert(initializer_file_path);
      }

      checked_files.insert(std::move(initializer_file_path));
    }
  }

  return Status::OK();
}

Status GetMissingExternalInitializerFiles(const onnxruntime::Graph& graph,
                                          std::unordered_set<std::filesystem::path>& invalid_files) {
  std::unordered_set<std::filesystem::path> checked_files;
  const std::filesystem::path model_dir = graph.GetModel().ModelPath().parent_path();

  return GetMissingExternalInitializerFilesImpl(graph, model_dir, checked_files, invalid_files);
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
