// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <streambuf>
#include <vector>

#include "core/common/status.h"
#include "core/framework/ep_context_options.h"
#include "core/graph/model.h"

namespace onnxruntime {
namespace epctx {

/// <summary>
/// Serialize an EPContext model into a onnx::ModelProto based on the provided options.
/// </summary>
/// <param name="ep_context_model">The EP Context model to serialize.</param>
/// <param name="validated_model_path">The path into which to save the model. May be empty if serialized into a
/// buffer or output stream.</param>
/// <param name="ep_context_gen_options">The model generation options.</param>
/// <param name="model_proto">Output parameter set to the serialized onnx::ModelProto.</param>
/// <returns>A status indicating success or an error.</returns>
Status EpContextModelToProto(const onnxruntime::Model& ep_context_model,
                             const std::filesystem::path& validated_model_path,
                             const epctx::ModelGenOptions& ep_context_gen_options,
                             /*out*/ ONNX_NAMESPACE::ModelProto& model_proto);

/// <summary>
/// Validate an output-model path: ensure it points to a file (not a folder), derive a default
/// "<model>_ctx.onnx" name from the source model path when none is given, and optionally fail if the
/// target file already exists.
/// </summary>
/// <param name="ep_context_path">The requested output path. May be empty.</param>
/// <param name="model_path">The source model path, used to derive a default output name.</param>
/// <param name="context_cache_path">Output parameter set to the validated path.</param>
/// <param name="error_if_output_file_exists">Whether to fail if the target file already exists.</param>
/// <returns>A status indicating success or an error.</returns>
Status GetValidatedEpContextPath(const std::filesystem::path& ep_context_path,
                                 const std::filesystem::path& model_path,
                                 std::filesystem::path& context_cache_path,
                                 bool error_if_output_file_exists = true);

/// <summary>
/// Write a serialized onnx::ModelProto to the location configured in the generation options: a buffer
/// ORT allocates for the caller, the caller's output-stream write function, or a file.
/// </summary>
/// <param name="model_proto">The model proto to serialize.</param>
/// <param name="gen_options">The model generation options selecting the output location.</param>
/// <param name="valid_output_model_path">The validated file path (used only for the file case).</param>
/// <param name="logger">Session logger.</param>
/// <returns>A status indicating success or an error.</returns>
Status SaveModelProtoToLocation(ONNX_NAMESPACE::ModelProto& model_proto,
                                const epctx::ModelGenOptions& gen_options,
                                const std::filesystem::path& valid_output_model_path,
                                const logging::Logger& logger);

/// <summary>
/// Build and save a plain optimized (non-EPContext) output model from an already optimized,
/// in-memory model. Used by the Compile API when no nodes were compiled (the kGenerateModel case, e.g. a
/// non-compiling EP such as WebGPU), so the emitted model captures whatever optimizations the session's
/// configured level produced. Unlike CreateEpContextModel this does no EPContext-node substitution - it
/// serializes the model as-is.
/// </summary>
/// <param name="model">The in-memory model to serialize.</param>
/// <param name="gen_options">The model generation options.</param>
/// <param name="logger">Session logger.</param>
/// <returns>A status indicating success or an error.</returns>
Status BuildAndSaveOptimizedModel(const onnxruntime::Model& model,
                                  const epctx::ModelGenOptions& gen_options,
                                  const logging::Logger& logger);

// Class that wraps the user's OrtBufferWriteFunc function to enable use with
// C++'s std::ostream.
// Example:
//    BufferWriteFuncHolder write_func_holder{write_func, stream_state};
//    std::unique_ptr<OutStreamBuf> out_stream_buf = std::make_unique<OutStreamBuf>(write_func_holder);
//    std::ostream out_stream(out_stream_buf.get());
class OutStreamBuf : public std::streambuf {
 public:
  explicit OutStreamBuf(BufferWriteFuncHolder write_func_holder);
  ~OutStreamBuf();

  const Status& GetStatus() const {
    return last_status_;
  }

 protected:
  int_type overflow(int_type ch) override;
  int sync() override;

 private:
  BufferWriteFuncHolder write_func_holder_{};
  std::vector<char> buffer_;
  Status last_status_{};
};

}  // namespace epctx
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
