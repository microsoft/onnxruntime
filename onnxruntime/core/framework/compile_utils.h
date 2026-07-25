// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <streambuf>
#include <vector>

#include "core/common/logging/logging.h"
#include "core/common/status.h"
#include "core/framework/ep_context_options.h"
#include "core/graph/model.h"

namespace onnxruntime {

class ExecutionProviders;
class Graph;
namespace epctx {

/// <summary>
/// Compute and validate the output model file path for an EPContext (or non-compiling) model save.
/// If ep_context_path is non-empty it is used as-is (after validation). Otherwise a path is derived
/// from model_path by replacing the extension with "_ctx.onnx".
/// </summary>
/// <param name="ep_context_path">Explicit output model path (may be empty).</param>
/// <param name="model_path">Source model path used to derive a default output path when needed.</param>
/// <param name="context_cache_path">Output: the resolved and validated output model path.</param>
/// <param name="error_if_output_file_exists">When true, returns an error if the output file already exists.</param>
/// <returns>A status indicating success or an error.</returns>
Status GetValidatedEpContextPath(const std::filesystem::path& ep_context_path,
                                 const std::filesystem::path& model_path,
                                 std::filesystem::path& context_cache_path,
                                 bool error_if_output_file_exists = true);

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
/// Write a serialized ModelProto to the output location specified in gen_options (buffer, write-func, or file).
/// </summary>
/// <param name="model_proto">The ModelProto to write. Must already be serialized.</param>
/// <param name="gen_options">Options that specify the output location.</param>
/// <param name="valid_output_model_path">Validated file system path for file-output mode; may be empty for
/// buffer/write-func modes.</param>
/// <param name="logger">Logger for diagnostic messages.</param>
/// <returns>A status indicating success or an error.</returns>
Status SaveModelProtoToLocation(ONNX_NAMESPACE::ModelProto& model_proto,
                                const epctx::ModelGenOptions& gen_options,
                                const std::filesystem::path& valid_output_model_path,
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

/// <summary>
/// Build a ModelProto from the fully-optimized in-memory model and save it to the output location
/// (buffer, write-func, or file) specified in gen_options.
/// </summary>
/// <param name="model">The fully-optimized in-memory model.</param>
/// <param name="gen_options">Compile API generation options that control initializer handling and output target.</param>
/// <param name="logger">Logger for diagnostic messages.</param>
/// <returns>A status indicating success or an error.</returns>
Status BuildAndSaveOptimizedModel(const onnxruntime::Model& model,
                                  const epctx::ModelGenOptions& gen_options,
                                  const logging::Logger& logger);

/// <summary>
/// Build and save an EPContext model from the compiled execution providers and graph.
/// </summary>
Status CreateEpContextModel(const ExecutionProviders& execution_providers,
                            const Graph& graph,
                            const epctx::ModelGenOptions& ep_context_gen_options,
                            const logging::Logger& logger);

}  // namespace epctx
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
