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
