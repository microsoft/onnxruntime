// Copyright(C) Xilinx Inc.
// Licensed under the MIT License

#pragma once

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#if defined(_MSC_VER)
#pragma warning(default : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic pop
#endif

#include <pyxir/pyxir.hpp>

#include "core/session/onnxruntime_c_api.h"
#include "core/framework/func_api.h"
#include "core/graph/graph_viewer.h"
#include "core/common/logging/logging.h"
#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {
namespace vitisai_ep {

class VitisAICustomOp {
 public:
  VitisAICustomOp(const ComputeContext* context,
                  const onnxruntime::Node* fused_node,
                  const std::string &backend_type,
                  const logging::Logger* logger);

  Status Compute(const OrtApi* api, OrtKernelContext* context) const;

  ~VitisAICustomOp();

  void SetLogger(const logging::Logger* logger) {
    logger_ = logger;
  }

  const logging::Logger* GetLogger() const {
    return logger_;
  }

 private:

  std::vector<std::string> in_tensor_names_;
  std::vector<std::string> out_tensor_names_;
  pyxir::XGraphHolder xg_;

  std::string backend_type_;

  pyxir::RtModHolder rt_mod_ = nullptr;

  AllocateFunc allocate_func_ = nullptr;

  DestroyFunc release_func_ = nullptr;

  AllocatorHandle allocator_ = nullptr;

  std::string name_;

  mutable std::mutex compute_lock_;

  const logging::Logger* logger_ = nullptr;

  ONNX_NAMESPACE::ModelProto model_proto_;
};
}  // namespace vitisai_ep
}  // namespace onnxruntime
