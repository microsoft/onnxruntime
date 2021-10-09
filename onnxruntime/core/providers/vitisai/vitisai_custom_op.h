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
                  const std::string& backend_type,
                  const std::string& export_runtime_module,
                  const std::string& load_runtime_module,
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
  // The partition input tensor names
  std::vector<std::string> in_tensor_names_;
  // The partition output tensor names
  std::vector<std::string> out_tensor_names_;
  // The PyXIR graph data structure
  pyxir::XGraphHolder xg_;
  // The Vitis AI DPU target
  std::string backend_type_;
  // If not empty, the path to the file where the PyXIR runtime module
  //	should be exported to (used for cross compilation)
  std::string export_runtime_module_;
  // If not empty, the path to the file where the PyXIR runtime module should
  // be loaded from
  std::string load_runtime_module_;
  // The PyXIR runtime module
  pyxir::RtModHolder rt_mod_ = nullptr;
  // The EP ComputeContext allocation function
  AllocateFunc allocate_func_ = nullptr;
  // The EP ComputeContext release function
  DestroyFunc release_func_ = nullptr;
  // The EP ComputeContext allocator
  AllocatorHandle allocator_ = nullptr;
  // The EP ComputeContext node name
  std::string name_;
  // The compute lock
  mutable std::mutex compute_lock_;
  // The logger
  const logging::Logger* logger_ = nullptr;
  // The ONNX ModelProto to go from fused node -> ModelProto -> PyXIR
  ONNX_NAMESPACE::ModelProto model_proto_;
};
}  // namespace vitisai_ep
}  // namespace onnxruntime
