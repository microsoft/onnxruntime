#pragma once

#include <core/framework/op_kernel.h>
#include "core/graph/graph.h"
#include <iostream>

namespace onnxruntime {

// An ExecutionFrame that only executes a single kernel
class SingleKernelExecutionFrame final : public IExecutionFrame {
 public:

  class Info {
   public:
    Info(std::unique_ptr<OpKernel> kernel,
         const logging::Logger &logger, const std::unique_ptr<IExecutionProvider>& provider);

    AllocatorPtr GetAllocator() const {
      return allocator_;
    }

    Status AddOutput(OrtValue value, int index, const std::string &name);

    Status AddInput(OrtValue value, int index, const std::string &name);

   protected:

    NodeIndexInfo &GetNodeIndexInfo() {
      if (!node_index_info_) {
        node_index_info_ = std::unique_ptr<NodeIndexInfo>(
            new NodeIndexInfo({&kernel_->Node()}, value_name_idx_map_));
      }
      return *node_index_info_;
    }

    std::unique_ptr<OpKernel> kernel_;
    const logging::Logger *const logger_;

    friend SingleKernelExecutionFrame;

    OrtValueNameIdxMap value_name_idx_map_;
    std::unordered_map<int, const ONNX_NAMESPACE::TypeProto *> ort_value_idx_nodearg_map_;

    std::unique_ptr<NodeIndexInfo> node_index_info_;

    std::vector<int> input_index_to_mlvalue_map_;
    std::vector<int> output_index_to_mlvalue_map_;
    std::vector<int> fetches_mlvalue_idxs_;
    std::vector<OrtValue> fetches_;
    std::vector<int> feed_mlvalue_idxs_;
    std::vector<OrtValue> feeds_;

    const std::unique_ptr<IExecutionProvider>& provider_;
    AllocatorPtr allocator_;
  };

  SingleKernelExecutionFrame(std::unique_ptr<Info> info) :
  // Ideally we would remove the NodeIndexInfo from the constructor, since we only have one node
      IExecutionFrame(info->value_name_idx_map_, info->GetNodeIndexInfo(), info->fetches_mlvalue_idxs_),
      info_(std::move(info)) {
    Init(info_->feed_mlvalue_idxs_, info_->feeds_, std::unordered_map<int, OrtValue>(), info_->fetches_);
  };

  Status SetInput(OrtValue &value, int index) {
    return SetOrtValue(value, info_->input_index_to_mlvalue_map_[index]);
  }

  Status SetOutput(OrtValue &value, int index) {
    return SetOrtValue(value, info_->output_index_to_mlvalue_map_[index]);
  }

  Status Compute() {
    OpKernelContext context(this, info_->kernel_.get(), nullptr, *info_->logger_);
    return info_->kernel_->Compute(&context);
  }


 protected:

  AllocatorPtr GetAllocatorImpl(const OrtMemoryInfo &info) const override {
    return info_->provider_->GetAllocator(info.id, info.mem_type);
  }

  Status
  CopyTensor(__attribute__((unused)) const Tensor &src, __attribute__((unused)) Tensor &dest) const override {
    return Status(ONNXRUNTIME, NOT_IMPLEMENTED, "CopyTensor is not implemented for Single Kernel Execution.");
  }

  Status CreateNodeOutputMLValueImpl(OrtValue &ort_value, int ort_value_idx, const TensorShape *shape,
                                     size_t nnz) override;

 private:
  const std::unique_ptr<const Info> info_;

  const int device_id_{0};
  const OrtMemType mem_type_{OrtMemTypeDefault};

};


struct KernelSessionImpl {
  KernelSessionImpl(std::unique_ptr<Model> model) : model(
      std::move(model)) {}

  // the model who's MainGraph holds the nodes for the kernels that we will execute
  std::unique_ptr<Model> model;

  // providers for the session
  std::vector<std::unique_ptr<IExecutionProvider>> provider_list;
};

class ExecutableKernelContextImpl {
 public:

  Status SetName(const std::string& name) {
    name_ = name;
    return Status::OK();
  }

  Status SetOpType(const std::string& op_type) {
    op_type_ = op_type;
    return Status::OK();
  }

  Status AddAttribute(std::string name, ONNX_NAMESPACE::AttributeProto& attribute){
    attributes_.insert({name, attribute});
    return Status::OK();
  }

  Status AddInput(ONNXTensorElementDataType type);

  Status AddOutput(ONNXTensorElementDataType type);

  Status CreateExecutionFrame(KernelSessionImpl* session, SingleKernelExecutionFrame** frame, size_t provider_id);

 private:

  Status SetupTensorType(std::unique_ptr<ONNX_NAMESPACE::TypeProto> const &type_proto, ONNXTensorElementDataType type);

  // Node attributes
  std::string name_;
  std::string op_type_;
  NodeAttributes attributes_;

  // AddNode takes a vector of raw pointers, so we store the unique pointers separately here and add references to them to the input and output
  // arg vectors
  std::vector<std::unique_ptr<NodeArg>> args_;
  std::vector<std::unique_ptr<ONNX_NAMESPACE::TypeProto>> types_;
  std::vector<NodeArg*> input_args_;
  std::vector<NodeArg*> output_args_;

  // before context is finalized, this op will be null
  std::unique_ptr<SingleKernelExecutionFrame> frame_;

};

}  // namespace onnxruntime