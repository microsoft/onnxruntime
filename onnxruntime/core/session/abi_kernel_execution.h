#pragma once

#include <core/framework/op_kernel.h>
#include "core/graph/graph.h"
#include <iostream>
#include <core/graph/model.h>
#include <core/framework/mldata_type_utils.h>
#include "core/graph/onnx_protobuf.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/session/abi_kernel_execution.h"
#include "core/session/inference_session.h"
#include "core/session/ort_apis.h"
#include "core/framework/customregistry.h"
#include "core/framework/data_types.h"
#include "core/framework/execution_providers.h"
#include "core/framework/op_kernel_info.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/TensorSeq.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/framework/tensorprotoutils.h"
#include "abi_session_options_impl.h"

namespace onnxruntime {

// An ExecutionFrame that only executes a single kernel
class SingleKernelExecutionFrame final : public IExecutionFrame {
 public:

  class Info {
   public:
    Info(std::unique_ptr<OpKernel> kernel,
         const logging::Logger &logger, const std::unique_ptr<IExecutionProvider>& provider)
        : kernel_(std::move(kernel)),
          logger_(&logger),
          provider_(provider)
    {
      ORT_ENFORCE(kernel_, "kernel cannot be null");
      ORT_ENFORCE(provider_, "provider cannot be null");

      allocator_ = provider_->GetAllocator(provider_->GetDeviceId(), OrtMemTypeDefault);

      auto &node = kernel_->Node();

      input_index_to_mlvalue_map_ = std::vector<int> (node.InputDefs().size(), -1);
      output_index_to_mlvalue_map_ = std::vector<int> (node.OutputDefs().size(), -1);

      if (node.ImplicitInputDefs().size()) {
        // not sure how to handle this correctly
        throw new NotImplementedException("Implicit inputs are not supported");
      }

      // initialize inputs and outputs with null values
      OrtValue null_value;
      node.ForEachWithIndex(node.InputDefs(),
                            [this](const NodeArg &arg, size_t index) {
                              this->AddInput(OrtValue(), index, arg.Name());
                              return Status::OK();
                            });

      node.ForEachWithIndex(node.OutputDefs(),
                            [this](const NodeArg &arg, size_t index) {
                              this->AddOutput(OrtValue(), index, arg.Name());
                              return Status::OK();
                            });
    }

    AllocatorPtr GetAllocator() const {
      return allocator_;
    }

    Status AddOutput(OrtValue value, int index, const std::string &name) {
      int mlvalue_idx = value_name_idx_map_.Add(name);
      ort_value_idx_nodearg_map_[mlvalue_idx] = kernel_->Info().GetOutputType(index);

      output_index_to_mlvalue_map_[index] = mlvalue_idx;
      fetches_.push_back(value);
      fetches_mlvalue_idxs_.push_back(mlvalue_idx);
      return Status::OK();
    }

    Status AddInput(OrtValue value, int index, const std::string &name) {
      int mlvalue_idx = value_name_idx_map_.Add(name);

      input_index_to_mlvalue_map_[index] = mlvalue_idx;
      ort_value_idx_nodearg_map_[mlvalue_idx] = kernel_->Info().GetInputType(index);
      feeds_.push_back(value);
      feed_mlvalue_idxs_.push_back(mlvalue_idx);
      return Status::OK();
    }

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
    Status status = info_->kernel_->Compute(&context);
    return status;
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


  Status AddInput(ONNXTensorElementDataType type) {
    std::unique_ptr<ONNX_NAMESPACE::TypeProto> type_proto = std::make_unique<ONNX_NAMESPACE::TypeProto>();

    Status status = SetupTensorType(type_proto, type);
    if (!status.IsOK()) {
      return status;
    }

    std::ostringstream oss;
    oss << "Input_" << input_args_.size();
    std::string name = oss.str();

    std::unique_ptr<NodeArg> arg_ptr = std::make_unique<NodeArg>(name, type_proto.get());
    input_args_.push_back(arg_ptr.get());
    types_.push_back(std::move(type_proto));
    args_.push_back(std::move(arg_ptr));
    return Status::OK();
  }

  Status AddOutput(ONNXTensorElementDataType type) {

    std::unique_ptr<ONNX_NAMESPACE::TypeProto> type_proto = std::make_unique<ONNX_NAMESPACE::TypeProto>();

    Status status = SetupTensorType(type_proto, type);
    if (!status.IsOK()) {
      return status;
    }

    std::ostringstream oss;
    oss << "Output_" << output_args_.size();
    std::string name = oss.str();

    std::unique_ptr<NodeArg> arg_ptr = std::make_unique<NodeArg>(name, type_proto.get());

    output_args_.push_back(arg_ptr.get());
    types_.push_back(std::move(type_proto));
    args_.push_back(std::move(arg_ptr));
    return Status::OK();
  }

  Status CreateExecutionFrame(KernelSessionImpl* session, SingleKernelExecutionFrame** frame, size_t provider_id) {
    auto& graph = session->model->MainGraph();
    std::string description;
    Node& node = graph.AddNode(
        name_,
        op_type_,
        description,
        input_args_,
        output_args_);
    Status status = graph.Resolve();
    if (!status.IsOK()){
      return status;
    }

    auto const& execution_provider = session->provider_list[provider_id];

    node.SetExecutionProviderType(execution_provider->Type());

    std::shared_ptr<KernelRegistry> registry = execution_provider->GetKernelRegistry();
    std::unique_ptr<OpKernel> op_kernel;
    status = registry->TryCreateKernel(node,
                                       *execution_provider,
                                       std::unordered_map<int, OrtValue>(),
                                       OrtValueNameIdxMap(),
                                       FuncManager(),
                                       DataTransferManager(),
                                       op_kernel);

    if (!status.IsOK()) {
      return status;
    }

    // create the context info
    std::unique_ptr<SingleKernelExecutionFrame::Info> info = std::make_unique<SingleKernelExecutionFrame::Info>(
        std::move(op_kernel),
        logging::LoggingManager::DefaultLogger(),
        execution_provider);

    *frame = new SingleKernelExecutionFrame(std::move(info));
    return Status::OK();
  }

 private:

  Status SetupTensorType(std::unique_ptr<ONNX_NAMESPACE::TypeProto> const &type_proto, ONNXTensorElementDataType type) {
    ONNX_NAMESPACE::TypeProto::Tensor *tensor_type = type_proto->mutable_tensor_type();

    switch (type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::UINT8);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::INT8);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::UINT16);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::INT16);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::INT32);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::UINT32);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::INT64);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::UINT64);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::STRING);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::BOOL);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT16);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::BFLOAT16);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::DOUBLE);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
        tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::COMPLEX64);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
        tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::COMPLEX128);
        break;
      default: {
        std::ostringstream oss;
        oss << "type " << type << " is not supported in this function";
        std::string errmsg = oss.str();
        return Status(ONNXRUNTIME, NOT_IMPLEMENTED, errmsg);
      }
    }
    return Status::OK();
  }

  // Node attributes
  std::string name_;
  std::string op_type_;

  // AddNode takes a vector of raw pointers, so we store the unique pointers separately here and add references to them to the input and output
  // arg vectors
  std::vector<std::unique_ptr<NodeArg>> args_;
  std::vector<std::unique_ptr<ONNX_NAMESPACE::TypeProto>> types_;
  std::vector<NodeArg*> input_args_;
  std::vector<NodeArg*> output_args_;
  std::unique_ptr<NodeAttributes> attributes_;

  // before context is finalized, this op will be null
  std::unique_ptr<SingleKernelExecutionFrame> frame_;

};

}  // namespace onnxruntime
