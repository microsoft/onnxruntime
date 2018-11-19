// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/tvm/tvm_compiler.h"
#include "core/graph/function.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

// TVMScheduleCreator is the function that create a tvm schedule based on given TVM graph.
// Different hardware may have different schedule strategy.
typedef tvm::Schedule (*TVMScheduleCreator)(const TVMGraph&);
// TVMModuleBuilder is the function that build a tvm module, given a schedule and args.
// Different tvm kernel may chose different way to build the module, like target to LLVM or other backend.
typedef tvm::runtime::Module (*TVMModuleBuilder)(tvm::Schedule schedule, tvm::BuildConfig config, tvm::Array<tvm::Tensor> args, std::vector<std::string>& target_func_names);

template <TVMScheduleCreator S, TVMModuleBuilder M>
class TVMKernel : public OpKernel {
 public:
  explicit TVMKernel(const OpKernelInfo& info) : OpKernel(info), tvm_values_(nullptr), dl_tensors_(nullptr), tvm_type_codes_(nullptr) {
    auto& node = info.node();
    ONNXRUNTIME_ENFORCE(node.NodeType() == Node::Type::Fused);
    auto func = node.GetFunctionBody();
    const onnxruntime::Graph& func_body = func->Body();
    //1. compile the onnxruntime Graph to tvm graph. This step is common for all hardware, and provided by onnxruntime framework.
    tvm_graph_ = CompileToTVM(func_body, node.GetExecutionProviderType());
    //2. create schedule for tvm graph, this step is depends on the execution provider/hardware.
    auto s = S(tvm_graph_);
    //3. Build module
    std::vector<tvm::Tensor> tvm_args;
    for (auto& t : tvm_graph_.inputs_) {
      tvm_args.push_back(t.tvm_tensor_);
    }
    for (auto& t : tvm_graph_.outputs_) {
      tvm_args.push_back(t.tvm_tensor_);
    }

    std::vector<std::string> func_names;
    tvm_module_ = M(s, tvm::build_config(), tvm_args, func_names);
    //TODO: do we have case that need more than 1 evaluation function?
    evaluate_func_ = tvm_module_.GetFunction(func_names[0]);
    //4. prepare args according to the type
    n_args_ = tvm_args.size();
    tvm_values_ = new TVMValue[n_args_];
    tvm_type_codes_ = new int[n_args_];
    dl_tensors_ = new DLTensor[n_args_];
    int i = 0;
    for (auto& tensor : tvm_graph_.inputs_) {
      tvm_type_codes_[i] = kNDArrayContainer;
      dl_tensors_[i].ctx = tensor.ctx_;
      dl_tensors_[i].dtype = tensor.dtype_;
      dl_tensors_[i].strides = nullptr;
      dl_tensors_[i].byte_offset = 0;
      tvm_values_[i].v_handle = &dl_tensors_[i];
      i++;
    }

    for (auto& tensor : tvm_graph_.outputs_) {
      tvm_type_codes_[i] = kNDArrayContainer;
      dl_tensors_[i].ctx = tensor.ctx_;
      dl_tensors_[i].dtype = tensor.dtype_;
      dl_tensors_[i].strides = nullptr;
      dl_tensors_[i].byte_offset = 0;
      tvm_values_[i].v_handle = &dl_tensors_[i];
      i++;
    }
    ONNXRUNTIME_ENFORCE(i == n_args_);
  }

  virtual ~TVMKernel() {
    if (!tvm_values_)
      delete[] tvm_values_;
    if (!tvm_type_codes_)
      delete[] tvm_type_codes_;
    if (!dl_tensors_)
      delete[] dl_tensors_;
  }

  virtual Status Compute(OpKernelContext* context) const override {
    for (int i = 0; i < tvm_graph_.inputs_.size(); ++i) {
      auto t = context->Input<Tensor>(i);
      dl_tensors_[i].data = const_cast<Tensor*>(t)->MutableDataRaw();
      dl_tensors_[i].ndim = static_cast<int>(t->Shape().NumDimensions());
      dl_tensors_[i].shape = dl_tensors_[i].ndim > 0 ? const_cast<int64_t*>(&(t->Shape().GetDims()[0])) : nullptr;
    }

    int num_inputs = static_cast<int>(tvm_graph_.inputs_.size());

    for (int i = 0; i < tvm_graph_.outputs_.size(); ++i) {
      //TODO: we need to have a shape inference function that could calculate output shape based on the symbolic formular in tvm
      //We could build that function as part of tvm module, or reuse the shape inference in onnx funciton.
      //Here for testing purpose, assume the output shape is same to input shape.
      auto t = context->Output(i, GetOutputShape(context, i));
      dl_tensors_[num_inputs + i].data = t->MutableDataRaw();
      dl_tensors_[num_inputs + i].ndim = static_cast<int>(t->Shape().NumDimensions());
      dl_tensors_[num_inputs + i].shape = dl_tensors_[i].ndim > 0 ? const_cast<int64_t*>(&(t->Shape().GetDims()[0])) : nullptr;
    }

    tvm::TVMArgs tvm_args(&tvm_values_[0], &tvm_type_codes_[0], static_cast<int>(n_args_));
    tvm::TVMRetValue rvalue;
    try {
      evaluate_func_.CallPacked(tvm_args, &rvalue);
    } catch (std::exception ex) {
      return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, "TVM run failed.");
    }
    if (rvalue.type_code() != kNull) {
      return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, "TVM return not null");  // TODO: get error code.
    } else {
      return Status::OK();
    }
  }

 protected:
  virtual const TensorShape& GetOutputShape(OpKernelContext* context, int i) const = 0;

  TVMGraph tvm_graph_;
  tvm::runtime::Module tvm_module_;
  tvm::PackedFunc evaluate_func_;

  size_t n_args_;
  TVMValue* tvm_values_;
  DLTensor* dl_tensors_;
  int* tvm_type_codes_;
};
}  // namespace onnxruntime
