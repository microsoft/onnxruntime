// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/dnnl/dnnl_fwd.h"
#include "core/providers/dnnl/dnnl_execution_provider.h"
#include "core/providers/dnnl/subgraph/dnnl_kernel.h"

namespace onnxruntime {
namespace ort_dnnl {

template <typename T>
class DnnlLrn : public DnnlKernel {
 public:
  DnnlLrn(const DnnlNode& node,
          DNNLExecutionProvider* provider,
          const NodeAttributes& attributes,
          const std::string attributes_prefix = "") : DnnlKernel(node, provider) {
    ReadAttributes(attributes, attributes_prefix);
  }

  void CreatePrimitives(const OrtCustomOpApi* api,
                        OrtKernelContext* context,
                        const std::unordered_map<dnnl::engine::kind, dnnl::engine>& dnnl_engine,
                        std::vector<dnnl::primitive>& net,
                        std::vector<std::unordered_map<int, dnnl::memory>>& net_args) override {
    dnnl::engine cpu_engine;
    dnnl::engine engine_to_use;
    std::unordered_map<dnnl::engine::kind, dnnl::engine>::const_iterator iter = dnnl_engine.find(dnnl::engine::kind::cpu);
    if (iter != dnnl_engine.end()) {
      cpu_engine = (dnnl::engine)iter->second;
      engine_to_use = cpu_engine;
    }
    gpu_available_ = false;
    dnnl::engine gpu_engine;
    iter = dnnl_engine.find(dnnl::engine::kind::gpu);
    if (iter != dnnl_engine.end()) {
      gpu_engine = (dnnl::engine)(iter->second);
      gpu_available_ = true;
      engine_to_use = gpu_engine;
      LOGS_DEFAULT(INFO) << "gpu engine found" << std::endl;
    }
    Ort::CustomOpApi ort{*api};
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    TensorShape x_shape;
    if (mklnode_ptr_->parent_nodes.empty()) {
      const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index);
      auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
      auto tensor_shape = ort.GetTensorShape(tensor_info);
      ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
      auto xshape = tensor_shape.data();
      auto xdim = tensor_shape.size();

      ort_source_format_ = GetSourceFormat(static_cast<int>(xdim));
      x_shape = TensorShape(xshape, xdim);

      dnnl::memory::dims src_dims(
          x_shape.GetDims().begin(), x_shape.GetDims().end());

      ort_source_desc_ = dnnl::memory::desc(
          {src_dims}, DnnnType<T>(), ort_source_format_);
      src_md_ = std::make_unique<dnnl::memory::desc>(
          dnnl::memory::desc({src_dims}, DnnnType<T>(), ort_source_format_));
      src_mem_ = std::make_unique<dnnl::memory>(
          dnnl::memory(*src_md_, cpu_engine, nullptr));
      if (gpu_available_) {
        src_mem_gpu_ = std::make_unique<dnnl::memory>(*src_md_, gpu_engine);
        net.push_back(mkldnn::reorder(*src_mem_, *src_mem_gpu_));
        net_args.push_back({{MKLDNN_ARG_SRC, *src_mem_},
                            {MKLDNN_ARG_DST, *src_mem_gpu_}});
      }
    } else {
      src_md_ = std::make_unique<dnnl::memory::desc>(
          dnnl::memory::desc(parents_[0].get()->primitive_dst_desc_));
      if (!gpu_available_) {
        src_mem_ = parents_[0].get()->primitive_dst_mem_;
      } else {  // gpu_available_
        src_mem_gpu_ = parents_[0].get()->primitive_dst_mem_;
      }
      x_shape = parents_[0].get()->primitive_dst_shape_;
      ort_source_format_ = parents_[0].get()->ort_source_format_;
      ort_source_desc_ = parents_[0].get()->ort_source_desc_;
      source_desc_ = parents_[0].get()->primitive_dst_desc_;
    }

    primitive_dst_shape_ = TensorShape(x_shape);

    dnnl::algorithm algo = dnnl::algorithm::lrn_across_channels;
    fwd_desc_ = std::make_unique<dnnl::lrn_forward::desc>(
        dnnl::lrn_forward::desc(dnnl::prop_kind::forward_scoring, algo, *src_md_,
                                size_, alpha_, beta_, bias_));

    fwd_primitive_desc_ = std::make_unique<dnnl::lrn_forward::primitive_desc>(
        dnnl::lrn_forward::primitive_desc(*fwd_desc_, engine_to_use));

    primitive_src_desc_ = fwd_primitive_desc_.get()->src_desc();
    primitive_dst_desc_ = fwd_primitive_desc_.get()->dst_desc();

    if (!gpu_available_) {
      if (mklnode_ptr_->output_index >= 0) {
        // last node of sub-graph. need to allocate memory for output_tensor
        if (primitive_dst_desc_ != ort_source_desc_) {
          // reorder neded. Use primitive output as input to reorder and
          // allocate buffer for reorder output, final output of this subgraph
          primitive_dst_mem_ = std::make_unique<dnnl::memory>(
              dnnl::memory(fwd_primitive_desc_.get()->dst_desc(), cpu_engine));
        } else {
          // Last node but re-order not needed. Allocate buffer to output of this node
          primitive_dst_mem_ = std::make_unique<dnnl::memory>(
              dnnl::memory(fwd_primitive_desc_.get()->dst_desc(), cpu_engine, nullptr));
        }
      } else {
        // Intermediate node. Use Dnnl kernel internal memory for output and
        // use this as input to next node.
        primitive_dst_mem_ = std::make_unique<dnnl::memory>(
            dnnl::memory(fwd_primitive_desc_.get()->dst_desc(), cpu_engine));
      }
    } else {  // gpu_available_
      primitive_dst_mem_ = std::make_unique<dnnl::memory>(
          dnnl::memory(fwd_primitive_desc_.get()->dst_desc(), gpu_engine));
    }

    lrn_fwd_ = std::make_unique<dnnl::lrn_forward>(
        dnnl::lrn_forward(*fwd_primitive_desc_));
    if (!gpu_available_) {
      net.push_back(*lrn_fwd_);
      net_args.push_back({{DNNL_ARG_SRC, *src_mem_},
                          {DNNL_ARG_DST, *primitive_dst_mem_}});
    } else {  // gpu_available_
      net.push_back(*lrn_fwd_);
      net_args.push_back({{DNNL_ARG_SRC, *src_mem_gpu_},
                          {DNNL_ARG_DST, *primitive_dst_mem_}});
    }
    if (mklnode_ptr_->output_index >= 0) {
      // one of the end nodes. Allocate output buffer memory and
      // reorder is necessary
      dnnl::memory::data_type t = DnnnType<T>();
      InitDstReorderOutput(cpu_engine, t, net, net_args, gpu_available_);
    }
  }

  Status Bind(const OrtCustomOpApi* api, OrtKernelContext* context) override {
    Ort::CustomOpApi ort{*api};
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    if (mklnode_ptr_->parent_nodes.empty()) {
      // Sub-graph's first node. Read input from input buffer
      const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index);
      const T* src_data = const_cast<T*>(ort.GetTensorData<T>(input_tensor));
      src_mem_->set_data_handle(static_cast<void*>(const_cast<T*>(src_data)));
    }

    if (mklnode_ptr_->output_index >= 0) {
      auto& y_dims = primitive_dst_shape_.GetDims();
      // Allocate memory for output bufffer
      OrtValue* output = ort.KernelContext_GetOutput(context, mklnode_ptr_->output_index, &y_dims[0], static_cast<int>(primitive_dst_shape_.GetDims().size()));
      T* dst_data = ort.GetTensorMutableData<T>(output);

      if (!gpu_available_) {
        if (primitive_dst_desc_ != ort_source_desc_) {
          reorder_dst_mem_to_->set_data_handle(dst_data);
        } else {
          primitive_dst_mem_->set_data_handle(dst_data);
        }
      } else {  // gpu_available_
        reorder_dst_mem_to_->set_data_handle(dst_data);
      }
    }

    return Status::OK();
  }

 private:
  void ReadAttributes(const NodeAttributes& attributes,
                      const std::string attributes_prefix = "") override {
    auto attr = attributes.find(attributes_prefix + "size");
    if (attr != attributes.end() &&
        attr->second().type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT) {
      size_ = attr->second().i();
    }
    ORT_ENFORCE(size_ > 0);
    ORT_ENFORCE(size_ % 2 == 1);

    attr = attributes.find(attributes_prefix + "alpha");
    if (attr != attributes.end() &&
        attr->second().type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT) {
      alpha_ = attr->second().f();
    }

    attr = attributes.find(attributes_prefix + "beta");
    if (attr != attributes.end() &&
        attr->second().type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT) {
      beta_ = attr->second().f();
    }

    bias_ = 1.0f;
    attr = attributes.find(attributes_prefix + "bias");
    if (attr != attributes.end() &&
        attr->second().type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT) {
      bias_ = attr->second().f();
    }
  }

 private:
  float alpha_ = 0;
  float beta_ = 0;
  float bias_ = 0;
  int size_ = 0;

 private:
  std::shared_ptr<dnnl::memory> src_mem_;
  std::shared_ptr<dnnl::memory> src_mem_gpu_;

  std::unique_ptr<dnnl::lrn_forward::desc> fwd_desc_;
  std::unique_ptr<dnnl::lrn_forward::primitive_desc> fwd_primitive_desc_;
  std::unique_ptr<dnnl::primitive> lrn_fwd_;

  std::unique_ptr<dnnl::memory::desc> src_md_;

  bool gpu_available_;
};
}  // namespace ort_dnnl
}  // namespace onnxruntime
