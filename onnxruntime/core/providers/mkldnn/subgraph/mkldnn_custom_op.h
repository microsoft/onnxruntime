// Copyright(C) 2018 Intel Corporation
// Licensed under the MIT License
#pragma once

#include "core/graph/onnx_protobuf.h"
#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/providers/mkldnn/mkldnn_common.h"
#include "core/providers/mkldnn/mkldnn_execution_provider.h"
#include "core/providers/mkldnn/subgraph/mkl_kernel.h"
#include "core/providers/mkldnn/subgraph/MklConv.h"
#include "core/providers/mkldnn/subgraph/MklBatchNorm.h"
#include "core/providers/mkldnn/subgraph/MklActivations.h"
#include "core/providers/mkldnn/subgraph/MklPool.h"
#include "core/providers/mkldnn/subgraph/MklSum.h"
#include "core/providers/mkldnn/subgraph/MklLrn.h"
#include "core/providers/cpu/nn/autopad_type.h"

namespace onnxruntime {
namespace mkl_dnn {

namespace {
struct SubgraphParams {
  std::unordered_map<std::string, ONNX_NAMESPACE::AttributeProto> attributes;
  MKLDNNExecutionProvider* provider;
  std::shared_ptr<Subgraph> subgraph;
  std::shared_ptr<MKLContext> mkl_context;

  SubgraphParams() {}
};

template <typename T>
class SubgraphPrimitive : public PrimitiveBase {
 public:
  SubgraphPrimitive(const ONNXRunTimeTensor* input_tensors,
                             const SubgraphParams& params)
      : cpu_engine_(GetEngine()) {
    context_.stream.reset(new mkldnn::stream(mkldnn::stream::kind::eager));

    if (context_.net.size() == 0) {
      CreateKernels(params);
      Initialize(input_tensors);
    }
  }

  void CreateKernels(const SubgraphParams& params) {
    for (auto& mklnode : params.subgraph->mklnodes) {
      if (mklnode.name == "Conv") {
        std::ostringstream os;
        os << "Conv-" << mklnode.node_index << "-";
        std::shared_ptr<MklConv<T>> kernel;
        kernel.reset(new MklConv<T>(mklnode, params.provider, params.mkl_context));
        kernel->ReadAttributes(params.attributes, os.str());
        for (auto& index : mklnode.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mklnode.name == "Conv-Relu") {
        std::ostringstream os;
        os << "Conv-" << mklnode.node_index << "-";
        std::shared_ptr<MklConv<T>> kernel;
        kernel.reset(new MklConv<T>(mklnode, params.provider, params.mkl_context));
        kernel->ReadAttributes(params.attributes, os.str());
        kernel->fuse_relu_ = true;
        for (auto& index : mklnode.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mklnode.name == "Relu") {
        std::ostringstream os;
        os << "Relu-" << mklnode.node_index << "-";
        std::shared_ptr<MklRelu<T>> kernel;
        kernel.reset(new MklRelu<T>(mklnode, params.provider, params.mkl_context));
        kernel->ReadAttributes(params.attributes, os.str());
        for (auto& index : mklnode.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mklnode.name == "BatchNormalization") {
        std::ostringstream os;
        os << "BatchNormalization-" << mklnode.node_index << "-";
        std::shared_ptr<MklBatchNorm<T>> kernel;
        kernel.reset(new MklBatchNorm<T>(mklnode, params.provider, params.mkl_context));
        kernel->ReadAttributes(params.attributes, os.str());
        for (auto& index : mklnode.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mklnode.name == "BatchNormalization-Relu") {
        std::ostringstream os;
        os << "BatchNormalization-" << mklnode.node_index << "-";
        std::shared_ptr<MklBatchNorm<T>> kernel;
        kernel.reset(new MklBatchNorm<T>(mklnode, params.provider, params.mkl_context));
        kernel->fuse_relu_ = true;
        kernel->ReadAttributes(params.attributes, os.str());
        for (auto& index : mklnode.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mklnode.name == "MaxPool") {
        std::ostringstream os;
        os << "MaxPool-" << mklnode.node_index << "-";
        std::shared_ptr<MklPool<T>> kernel;
        kernel.reset(new MklPool<T>(mklnode, params.provider, params.mkl_context));
        kernel->ReadAttributes(params.attributes, os.str());
        for (auto& index : mklnode.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mklnode.name == "GlobalMaxPool") {
        std::ostringstream os;
        os << "GlobalMaxPool-" << mklnode.node_index << "-";
        std::shared_ptr<MklPool<T>> kernel;
        kernel.reset(new MklPool<T>(mklnode, params.provider, params.mkl_context));
        kernel->ReadAttributes(params.attributes, os.str());
        for (auto& index : mklnode.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mklnode.name == "AveragePool") {
        std::ostringstream os;
        os << "AveragePool-" << mklnode.node_index << "-";
        std::shared_ptr<MklPool<T>> kernel;
        kernel.reset(new MklPool<T>(mklnode, params.provider, params.mkl_context));
        kernel->ReadAttributes(params.attributes, os.str());
        for (auto& index : mklnode.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mklnode.name == "GlobalAveragePool") {
        std::ostringstream os;
        os << "GlobalAveragePool-" << mklnode.node_index << "-";
        std::shared_ptr<MklPool<T>> kernel;
        kernel.reset(new MklPool<T>(mklnode, params.provider, params.mkl_context));
        kernel->ReadAttributes(params.attributes, os.str());
        for (auto& index : mklnode.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mklnode.name == "Sum") {
        std::ostringstream os;
        os << "Sum-" << mklnode.node_index << "-";
        std::shared_ptr<MklSum<T>> kernel;
        kernel.reset(new MklSum<T>(mklnode, params.provider, params.mkl_context));
        kernel->ReadAttributes(params.attributes, os.str());
        for (auto& index : mklnode.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mklnode.name == "LRN") {
        std::ostringstream os;
        os << "LRN-" << mklnode.node_index << "-";
        std::shared_ptr<MklLrn<T>> kernel;
        kernel.reset(new MklLrn<T>(mklnode, params.provider, params.mkl_context));
        kernel->ReadAttributes(params.attributes, os.str());
        for (auto& index : mklnode.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      }
    }
  }

  Status Compute(const ONNXRunTimeTensor* input_tensors, ONNXRunTimeTensor* const output_tensors) {
    Status status;
    for (auto& kernel : context_.kernels) {
      status = kernel->Compute(input_tensors, output_tensors);
      if (!status.IsOK())
        break;
    }
    if (status.IsOK())
      context_.stream->submit(context_.net);
    return status;
  }

  ~SubgraphPrimitive() = default;

 private:
  struct SubgraphContext {
    std::unique_ptr<mkldnn::stream> stream;
    std::vector<mkldnn::primitive> net;
    std::vector<std::shared_ptr<MklKernel>> kernels;

    SubgraphContext() : stream(nullptr) {}
  };

  void Initialize(const ONNXRunTimeTensor* input_tensors) {
    // Propagate mkldnn block format
    // dst format of current node to src format of next node
    mkldnn::memory::format source_format = mkldnn::memory::format::any;  // ONNXRuntime format
    for (auto& kernel : context_.kernels) {
      Status status = kernel->CreatePrimitives(input_tensors, cpu_engine_, context_.net, source_format);
      if (status.IsOK())
		kernel->ReorderWeights(input_tensors, cpu_engine_);
    }
  }

  SubgraphContext context_;
  mkldnn::engine& cpu_engine_;
};

// Pool which allows for reuse of MKLDNN Conv primitives which are expensive to instantiate.
// To address thread safety, the primitives are stored in a map on thread local storage.
template <typename T>
class SubgraphPrimitivePool : public PrimitivePool<T> {
 public:
  static SubgraphPrimitive<T>* Get(const ONNXRunTimeTensor* input_tensors,
                                   const std::string subgraph_key,
                                   const SubgraphParams& params) {
    SubgraphPrimitive<T>* primitive = dynamic_cast<SubgraphPrimitive<T>*>(
        SubgraphPrimitivePool<T>::GetInstance().GetPrimitive(subgraph_key));

    if (primitive == nullptr) {
      auto subgraph_primitive = std::make_unique<SubgraphPrimitive<T>>(input_tensors, params);
      primitive = subgraph_primitive.get();
      SubgraphPrimitivePool<T>::GetInstance().SetPrimitive(subgraph_key, std::move(subgraph_primitive));
    }
    return primitive;
  }

 private:
  SubgraphPrimitivePool() = default;
  ~SubgraphPrimitivePool() = default;

  static SubgraphPrimitivePool& GetInstance() {
    static SubgraphPrimitivePool pool;
    return pool;
  }
};
}  // namespace

template <typename T>
class MkldnnCustomOp {
 public:
  explicit MkldnnCustomOp(const ComputeContext* context,
                          const std::unordered_map<std::string, ONNX_NAMESPACE::AttributeProto>& attributes,
                          MKLDNNExecutionProvider* provider) {
    params_.provider = provider;
    params_.attributes = attributes;
    params_.mkl_context.reset(new MKLContext(context->allocate_func, context->release_func, context->allocator_handle));

    auto sub_it = attributes.find("subgraph_id");
    if (sub_it->second.type() != ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING) {
      subgraph_id_ = sub_it->second.s();
      params_.subgraph = provider->GetMklSubgraph(subgraph_id_);
      std::ostringstream key_os;
      key_os << subgraph_id_ << "-" << params_.subgraph->mklnodes.back().name << "-" << params_.subgraph->mklnodes.back().output_name;
      subgraph_key_ = key_os.str();
    }
  }

  Status Compute(const ONNXRunTimeTensor* input_tensors, const size_t num_inputs,
                 ONNXRunTimeTensor* const output_tensors, const size_t num_outputs) const {
    Status status;
    ORT_UNUSED_PARAMETER(num_inputs);
    ORT_UNUSED_PARAMETER(num_outputs);

    try {
      auto xshape = input_tensors[0].shape;
      auto xdim = input_tensors[0].ndim;
      TensorShape x_shape(xshape, xdim);
      mkldnn::memory::dims src_dims(x_shape.GetDims().begin(), x_shape.GetDims().end());
      std::string key;
      key.reserve(128);
      key = subgraph_key_;
      AddDimsToKey(key, src_dims);

      SubgraphPrimitive<T>* primitive = SubgraphPrimitivePool<T>::Get(input_tensors, key, params_);
      status = primitive->Compute(input_tensors, output_tensors);
    } catch (mkldnn::error& e) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Status: ", e.status,
                             ", message: ", e.message.c_str());
    }
    return status;
  }

 private:
  std::string subgraph_id_;
  std::string subgraph_key_;
  SubgraphParams params_;
};
}  // namespace mkl_dnn
}  // namespace onnxruntime