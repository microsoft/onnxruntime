// Copyright(C) 2018 Intel Corporation
// Licensed under the MIT License
#ifdef _MSC_VER
#pragma warning(disable : 4505)  //Unreferenced local function has been removed
#endif

#include "mkldnn_func_kernel.h"
#include "core/common/exceptions.h"
#include "core/providers/mkldnn/mkldnn_common.h"
#include "core/providers/mkldnn/subgraph/mkldnn_conv.h"
#include "core/providers/mkldnn/subgraph/mkldnn_batchnorm.h"
#include "core/providers/mkldnn/subgraph/mkldnn_activations.h"
#include "core/providers/mkldnn/subgraph/mkldnn_pool.h"
#include "core/providers/mkldnn/subgraph/mkldnn_sum.h"
#include "core/providers/mkldnn/subgraph/mkldnn_lrn.h"

namespace onnxruntime {
namespace mkl_dnn {

namespace {
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

  Status Compute(const ONNXRunTimeTensor* input_tensors, ONNXRunTimeTensor* const output_tensors) {
    Status status;
    for (auto& kernel : context_.kernels) {
      status = kernel->Bind(input_tensors, output_tensors);
      if (!status.IsOK())
        break;
    }
    if (status.IsOK())
      context_.stream->submit(context_.net);
    return status;
  }

  ~SubgraphPrimitive() = default;

 private:
  void CreateKernels(const SubgraphParams& params) {
    for (auto& mklnode : params.subgraph->mklnodes) {
      if (mklnode.name == "Conv") {
        std::ostringstream os;
        os << "Conv-" << mklnode.node_index << "-";
        std::shared_ptr<MklDnnConv<T>> kernel;
        kernel.reset(new MklDnnConv<T>(mklnode, params.provider, params.mkl_context, params.attributes, os.str()));
        for (auto& index : mklnode.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mklnode.name == "Conv-Relu") {
        std::ostringstream os;
        os << "Conv-" << mklnode.node_index << "-";
        std::shared_ptr<MklDnnConv<T>> kernel;
        kernel.reset(new MklDnnConv<T>(mklnode, params.provider, params.mkl_context, params.attributes, os.str()));
        kernel->fuse_relu_ = true;
        for (auto& index : mklnode.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mklnode.name == "Relu") {
        std::ostringstream os;
        os << "Relu-" << mklnode.node_index << "-";
        std::shared_ptr<MklDnnRelu<T>> kernel;
        kernel.reset(new MklDnnRelu<T>(mklnode, params.provider, params.mkl_context, params.attributes, os.str()));
        for (auto& index : mklnode.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mklnode.name == "BatchNormalization") {
        std::ostringstream os;
        os << "BatchNormalization-" << mklnode.node_index << "-";
        std::shared_ptr<MklDnnBatchNorm<T>> kernel;
        kernel.reset(new MklDnnBatchNorm<T>(mklnode, params.provider, params.mkl_context, params.attributes, os.str()));
        for (auto& index : mklnode.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mklnode.name == "BatchNormalization-Relu") {
        std::ostringstream os;
        os << "BatchNormalization-" << mklnode.node_index << "-";
        std::shared_ptr<MklDnnBatchNorm<T>> kernel;
        kernel.reset(new MklDnnBatchNorm<T>(mklnode, params.provider, params.mkl_context, params.attributes, os.str()));
        kernel->fuse_relu_ = true;
        for (auto& index : mklnode.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mklnode.name == "MaxPool") {
        std::ostringstream os;
        os << "MaxPool-" << mklnode.node_index << "-";
        std::shared_ptr<MklDnnPool<T>> kernel;
        kernel.reset(new MklDnnPool<T>(mklnode, params.provider, params.mkl_context, params.attributes, os.str()));
        for (auto& index : mklnode.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mklnode.name == "GlobalMaxPool") {
        std::ostringstream os;
        os << "GlobalMaxPool-" << mklnode.node_index << "-";
        std::shared_ptr<MklDnnPool<T>> kernel;
        kernel.reset(new MklDnnPool<T>(mklnode, params.provider, params.mkl_context, params.attributes, os.str()));
        for (auto& index : mklnode.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mklnode.name == "AveragePool") {
        std::ostringstream os;
        os << "AveragePool-" << mklnode.node_index << "-";
        std::shared_ptr<MklDnnPool<T>> kernel;
        kernel.reset(new MklDnnPool<T>(mklnode, params.provider, params.mkl_context, params.attributes, os.str()));
        for (auto& index : mklnode.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mklnode.name == "GlobalAveragePool") {
        std::ostringstream os;
        os << "GlobalAveragePool-" << mklnode.node_index << "-";
        std::shared_ptr<MklDnnPool<T>> kernel;
        kernel.reset(new MklDnnPool<T>(mklnode, params.provider, params.mkl_context, params.attributes, os.str()));
        for (auto& index : mklnode.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mklnode.name == "Sum") {
        std::ostringstream os;
        os << "Sum-" << mklnode.node_index << "-";
        std::shared_ptr<MklDnnSum<T>> kernel;
        kernel.reset(new MklDnnSum<T>(mklnode, params.provider, params.mkl_context, params.attributes, os.str()));
        for (auto& index : mklnode.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mklnode.name == "LRN") {
        std::ostringstream os;
        os << "LRN-" << mklnode.node_index << "-";
        std::shared_ptr<MklDnnLrn<T>> kernel;
        kernel.reset(new MklDnnLrn<T>(mklnode, params.provider, params.mkl_context, params.attributes, os.str()));
        for (auto& index : mklnode.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      }
    }
  }

 private:
  struct SubgraphContext {
    std::unique_ptr<mkldnn::stream> stream;
    std::vector<mkldnn::primitive> net;
    std::vector<std::shared_ptr<MklDnnKernel>> kernels;

    SubgraphContext() : stream(nullptr) {}
  };

  Status Initialize(const ONNXRunTimeTensor* input_tensors) {
    // Propagate mkldnn block format
    // dst format of current node to src format of next node
    mkldnn::memory::format source_format = mkldnn::memory::format::any;  // ONNXRuntime format
    for (auto& kernel : context_.kernels) {
      Status status = kernel->CreatePrimitives(input_tensors, cpu_engine_, context_.net, source_format);
      if (status.IsOK())
        kernel->ReorderWeights(input_tensors, cpu_engine_);
      else
        return status;
    }
    return Status::OK();
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
                                   const SubgraphParams& params) {
    auto xshape = input_tensors[0].shape;
    auto xdim = input_tensors[0].ndim;
    TensorShape x_shape(xshape, xdim);
    mkldnn::memory::dims src_dims(x_shape.GetDims().begin(), x_shape.GetDims().end());
    std::string dims_str;
    AddDimsToKey(dims_str, src_dims);

    SubgraphPrimitive<T>* primitive = dynamic_cast<SubgraphPrimitive<T>*>(
        SubgraphPrimitivePool<T>::GetInstance().GetPrimitive(params.subgraph_key + dims_str));

    if (primitive == nullptr) {
      auto subgraph_primitive = std::make_unique<SubgraphPrimitive<T>>(input_tensors, params);
      primitive = subgraph_primitive.get();
      SubgraphPrimitivePool<T>::GetInstance().SetPrimitive(params.subgraph_key + dims_str, std::move(subgraph_primitive));
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
Status MkldnnFuncKernel<T>::Compute(const ONNXRunTimeTensor* input_tensors, const size_t num_inputs,
                                    ONNXRunTimeTensor* const output_tensors, const size_t num_outputs) const {
  Status status;
  ORT_UNUSED_PARAMETER(num_inputs);
  ORT_UNUSED_PARAMETER(num_outputs);

  try {
    SubgraphPrimitive<T>* primitive = SubgraphPrimitivePool<T>::Get(input_tensors, params_);
    status = primitive->Compute(input_tensors, output_tensors);
  } catch (const mkldnn::error& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Status: ", e.status,
                           ", message: ", e.message.c_str());
  }
  return status;
}

template class MkldnnFuncKernel<float>;

}  // namespace mkl_dnn
}  // namespace onnxruntime