// Copyright(C) 2018 Intel Corporation
// Licensed under the MIT License
#ifdef _MSC_VER
#pragma warning(disable : 4505)  //Unreferenced local function has been removed
#endif

#include "dnnl_func_kernel.h"
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/dnnl/dnnl_common.h"
#include "core/providers/dnnl/subgraph/dnnl_conv.h"
#include "core/providers/dnnl/subgraph/dnnl_batchnorm.h"
#include "core/providers/dnnl/subgraph/dnnl_conv_batchnorm.h"
#include "core/providers/dnnl/subgraph/dnnl_activations.h"
#include "core/providers/dnnl/subgraph/dnnl_pool.h"
#include "core/providers/dnnl/subgraph/dnnl_sum.h"
#include "core/providers/dnnl/subgraph/dnnl_lrn.h"

namespace onnxruntime {
namespace ort_dnnl {

namespace {
template <typename T>
class SubgraphPrimitive : public PrimitiveBase {
 public:
  SubgraphPrimitive(const OrtCustomOpApi* api,
                    OrtKernelContext* context,
                    const SubgraphParams& params)
      : cpu_engine_(GetEngine()) {
    context_.stream = onnxruntime::make_unique<dnnl::stream>(dnnl::stream(cpu_engine_));

    if (context_.net.size() == 0) {
      CreateKernels(params);
      Initialize(api, context);
    }
  }

  void UpdateProvider(const SubgraphParams& params) {
    if (context_.kernels.size() > 0 && context_.kernels[0]->GetProvider() != params.provider)
      for (auto& kernel : context_.kernels) {
        kernel->SetProvider(params.provider);
      }
  }

  Status Compute(const OrtCustomOpApi* api, OrtKernelContext* context) {
    Status status;

    for (auto& kernel : context_.kernels) {
      ORT_RETURN_IF_ERROR(kernel->Bind(api, context));
    }
    for (size_t i = 0; i < context_.net.size(); ++i) {
      context_.net.at(i).execute(*context_.stream, context_.net_args.at(i));
    }
    return Status::OK();
  }

  ~SubgraphPrimitive() = default;

 private:
  void CreateKernels(const SubgraphParams& params) {
    for (const auto& dnnl_node : params.subgraph->dnnl_nodes) {
      if (dnnl_node.name == "Conv") {
        std::ostringstream os;
        os << "Conv-" << dnnl_node.node_index << "-";
        std::shared_ptr<DnnlConv<T>> kernel;
        kernel = std::make_shared<DnnlConv<T>>(dnnl_node, params.provider, *params.attributes, os.str());
        for (auto index : dnnl_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (dnnl_node.name == "Conv-Relu") {
        std::ostringstream os;
        os << "Conv-" << dnnl_node.node_index << "-";
        std::shared_ptr<DnnlConv<T>> kernel;
        kernel = std::make_shared<DnnlConv<T>>(dnnl_node, params.provider, *params.attributes, os.str());
        kernel->fuse_relu_ = true;
        for (auto index : dnnl_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (dnnl_node.name == "Relu") {
        std::ostringstream os;
        os << "Relu-" << dnnl_node.node_index << "-";
        std::shared_ptr<DnnlRelu<T>> kernel;
        kernel = std::make_shared<DnnlRelu<T>>(dnnl_node, params.provider, *params.attributes, os.str());
        for (auto index : dnnl_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (dnnl_node.name == "BatchNormalization") {
        std::ostringstream os;
        os << "BatchNormalization-" << dnnl_node.node_index << "-";
        std::shared_ptr<DnnlBatchNorm<T>> kernel;
        kernel = std::make_shared<DnnlBatchNorm<T>>(dnnl_node, params.provider, *params.attributes, os.str());
        for (auto index : dnnl_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (dnnl_node.name == "BatchNormalization-Relu") {
        std::ostringstream os;
        os << "BatchNormalization-" << dnnl_node.node_index << "-";
        std::shared_ptr<DnnlBatchNorm<T>> kernel;
        kernel = std::make_shared<DnnlBatchNorm<T>>(dnnl_node, params.provider, *params.attributes, os.str());
        kernel->fuse_relu_ = true;
        for (auto index : dnnl_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (dnnl_node.name == "Conv-BatchNormalization") {
        std::ostringstream os;
        os << "Conv-" << dnnl_node.node_index << "-";
        std::shared_ptr<DnnlConvBatchNorm<T>> kernel;
        kernel = std::make_shared<DnnlConvBatchNorm<T>>(dnnl_node, params.provider, *params.attributes, os.str());
        for (auto index : dnnl_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (dnnl_node.name == "Conv-BatchNormalization-Relu") {
        std::ostringstream os;
        os << "Conv-" << dnnl_node.node_index << "-";
        std::shared_ptr<DnnlConvBatchNorm<T>> kernel;
        kernel = std::make_shared<DnnlConvBatchNorm<T>>(dnnl_node, params.provider, *params.attributes, os.str());
        kernel->fuse_relu_ = true;
        for (auto index : dnnl_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (dnnl_node.name == "MaxPool") {
        std::ostringstream os;
        os << "MaxPool-" << dnnl_node.node_index << "-";
        std::shared_ptr<DnnlPool<T>> kernel;
        kernel = std::make_shared<DnnlPool<T>>(dnnl_node, params.provider, *params.attributes, os.str());
        for (auto index : dnnl_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (dnnl_node.name == "GlobalMaxPool") {
        std::ostringstream os;
        os << "GlobalMaxPool-" << dnnl_node.node_index << "-";
        std::shared_ptr<DnnlPool<T>> kernel;
        kernel = std::make_shared<DnnlPool<T>>(dnnl_node, params.provider, *params.attributes, os.str());
        for (auto index : dnnl_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (dnnl_node.name == "AveragePool") {
        std::ostringstream os;
        os << "AveragePool-" << dnnl_node.node_index << "-";
        std::shared_ptr<DnnlPool<T>> kernel;
        kernel = std::make_shared<DnnlPool<T>>(dnnl_node, params.provider, *params.attributes, os.str());
        for (auto index : dnnl_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (dnnl_node.name == "GlobalAveragePool") {
        std::ostringstream os;
        os << "GlobalAveragePool-" << dnnl_node.node_index << "-";
        std::shared_ptr<DnnlPool<T>> kernel;
        kernel = std::make_shared<DnnlPool<T>>(dnnl_node, params.provider, *params.attributes, os.str());
        for (auto index : dnnl_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (dnnl_node.name == "LRN") {
        std::ostringstream os;
        os << "LRN-" << dnnl_node.node_index << "-";
        std::shared_ptr<DnnlLrn<T>> kernel;
        kernel = std::make_shared<DnnlLrn<T>>(dnnl_node, params.provider, *params.attributes, os.str());
        for (auto index : dnnl_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (dnnl_node.name == "Sum") {
        std::ostringstream os;
        os << "Sum-" << dnnl_node.node_index << "-";
        std::shared_ptr<DnnlSum<T>> kernel;
        kernel = std::make_shared<DnnlSum<T>>(dnnl_node, params.provider, *params.attributes, os.str());
        for (auto index : dnnl_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      }
    }
  }

  struct SubgraphContext {
    std::unique_ptr<dnnl::stream> stream;
    std::vector<dnnl::primitive> net;
    std::vector<std::unordered_map<int, dnnl::memory>> net_args;
    std::vector<std::shared_ptr<DnnlKernel>> kernels;

    SubgraphContext() : stream(nullptr) {}
  };

  void Initialize(const OrtCustomOpApi* api, OrtKernelContext* context) {
    // Propagate Dnnl block format
    // dst format of current node to src format of next node
    for (auto& kernel : context_.kernels) {
      kernel->CreatePrimitives(api, context, cpu_engine_, context_.net, context_.net_args);
      if (kernel->primitive_created_status_.IsOK()) {
        kernel->ReorderWeights(api, context, cpu_engine_);
      }
    }
  }

  SubgraphContext context_;
  dnnl::engine& cpu_engine_;
};

// Pool which allows for reuse of DNNL Conv primitives which are expensive to instantiate.
// To address thread safety, the primitives are stored in a map on thread local storage.
template <typename T>
class SubgraphPrimitivePool : public PrimitivePool<T> {
 public:
  static SubgraphPrimitive<T>* Get(const OrtCustomOpApi* api,
                                   OrtKernelContext* context,
                                   const SubgraphParams& params) {
    Ort::CustomOpApi ort{*api};
    std::string dims_str;
    for (auto i = 0; i < params.subgraph->dnnl_nodes[0].num_inputs; i++) {
      const OrtValue* input_tensor = ort.KernelContext_GetInput(context, i);
      auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
      auto tensor_shape = ort.GetTensorShape(tensor_info);
      ort.ReleaseTensorTypeAndShapeInfo(tensor_info);

      dnnl::memory::dims src_dims(tensor_shape);
      AddDimsToKey(dims_str, src_dims);
    }

    SubgraphPrimitive<T>* primitive = dynamic_cast<SubgraphPrimitive<T>*>(
        SubgraphPrimitivePool<T>::GetInstance().GetPrimitive(params.subgraph_key + dims_str));

    if (primitive == nullptr) {
      auto subgraph_primitive = onnxruntime::make_unique<SubgraphPrimitive<T>>(api, context, params);
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
Status DnnlFuncKernel<T>::Compute(const OrtCustomOpApi* api, OrtKernelContext* context) const {
  Status status;
  try {
    SubgraphPrimitive<T>* primitive = SubgraphPrimitivePool<T>::Get(api, context, params_);
    primitive->UpdateProvider(params_);
    status = primitive->Compute(api, context);
  } catch (const dnnl::error& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Status: ", e.status, ", message: ", e.what());
  }
  return status;
}

template class DnnlFuncKernel<float>;

}  // namespace ort_dnnl
}  // namespace onnxruntime
