// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef TVM_RUNNER_H
#define TVM_RUNNER_H

#include <string>
#include <vector>
#include <memory>
#include <map>

#include "tvm_runner_impl.h"
#include "tvm_ep_options.h"


namespace ONNX_NAMESPACE {
    struct TensorShapeProto;
}

namespace onnxruntime {
    class Graph;
    class NodeArg;
namespace tvm {
    class TVMCompiler;

class TVMRunner {
public:
    using TVMTensorShape = std::vector<int64_t>;
    using TVMTensorShapes = std::vector<TVMTensorShape>;
    using InputsInfoMap = std::map<size_t, TVMTensorShape>;
    using ORTGraphNodes = std::vector<const NodeArg*>;

    TVMRunner() = delete;
    virtual ~TVMRunner() = default;

    TVMRunner(TvmEPOptions options,
              std::shared_ptr<TVMCompiler> compiler,
              const Graph& graph);

    common::Status operator()(FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context);

private:
    void getTensorInfo(const ONNX_NAMESPACE::TensorShapeProto& shape_proto,
                       TVMTensorShape& ishape,
                       size_t indx);

    bool compare_shapes(const TVMTensorShape& shape1, const TVMTensorShape& shape2);

private:
    std::shared_ptr<RunnerImpl> runner_;
    std::shared_ptr<TvmModule> mod_;
    bool use_vm_ = true;
    bool probe_infer_ = false;
    InputsInfoMap inputs_info_{};
    bool update_output_shapes_ = false;
    TVMTensorShapes output_shapes_;
    std::vector<DLTensor> tensors_outputs_;
};

}   // namespace tvm
}   // namespace onnxruntime

#endif  // TVM_TVM_RUNNER_H
