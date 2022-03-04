// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef TVM_RUNNER_IMPL_H
#define TVM_RUNNER_IMPL_H

#include <string>
#include <memory>
#include <map>

#include "core/framework/func_api.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "tvm_common.h"

namespace onnxruntime {
namespace tvm {

class RunnerImpl {
public:
    using TVMTensorShape = std::vector<int64_t>;
    using TVMTensorShapes = std::vector<TVMTensorShape>;
    using InputsInfoMap = std::map<size_t, TVMTensorShape>;

    RunnerImpl() = delete;
    RunnerImpl(const std::shared_ptr<TvmModule>& mod);
    virtual ~RunnerImpl() = default;

    virtual common::Status run(FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context);

    virtual void set_input(Ort::CustomOpApi& ort, OrtKernelContext* context) = 0;
    virtual void connect_output_tensors2ort(Ort::CustomOpApi& ort, OrtKernelContext* context) = 0;
    virtual void run_and_get_output() = 0;

protected:
    void convert_input_tensors2dl_tensors(Ort::CustomOpApi& ort,
                                          OrtKernelContext* context,
                                          std::vector<DLTensor>& dst,
                                          std::vector<size_t>& dst_inds);
    void add_device_type_data2output_tensors(Ort::CustomOpApi& ort,
                                             OrtKernelContext* context);

private:
    bool compare_shapes(const TVMTensorShape& shape1, const TVMTensorShape& shape2) const;

protected:
    std::shared_ptr<TvmModule> mod_;
    InputsInfoMap inputs_info_{};
    TVMTensorShapes output_shapes_;
    std::vector<DLTensor> tensors_outputs_;
};

class GERunnerImpl : public RunnerImpl {
public:
    GERunnerImpl() = delete;
    GERunnerImpl(const std::shared_ptr<TvmModule>& mod);
    virtual ~GERunnerImpl() = default;

    virtual void set_input(Ort::CustomOpApi& ort, OrtKernelContext* context) override final;
    virtual void connect_output_tensors2ort(Ort::CustomOpApi& ort, OrtKernelContext* context) override final;
    virtual void run_and_get_output() override final;
};

class VMRunnerImpl : public RunnerImpl {
public:
    VMRunnerImpl() = delete;
    VMRunnerImpl(const std::shared_ptr<TvmModule>& mod);
    virtual ~VMRunnerImpl() = default;

    virtual void set_input(Ort::CustomOpApi& ort, OrtKernelContext* context) override final;
    virtual void connect_output_tensors2ort(Ort::CustomOpApi& ort, OrtKernelContext* context) override final;
    virtual void run_and_get_output() override final;

private:
    void infer_once_to_get_output_shapes();

private:
    bool probe_infer_ = false;
};

std::shared_ptr<RunnerImpl> getTVMRunnerImpl(const std::string& name, const std::shared_ptr<TvmModule>& mod);

}   // namespace tvm
}   // namespace onnxruntime

#endif  // TVM_TVM_RUNNER_IMPL_H
