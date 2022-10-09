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
#include "tvm_ep_options.h"


namespace onnxruntime {
namespace tvm {

class RunnerImpl {
public:
  RunnerImpl() = delete;
  RunnerImpl(const std::shared_ptr<TvmModule>& mod,
             const InputsInfoMap& inputs_info,
             const TVMTensorShapes output_shapes,
             const std::vector<DLTensor> tensors_outputs);
  virtual ~RunnerImpl() = default;

  virtual common::Status run(const OrtApi* /* api */, OrtKernelContext* context) {

    Ort::KernelContext ctx{context};
    set_input(ctx);
    connect_output_tensors2ort(ctx);
    run_and_get_output();

    return Status::OK();
  }

  virtual void set_input(Ort::KernelContext& ctx) = 0;
  virtual void connect_output_tensors2ort(Ort::KernelContext& context) = 0;
  virtual void run_and_get_output() = 0;

protected:
  void convert_input_tensors2dl_tensors(Ort::KernelContext& context,
                                        std::vector<DLTensor>& dst,
                                        std::vector<size_t>& dst_inds);
  void add_device_type_data2output_tensors(Ort::KernelContext& context);

protected:
  std::shared_ptr<TvmModule> mod_;
  InputsInfoMap inputs_info_;
  TVMTensorShapes output_shapes_;
  std::vector<DLTensor> output_tensors_;
};


class GERunnerImpl : public RunnerImpl {
public:
  GERunnerImpl() = delete;
  GERunnerImpl(const std::shared_ptr<TvmModule>& mod,
               const InputsInfoMap& inputs_info,
               const TVMTensorShapes output_shapes,
               const std::vector<DLTensor> tensors_outputs);
  virtual ~GERunnerImpl() = default;

  void set_input(Ort::KernelContext& context) override final;
  void connect_output_tensors2ort(Ort::KernelContext& context) override final;
  void run_and_get_output() override final;
};


class VMRunnerImpl : public RunnerImpl {
public:
  VMRunnerImpl() = delete;
  VMRunnerImpl(const std::shared_ptr<TvmModule>& mod,
               const InputsInfoMap& inputs_info,
               const TVMTensorShapes output_shapes,
               const std::vector<DLTensor> tensors_outputs);
  virtual ~VMRunnerImpl() = default;

  void set_input(Ort::KernelContext& context) override final;
  void connect_output_tensors2ort(Ort::KernelContext& context) override final;
  void run_and_get_output() override final;

private:
    void infer_once_to_get_output_shapes();

private:
    bool probe_infer_ = false;
};


std::shared_ptr<RunnerImpl> getTVMRunnerImpl(const std::shared_ptr<TvmModule>& mod,
                                             const TvmEPOptions& options,
                                             const InputsInfoMap& inputs_info,
                                             const std::vector<DLTensor> output_tensors);

}   // namespace tvm
}   // namespace onnxruntime

#endif  // TVM_TVM_RUNNER_IMPL_H
