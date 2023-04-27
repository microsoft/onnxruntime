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
             const std::vector<DLTensor> tensors_outputs,
             bool set_output_zero_copy);
  virtual ~RunnerImpl() = default;

  virtual common::Status run(Ort::KernelContext& context) {
    common::Status res;
    if (set_output_zero_copy_) {
      res = run_without_output_copying(context);
    } else {
      res = run_with_output_copying(context);
    }
    return res;
  }

  virtual common::Status run_without_output_copying(Ort::KernelContext& context) {
    set_input(context);
    connect_output_tensors2ort(context);
    set_output_zero_copy();
    run();

    return Status::OK();
  }

  virtual common::Status run_with_output_copying(Ort::KernelContext& context) {
    set_input(context);
    connect_output_tensors2ort(context);
    run();
    get_outputs();

    return Status::OK();
  }

  virtual void set_input(Ort::KernelContext& context) = 0;
  virtual void connect_output_tensors2ort(Ort::KernelContext& context) = 0;
  virtual void set_output_zero_copy() = 0;
  virtual void run() = 0;
  virtual void get_outputs() = 0;

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
  bool set_output_zero_copy_;
};

class GERunnerImpl : public RunnerImpl {
 public:
  GERunnerImpl() = delete;
  GERunnerImpl(const std::shared_ptr<TvmModule>& mod,
               const InputsInfoMap& inputs_info,
               const TVMTensorShapes output_shapes,
               const std::vector<DLTensor> tensors_outputs,
               bool set_output_zero_copy);
  virtual ~GERunnerImpl() = default;

  void set_input(Ort::KernelContext& context) final;
  void connect_output_tensors2ort(Ort::KernelContext& context) final;
  void set_output_zero_copy() final;
  void run() final;
  void get_outputs() final;
};

class VMRunnerImpl : public RunnerImpl {
 public:
  VMRunnerImpl() = delete;
  VMRunnerImpl(const std::shared_ptr<TvmModule>& mod,
               const InputsInfoMap& inputs_info,
               const TVMTensorShapes output_shapes,
               const std::vector<DLTensor> tensors_outputs,
               bool set_output_zero_copy);
  virtual ~VMRunnerImpl() = default;

  void set_input(Ort::KernelContext& context) final;
  void connect_output_tensors2ort(Ort::KernelContext& context) final;
  void set_output_zero_copy() final;
  void run() final;
  void get_outputs() final;

 private:
  void infer_once_to_get_output_shapes();

 private:
  bool probe_infer_ = false;
};

std::shared_ptr<RunnerImpl> getTVMRunnerImpl(const std::shared_ptr<TvmModule>& mod,
                                             const TvmEPOptions& options,
                                             const InputsInfoMap& inputs_info,
                                             const std::vector<DLTensor> output_tensors);

}  // namespace tvm
}  // namespace onnxruntime

#endif  // TVM_TVM_RUNNER_IMPL_H
