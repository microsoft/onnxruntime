// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef TRACE_EXECUTION

#include "session_profiler_base.h"
namespace onnxruntime {

//for session profiling
class TraceSessProfiler : public ISessProfiler {
 public:
  TraceSessProfiler(const SessionState& sess_state,
                    const ExecutionFrame& frame) : ISessProfiler(sess_state, frame)
#ifdef ANOTHER_SESS_PROFILER
                                                   ,
                                                   another_sess_profiler_(sess_state, frame)
#endif

  {
    const auto& seq_exec_plan = sess_state_.GetExecutionPlan();
    std::cout << std::make_pair(&seq_exec_plan, &sess_state_) << std::endl;
  }
#ifdef ANOTHER_SESS_PROFILER
  ISessProfiler& GetAnotherSessProfiler() override { return another_sess_profiler_; }
  ANOTHER_SESS_PROFILER another_sess_profiler_;
#endif
};
#undef ANOTHER_SESS_PROFILER
#define ANOTHER_SESS_PROFILER TraceSessProfiler
//for kernel profiling
struct TraceKernelProfiler : public IKernelProfiler {
  TraceKernelProfiler(ISessProfiler& sess_profiler,
                      const OpKernelContextInternal& context,
                      const OpKernel& kernel) : IKernelProfiler(sess_profiler, context, kernel),
#ifdef ANOTHER_KERNEL_PROFILER
                                                ,
                                                another_kernel_profiler_(sess_profiler.GetAnotherSessProfiler(), context, kernel)
#endif
  {
    const int input_count = context_.InputCount();
    for (auto i = 0; i < input_count; i++) {
      const OrtValue* p_input = context_.GetInputMLValue(i);
      if (p_input && p_input->IsTensor()) {
        const OpKernelInfo& op_kernel_info = kernel_.Info();
        const Tensor* p_tensor = nullptr;
        bool is_param = op_kernel_info.TryGetConstantInput(i, &p_tensor);
        if (!is_param) {
          p_tensor = &(p_input->Get<Tensor>());
        }
        size_t tensor_size = p_tensor->SizeInBytes();
        const TensorShape& tensor_shape = p_tensor->Shape();
        size_t element_size = p_tensor->DataType()->Size();
        std::cout << kernel_.Node().Name() << " input[" << i << "]"
                  << " is_param=" << is_param
                  << " size=" << tensor_size
                  << " shape=" << tensor_shape.ToString()
                  << " element_size=" << element_size
                  << std::endl;
      }
    }
  }
  ~TraceKernelProfiler() {
    int output_count = context_.OutputCount();
    for (auto i = 0; i < output_count; i++) {
      const OrtValue* p_output = context_.GetOutputMLValue(i);
      if (p_output != nullptr && p_output->IsTensor()) {
        const auto& tensor = p_output->Get<Tensor>();
        size_t tensor_size = tensor.SizeInBytes();
        const TensorShape& tensor_shape = tensor.Shape();
        std::cout << node_name << " output[" << i << "]"
                  << " size=" << tensor_size
                  << " shape=" << tensor_shape.ToString()
                  << " element_size=" << tensor.DataType()->Size()
                  << std::endl;
      }
    }
    auto& node = kernel_.Node();
    std::cout << "Executed op kernel node " << node.Name()
              << " Index=" << node.Index()
              << " OpType=" << node.OpType()
              << " Name=" << node.Name()
              << std::endl;
  }
#ifdef ANOTHER_KERNEL_PROFILER
  ANOTHER_KERNEL_PROFILER another_kernel_profiler_;
#endif
};
#undef ANOTHER_KERNEL_PROFILER
#define ANOTHER_KERNEL_PROFILER TraceKernelProfiler
}  // namespace onnxruntime
#endif