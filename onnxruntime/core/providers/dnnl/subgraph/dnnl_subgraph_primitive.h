// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"
#include "dnnl.hpp"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {
namespace ort_dnnl {

// borrow from coreml ep's data structures to organize data handle, shape and data type
struct OnnxTensorInfo {
  const int32_t data_type;  // Uses TensorProto::DataType
  const std::vector<int64_t> shape;
};

struct OnnxTensorData {
  OnnxTensorInfo tensor_info;
  void* buffer{nullptr};
};

class DnnlSubgraphPrimitive {
 public:
  DnnlSubgraphPrimitive(ort_dnnl::DnnlSubgraph& dnnl_subgraph);
  ~DnnlSubgraphPrimitive() = default;

  // compile subgraph primitive with runtime input information
  void Compile(const std::unordered_map<std::string, OnnxTensorData>& inputs);
  void AddInitializers();
  void AddOutputs();
  void AddKernels();

  // run inference
  onnxruntime::common::Status Predict(const std::unordered_map<std::string, OnnxTensorData>& inputs, const std::unordered_map<std::string, OnnxTensorData>& outputs);

  void SetOrderedInputs(std::vector<std::string>&& inputs);
  void SetOrderedOutputs(std::vector<std::string>&& outputs);
  const std::vector<std::string>& GetOrderedInputs() const;
  const std::vector<std::string>& GetOrderedOutputs() const;

  // get corresponding DNNL format from dim size in onnxruntime
  dnnl::memory::format_tag GetDnnlFormat(size_t dim_size);
  dnnl::engine GetCPUEngine();
  dnnl::engine GetEngine();
  dnnl::stream GetStream();

  // obtain a dnnl::memory with specified name, memory descriptor and engine, will perform extra reorder/reshape if necessary before returning
  dnnl::memory GetMemoryAndReshape(const DnnlTensor& tensor, dnnl::memory::desc mem_desc, dnnl::engine eng, bool transpose = false);
  // add dnnl primitive and memory map to subgraph primitive
  // when you add primitive, you can optionally specify a vector of indexes to be printed in runtime for debug purpose
  // eg, sp.AddPrimitve(prim,mem_map,{DNNL_ARG_SRC})
  void AddPrimitive(dnnl::primitive prim, std::unordered_map<int, dnnl::memory> mem_map, std::vector<int> items_to_print = {});
  // add a reshape (e.g. squeeze, unsqueeze) to subgraph primitive
  void AddReshape(dnnl::memory src, dnnl::memory dst);
  bool HasMemory(std::string memory_name, dnnl::memory::desc mem_desc, dnnl::engine eng);
  dnnl::memory GetMemory(const DnnlTensor& tensor);
  dnnl::memory GetMemory(const DnnlTensor& tensor, dnnl::memory::desc mem_desc, dnnl::engine eng);
  // set memory to a tensor (output)
  // if always_copy_output is true a copy of the memory will be made when the output is leaving the subgraph.
  // is_scalar is true to indicate a scalar output in order to allocate the correct onnxruntime output buffer
  void SetMemory(const DnnlTensor& tensor, dnnl::memory mem, bool always_copy_output = false, bool is_scalar = false);
  void SetMemory(std::string memory_name, dnnl::memory mem);
  void SetInitializer(std::string memory_name, dnnl::memory mem);
  dnnl::memory::desc GetOutputInfo(std::string name);
  bool IsScalarOutput(const std::string& name);
  bool IsDynamic();
  // All Scalar inputs are automatically converterted to a one dimentional tensor when used in OneDNN
  // If the input being a scalar affects the operator this function can be used to determine if the
  // original input from ORT was a scalar.
  bool IsScalar(const DnnlTensor& tensor);
  OrtMutex& GetMutex() { return mutex_; }

  // GetMemory in OrtFormat if the memory is not in the OrtFormat this will reorder the memory.
  // All memory will be moved to the dnnl_engine even if it is already in OrtFormat.
  dnnl::memory GetMemoryInOrtFormat(const DnnlTensor& tensor, const dnnl::engine& eng);
  bool IsMemoryInExpectedOrtFormat(const dnnl::memory::desc& desc) const;

  template <typename T>
  void WriteToDnnlMemory(dnnl::memory& mem, std::vector<T> values) {
    if (mem.get_engine().get_kind() == dnnl::engine::kind::gpu) {
      // Create a CPU memory
      auto cpu_memory = dnnl::memory(mem.get_desc(), GetCPUEngine());
      // Copy data from the vector into the CPU memory data handle
      std::copy(values.begin(), values.end(), static_cast<T*>(cpu_memory.get_data_handle()));
      // Use reorder to copy data from CPU to GPU
      dnnl::stream s{mem.get_engine()};
      // mem now contains all zero
      dnnl::reorder(cpu_memory, mem).execute(s, cpu_memory, mem);
      // wait for reorder to complete
      s.wait();
    } else {
      // Copy data from the vector into the memory data handle
      std::copy(values.begin(), values.end(), static_cast<T*>(mem.get_data_handle()));
    }
  }

 private:
  std::string shape_key_;

  std::unordered_map<std::string, std::vector<dnnl::memory>> intermediates_;

  std::unordered_map<std::string, dnnl::memory> inputs_;
  std::unordered_map<std::string, dnnl::memory::desc> inputs_md_;
  std::unordered_set<std::string> input_is_scalar_;

  std::unordered_map<std::string, dnnl::memory> outputs_;
  std::unordered_map<std::string, dnnl::memory::desc> outputs_md_;
  std::unordered_set<std::string> outputs_are_always_copied_;

  // initializer should not be dynamic
  std::unordered_map<std::string, std::vector<dnnl::memory>> initializers_;

  std::vector<dnnl::primitive> net_;
  std::vector<std::unordered_map<int, dnnl::memory>> net_args_;

  std::vector<std::pair<dnnl::memory, dnnl::memory>> reshapes_;
  std::unordered_set<std::string> scalar_outputs_;

  ort_dnnl::DnnlSubgraph* subgraph_;

  std::vector<std::string> ordered_inputs_;
  std::vector<std::string> ordered_outputs_;

  dnnl::engine cpu_engine_;
  dnnl::engine gpu_engine_;

  OrtMutex mutex_;

  // for memory debug purpose
  std::vector<std::pair<int, int>> items_to_print_;
  void PrintMemory(const dnnl::memory& mem);
};

}  // namespace ort_dnnl

inline std::ostream& operator<<(std::ostream& os, const dnnl::memory::dims& dims) {
  std::copy(dims.cbegin(), dims.cend(), std::ostream_iterator<dnnl::memory::dim>(os, " "));
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const gsl::span<const int64_t>& span) {
  std::copy(span.begin(), span.end(), std::ostream_iterator<int64_t>(os, " "));
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const gsl::span<int64_t>& span) {
  std::copy(span.begin(), span.end(), std::ostream_iterator<int64_t>(os, " "));
  return os;
}

}  // namespace onnxruntime
