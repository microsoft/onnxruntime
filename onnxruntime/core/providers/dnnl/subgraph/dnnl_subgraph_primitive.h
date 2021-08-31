// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"
#include "dnnl.hpp"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {
namespace ort_dnnl {

//borrow from coreml ep's data structures to organize data handle, shape and data type
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

  //compile subgraph primitive with runtime input information
  void Compile(const std::unordered_map<std::string, OnnxTensorData>& inputs);
  void AddInitializers();
  void AddOutputs();
  void AddKernels();

  //run inference
  onnxruntime::common::Status Predict(const std::unordered_map<std::string, OnnxTensorData>& inputs, const std::unordered_map<std::string, OnnxTensorData>& outputs);

  void SetOrderedInputs(std::vector<std::string>&& inputs);
  void SetOrderedOutputs(std::vector<std::string>&& outputs);
  const std::vector<std::string>& GetOrderedInputs() const;
  const std::vector<std::string>& GetOrderedOutputs() const;

  //get corresponding DNNL format from dim size in onnxruntime
  dnnl::memory::format_tag GetDnnlFormat(size_t dim_size);
  dnnl::engine GetCPUEngine();
  dnnl::engine GetEngine();
  dnnl::stream GetStream();

  //obtain a dnnl::memory with specified name, memory descriptor and engine, will perform extra reorder/reshape if necessary before returning
  dnnl::memory GetMemoryAndReshape(const DnnlTensor& tensor, dnnl::memory::desc mem_desc, dnnl::engine eng, bool transpose = false);
  //add dnnl primitive and memory map to subgraph primitive
  void AddPrimitive(dnnl::primitive prim, std::unordered_map<int, dnnl::memory> mem_map);
  //add a reshape (e.g. squeeze, unsqueeze) to subgraph primitive
  void AddReshape(dnnl::memory src, dnnl::memory dst);
  bool HasMemory(std::string memory_name, dnnl::memory::desc mem_desc, dnnl::engine eng);
  dnnl::memory GetMemory(const DnnlTensor& tensor);
  dnnl::memory GetMemory(const DnnlTensor& tensor, dnnl::memory::desc mem_desc, dnnl::engine eng);
  //set memory to a tensor (output)
  // if always_copy_output is true a copy of the memory will be made when the output is leaving the subgraph.
  void SetMemory(DnnlTensor tensor, dnnl::memory mem, bool always_copy_output = false);
  void SetMemory(std::string memory_name, dnnl::memory mem);
  void SetInitializer(std::string memory_name, dnnl::memory mem);
  dnnl::memory::desc GetOutputInfo(std::string name);
  bool IsDynamic();
  OrtMutex& GetMutex() { return mutex_; }

 private:
  std::string shape_key_;

  std::unordered_map<std::string, std::vector<dnnl::memory>> intermediates_;

  std::unordered_map<std::string, dnnl::memory> inputs_;
  std::unordered_map<std::string, dnnl::memory::desc> inputs_md_;

  std::unordered_map<std::string, dnnl::memory> outputs_;
  std::unordered_map<std::string, dnnl::memory::desc> outputs_md_;
  std::unordered_set<std::string> outputs_are_always_copied_;

  //initializer should not be dynamic
  std::unordered_map<std::string, std::vector<dnnl::memory>> initializers_;

  std::vector<dnnl::primitive> net_;
  std::vector<std::unordered_map<int, dnnl::memory>> net_args_;

  std::vector<std::pair<dnnl::memory, dnnl::memory>> reshapes_;

  ort_dnnl::DnnlSubgraph* subgraph_;

  std::vector<std::string> ordered_inputs_;
  std::vector<std::string> ordered_outputs_;

  dnnl::engine cpu_engine_;
  dnnl::engine gpu_engine_;

  OrtMutex mutex_;
};

}  // namespace ort_dnnl
}  // namespace onnxruntime
