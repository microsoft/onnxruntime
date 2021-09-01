// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_subgraph_primitive.h"

#include "dnnl_batchnorm.h"
#include "dnnl_binary.h"
#include "dnnl_conv.h"
#include "dnnl_elementwise.h"
#include "dnnl_gemm.h"
#include "dnnl_lrn.h"
#include "dnnl_matmul.h"
#include "dnnl_matmul_integer.h"
#include "dnnl_pool.h"
#include "dnnl_reducemean.h"
#include "dnnl_softmax.h"
#include "dnnl_sum.h"

#if defined(ENABLE_TRAINING)
#include "dnnl_convgrad.h"
#include "dnnl_poolgrad.h"
#include "dnnl_relugrad.h"
#endif

namespace onnxruntime {
namespace ort_dnnl {

template <class Map, class Key>
inline bool Contains(const Map& map, const Key& key) {
  return map.find(key) != map.end();
}

int Product(dnnl::memory::dims d) {
  int result = 1;
  for (const auto& e : d)
    result *= (int)e;
  return result;
}

void DnnlSubgraphPrimitive::AddKernels() {
  std::unordered_set<std::string> binary_ops = {"Add", "Div", "Mul", "Sub"};
  std::unordered_set<std::string> elementwise_ops = {"Abs", "Elu", "Exp","Log", "Relu", "Round", "Sigmoid", "Softplus", "Sqrt", "Tanh"};
  std::unordered_set<std::string> pool_ops = {"AveragePool", "GlobalAveragePool", "GlobalMaxPool", "MaxPool"};
  for (auto& node : subgraph_->GetDnnlNodes()) {
    if (node.OpType() == "BatchNormalization") {
      DnnlBatchNorm().CreatePrimitive(*this, node);
    } else if (binary_ops.count(node.OpType())) {
      DnnlBinary().CreatePrimitive(*this, node);
    } else if (node.OpType() == "Conv") {
      DnnlConv().CreatePrimitive(*this, node);
    } else if (elementwise_ops.count(node.OpType())) {
      DnnlElementwise().CreatePrimitive(*this, node);
    } else if (node.OpType() == "Gemm") {
      DnnlGemm().CreatePrimitive(*this, node);
    } else if (node.OpType() == "LRN") {
      DnnlLrn().CreatePrimitive(*this, node);
    } else if (node.OpType() == "MatMul") {
      DnnlMatMul().CreatePrimitive(*this, node);
    } else if (node.OpType() == "MatMulInteger") {
      DnnlMatMulInteger().CreatePrimitive(*this, node);
    } else if (pool_ops.count(node.OpType())) {
      DnnlPool().CreatePrimitive(*this, node);
    } else if (node.OpType() == "ReduceMean") {
      DnnlReduceMean().CreatePrimitive(*this, node);
    } else if (node.OpType() == "Softmax") {
      DnnlSoftmax().CreatePrimitive(*this, node);
    } else if (node.OpType() == "Sum") {
      DnnlSum().CreatePrimitive(*this, node);
#if defined(ENABLE_TRAINING)
    } else if (node.OpType() == "AveragePoolGrad" || node.OpType() == "MaxPoolGrad") {
      DnnlPoolGrad().CreatePrimitive(*this, node);
    } else if (node.OpType() == "ConvGrad") {
      DnnlConvGrad().CreatePrimitive(*this, node);
    } else if (node.OpType() == "ReluGrad") {
      DnnlReluGrad().CreatePrimitive(*this, node);
#endif
    } else {
      throw std::invalid_argument("Kernel not found");
    }
  }
}

DnnlSubgraphPrimitive::DnnlSubgraphPrimitive(ort_dnnl::DnnlSubgraph& dnnl_subgraph) {
  subgraph_ = &dnnl_subgraph;
  if (dnnl_engine_get_count(dnnl_engine_kind_t::dnnl_cpu)) {
    cpu_engine_ = dnnl::engine(dnnl::engine::kind::cpu, 0);
  }

  if (dnnl_engine_get_count(dnnl_engine_kind_t::dnnl_gpu)) {
    gpu_engine_ = dnnl::engine(dnnl::engine::kind::gpu, 0);
  }
}

bool DnnlSubgraphPrimitive::IsDynamic() {
  return subgraph_->IsDynamic();
}

void DnnlSubgraphPrimitive::Compile(const std::unordered_map<std::string, OnnxTensorData>& inputs) {
  //if already compiled once and is not dynamic, then don't compile again
  if (!shape_key_.empty() && !IsDynamic()) {
    return;
  }

  std::string key;
  for (auto input : inputs) {
    for (auto dim : input.second.tensor_info.shape) {
      std::ostringstream o;
      o << dim;
      key += o.str();
      key += ",";
    }
    key += "|";
  }
  // if key different from shape key, update and recompile
  if (key != shape_key_) {
    shape_key_ = key;
  } else {
    return;
  }
  if (IsDynamic()) {
    LOGS_DEFAULT(INFO) << "Dynamic Compile";
  } else {
    LOGS_DEFAULT(INFO) << "Static Compile";
  }

  inputs_.clear();
  intermediates_.clear();
  outputs_.clear();
  inputs_md_.clear();
  outputs_md_.clear();
  net_.clear();
  net_args_.clear();
  reshapes_.clear();
  //initializer should not be cleared upon recompile
  //initializers_.clear();

  for (auto& nodearg : subgraph_->GetDnnlInputs()) {
    auto dnnl_tensor_name = nodearg.Name();
    auto dnnl_data_type = nodearg.Type();
    dnnl::memory::dims dnnl_dims = inputs.at(dnnl_tensor_name).tensor_info.shape;
    if (dnnl_dims.size() == 0) {
      dnnl_dims.push_back(1);
    }
    auto dnnl_format = GetDnnlFormat(dnnl_dims.size());
    auto input_md = dnnl::memory::desc(dnnl_dims, dnnl_data_type, dnnl_format);
    inputs_md_.emplace(dnnl_tensor_name, input_md);
    auto engine = GetCPUEngine();
    auto input_mem = dnnl::memory(input_md, engine, inputs.at(dnnl_tensor_name).buffer);
    inputs_.emplace(dnnl_tensor_name, input_mem);
  }

  AddInitializers();
  AddKernels();
  AddOutputs();
}

dnnl::memory::format_tag DnnlSubgraphPrimitive::GetDnnlFormat(size_t dim_size) {
  dnnl::memory::format_tag source_format = dnnl::memory::format_tag::any;
  switch (dim_size) {
    case 1: {
      source_format = dnnl::memory::format_tag::x;
      break;
    }
    case 2: {
      source_format = dnnl::memory::format_tag::nc;
      break;
    }
    case 3: {
      source_format = dnnl::memory::format_tag::ncw;
      break;
    }
    case 4: {
      source_format = dnnl::memory::format_tag::nchw;
      break;
    }
    case 5: {
      source_format = dnnl::memory::format_tag::ncdhw;
      break;
    }
    case 6: {
      source_format = dnnl::memory::format_tag::abcdef;
      break;
    }
    case 7: {
      source_format = dnnl::memory::format_tag::abcdefg;
      break;
    }
    case 8: {
      source_format = dnnl::memory::format_tag::abcdefgh;
      break;
    }
    case 9: {
      source_format = dnnl::memory::format_tag::abcdefghi;
      break;
    }
    case 10: {
      source_format = dnnl::memory::format_tag::abcdefghij;
      break;
    }
    case 11: {
      source_format = dnnl::memory::format_tag::abcdefghijk;
      break;
    }
    case 12: {
      source_format = dnnl::memory::format_tag::abcdefghijkl;
      break;
    }
    default: {
      source_format = dnnl::memory::format_tag::any;
      break;
    }
  }
  return source_format;
}

dnnl::engine DnnlSubgraphPrimitive::GetCPUEngine() {
  return cpu_engine_;
}

dnnl::engine DnnlSubgraphPrimitive::GetEngine() {
  if (gpu_engine_) {
    return gpu_engine_;
  }
  return cpu_engine_;
}

dnnl::stream DnnlSubgraphPrimitive::GetStream() {
  return dnnl::stream(GetEngine());
}

void DnnlSubgraphPrimitive::AddInitializers() {
  for (auto& nodearg : subgraph_->GetDnnlInitializers()) {
    auto dnnl_tensor_name = nodearg.Name();
    if (!Contains(initializers_, dnnl_tensor_name)) {
      initializers_.insert(std::pair<std::string, std::vector<dnnl::memory> >(dnnl_tensor_name, std::vector<dnnl::memory>()));
    }
  }
}

void DnnlSubgraphPrimitive::AddOutputs() {
  for (auto& tensor : subgraph_->GetDnnlOutputs()) {
    auto dnnl_data_type = tensor.Type();
    auto dnnl_tensor_name = tensor.Name();
    auto engine = GetCPUEngine();
    auto output_mem_dnnl = GetMemory(dnnl_tensor_name);
    auto output_md = dnnl::memory::desc(output_mem_dnnl.get_desc().dims(), dnnl_data_type, GetDnnlFormat(output_mem_dnnl.get_desc().dims().size()));
    // if output already in correct memory format, just place it to outputs instead of reorder
    if (output_mem_dnnl.get_desc() == output_md && output_mem_dnnl.get_engine() == engine) {
      outputs_.emplace(dnnl_tensor_name, output_mem_dnnl);
    } else {
      auto output_mem = dnnl::memory(output_md, engine, nullptr);
      AddPrimitive(dnnl::reorder(output_mem_dnnl, output_mem), {{DNNL_ARG_FROM, output_mem_dnnl},
                                                                {DNNL_ARG_TO, output_mem}});
      outputs_.emplace(dnnl_tensor_name, output_mem);
    }
  }
}

bool DnnlSubgraphPrimitive::HasMemory(std::string memory_name, dnnl::memory::desc mem_desc, dnnl::engine eng) {
  if (Contains(initializers_, memory_name)) {
    for (auto& mem : initializers_.at(memory_name)) {
      if (mem.get_engine() == eng && mem.get_desc() == mem_desc) {
        return true;
      }
    }
  }

  if (Contains(inputs_, memory_name)) {
    auto& mem = inputs_.at(memory_name);
    if (mem.get_engine() == eng && mem.get_desc() == mem_desc) {
      return true;
    }
  }

  if (Contains(intermediates_, memory_name)) {
    for (auto& mem : intermediates_.at(memory_name)) {
      if (mem.get_engine() == eng && mem.get_desc() == mem_desc) {
        return true;
      }
    }
  }

  return false;
}

void DnnlSubgraphPrimitive::SetMemory(DnnlTensor tensor, dnnl::memory mem) {
  SetMemory(tensor.Name(), mem);
}

dnnl::memory DnnlSubgraphPrimitive::GetMemory(std::string memory_name) {
  if (Contains(initializers_, memory_name)) {
    if (!initializers_.at(memory_name).empty()) {
      return initializers_.at(memory_name)[0];
    }
  }

  if (Contains(inputs_, memory_name)) {
    return inputs_.at(memory_name);
  }

  if (Contains(intermediates_, memory_name)) {
    return intermediates_.at(memory_name)[0];
  }

  throw std::invalid_argument("cannot find memory");
}

dnnl::memory DnnlSubgraphPrimitive::GetMemory(std::string memory_name, dnnl::memory::desc mem_desc, dnnl::engine eng) {
  if (Contains(initializers_, memory_name)) {
    for (auto& mem : initializers_.at(memory_name)) {
      if (mem.get_engine() == eng && mem.get_desc() == mem_desc) {
        return mem;
      }
    }
  }

  if (Contains(inputs_, memory_name)) {
    auto& mem = inputs_.at(memory_name);
    if (mem.get_engine() == eng && mem.get_desc() == mem_desc) {
      return mem;
    }
  }

  if (Contains(intermediates_, memory_name)) {
    for (auto& mem : intermediates_.at(memory_name)) {
      if (mem.get_engine() == eng && mem.get_desc() == mem_desc) {
        return mem;
      }
    }
  }

  throw std::invalid_argument("cannot find memory");
}

void DnnlSubgraphPrimitive::SetMemory(std::string memory_name, dnnl::memory mem) {
  if (Contains(intermediates_, memory_name)) {
    for (auto& tmp_mem : intermediates_.at(memory_name)) {
      if (tmp_mem == mem) {
        throw std::invalid_argument("setting duplicate memory");
      }
    }
    intermediates_.at(memory_name).push_back(mem);
  } else {
    intermediates_.insert(std::pair<std::string, std::vector<dnnl::memory> >(memory_name, std::vector<dnnl::memory>()));
    intermediates_[memory_name].push_back(mem);
  }
}

void DnnlSubgraphPrimitive::SetInitializer(std::string memory_name, dnnl::memory mem) {
  if (Contains(initializers_, memory_name)) {
    for (auto& tmp_mem : initializers_.at(memory_name)) {
      if (tmp_mem == mem) {
        throw std::invalid_argument("setting duplicate initializer");
      }
    }
    initializers_.at(memory_name).push_back(mem);
  } else {
    initializers_.insert(std::pair<std::string, std::vector<dnnl::memory> >(memory_name, std::vector<dnnl::memory>()));
    initializers_[memory_name].push_back(mem);
  }
}

dnnl::memory DnnlSubgraphPrimitive::GetMemoryAndReshape(ort_dnnl::DnnlTensor tensor, dnnl::memory::desc mem_desc, dnnl::engine eng, bool transpose) {
  // if found just return
  if (HasMemory(tensor.Name(), mem_desc, eng)) {
    return GetMemory(tensor.Name(), mem_desc, eng);
  }

  // is non overridable constant initializer (assume already in memory (runtime))
  bool is_constant = Contains(initializers_, tensor.Name());
  if (is_constant) {
    LOGS_DEFAULT(INFO) << "initializer cache started";
  }
  // will get the first memory with matching name
  auto mem_from = GetMemory(tensor.Name());
  auto mem_to = dnnl::memory(mem_desc, eng);

  // if it is a reshape, ensure reorder is possible by making the same dims
  if (mem_from.get_desc().dims() != mem_to.get_desc().dims() || transpose) {
    auto mem_from_dims = mem_from.get_desc().dims();
    auto mem_to_dims = mem_to.get_desc().dims();
    if (Product(mem_from_dims) != Product(mem_to_dims)) {
      throw std::invalid_argument("not a valid reshape, inconsistent dim product");
    }
    auto mem_from_reshape = dnnl::memory(mem_desc, mem_from.get_engine(), nullptr);
    if (is_constant) {  // if constant, do reshape now
      LOGS_DEFAULT(INFO) << "reshaped now";
      mem_from_reshape.set_data_handle(mem_from.get_data_handle());
    } else {
      AddReshape(mem_from, mem_from_reshape);
    }
    if (mem_from_reshape.get_desc() == mem_to.get_desc() && mem_from_reshape.get_engine() == mem_to.get_engine()) {
      mem_to = mem_from_reshape;
    } else {              // after reshape still need to reorder
      if (is_constant) {  // execute reorder now if constant
        dnnl::stream s{eng};
        dnnl::reorder(mem_from_reshape, mem_to).execute(s, mem_from_reshape, mem_to);
        s.wait();
      } else {
        AddPrimitive(dnnl::reorder(mem_from_reshape, mem_to), {{DNNL_ARG_FROM, mem_from_reshape},
                                                               {DNNL_ARG_TO, mem_to}});
      }
    }
  } else {              // same shape, save to reorder
    if (is_constant) {  // execute reorder now if constant
      dnnl::stream s{eng};
      dnnl::reorder(mem_from, mem_to).execute(s, mem_from, mem_to);
      s.wait();
    } else {
      AddPrimitive(dnnl::reorder(mem_from, mem_to), {{DNNL_ARG_FROM, mem_from},
                                                     {DNNL_ARG_TO, mem_to}});
    }
  }

  if (is_constant) {  // initializer should stay even after dynamic recompile
    SetInitializer(tensor.Name(), mem_to);
  }
  return mem_to;
}

void DnnlSubgraphPrimitive::AddReshape(dnnl::memory src, dnnl::memory dst) {
  LOGS_DEFAULT(INFO) << "reshape queued";
  reshapes_.push_back({src, dst});
}

void DnnlSubgraphPrimitive::AddPrimitive(dnnl::primitive prim, std::unordered_map<int, dnnl::memory> mem_map) {
  net_.push_back(prim);
  net_args_.push_back(mem_map);
}

onnxruntime::common::Status DnnlSubgraphPrimitive::Predict(const std::unordered_map<std::string, OnnxTensorData>& inputs, const std::unordered_map<std::string, OnnxTensorData>& outputs) {
  for (auto& input : inputs) {
    if (Contains(inputs_, input.first)) {
      inputs_.at(input.first).set_data_handle(input.second.buffer);
    }
  }

  for (auto& output : outputs) {
    if (Contains(outputs_, output.first)) {
      outputs_.at(output.first).set_data_handle(output.second.buffer);
    }
  }

  // reshapes (eg, unsqueeze)
  // it is safe to set data handle because all external data handles have been set and onednn managed memory data handles will not change
  for (auto& reshape_pair : reshapes_) {
    reshape_pair.second.set_data_handle(reshape_pair.first.get_data_handle());
  }

  auto stream = GetStream();
  for (size_t i = 0; i < net_.size(); ++i) {
    net_.at(i).execute(stream, net_args_.at(i));
    stream.wait();
  }

  return Status::OK();
}

dnnl::memory::desc DnnlSubgraphPrimitive::GetOutputInfo(std::string name) {
  if (Contains(outputs_, name)) {
    return outputs_.at(name).get_desc();
  }
  throw std::invalid_argument("no such output exists");
}

void DnnlSubgraphPrimitive::SetOrderedInputs(std::vector<std::string>&& inputs) {
  ordered_inputs_ = std::move(inputs);
}

void DnnlSubgraphPrimitive::SetOrderedOutputs(std::vector<std::string>&& outputs) {
  ordered_outputs_ = std::move(outputs);
}

const std::vector<std::string>& DnnlSubgraphPrimitive::GetOrderedInputs() const {
  return ordered_inputs_;
}

const std::vector<std::string>& DnnlSubgraphPrimitive::GetOrderedOutputs() const {
  return ordered_outputs_;
}

}  // namespace ort_dnnl
}  // namespace onnxruntime