#include "core/providers/brainslice/brain_slice_execution_provider.h"
#include "core/common/common.h"
#include "core/providers/brainslice/fpga_util.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "core/providers/brainslice/brainslice_kernel.h"
#include "core/providers/brainslice/brainslice_fwd.h"

namespace onnxruntime {
namespace brainslice {
using namespace onnxruntime;
class BrainSlicePinnedAllocator : public CPUAllocator {
 public:
  virtual const OrtMemoryInfo& Info() const override {
    static OrtMemoryInfo bs_cpu_allocator_info("BrainSlice",
                                                  OrtAllocatorType::OrtDeviceAllocator, 0,
                                                  OrtMemType::OrtMemTypeCPU);
    return bs_cpu_allocator_info;
  }
};

//In current framework, I have to provide a default allocator, otherwise our allocation planner can't get the default allocator info
//Altough we won't use this default allocator to allocate anything....
//Looks something wrong is the design, may need to take a look later.
class BrainSliceAllocator : public IAllocator {
 public:
  virtual const OrtMemoryInfo& Info() const override {
    static OrtMemoryInfo bs_default_allocator_info("BrainSlice",
                                                      OrtAllocatorType::OrtDeviceAllocator, 0,
                                                      OrtMemType::OrtMemTypeDefault);
    return bs_default_allocator_info;
  }

  virtual void* Alloc(size_t) override {
    ORT_THROW("BrainSlice has no default allocator.");
  }

  virtual void Free(void*) override {
    ORT_THROW("BrainSlice has no default allocator.");
  }
};

BrainSliceExecutionProvider::BrainSliceExecutionProvider(const fpga::FPGAInfo& info) : handle_(info),
                                                                                       matrix_rf_planner_(onnxruntime::make_unique<BrainSliceMemoryPlanner>(ISA_Mem_MatrixRf, handle_.GetParameters().MATRIX_RF_SIZE, 0)),
                                                                                       multiply_vrf_planner_(onnxruntime::make_unique<BrainSliceMemoryPlanner>(ISA_Mem_MultiplyVrf, handle_.GetParameters().MULTIPLY_VRF_SIZE, 0)),
                                                                                       add_sub_vrf_planner_(onnxruntime::make_unique<BrainSliceMemoryPlanner>(ISA_Mem_AddSubVrf_0, handle_.GetParameters().ADDSUB_VRF_0_SIZE, 0)),
                                                                                       m_dram_planner_(onnxruntime::make_unique<BrainSliceMemoryPlanner>(ISA_Mem_Dram, handle_.GetParameters().MATRIX_MEM_SIZE, 1)),
                                                                                       v_dram_planner_(onnxruntime::make_unique<BrainSliceMemoryPlanner>(ISA_Mem_AddSubVrf, handle_.GetParameters().VECTOR_MEM_SIZE, 1)) {
  // insert cpu memory allocator
  AllocatorPtr cpu_allocator(new BrainSlicePinnedAllocator());
  AllocatorPtr bs_allocator(new BrainSliceAllocator());
  InsertAllocator(cpu_allocator);
  InsertAllocator(bs_allocator);
}

// TODO: remove once we mark all BrainSliceOperator inputs as CPU memory type.
Status BrainSliceExecutionProvider::CopyTensor(const Tensor& /*src*/, Tensor& /*dst*/) const {
  return Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED, "explict copy between FPGA and other device is not implement, please mark your kernel on cpu memory.");
}

static std::string NormalizeFuncName(const std::string& func_name) {
  std::string name(func_name);
  std::transform(name.begin(), name.end(), name.begin(),
                 [](const unsigned char i) { return static_cast<char>(::tolower(i)); });
  return name;
}

bool BrainSliceExecutionProvider::CheckNodeWithCapacity(const onnxruntime::GraphViewer& graph, const onnxruntime::Node& node) const {
  //TODO: right now we only handle GRU node (maybe LSTM later) because they are built-in firmware.
  //Wil need more work to support different node's capacity check.
  if (node.OpType() == "GRU" || node.OpType() == "LSTM") {
    //1. check batch size is 1
    auto inputs = node.InputDefs();
    // sequence_length / initial_h is not supported yet.
    if (node.OpType() == "GRU" && (inputs.size() < 3 || inputs.size() > 4 || !inputs[0] || !inputs[0]->Shape()))
      return false;
    if (node.OpType() == "LSTM" && (inputs.size() < 3 || inputs.size() > 7 || !inputs[0] || !inputs[0]->Shape() || (inputs.size() >= 5 && inputs[4]->Exists())))
      return false;
    auto x_shape = inputs[0]->Shape();
    if (x_shape->dim_size() != 3 || x_shape->dim()[1].dim_value() != 1)
      return false;
    //2. check W and R is initializer
    auto W = inputs[1];
    auto R = inputs[2];
    const onnx::TensorProto* tmp;
    if (!graph.GetInitializedTensor(W->Name(), tmp) || !graph.GetInitializedTensor(R->Name(), tmp))
      return false;
    //3. check B is we have bias
    if (inputs.size() >= 3 && inputs[3] && !graph.GetInitializedTensor(inputs[3]->Name(), tmp))
      return false;
    auto& attributes = node.GetAttributes();
    //5. check activate function
    auto it = attributes.find("activations");
    if (it != attributes.end()) {
      if ((node.OpType() == "GRU" &&
           !((it->second.strings_size() == 2 && NormalizeFuncName(it->second.strings()[0]) == "sigmoid" &&
              NormalizeFuncName(it->second.strings()[1]) == "tanh") ||
             (it->second.strings_size() == 4 && NormalizeFuncName(it->second.strings()[0]) == "sigmoid" &&
              NormalizeFuncName(it->second.strings()[1]) == "tanh" &&
              NormalizeFuncName(it->second.strings()[2]) == "sigmoid" &&
              NormalizeFuncName(it->second.strings()[3]) == "tanh"))) ||
          (node.OpType() == "LSTM" &&
           !((it->second.strings_size() == 3 && NormalizeFuncName(it->second.strings()[0]) == "sigmoid" &&
              NormalizeFuncName(it->second.strings()[1]) == "tanh" &&
              NormalizeFuncName(it->second.strings()[2]) == "tanh") ||
             (it->second.strings_size() == 6 && NormalizeFuncName(it->second.strings()[0]) == "sigmoid" &&
              NormalizeFuncName(it->second.strings()[1]) == "tanh" &&
              NormalizeFuncName(it->second.strings()[2]) == "tanh" &&
              NormalizeFuncName(it->second.strings()[3]) == "sigmoid" &&
              NormalizeFuncName(it->second.strings()[4]) == "tanh" &&
              NormalizeFuncName(it->second.strings()[5]) == "tanh"))))
        return false;
    }
    // 6. clip not supported now.
    if (attributes.count("clip") > 0)
      return false;

    // 7. linear_before_reset not supported
    it = attributes.find("linear_before_reset");
    if (it != attributes.end() && it->second.i() != 0)
      return false;
    // TODO: check capacity and the weight size.
    BrainSlice_Parameters fpga_parameters = this->handle_.GetParameters();
    // calculate size of W
    auto* shape = W->Shape();
    int64_t w_size = 1;
    int64_t input_dims = 1;
    for (auto& dim : shape->dim()) {
      if (dim.has_dim_value())
        w_size *= dim.dim_value();
      else
        return false;
      input_dims = dim.dim_value();
    }
    uint16_t w_block_rows = static_cast<uint16_t>((input_dims + fpga_parameters.NATIVE_DIM - 1) / fpga_parameters.NATIVE_DIM);
    uint16_t w_block_cols = static_cast<uint16_t>((w_size / input_dims + fpga_parameters.NATIVE_DIM - 1) / fpga_parameters.NATIVE_DIM);
    auto* r_shape = R->Shape();
    int64_t r_size = 1;
    int64_t hidden = 1;
    for (auto& dim : r_shape->dim()) {
      if (dim.has_dim_value())
        r_size *= dim.dim_value();
      else
        return false;
      hidden = dim.dim_value();
    }
    uint16_t r_block_rows = static_cast<uint16_t>((hidden + fpga_parameters.NATIVE_DIM - 1) / fpga_parameters.NATIVE_DIM);
    uint16_t r_block_cols = static_cast<uint16_t>((r_size / hidden + fpga_parameters.NATIVE_DIM - 1) / fpga_parameters.NATIVE_DIM);
    // check matrix register file
    if (fpga_parameters.MATRIX_RF_SIZE < (w_block_rows * w_block_cols + r_block_rows * r_block_cols)) {
      return false;
    }
    if (inputs.size() >= 3 && inputs[3]) {
      auto* b_shape = inputs[3]->Shape();
      int64_t b_size = 1;
      for (auto& dim : b_shape->dim()) {
        if (dim.has_dim_value())
          b_size *= dim.dim_value();
        else
          return false;
      }
      // check bias vector size
      // todo: actually we did some pre-processing for bias, so only have of the bias is uploaded to fpga.
      if (fpga_parameters.ADDSUB_VRF_0_SIZE < static_cast<uint16_t>(b_size / fpga_parameters.NATIVE_DIM / 2)) {
        return false;
      }
    }
    return true;
  }

  if (node.OpType() == "BrainSlice") {
    return true;
  }

  return false;
}

std::vector<std::unique_ptr<ComputeCapability>>
BrainSliceExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                           const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (auto& node : graph.Nodes()) {
    if (CheckNodeWithCapacity(graph, node)) {
      std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();
      sub_graph->nodes.push_back(node.Index());
      result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
      //TODO: right now BrainSlice only support one node on the device.
      //Will fix it later.
      break;
    }
  }

  return result;
}

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kBrainSliceExecutionProvider, kOnnxDomain, 7, GRU);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kBrainSliceExecutionProvider, kOnnxDomain, 7, LSTM);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kBrainSliceExecutionProvider, kMSDomain, 1, BrainSlice);

static void RegisterBrainSliceKernels(KernelRegistry& kernel_registry) {
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kBrainSliceExecutionProvider, kOnnxDomain, 7, GRU)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kBrainSliceExecutionProvider, kOnnxDomain, 7, LSTM)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kBrainSliceExecutionProvider, kMSDomain, 1, BrainSlice)>());
}

std::shared_ptr<KernelRegistry> GetBrainSliceKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  RegisterBrainSliceKernels(*kernel_registry);
  return kernel_registry;
}

std::shared_ptr<KernelRegistry> BrainSliceExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = GetBrainSliceKernelRegistry();
  return kernel_registry;
}

BrainSliceMemoryPlanner* BrainSliceExecutionProvider::GetBrainSliceMemoryPlanner(ISA_Mem mem_type, ParameterUsage usage) {
  switch (mem_type) {
    case ISA_Mem_AddSubVrf_0:
      return add_sub_vrf_planner_.get();
    case ISA_Mem_MatrixRf:
      return matrix_rf_planner_.get();
    case ISA_Mem_MultiplyVrf:
      return multiply_vrf_planner_.get();
    case ISA_Mem_Dram: {
      if (usage == ParameterUsage::USE_AS_MATRIX)
        return m_dram_planner_.get();
      else
        return v_dram_planner_.get();
    }
    default:
      return nullptr;
  }
}
}  // namespace brainslice
}  // namespace onnxruntime
