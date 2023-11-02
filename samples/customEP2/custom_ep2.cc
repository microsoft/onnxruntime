#include <memory>
#include <cmath>
#include <iostream>
#include "custom_ep2.h"
#include "core/session/onnxruntime_lite_custom_op.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/ortdevice.h"
#include "core/framework/ortmemoryinfo.h"

namespace onnxruntime {

/////////////////////////////////////// POC Kernels ////////////////////////////////////////////
onnxruntime::Status Identity(onnxruntime::interface::TensorView<float>& input,
                             onnxruntime::interface::Tensor<float>& output) {
  const auto& shape = input.Shape();
  const float* input_data = input.Data();
  float* output_data = output.Allocate(shape);
  size_t number_of_elements = input.NumberOfElements();
  for (size_t i = 0; i < number_of_elements; ++i) {
      output_data[i] = input_data[i];
  }
  return onnxruntime::Status::OK();
}
struct Celu {
  Celu(const interface::IKernelInfo&) {}
  onnxruntime::Status Compute(onnxruntime::interface::TensorView<float>& input,
                              onnxruntime::interface::Tensor<float>& output) {
      const auto& shape = input.Shape();
      const float* input_data = input.Data();
      float* output_data = output.Allocate(shape);
      size_t number_of_elements = input.NumberOfElements();

      onnxruntime::interface::TensorView<float> identity_input(input_data, shape);
      onnxruntime::interface::Tensor<float> identity_output(output_data, shape);
      auto status = Identity(identity_input, identity_output);
      if (!status.IsOK()) {
        return status;
      }

      for (size_t i = 0; i < number_of_elements; ++i) {
        if (input_data[i] < 0) {
          output_data[i] = 0.f;
        }
      }

      return onnxruntime::Status::OK();
  }
  float alpha = 0.f;
};
/////////////////////////////////////////////////////////////////////////////////////////////////

struct CustomCPUAllocator : public OrtAllocator {
  CustomCPUAllocator() {
    mem_info = new OrtMemoryInfo("", OrtDeviceAllocator, OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, 0));
    OrtAllocator::version = ORT_API_VERSION;
    OrtAllocator::Alloc = [](OrtAllocator* this_, size_t size) { return static_cast<CustomCPUAllocator*>(this_)->Alloc(size); };
    OrtAllocator::Free = [](OrtAllocator* this_, void* p) { static_cast<CustomCPUAllocator*>(this_)->Free(p); };
    OrtAllocator::Info = [](const OrtAllocator* this_) { return static_cast<const CustomCPUAllocator*>(this_)->Info(); };
  }

  virtual ~CustomCPUAllocator() { Ort::GetApi().ReleaseMemoryInfo(mem_info); }

  void* Alloc(size_t size) {
    void* device_address = new (std::nothrow) uint8_t[size];
    return device_address;
  }
  void Free(void* p) {
    delete[] reinterpret_cast<uint8_t*>(p);
  }
  const OrtMemoryInfo* Info() const {
    return mem_info;
  }

private:
  OrtMemoryInfo* mem_info;
};

CustomEp2::CustomEp2(const CustomEp2Info& info) : info_{info} {
    type_ = "customEp2";
    /*
    std::unique_ptr<Ort::Custom::ExternalKernelDef> p(Ort::Custom::CreateExternalKernelDef("Relu", type_.c_str(), MyRelu, "ai.onnx", 14));
    std::unique_ptr<Ort::Custom::ExternalKernelDef> pp(Ort::Custom::CreateExternalKernelDef("Max", type_.c_str(), MyMax, "ai.onnx", 13));
    kernel_definitions_.push_back(std::move(p));
    kernel_definitions_.push_back(std::move(pp));
    */
    allocators_.push_back(std::make_unique<CustomCPUAllocator>().release());  // TODO: release resource
}

bool CustomEp2::CanCopy(const OrtDevice& src, const OrtDevice& dest) {
  std::cout<<"Custom2's CanCopy()\n";
  return true;
}

void CustomEp2::MemoryCpy(Ort::UnownedValue&, Ort::ConstValue const&) {
  std::cout<<"Custom2's MemoryCpy()\n";
  //memcpy(dst, src, bytes_count);
}

std::vector<std::unique_ptr<SubGraphDef>> CustomEp2::GetCapability(interface::GraphViewRef* graph) {
  std::vector<std::unique_ptr<SubGraphDef>> ret;
  for (std::unique_ptr<interface::NodeViewRef>& node : graph->NodeViews()) {
    if (node->IsOp("Celu")) {
      std::vector<std::string_view> inputs = node->Inputs();
      assert(inputs.size() == 1);
      std::unique_ptr<interface::NodeViewRef> producer = graph->GetNodeViewProducingOutput(inputs[0]);
      if (producer != nullptr && producer->IsOp("Identity")) {
        std::unique_ptr<SubGraphDef> subgraph = std::make_unique<SubGraphDef>();
        subgraph->nodes.push_back(producer->Index());
        subgraph->nodes.push_back(node->Index());
        std::unique_ptr<SubGraphDef::MetaDef> meta_data = std::make_unique<SubGraphDef::MetaDef>();
        meta_data->name = "Identity_Celu";
        for (std::string_view& input : producer->Inputs()) meta_data->inputs.emplace_back(std::string(input));
        for (std::string_view& output : node->Outputs())   meta_data->outputs.emplace_back(std::string(output));
        subgraph->SetMetaDef(std::move(meta_data));
        ret.emplace_back(std::move(subgraph));
      }
    }
  }
  std::cout<<"CustomEp2::GetCapability() has "<<ret.size()<<" subgraphs\n";
  return ret;
}

common::Status CustomEp2::Compile(std::vector<std::unique_ptr<interface::GraphViewRef>>& partial_graph, std::vector<std::unique_ptr<interface::NodeViewRef>>& fused_nodes, std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (size_t i = 0; i < partial_graph.size(); i++) {

    NodeComputeInfo compute_info;
    compute_info.create_state_func = [](ComputeContext*, FunctionState*) {
      return 0;
    };
    compute_info.release_state_func = [](FunctionState) {
    };
    compute_info.compute_func = [](void* state, const OrtApi*, OrtKernelContext* context) {
      Ort::KernelContext ctx(context);
      assert(ctx.GetInputCount() == 1);
      assert(ctx.GetOutputCount() == 1);
      Ort::ConstValue input = ctx.GetInput(0);
      const float* X = input.GetTensorData<float>();
      size_t len = 1;
      for (int64_t& elem : input.GetTensorTypeAndShapeInfo().GetShape()) len *= elem;
      Ort::UnownedValue output = ctx.GetOutput(0, input.GetTensorTypeAndShapeInfo().GetShape());
      float* Y = output.GetTensorMutableData<float>();

      float alpha = 1.0;
      for (size_t j = 0; j < len; j++) {
        Y[j] = std::max(0.f, X[j]) + std::min<float>(0.f, alpha*(exp(X[j]/alpha)-1));
      }
      std::cout<<"CustomEP2::Compile()\n";
      return Status::OK();
    };
    node_compute_funcs.push_back(compute_info);
  }
  return common::Status::OK();
}

void CustomEp2::RegisterKernels(interface::IKernelRegistry& kernel_registry) {
  interface::IKernelBuilder& identity_builder = kernel_registry.RegisterKernel("customEp2", "ai.onnx", "Identity", 10, 19, Identity);
  identity_builder.TypeConstraint("V", interface::TensorDataType::float_tp).Alias(0,0);
  interface::IKernelBuilder& celu_builder = kernel_registry.RegisterKernel<Celu>("customEp2", "ai.onnx", "Celu", 10, 19);
  celu_builder.TypeConstraint("T", interface::TensorDataType::float_tp).Alias(0,0);
}

CustomEp2Info ProviderOption2CustomEpInfo(std::unordered_map<std::string, std::string>& provider_option) {
  CustomEp2Info ret;
  if (provider_option.find("int_property") != provider_option.end()) {
    ret.int_property = stoi(provider_option["int_property"]);
    std::cout<<"int_property="<<provider_option["int_property"]<<"\n";
  }
  if (provider_option.find("str_property") != provider_option.end()) {
    ret.str_property = provider_option["str_property"];
    std::cout<<"str_property="<<provider_option["str_property"]<<"\n";
  }
  return ret;
}

class CustomEP2Factory {
public:
  CustomEP2Factory() {}
  ~CustomEP2Factory() {}
  static CustomEp2* CreateCustomEp2(std::unordered_map<std::string, std::string>& provider_option) {
    return std::make_unique<CustomEp2>(ProviderOption2CustomEpInfo(provider_option)).release();
  }
};

}

#ifdef __cplusplus
extern "C" {
#endif

onnxruntime::CustomEp2* GetExternalProvider(const void* provider_options) {
  std::unordered_map<std::string, std::string>* options = (std::unordered_map<std::string, std::string>*)(provider_options);
  return onnxruntime::CustomEP2Factory::CreateCustomEp2(*options);
}

#ifdef __cplusplus
}
#endif
