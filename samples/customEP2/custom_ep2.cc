#include <memory>
#include <cmath>
#include <iostream>
#include "custom_ep2.h"
#include "core/session/onnxruntime_lite_custom_op.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/ortdevice.h"
#include "core/framework/ortmemoryinfo.h"

namespace onnxruntime {
void KernelTwo(const Ort::Custom::Tensor<float>& X,
               Ort::Custom::Tensor<int32_t>& Y) {
  const auto& shape = X.Shape();
  auto X_raw = X.Data();
  auto Y_raw = Y.Allocate(shape);
  auto total = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
  for (int64_t i = 0; i < total; i++) {
    Y_raw[i] = static_cast<int32_t>(round(X_raw[i]));
  }
  std::cout<<"In KernelTwo()\n";
}

void MyRelu(const Ort::Custom::Tensor<float>& X, Ort::Custom::Tensor<float>& Y) {
  const auto& shape = X.Shape();
  auto X_raw = X.Data();
  auto Y_raw = Y.Allocate(shape);
  auto total = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
  for (int64_t i = 0; i < total; i++) {
    Y_raw[i] = X_raw[i] > 0 ? X_raw[i] : 0;
  }
  std::cout<<"In MyRelu()\n";
}

void MyMax(const Ort::Custom::Variadic& inputs, Ort::Custom::Tensor<float>& output) {
  const auto& shape = inputs[0]->Shape();
  const float* raw_input0 = reinterpret_cast<const float*>(inputs[0]->DataRaw());
  auto raw = output.Allocate(shape);
  auto num_elements = inputs[0]->NumberOfElement();
  for (int64_t i = 0; i < num_elements; i++) raw[i] = raw_input0[i];

  for (size_t ith_input = 1; ith_input < inputs.Size(); ++ith_input) {
    const auto& input = inputs[ith_input];
    const float* raw_input = reinterpret_cast<const float*>(input->DataRaw());
    num_elements = input->NumberOfElement();
    for (int64_t jth_elem = 0; jth_elem < num_elements; ++jth_elem) {
      raw[jth_elem] = std::max(raw[jth_elem], raw_input[jth_elem]);
    }
  }
  std::cout<<"In MyMax()\n";
}

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
    std::unique_ptr<Ort::Custom::ExternalKernelDef> p(Ort::Custom::CreateExternalKernelDef("Relu", type_.c_str(), MyRelu, "ai.onnx", 14));
    std::unique_ptr<Ort::Custom::ExternalKernelDef> pp(Ort::Custom::CreateExternalKernelDef("Max", type_.c_str(), MyMax, "ai.onnx", 13));
    kernel_definitions_.push_back(std::move(p));
    kernel_definitions_.push_back(std::move(pp));

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

std::vector<std::unique_ptr<SubGraphDef>> CustomEp2::GetCapability(GraphViewRef* graph) {
  auto opset = graph->Opset("");
  return {};
}

common::Status CustomEp2::Compile(std::vector<std::unique_ptr<GraphViewRef>>&, std::vector<std::unique_ptr<NodeViewRef>>&, std::vector<NodeComputeInfo>&) {
  return common::Status::OK();
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

ORT_API(onnxruntime::CustomEp2*, GetExternalProvider, const void* provider_options) {
    std::unordered_map<std::string, std::string>* options = (std::unordered_map<std::string, std::string>*)(provider_options);
    return onnxruntime::CustomEP2Factory::CreateCustomEp2(*options);
}

#ifdef __cplusplus
}
#endif
