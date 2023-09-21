#include "intree_execution_provider.h"
#include <iostream>

namespace onnxruntime {

void IntreeRelu(const Ort::Custom::Tensor<float>& X, Ort::Custom::Tensor<float>& Y) {
  const auto& shape = X.Shape();
  auto X_raw = X.Data();
  auto Y_raw = Y.Allocate(shape);
  auto total = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
  for (int64_t i = 0; i < total; i++) {
    Y_raw[i] = X_raw[i] > 0 ? X_raw[i] : 0;
  }
  std::cout<<"In IntreeRelu()\n";
}

struct InTreeCPUAllocator : public OrtAllocator {
  InTreeCPUAllocator() {
    mem_info = new OrtMemoryInfo("", OrtDeviceAllocator, OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, 0));
    OrtAllocator::version = ORT_API_VERSION;
    OrtAllocator::Alloc = [](OrtAllocator* this_, size_t size) { return static_cast<InTreeCPUAllocator*>(this_)->Alloc(size); };
    OrtAllocator::Free = [](OrtAllocator* this_, void* p) { static_cast<InTreeCPUAllocator*>(this_)->Free(p); };
    OrtAllocator::Info = [](const OrtAllocator* this_) { return static_cast<const InTreeCPUAllocator*>(this_)->Info(); };
  }

  virtual ~InTreeCPUAllocator() { Ort::GetApi().ReleaseMemoryInfo(mem_info); }

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

InTreeExecutionProvider::InTreeExecutionProvider(const InTreeExecutionProviderInfo& info) : info_{info} {
    type_ = "InTreeExecutionProvider";
    std::unique_ptr<Ort::Custom::ExternalKernelDef> p(Ort::Custom::CreateExternalKernelDef("Relu", type_.c_str(), IntreeRelu, "ai.onnx", 14));
    kernel_definitions_.push_back(std::move(p));

    allocators_.push_back(std::make_unique<InTreeCPUAllocator>().release());  // TODO: release resource
}

InTreeExecutionProviderInfo ProviderOption2CustomEpInfo(const std::unordered_map<std::string, std::string>& provider_option) {
  InTreeExecutionProviderInfo ret;
  auto it = provider_option.find("int_property");
  if (it != provider_option.end()) {
    ret.int_property = stoi(it->second);
    std::cout<<"int_property="<<ret.int_property<<"\n";
  }
  it = provider_option.find("str_property");
  if (it != provider_option.end()) {
    ret.str_property = it->second;
    std::cout<<"str_property="<<ret.str_property<<"\n";
  }
  return ret;
}

InTreeExecutionProvider* InTreeExecutionProviderFactory::CreateInTreeExecutionProvider(const std::unordered_map<std::string, std::string>& provider_option) {
    return std::make_unique<InTreeExecutionProvider>(ProviderOption2CustomEpInfo(provider_option)).release();
}

}
