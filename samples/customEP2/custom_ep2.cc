#include <memory>
#include <cmath>
#include <iostream>
#include "custom_ep2.h"
#include "core/session/onnxruntime_lite_custom_op.h"

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
}

void MyRelu(const Ort::Custom::Tensor<float>& X, Ort::Custom::Tensor<float>& Y) {
  const auto& shape = X.Shape();
  auto X_raw = X.Data();
  auto Y_raw = Y.Allocate(shape);
  auto total = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
  for (int64_t i = 0; i < total; i++) {
    Y_raw[i] = X_raw[i] > 0 ? X_raw[i] : 0;
  }
}

CustomEp2::CustomEp2(const CustomEp2Info& info) : type_{"customEp2"}, info_{info} {
    //custom_ops_.push_back(Ort::Custom::CreateLiteCustomOp("CustomOpTwo", type_.c_str(), KernelTwo));  // TODO: should use smart pointer for vector custom_ops_
    kernel_definitions_.push_back(Ort::Custom::CreateLiteCustomOp("CustomOpTwo", type_.c_str(), MyRelu));  // TODO: should use smart pointer for vector custom_ops_
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
