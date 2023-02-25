#include "ft_custom_op_lib.h"
#include "ft_vit_int8_custom_op.h"
#include "core/framework/provider_options.h"

//namespace onnxruntime {

//common::Status CreateFTCustomOpDomainList(CUDAExecutionProviderInfo& info) {
  //std::unique_ptr<OrtProviderCustomOpDomain> custom_op_domain = std::make_unique<OrtProviderCustomOpDomain>();
  ////custom_op_domain->domain_ = "fastertransformer";
  //custom_op_domain->domain_ = "trt.plugins";

  //std::unique_ptr<FTViTCustomOp> vit_custom_op = std::make_unique<FTViTCustomOp>(onnxruntime::kCudaExecutionProvider, nullptr);
  //custom_op_domain->custom_ops_.push_back(vit_custom_op.release());
  //info.custom_op_domain_list.push_back(custom_op_domain.release());


  //return common::Status::OK();
//}

//}

static const char* c_OpDomain = "fastertransformer";
constexpr const char* c_ORTCudaExecutionProvider = "CUDAExecutionProvider";

static void AddOrtCustomOpDomain(Ort::CustomOpDomain&& domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {

  // Allow use of Ort::GetApi() in C++ API implementations.
  Ort::InitApi(api_base->GetApi(ORT_API_VERSION));
  Ort::UnownedSessionOptions session_options(options);

  static FTViTINT8CustomOp vit_int_custom_op(c_ORTCudaExecutionProvider, nullptr);

  OrtStatus* result = nullptr;

  try {
    Ort::CustomOpDomain domain{c_OpDomain};
    domain.Add(&vit_int_custom_op);

    session_options.Add(domain);
    AddOrtCustomOpDomain(std::move(domain));

  } catch(const std::exception& e) {
    Ort::Status status{e};
    result = status.release();
  }

  return result;

}
