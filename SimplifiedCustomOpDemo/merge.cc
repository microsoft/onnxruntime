#include "header.h"

#define ORT_API_MANUAL_INIT
#include "custom_op_lite.h"
#undef ORT_API_MANUAL_INIT

/////////////////////////////////// Merge ////////////////////////////////////////

struct Merge {
  Merge(const OrtApi* ort_api, const OrtKernelInfo* info) {
    int64_t reverse;
    auto status = ort_api->KernelInfoGetAttribute_int64(info, "reverse", &reverse);
    reverse_ = reverse != 0;
  }
  void Compute(const Ort::Custom2::TensorT<std::string>& strings_in,
               const std::string& string_in,
               Ort::Custom2::TensorT<std::string>& strings_out) {
    std::vector<std::string> string_pool = strings_in.Data();
    string_pool.push_back(string_in);
    if (reverse_) {
      for (auto& str : string_pool) {
        std::reverse(str.begin(), str.end());
      }
      std::reverse(string_pool.begin(), string_pool.end());
    }
    strings_out.SetStringOutput(string_pool, {static_cast<int64_t>(string_pool.size())});
  }
  bool reverse_ = false;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);

  static Ort::CustomOpDomain v2_domain{"v2"};
  static std::unique_ptr<OrtCustomOp> mrg_op_ptr{Ort::Custom2::CreateCustomOp<Merge>("Merge", "CPUExecutionProvider")};

  v2_domain.Add(mrg_op_ptr.get());

  Ort::UnownedSessionOptions session_options(options);
  session_options.Add(v2_domain);
  return nullptr;
}

OrtStatus* ORT_API_CALL RegisterCustomOpsAltName(OrtSessionOptions* options, const OrtApiBase* api) {
  return RegisterCustomOps(options, api);
}
