#include "header.h"

#define ORT_API_MANUAL_INIT
#include "custom_op_lite.h"
#undef ORT_API_MANUAL_INIT

/////////////////////////////////// Merge ////////////////////////////////////////
static const int64_t upper_bound = 10;

void Box(const Ort::Custom2::TensorT<float>& float_in_1,
         const Ort::Custom2::TensorT<float>& float_in_2,
         const Ort::Custom2::TensorT<float>& float_in_3,
         Ort::Custom2::TensorT<float>& float_out_1,
         Ort::Custom2::TensorT<float>& float_out_2) {
  auto raw_in_1 = float_in_1.Data();
  auto raw_in_2 = float_in_2.Data();

  auto l_in_1 = float_in_1.Shape()[0];
  auto l_in_2 = float_in_2.Shape()[0];
  auto l_out_1 = l_in_1 + l_in_2;

  auto raw_out_1 = float_out_1.Allocate({l_out_1});

  for (int64_t i = 0; i < l_out_1; ++i) {
    raw_out_1[i] = i < l_in_1 ? raw_in_1[i] : raw_in_2[i - l_in_1];
  }

  if (float_in_3) {
    auto raw_in_3 = float_in_3.Data();
    auto l_in_3 = float_in_3.Shape()[0];
    auto l_out_2 = l_in_2 + l_in_3;
    auto raw_out_2 = float_out_2.Allocate({l_out_2});
    for (int64_t i = 0; i < l_out_2; ++i) {
      raw_out_2[i] = i < l_in_2 ? raw_in_2[i] : raw_in_3[i - l_in_2];
    }
  }
}

void Tie(const Ort::Custom2::TensorT<float>& float_in_1,
         std::optional<const Ort::Custom2::TensorT<float>*> float_in_2,
         Ort::Custom2::TensorT<float>& float_out_1,
         std::optional<Ort::Custom2::TensorT<float>*> float_out_2) {
}

//void Tie(const Ort::Custom2::TensorT<float>& float_in_1,
//         Ort::Custom2::TensorT<float>& float_out_1) {
//}

/////////////////////////////////////////////////////////////////////////////////////////////////

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);

  static Ort::CustomOpDomain v2_domain{"v2"};
  static std::unique_ptr<OrtCustomOp> box_op_ptr{Ort::Custom2::CreateCustomOp("Box", "CPUExecutionProvider", Box)};
  static std::unique_ptr<OrtCustomOp> tie_op_ptr{Ort::Custom2::CreateCustomOp("Box", "CPUExecutionProvider", Tie)};

  v2_domain.Add(box_op_ptr.get());
  v2_domain.Add(tie_op_ptr.get());

  Ort::UnownedSessionOptions session_options(options);
  session_options.Add(v2_domain);
  return nullptr;
}

OrtStatus* ORT_API_CALL RegisterCustomOpsAltName(OrtSessionOptions* options, const OrtApiBase* api) {
  return RegisterCustomOps(options, api);
}
