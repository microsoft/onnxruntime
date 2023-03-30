#include "header.h"

#define ORT_API_MANUAL_INIT
#include "custom_op_lite.h"
#undef ORT_API_MANUAL_INIT

/////////////////////////////////// Fuse ////////////////////////////////////////

void Fuse(
    OrtKernelContext* ctx,
    const Ort::Custom2::Span<float>& vector_1,
    const Ort::Custom2::Span<float>& vector_2,
    int32_t alpha,
    Ort::Custom2::TensorT<float>& vector_output) {
  auto len_output = std::min(vector_1.size(), vector_2.size());
  float* floats_out = static_cast<float*>(vector_output.Allocate({(int64_t)len_output}));
  for (size_t i = 0; i < len_output; ++i) {
    floats_out[i] = vector_1[i] + vector_2[i];
  }
}

/////////////////////////////////// Select ////////////////////////////////////////

void Select(const Ort::Custom2::Span<int32_t>& indices_in,
            Ort::Custom2::TensorT<int32_t>& indices_out) {
  std::vector<int32_t> selected_indices;
  for (size_t i = 0; i < indices_in.size(); ++i) {
    if (indices_in[i] % 2 == 0) {
      selected_indices.push_back(indices_in[i]);
    }
  }

  int32_t* int_out = static_cast<int32_t*>(indices_out.Allocate({static_cast<int64_t>(selected_indices.size())}));
  for (size_t j = 0; j < selected_indices.size(); ++j) {
    int_out[j] = selected_indices[j];
  }
}

/////////////////////////////////// Filter ////////////////////////////////////////

void Filter(const Ort::Custom2::TensorT<float>& floats_in,
            Ort::Custom2::TensorT<float>& floats_out) {
  const float* in = floats_in.Data();
  auto in_len = floats_in.NumberOfElement();

  std::vector<float> filter_floats;
  for (int64_t i = 0; i < in_len; ++i) {
    if (in[i] > 1.f) {
      filter_floats.push_back(in[i]);
    }
  }

  float* out = static_cast<float*>(floats_out.Allocate({static_cast<int64_t>(filter_floats.size())}));
  for (size_t j = 0; j < filter_floats.size(); ++j) {
    out[j] = filter_floats[j];
  }
}

////////////////////////////////////////////////////////////////////////////////////

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);

  static Ort::CustomOpDomain v2_domain{"v2"};
  static std::unique_ptr<OrtCustomOp> fus_op_ptr{Ort::Custom2::CreateCustomOp("Fuse", "CPUExecutionProvider", Fuse)};
  static std::unique_ptr<OrtCustomOp> sel_op_ptr{Ort::Custom2::CreateCustomOp("Select", "CPUExecutionProvider", Select)};
  static std::unique_ptr<OrtCustomOp> fil_op_ptr{Ort::Custom2::CreateCustomOp("Filter", "CPUExecutionProvider", Filter)};

  v2_domain.Add(fus_op_ptr.get());
  v2_domain.Add(sel_op_ptr.get());
  v2_domain.Add(fil_op_ptr.get());

  Ort::UnownedSessionOptions session_options(options);
  session_options.Add(v2_domain);
  return nullptr;
}

OrtStatus* ORT_API_CALL RegisterCustomOpsAltName(OrtSessionOptions* options, const OrtApiBase* api) {
  return RegisterCustomOps(options, api);
}
