// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "onnxruntime_lite_custom_op.h"

using namespace Ort::Custom;

namespace Cpu {

void KernelOne(const Ort::Custom::Tensor<float>& X,
               const Ort::Custom::Tensor<float>& Y,
               Ort::Custom::Tensor<float>& Z) {
  auto input_shape = X.Shape();
  auto x_raw = X.Data();
  auto y_raw = Y.Data();
  auto z_raw = Z.Allocate(input_shape);
  for (int64_t i = 0; i < Z.NumberOfElement(); ++i) {
    z_raw[i] = x_raw[i] + y_raw[i];
  }
}

// lite custom op as a function
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

template <typename T>
void MulTop(const Ort::Custom::Span<T>& in, Ort::Custom::Tensor<T>& out) {
  out.Allocate({1})[0] = in[0] * in[1];
}

void Fuse(
    OrtKernelContext*,
    const Ort::Custom::Span<float>& vector_1,
    const Ort::Custom::Span<float>& vector_2,
    int32_t alpha,
    Ort::Custom::Tensor<float>& vector_output) {
  auto len_output = std::min(vector_1.size(), vector_2.size());
  float* floats_out = static_cast<float*>(vector_output.Allocate({(int64_t)len_output}));
  for (size_t i = 0; i < len_output; ++i) {
    floats_out[i] = (vector_1[i] + vector_2[i]) * alpha;
  }
}

void Select(const Ort::Custom::Span<int32_t>& indices_in,
            Ort::Custom::Tensor<int32_t>& indices_out) {
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

void Filter(const Ort::Custom::Tensor<float>& floats_in,
            Ort::Custom::Tensor<float>& floats_out) {
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

void Box(const Ort::Custom::Tensor<float>* float_in_1,
         const Ort::Custom::Tensor<float>* float_in_2,
         std::optional<const Ort::Custom::Tensor<float>*> float_in_3,
         Ort::Custom::Tensor<float>* float_out_1,
         std::optional<Ort::Custom::Tensor<float>*> float_out_2) {
  auto raw_in_1 = float_in_1->Data();
  auto raw_in_2 = float_in_2->Data();

  auto l_in_1 = float_in_1->Shape()[0];
  auto l_in_2 = float_in_2->Shape()[0];
  auto l_out_1 = l_in_1 + l_in_2;

  auto raw_out_1 = float_out_1->Allocate({l_out_1});

  for (int64_t i = 0; i < l_out_1; ++i) {
    raw_out_1[i] = i < l_in_1 ? raw_in_1[i] : raw_in_2[i - l_in_1];
  }

  if (float_in_3.has_value() && float_out_2.has_value()) {
    auto raw_in_3 = float_in_3.value()->Data();
    auto l_in_3 = float_in_3.value()->Shape()[0];
    auto l_out_2 = l_in_2 + l_in_3;
    auto raw_out_2 = float_out_2.value()->Allocate({l_out_2});
    for (int64_t i = 0; i < l_out_2; ++i) {
      raw_out_2[i] = i < l_in_2 ? raw_in_2[i] : raw_in_3[i - l_in_2];
    }
  }
}

#if !defined(DISABLE_FLOAT8_TYPES)
struct KernelOneFloat8 {
  void Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);
    auto input_X = ctx.GetInput(0);
    const Ort::Float8E4M3FN_t* X = input_X.GetTensorData<Ort::Float8E4M3FN_t>();
    auto dimensions = input_X.GetTensorTypeAndShapeInfo().GetShape();
    auto output = ctx.GetOutput(0, dimensions);
    Ort::Float8E4M3FN_t* out = output.GetTensorMutableData<Ort::Float8E4M3FN_t>();
    const size_t size = output.GetTensorTypeAndShapeInfo().GetElementCount();
    for (size_t i = 0; i < size; i++) {
      out[i] = X[i];
    }
  }
};
// legacy custom op registration
struct CustomOpOneFloat8 : Ort::CustomOpBase<CustomOpOneFloat8, KernelOneFloat8> {
  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* /* info */) const {
    return std::make_unique<KernelOneFloat8>().release();
  };
  const char* GetName() const { return "CustomOpOneFloat8"; };
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; };
  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN; };
  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN; };
};

void FilterFloat8(const Ort::Custom::Tensor<Ort::Float8E4M3FN_t>& floats_in,
                  Ort::Custom::Tensor<Ort::Float8E4M3FN_t>& floats_out) {
  const Ort::Float8E4M3FN_t* in = floats_in.Data();
  auto in_len = floats_in.NumberOfElement();

  std::vector<Ort::Float8E4M3FN_t> filter_floats;
  for (int64_t i = 0; i < in_len; ++i) {
    if (in[i] > 1.f) {
      filter_floats.push_back(in[i]);
    }
  }

  Ort::Float8E4M3FN_t* out = static_cast<Ort::Float8E4M3FN_t*>(floats_out.Allocate({static_cast<int64_t>(filter_floats.size())}));
  for (size_t j = 0; j < filter_floats.size(); ++j) {
    out[j] = filter_floats[j];
  }
}
#endif

void RegisterOps(Ort::CustomOpDomain& domain) {
  static const std::unique_ptr<OrtLiteCustomOp> c_CustomOpOne{Ort::Custom::CreateLiteCustomOp("CustomOpOne", "CPUExecutionProvider", KernelOne)};
  static const std::unique_ptr<OrtLiteCustomOp> c_CustomOpTwo{Ort::Custom::CreateLiteCustomOp("CustomOpTwo", "CPUExecutionProvider", KernelTwo)};
  static const std::unique_ptr<OrtLiteCustomOp> c_MulTopOpFloat{Ort::Custom::CreateLiteCustomOp("MulTop", "CPUExecutionProvider", MulTop<float>)};
  static const std::unique_ptr<OrtLiteCustomOp> c_MulTopOpInt32{Ort::Custom::CreateLiteCustomOp("MulTop", "CPUExecutionProvider", MulTop<int32_t>)};
  static const std::unique_ptr<OrtLiteCustomOp> c_Fuse{Ort::Custom::CreateLiteCustomOp("Fuse", "CPUExecutionProvider", Fuse)};
  static const std::unique_ptr<OrtLiteCustomOp> c_Select{Ort::Custom::CreateLiteCustomOp("Select", "CPUExecutionProvider", Select)};
  static const std::unique_ptr<OrtLiteCustomOp> c_Fill{Ort::Custom::CreateLiteCustomOp("Filter", "CPUExecutionProvider", Filter)};
  static const std::unique_ptr<OrtLiteCustomOp> c_Box{Ort::Custom::CreateLiteCustomOp("Box", "CPUExecutionProvider", Box)};

#if !defined(DISABLE_FLOAT8_TYPES)
  static const CustomOpOneFloat8 c_CustomOpOneFloat8;
  static const std::unique_ptr<OrtLiteCustomOp> c_FilterFloat8{Ort::Custom::CreateLiteCustomOp("FilterFloat8", "CPUExecutionProvider", FilterFloat8)};
#endif

  domain.Add(c_CustomOpOne.get());
  domain.Add(c_CustomOpTwo.get());
  domain.Add(c_MulTopOpFloat.get());
  domain.Add(c_MulTopOpInt32.get());
  domain.Add(c_Fuse.get());
  domain.Add(c_Select.get());
  domain.Add(c_Fill.get());
  domain.Add(c_Box.get());
#if !defined(DISABLE_FLOAT8_TYPES)
  domain.Add(&c_CustomOpOneFloat8);
  domain.Add(c_FilterFloat8.get());
#endif
}

}  // namespace Cpu
