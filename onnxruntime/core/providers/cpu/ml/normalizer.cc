// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/normalizer.h"

#include <algorithm>
#include "gsl/gsl"

/*
ONNX_OPERATOR_SCHEMA(Normalizer)
    .SetDomain("ai.onnx.ml")
    .SetDoc(R"DOC(
    Normalize the input.  There are three normalization modes,
    which have the corresponding formulas:
    Max .. math::     max(x_i)
    L1  .. math::  z = ||x||_1 = \sum_{i=1}^{n} |x_i|
    L2  .. math::  z = ||x||_2 = \sqrt{\sum_{i=1}^{n} x_i^2}
)DOC")
    .Input(0, "X", "Data to be encoded", "T")
    .Output(0, "Y", "encoded output data", "tensor(float)")
    .TypeConstraint(
        "T",
        {"tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"},
        " allowed types.")
    .Attr(
        "norm",
        "enum 'MAX', 'L1', 'L2'",
        AttributeProto::STRING,
        std::string("MAX"));
*/

namespace onnxruntime {
namespace ml {

ONNX_CPU_OPERATOR_ML_KERNEL(
    Normalizer,
    1,
    KernelDefBuilder().MayInplace(0, 0)  // input is 4 or 8 byte, output is 4 byte
        .TypeConstraint("T", std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>(),
                                                     DataTypeImpl::GetTensorType<double>(),
                                                     DataTypeImpl::GetTensorType<int32_t>(),
                                                     DataTypeImpl::GetTensorType<int64_t>()}),
    Normalizer);

template <typename T>
void NormalizeMax(const T* in, float* out, int64_t num_batches, int64_t batch_size) {
  for (int b = 0; b < num_batches; ++b) {
    float max = std::numeric_limits<float>::lowest();

    for (int i = 0; i < batch_size; ++i) {
      max = std::max(max, static_cast<float>(*in++));
    }

    in -= batch_size;

    if (max != 0.f) {
      for (int i = 0; i < batch_size; ++i) {
        *out++ = static_cast<float>(*in++) / max;
      }
    } else {
      for (int i = 0; i < batch_size; ++i) {
        *out++ = static_cast<float>(*in++);
      }
    }
  }
}

template <typename T>
static void NormalizeL1(const T* in, float* out, int64_t num_batches, int64_t batch_size) {
  for (int b = 0; b < num_batches; ++b) {
    float sum = 0.f;

    for (int i = 0; i < batch_size; ++i) {
      sum += static_cast<float>(std::abs(*in++));
    }

    in -= batch_size;

    if (sum != 0.f) {
      for (int i = 0; i < batch_size; ++i) {
        *out++ = static_cast<float>(*in++) / sum;
      }
    } else {
      for (int i = 0; i < batch_size; ++i) {
        *out++ = static_cast<float>(*in++);
      }
    }
  }
}

template <typename T>
void NormalizeL2(const T* in, float* out, int64_t num_batches, int64_t batch_size) {
  for (int b = 0; b < num_batches; ++b) {
    float sum = 0.f;

    for (int i = 0; i < batch_size; ++i) {
      auto x = *in++;
      auto x_sq = static_cast<float>(x * x);
      *out++ = x_sq;
      sum += x_sq;
    }

    in -= batch_size;
    out -= batch_size;

    if (sum != 0.f) {
      for (int i = 0; i < batch_size; ++i) {
        auto x = *in++;
        auto x_sq = *out;

        *out++ = (x < 0) ? std::sqrt(x_sq / sum) * -1 : std::sqrt(x_sq / sum);
      }
    } else {
      for (int i = 0; i < batch_size; ++i) {
        *out++ = static_cast<float>(*in++);
      }
    }
  }
}

template <typename T>
Status Normalizer::Normalize(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);
  const TensorShape& x_shape = X.Shape();

  if (x_shape.NumDimensions() > 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Rank of input to Normalized must be less than 2. Got ",
                           x_shape.NumDimensions());
  }

  const auto& x_dims = x_shape.GetDims();
  int64_t num_batches = x_dims.size() == 1 ? 1 : x_dims[0];
  int64_t batch_size = x_dims.size() == 1 ? x_dims[0] : x_dims[1];

  Tensor* Y = context->Output(0, x_shape);

  const T* input = X.template Data<T>();
  float* output = Y->MutableData<float>();

  switch (normalization_) {
    case NORMALIZE::NMAX: {
      NormalizeMax(input, output, num_batches, batch_size);
      break;
    }
    case NORMALIZE::L1: {
      NormalizeL1(input, output, num_batches, batch_size);
      break;
    }
    case NORMALIZE::L2: {
      NormalizeL2(input, output, num_batches, batch_size);
      break;
    }
    default: {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unexpected NORMALIZE value of ", normalization_);
    }
  }

  return Status::OK();
}

// MLTypeCallDispather implementation wrapper
template <class T>
struct Normalizer::CallNormalizerImpl {
  Status operator()(const Normalizer* norm, OpKernelContext* ctx) const {
    return norm->Normalize<T>(ctx);
  }
};

Status Normalizer::Compute(OpKernelContext* context) const {
  const auto& input_tensor_ptr = *context->Input<Tensor>(0);

  utils::MLTypeCallDispatcher<float, double, int64_t, int32_t>
      t_disp(input_tensor_ptr.GetElementType());

  auto status = t_disp.InvokeRet<Status, CallNormalizerImpl>(this, context);
  return status;
}

}  // namespace ml
}  // namespace onnxruntime
