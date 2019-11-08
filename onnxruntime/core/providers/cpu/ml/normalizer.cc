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
void NormalizeMax(const gsl::span<const T>& in, gsl::span<float>& out,
                  int64_t offset, int64_t stride, int64_t increment_by) {
  float max = std::numeric_limits<float>::lowest();

  for (int64_t i = offset, s = 0; s < stride; ++s, i += increment_by) {
    max = std::max(max, static_cast<float>(in[i]));
  }

  if (max != 0.f) {
    for (int64_t i = offset, s = 0; s < stride; ++s, i += increment_by) {
      out[i] = static_cast<float>(in[i]) / max;
    }
  } else {
    for (int64_t i = offset, s = 0; s < stride; ++s, i += increment_by) {
      out[i] = static_cast<float>(in[i]);
    }
  }
}

template <typename T>
void NormalizeL1(const gsl::span<const T>& in, gsl::span<float>& out,
                 int64_t offset, int64_t stride, int64_t increment_by) {
  float sum = 0.f;

  for (int64_t i = offset, s = 0; s < stride; ++s, i += increment_by) {
    sum += static_cast<float>(std::abs(in[i]));
  }

  if (sum != 0.f) {
    for (int64_t i = offset, s = 0; s < stride; ++s, i += increment_by) {
      out[i] = static_cast<float>(in[i]) / sum;
    }
  } else {
    for (int64_t i = offset, s = 0; s < stride; ++s, i += increment_by) {
      out[i] = static_cast<float>(in[i]);
    }
  }
}

template <typename T>
void NormalizeL2(const gsl::span<const T>& in, gsl::span<float>& out,
                 int64_t offset, int64_t stride, int64_t increment_by) {
  float sum = 0.f;
  for (int64_t i = offset, s = 0; s < stride; ++s, i += increment_by) {
    auto x = in[i];
    auto x_sq = static_cast<float>(x * x);
    out[i] = x_sq;
    sum += x_sq;
  }

  if (sum != 0.f) {
    for (int64_t i = offset, s = 0; s < stride; ++s, i += increment_by) {
      auto x = in[i];
      auto x_sq = out[i];

      if (x < 0)
        out[i] = std::sqrt(x_sq / sum) * -1;
      else
        out[i] = std::sqrt(x_sq / sum);
    }
  } else {
    for (int64_t i = offset, s = 0; s < stride; ++s, i += increment_by) {
      out[i] = static_cast<float>(in[i]);
    }
  }
}

template <typename T>
void Normalizer::Normalize(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);
  const TensorShape& x_shape = X.Shape();
  const auto data_size = x_shape.Size();
  const auto& x_dims = x_shape.GetDims();

  Tensor* Y = context->Output(0, x_shape);

  auto input = gsl::make_span(X.template Data<T>(), data_size);
  auto output = gsl::make_span(Y->template MutableData<float>(), data_size);

  int64_t stride = x_dims.size() == 1 ? x_dims[0] : x_dims[1];
  int64_t loops = data_size / stride;

  // we normalize on axis 1 so if there are more than 2 dimensions we need to increment the index
  // by more than 1 as we process the stride
  // for 1 and 2 dimension tensors we're normalizing across the row/s, so increment_by is 1
  //
  // e.g. if you have a tensor of shape {2, 2, 3}
  // [[[ 1,  2,  3],
  //   [ 4,  5,  6]],
  //  [[ 7,  8,  9],
  //   [10, 11, 12]]]
  //
  // we want to normalize (1,  4), (2,  5), (3,  6),
  //                      (7, 10), (8, 11), (9, 12)
  // so the stride would be 2, and the increment_by would be 3.
  //
  // we process a block of stride * increment_by entries before we need to skip to the next row of the 2nd dimension.
  // the offset starts at 0 and increases by 1 each loop, for increment_by loops.
  // the offset then jumps by stride * increment

  int64_t increment_by = x_dims.size() > 1 ? x_shape.SizeFromDimension(2) : 1;

  for (int64_t n = 0; n < loops; ++n) {
    int64_t offset = (n % increment_by) + ((n / increment_by) * (stride * increment_by));

    switch (normalization_) {
      case NORMALIZE::NMAX: {
        NormalizeMax(input, output, offset, stride, increment_by);
        break;
      }
      case NORMALIZE::L1: {
        NormalizeL1(input, output, offset, stride, increment_by);
        break;
      }
      case NORMALIZE::L2: {
        NormalizeL2(input, output, offset, stride, increment_by);
        break;
      }
      default: {
        ORT_THROW("Unexpected NORMALIZE value of ", normalization_);
      }
    }
  }
}

// MLTypeCallDispather implementation wrapper
template <class T>
struct Normalizer::CallNormalizerImpl {
  void operator()(const Normalizer* norm, OpKernelContext* ctx) const {
    norm->Normalize<T>(ctx);
  }
};

Status Normalizer::Compute(OpKernelContext* context) const {
  const auto* input_tensor_ptr = context->Input<Tensor>(0);
  ORT_ENFORCE(input_tensor_ptr != nullptr);

  utils::MLTypeCallDispatcher<CallNormalizerImpl, float, double, int64_t, int32_t> t_disp(input_tensor_ptr->GetElementType());
  t_disp.Invoke(this, context);
  return Status::OK();
}

}  // namespace ml
}  // namespace onnxruntime
