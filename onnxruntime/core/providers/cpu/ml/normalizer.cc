// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/normalizer.h"

#include <algorithm>

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
  for (int64_t b = 0; b < num_batches; ++b) {
    double max = std::numeric_limits<double>::lowest();

    for (int64_t i = 0; i < batch_size; ++i) {
      max = std::max(max, static_cast<double>(*in++));
    }

    in -= batch_size;

    if (max != 0.0) {
      for (int64_t i = 0; i < batch_size; ++i) {
        *out++ = static_cast<float>(static_cast<double>(*in++) / max);
      }
    } else {
      for (int64_t i = 0; i < batch_size; ++i) {
        *out++ = static_cast<float>(*in++);
      }
    }
  }
}

template <typename T>
static void NormalizeL1(const T* in, float* out, int64_t num_batches, int64_t batch_size) {
  for (int64_t b = 0; b < num_batches; ++b) {
    double scale = 0.0;

    for (int64_t i = 0; i < batch_size; ++i) {
      scale = std::max(scale, std::abs(static_cast<double>(*in++)));
    }

    in -= batch_size;

    if (scale != 0.0) {
      double scaled_sum = 0.0;
      for (int64_t i = 0; i < batch_size; ++i) {
        scaled_sum += std::abs(static_cast<double>(*in++) / scale);
      }

      in -= batch_size;
      for (int64_t i = 0; i < batch_size; ++i) {
        *out++ = static_cast<float>((static_cast<double>(*in++) / scale) / scaled_sum);
      }
    } else {
      for (int64_t i = 0; i < batch_size; ++i) {
        *out++ = static_cast<float>(*in++);
      }
    }
  }
}

template <typename T>
void NormalizeL2(const T* in, float* out, int64_t num_batches, int64_t batch_size) {
  for (int64_t b = 0; b < num_batches; ++b) {
    double scale = 0.0;

    for (int64_t i = 0; i < batch_size; ++i) {
      scale = std::max(scale, std::abs(static_cast<double>(*in++)));
    }

    in -= batch_size;
    if (scale != 0.0) {
      double scaled_sum_squares = 0.0;
      for (int64_t i = 0; i < batch_size; ++i) {
        const double scaled = static_cast<double>(*in++) / scale;
        scaled_sum_squares += scaled * scaled;
      }

      in -= batch_size;
      const double scaled_norm = std::sqrt(scaled_sum_squares);
      for (int64_t i = 0; i < batch_size; ++i) {
        *out++ = static_cast<float>((static_cast<double>(*in++) / scale) / scaled_norm);
      }
    } else {
      for (int64_t i = 0; i < batch_size; ++i) {
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
  if (x_dims.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input to Normalizer must have rank 1 or 2. Got rank 0.");
  }

  int64_t num_batches = x_dims.size() == 1 ? 1 : x_dims[0];
  int64_t batch_size = x_dims.size() == 1 ? x_dims[0] : x_dims[1];

  Tensor* Y = context->Output(0, x_shape);

  const T* input = X.Data<T>();
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
