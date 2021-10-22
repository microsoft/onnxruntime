// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qlinear_activations.h"

#include "qlinear_lookup_table.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
template <typename Transformer>
void QLinearLookupBase<T>::BuildLookupTableIfFixed(const OpKernelInfo& info, Transformer fn) {
  const Tensor* tensor_x_scale = nullptr;
  const Tensor* tensor_x_zero_point = nullptr;
  const Tensor* tensor_y_scale = nullptr;
  const Tensor* tensor_y_zero_point = nullptr;

  bool get_x_scale = info.TryGetConstantInput(1, &tensor_x_scale);
  bool get_x_zero_point = !info.node().InputDefs()[2]->Exists() || info.TryGetConstantInput(2, &tensor_x_zero_point);
  bool get_y_scale = info.TryGetConstantInput(3, &tensor_y_scale);
  bool get_y_zero_point = !info.node().InputDefs()[4]->Exists() || info.TryGetConstantInput(4, &tensor_y_zero_point);
  bool is_fixed_parameters = get_x_scale && get_x_zero_point && get_y_scale && get_y_zero_point;

  if (is_fixed_parameters) {
    fixed_lookup_table_.resize(256);
    QlinearBuildLookupTable<T>(
        fixed_lookup_table_.data(), tensor_x_scale, tensor_x_zero_point,
        tensor_y_scale, tensor_y_zero_point, fn);
  }
}

template <typename T>
template <typename Transformer>
Status QLinearLookupBase<T>::ComputeBase(OpKernelContext* context, Transformer fn) const {
  const auto& X = *context->Input<Tensor>(0);
  const auto& input_shape = X.Shape();
  const auto N = input_shape.Size();
  auto& Y = *context->Output(0, input_shape);

  uint8_t table[256];
  if (fixed_lookup_table_.size() == 0) {
    QlinearBuildLookupTable<T>(
        table, context->Input<Tensor>(1), context->Input<Tensor>(2),
        context->Input<Tensor>(3), context->Input<Tensor>(4), fn);
  }

  using onnxruntime::TensorOpCost;
  using onnxruntime::concurrency::ThreadPool;
  ThreadPool* tp = context->GetOperatorThreadPool();
  const uint8_t* x_data = reinterpret_cast<const uint8_t*>(X.template Data<T>());
  uint8_t* y_data = reinterpret_cast<uint8_t*>(Y.template MutableData<T>());
  ThreadPool::TryParallelFor(
      tp, N, TensorOpCost{1.0, 1.0, 1.0},
      [this, x_data, y_data, &table](std::ptrdiff_t first, std::ptrdiff_t last) {
        QLinearLookupTableTransform(
            x_data + first,
            fixed_lookup_table_.size() ? fixed_lookup_table_.data() : table,
            y_data + first,
            last - first);
      });

  return Status::OK();
}

// Derived classes from QLinearLookupBase
template <typename T>
QLinearLeakyRelu<T>::QLinearLeakyRelu(const OpKernelInfo& info)
    : QLinearLookupBase<T>(info), alpha_(info.GetAttrOrDefault("alpha", 0.01f)) {
  this->BuildLookupTableIfFixed(info, [this](float v) -> float {
    return v >= 0.0f ? v : alpha_ * v;
  });
}

template <typename T>
Status QLinearLeakyRelu<T>::Compute(OpKernelContext* context) const {
  return this->ComputeBase(context, [this](float v) -> float {
    return v >= 0.0f ? v : alpha_ * v;
  });
}

template <typename T>
QLinearSigmoid<T>::QLinearSigmoid(const OpKernelInfo& info)
    : QLinearLookupBase<T>(info) {
  this->BuildLookupTableIfFixed(info, [](const float* input, float* output, size_t length) {
    MlasComputeLogistic(input, output, length);
  });
}

template <typename T>
Status QLinearSigmoid<T>::Compute(OpKernelContext* context) const {
  return this->ComputeBase(context, [](const float* input, float* output, size_t length) {
    MlasComputeLogistic(input, output, length);
  });
}

#define REGISTER_QLINEAR_LOOKUPTABLE_TYPED_KERNEL(op_name, version, data_type, KERNEL_CLASS) \
  ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(                                                         \
      op_name, version, data_type,                                                           \
      KernelDefBuilder()                                                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()),                    \
      KERNEL_CLASS<data_type>);

REGISTER_QLINEAR_LOOKUPTABLE_TYPED_KERNEL(QLinearLeakyRelu, 1, int8_t, QLinearLeakyRelu);
REGISTER_QLINEAR_LOOKUPTABLE_TYPED_KERNEL(QLinearLeakyRelu, 1, uint8_t, QLinearLeakyRelu);
REGISTER_QLINEAR_LOOKUPTABLE_TYPED_KERNEL(QLinearSigmoid, 1, int8_t, QLinearSigmoid);
REGISTER_QLINEAR_LOOKUPTABLE_TYPED_KERNEL(QLinearSigmoid, 1, uint8_t, QLinearSigmoid);

}  // namespace contrib
}  // namespace onnxruntime

