#include "qlinear_where.h"
#include "qlinear_lookup_table.h"

#include "core/providers/common.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

QLinearWhere::QLinearWhere(const OpKernelInfo& info) : OpKernel(info) {
  size_t input_def_count = info.node().InputDefs().size();
  ORT_ENFORCE(input_def_count == expected_input_count,
              "There must be ", expected_input_count, " inputs! (condition, x, x_scale, x_zero_point, y, y_scale, y_zero_point, z_scale, z_zero_point)");
  const Tensor* tensor_x_scale = nullptr;
  const Tensor* tensor_x_zero_point = nullptr;
  const Tensor* tensor_y_scale = nullptr;
  const Tensor* tensor_y_zero_point = nullptr;
  const Tensor* tensor_z_scale = nullptr;
  const Tensor* tensor_z_zero_point = nullptr;

  bool get_x_scale = info.TryGetConstantInput(2, &tensor_x_scale);
  bool get_x_zero_point = info.TryGetConstantInput(3, &tensor_x_zero_point);
  bool get_y_scale = info.TryGetConstantInput(5, &tensor_y_scale);
  bool get_y_zero_point = info.TryGetConstantInput(6, &tensor_y_zero_point);
  bool get_z_scale = info.TryGetConstantInput(7, &tensor_z_scale);
  bool get_z_zero_point = info.TryGetConstantInput(8, &tensor_z_zero_point);
  if (!get_z_scale || !get_z_zero_point) {
    // Can not build fix lookup table
    return;
  }
  ORT_ENFORCE(
      tensor_x_zero_point->GetElementType() == tensor_y_zero_point->GetElementType() &&
          tensor_x_zero_point->GetElementType() == tensor_z_zero_point->GetElementType() &&
          tensor_y_zero_point->GetElementType() == tensor_z_zero_point->GetElementType(),
      "Wrong input type encountered for zero point input def of x, y, z");
  bool is_signed_int8 = tensor_z_zero_point->IsDataType<int8_t>();
  const auto identity_float = [](float v) -> float { return v; };

  if (get_x_scale && get_x_zero_point) {
    // Build fix lookup table for x
    is_x_fixed_copy_ = has_same_scale(tensor_x_scale, tensor_z_scale) &&
                       has_same_zero_point(is_signed_int8, tensor_x_zero_point, tensor_z_zero_point);
    if (!is_x_fixed_copy_) {
      x_fixed_lookup_table_.resize(256);
      if (is_signed_int8) {
        QlinearBuildLookupTable<int8_t>(
            x_fixed_lookup_table_.data(), tensor_x_scale, tensor_x_zero_point,
            tensor_z_scale, tensor_z_zero_point, identity_float);
      } else {
        QlinearBuildLookupTable<uint8_t>(
            x_fixed_lookup_table_.data(), tensor_x_scale, tensor_x_zero_point,
            tensor_z_scale, tensor_z_zero_point, identity_float);
      }
    }
    is_x_dynamic_ = false;
  }

  if (get_y_scale && get_y_zero_point) {
    // Build fix lookup table for y
    is_y_fixed_copy_ = has_same_scale(tensor_y_scale, tensor_z_scale) &&
                       has_same_zero_point(is_signed_int8, tensor_y_zero_point, tensor_z_zero_point);
    if (!is_y_fixed_copy_) {
      y_fixed_lookup_table_.resize(256);
      if (is_signed_int8) {
        QlinearBuildLookupTable<int8_t>(
            y_fixed_lookup_table_.data(), tensor_y_scale, tensor_y_zero_point,
            tensor_z_scale, tensor_z_zero_point, identity_float);
      } else {
        QlinearBuildLookupTable<uint8_t>(
            y_fixed_lookup_table_.data(), tensor_y_scale, tensor_y_zero_point,
            tensor_z_scale, tensor_z_zero_point, identity_float);
      }
    }
    is_y_dynamic_ = false;
  }
}

Status QLinearWhere::Compute(OpKernelContext* ctx) const {
  const auto* tensor_condition = ctx->Input<Tensor>(0);
  const auto* tensor_x_input = ctx->Input<Tensor>(1);
  const auto* tensor_x_scale = ctx->Input<Tensor>(2);
  const auto* tensor_x_zero_point = ctx->Input<Tensor>(3);
  const auto* tensor_y_input = ctx->Input<Tensor>(4);
  const auto* tensor_y_scale = ctx->Input<Tensor>(5);
  const auto* tensor_y_zero_point = ctx->Input<Tensor>(6);
  const auto* tensor_z_scale = ctx->Input<Tensor>(7);
  const auto* tensor_z_zero_point = ctx->Input<Tensor>(8);
  auto* tensor_output = ctx->Output(0, tensor_condition->Shape());
  ORT_ENFORCE(tensor_x_scale->IsDataType<float>(), "Input scale is not float for quantized input x @ 2");
  ORT_ENFORCE(tensor_y_scale->IsDataType<float>(), "Input scale is not float for quantized input y @ 5");
  ORT_ENFORCE(tensor_z_scale->IsDataType<float>(), "Input scale is not float for quantized output z @ 7");
  ORT_ENFORCE(tensor_x_zero_point->GetElementType() == tensor_y_zero_point->GetElementType() &&
                  tensor_x_zero_point->GetElementType() == tensor_z_zero_point->GetElementType() &&
                  tensor_y_zero_point->GetElementType() == tensor_z_zero_point->GetElementType(),
              "Wrong input type encountered for zero point of quantized input @", 3, 6, 8);
  bool is_signed_int8 = tensor_z_zero_point->IsDataType<int8_t>();
  const auto identity_float = [](float v) -> float { return v; };

  std::vector<uint8_t> x_dynamic_lookup_table;
  bool is_x_copy = !is_x_dynamic_ ? is_x_fixed_copy_ : has_same_scale(tensor_x_scale, tensor_z_scale) && has_same_zero_point(is_signed_int8, tensor_x_zero_point, tensor_z_zero_point);
  if (is_x_dynamic_ && !is_x_copy) {
    x_dynamic_lookup_table.resize(256);
    if (is_signed_int8) {
      QlinearBuildLookupTable<int8_t>(
          x_dynamic_lookup_table.data(), tensor_x_scale, tensor_x_zero_point,
          tensor_z_scale, tensor_z_zero_point, identity_float);
    } else {
      QlinearBuildLookupTable<uint8_t>(
          x_dynamic_lookup_table.data(), tensor_x_scale, tensor_x_zero_point,
          tensor_z_scale, tensor_z_zero_point, identity_float);
    }
  }

  // Build dynamic lookup table for y
  std::vector<uint8_t> y_dynamic_lookup_table;
  bool is_y_copy = !is_y_dynamic_ ? is_y_fixed_copy_ : has_same_scale(tensor_y_scale, tensor_z_scale) && has_same_zero_point(is_signed_int8, tensor_y_zero_point, tensor_z_zero_point);
  if (is_y_dynamic_ && !is_y_copy) {
    y_dynamic_lookup_table.resize(256);
    if (is_signed_int8) {
      QlinearBuildLookupTable<int8_t>(
          y_dynamic_lookup_table.data(), tensor_y_scale, tensor_y_zero_point,
          tensor_z_scale, tensor_z_zero_point, identity_float);
    } else {
      QlinearBuildLookupTable<uint8_t>(
          y_dynamic_lookup_table.data(), tensor_y_scale, tensor_y_zero_point,
          tensor_z_scale, tensor_z_zero_point, identity_float);
    }
  }
  const uint8_t* x_table = is_x_dynamic_ ? x_dynamic_lookup_table.data() : x_fixed_lookup_table_.data();
  const uint8_t* y_table = is_y_dynamic_ ? y_dynamic_lookup_table.data() : y_fixed_lookup_table_.data();
  // Todo: compute output z using ProcessBroadcastSpanFuncs
  return Status();
}

ONNX_CPU_OPERATOR_MS_KERNEL(
    QLinearWhere,
    1,
    KernelDefBuilder().TypeConstraint(
        "T",
        {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()}),
    QLinearWhere)
}  // namespace contrib
}  // namespace onnxruntime