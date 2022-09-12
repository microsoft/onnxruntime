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
  if (!get_x_scale || !get_x_zero_point || !get_y_scale || !get_y_zero_point || !get_z_scale || !get_z_zero_point) {
    // Can not build const lookup table
    return;
  }
  ORT_ENFORCE(
      tensor_x_zero_point->GetElementType() == tensor_y_zero_point->GetElementType() &&
          tensor_x_zero_point->GetElementType() == tensor_z_zero_point->GetElementType() &&
          tensor_y_zero_point->GetElementType() == tensor_z_zero_point->GetElementType(),
      "Wrong input type encountered for zero point input def of x, y, z");
  bool is_signed_int8 = tensor_z_zero_point->IsDataType<int8_t>();
  const auto identity_float = [](float v) -> float { return v; };

  x_fixed_table_attr_ |= LOOKUP_TABLE_IS_FIXED;
  if (has_same_scale(tensor_x_scale, tensor_z_scale) && has_same_zero_point(is_signed_int8, tensor_x_zero_point, tensor_z_zero_point)) {
    x_fixed_table_attr_ |= LOOKUP_TABLE_IS_COPY;
  } else {
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

  y_fixed_table_attr_ |= LOOKUP_TABLE_IS_FIXED;
  if (has_same_scale(tensor_y_scale, tensor_z_scale) && has_same_zero_point(is_signed_int8, tensor_y_zero_point, tensor_z_zero_point)) {
    y_fixed_table_attr_ |= LOOKUP_TABLE_IS_COPY;
  } else {
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
}

Status QLinearWhere::Compute(OpKernelContext* context) const {
  return Status();
}

ONNX_OPERATOR_KERNEL_EX(QLinearWhere, kMSDomain, 1, kCpuExecutionProvider, KernelDefBuilder(), QLinearWhere);
}  // namespace contrib
}  // namespace onnxruntime