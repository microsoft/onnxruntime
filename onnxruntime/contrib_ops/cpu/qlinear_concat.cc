// Copyright (c Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qlinear_concat.h"
#include "contrib_ops/cpu/qlinear_lookup_table.h"
#include "core/providers/common.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

constexpr int LOOKUP_TABLE_IS_FIXED = 1;
constexpr int LOOKUP_TABLE_IS_COPY = 2;

static inline bool has_same_scale(const Tensor* tensor_x_scale, const Tensor* tensor_y_scale) {
  return *(tensor_x_scale->Data<float>()) == *(tensor_y_scale->Data<float>());
}

static inline bool has_same_zero_point(bool is_signed, const Tensor* tensor_x_zero_point, const Tensor* tensor_y_zero_point) {
  if (is_signed) {
    const int8_t X_zero_point = (tensor_x_zero_point == nullptr) ? static_cast<int8_t>(0) : *(tensor_x_zero_point->template Data<int8_t>());
    const int8_t Y_zero_point = (tensor_y_zero_point == nullptr) ? static_cast<int8_t>(0) : *(tensor_y_zero_point->template Data<int8_t>());
    return X_zero_point == Y_zero_point;
  } else {
    const uint8_t X_zero_point = (tensor_x_zero_point == nullptr) ? static_cast<uint8_t>(0) : *(tensor_x_zero_point->template Data<uint8_t>());
    const uint8_t Y_zero_point = (tensor_y_zero_point == nullptr) ? static_cast<uint8_t>(0) : *(tensor_y_zero_point->template Data<uint8_t>());
    return X_zero_point == Y_zero_point;
  }
}

QLinearConcat::QLinearConcat(const OpKernelInfo& info) : OpKernel(info), ConcatBase(info) {
  size_t input_def_count = info.node().InputDefs().size();
  ORT_ENFORCE(input_def_count >= 8 && (input_def_count - 2) % 3 == 0,
              "At least two inputs are needed, and each input must be (tensor, scale, zero_point) tuple!");

  size_t input_count = (input_def_count - 2) / 3;
  fixed_lookup_tables_.resize(input_count);
  fixed_table_attrs_.resize(input_count, 0);

  const Tensor* tensor_y_scale = nullptr;
  const Tensor* tensor_y_zero_point = nullptr;
  bool get_y_scale = info.TryGetConstantInput(0, &tensor_y_scale);
  bool get_y_zero_point = info.TryGetConstantInput(1, &tensor_y_zero_point);
  if (!get_y_scale || !get_y_zero_point) {
    // Can not build const lookup table
    return;
  }

  // Initialize lookup table given constant input/output scales and zero points
  bool is_signed_int8 = tensor_y_zero_point->IsDataType<int8_t>();
  const auto identity_float = [](float v) -> float { return v; };
  for (size_t def_index = 2; def_index < input_def_count; def_index += 3) {
    const Tensor* tensor_x_scale = nullptr;
    const Tensor* tensor_x_zero_point = nullptr;
    bool get_x_scale = info.TryGetConstantInput(static_cast<int>(def_index) + 1, &tensor_x_scale);
    bool get_x_zero_point = info.TryGetConstantInput(static_cast<int>(def_index) + 2, &tensor_x_zero_point);
    if (!get_x_scale || !get_x_zero_point) {
      continue;  // try to optimize next one
    }
    ORT_ENFORCE(tensor_x_scale->IsDataType<float>(), "Input scale is not float for input def @", def_index + 1);
    ORT_ENFORCE(tensor_x_zero_point->GetElementType() == tensor_y_zero_point->GetElementType(),
                "Wrong input type encountered for zero point input def @", def_index + 2);

    size_t input_idx = (def_index - 2) / 3;
    fixed_table_attrs_[input_idx] |= LOOKUP_TABLE_IS_FIXED;
    if (has_same_scale(tensor_x_scale, tensor_y_scale) && has_same_zero_point(is_signed_int8, tensor_x_zero_point, tensor_y_zero_point)) {
      fixed_table_attrs_[input_idx] |= LOOKUP_TABLE_IS_COPY;
    } else {
      fixed_lookup_tables_[input_idx].resize(256);
      if (is_signed_int8) {
        QlinearBuildLookupTable<int8_t>(
            fixed_lookup_tables_[input_idx].data(), tensor_x_scale, tensor_x_zero_point,
            tensor_y_scale, tensor_y_zero_point, identity_float);
      } else {
        QlinearBuildLookupTable<uint8_t>(
            fixed_lookup_tables_[input_idx].data(), tensor_x_scale, tensor_x_zero_point,
            tensor_y_scale, tensor_y_zero_point, identity_float);
      }
    }
  }
}

Status QLinearConcat::Compute(OpKernelContext* ctx) const {
  const Tensor* tensor_y_scale = ctx->Input<Tensor>(0);
  const Tensor* tensor_y_zero_point = ctx->Input<Tensor>(1);
  bool is_signed_int8 = tensor_y_zero_point->IsDataType<int8_t>();
  const auto identity_float = [](float v) -> float { return v; };

  // Number of input tensors to concatenate (tupled)
  auto input_count_x3 = Node().InputArgCount()[2];
  ORT_ENFORCE(input_count_x3 >= 6 && input_count_x3 % 3 == 0,
              "At least two inputs are needed, and each input must be (tensor, scale, zero_point) tuple!");

  // Hold pointers to the input tensors to be used in the PrepareForCompute() step
  auto input_count = input_count_x3 / 3;
  std::vector<std::vector<uint8_t>> dynamic_lookup_tables(input_count);
  std::vector<int> dynamic_table_attrs(input_count, 0);

  std::vector<const Tensor*> input_tensors(input_count);
  for (auto input_index = 0; input_index < input_count; ++input_index) {
    auto tuple_start = 2 + input_index * 3;
    input_tensors[input_index] = ctx->Input<Tensor>(static_cast<int>(tuple_start));

    // Build lookup table for non-const parameters (scale + zero point)
    if (!(fixed_table_attrs_[input_index] & LOOKUP_TABLE_IS_FIXED)) {
      // Check tensor type first
      const Tensor* tensor_x_scale = ctx->Input<Tensor>(tuple_start + 1);
      const Tensor* tensor_x_zero_point = ctx->Input<Tensor>(tuple_start + 2);

      ORT_ENFORCE(tensor_x_scale->IsDataType<float>(), "Input scale is not float for quantized input @", tuple_start + 1);
      ORT_ENFORCE(tensor_x_zero_point->GetElementType() == tensor_y_zero_point->GetElementType(),
                  "Wrong input type encountered for zero point of quantized input @", tuple_start + 2);

      if (has_same_scale(tensor_x_scale, tensor_y_scale) && has_same_zero_point(is_signed_int8, tensor_x_zero_point, tensor_y_zero_point)) {
        dynamic_table_attrs[input_index] |= LOOKUP_TABLE_IS_COPY;
      } else {
        dynamic_lookup_tables[input_index].resize(256);
        if (is_signed_int8) {
          QlinearBuildLookupTable<int8_t>(
              dynamic_lookup_tables[input_index].data(), tensor_x_scale, tensor_x_zero_point,
              tensor_y_scale, tensor_y_zero_point, identity_float);
        } else {
          QlinearBuildLookupTable<uint8_t>(
              dynamic_lookup_tables[input_index].data(), tensor_x_scale, tensor_x_zero_point,
              tensor_y_scale, tensor_y_zero_point, identity_float);
        }
      }
    }
  }

  // Validate inputs and prepare some metadata used during actual compute
  Prepare p;
  auto status = PrepareForCompute(ctx, input_tensors, p);
  if (!status.IsOK())
    return status;

  // Return at this point if output tensor is going to be empty
  if (p.output_num_elements == 0)
    return Status::OK();

  int64_t initial_output_offset = 0;  // initial offset for each input
  for (int input_index = 0; input_index < input_count; input_index++) {
    const auto& prep = p.inputs[input_index];
    if (prep.num_elements == 0)
      continue;

    const uint8_t* table = (fixed_table_attrs_[input_index] & LOOKUP_TABLE_IS_FIXED)
                               ? fixed_lookup_tables_[input_index].data()
                               : dynamic_lookup_tables[input_index].data();
    const bool is_copy = (fixed_table_attrs_[input_index] & LOOKUP_TABLE_IS_FIXED)
                             ? bool(fixed_table_attrs_[input_index] & LOOKUP_TABLE_IS_COPY)
                             : bool(dynamic_table_attrs[input_index] & LOOKUP_TABLE_IS_COPY);

    auto input_axis_pitch = prep.axis_pitch;
    const uint8_t* input = static_cast<const uint8_t*>(prep.tensor->DataRaw());
    uint8_t* output = static_cast<uint8_t*>(p.output_tensor->MutableDataRaw()) + initial_output_offset;
    for (int64_t cur_in_offset = 0; cur_in_offset < prep.num_elements; cur_in_offset += input_axis_pitch) {
      if (is_copy) {
        memcpy(output, input + cur_in_offset, input_axis_pitch);
      } else {
        QLinearLookupTableTransform(input + cur_in_offset, table, output, input_axis_pitch);
      }
      output += p.output_axis_pitch;
    }

    initial_output_offset += input_axis_pitch;
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(QLinearConcat, kMSDomain, 1, kCpuExecutionProvider, KernelDefBuilder(), QLinearConcat);

}  // namespace contrib
}  // namespace onnxruntime
