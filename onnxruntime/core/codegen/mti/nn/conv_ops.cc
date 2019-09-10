#include "core/codegen/mti/nn/conv_ops.h"

#include "core/codegen/mti/math/matmul_ops.h"
#include "core/codegen/mti/tensor/pad_ops.h"
#include "core/codegen/mti/tensor/reshape_ops.h"
#include "core/codegen/mti/tensor/transpose.h"

namespace onnxruntime {
namespace tvm_codegen {

static tvm::Tensor PadTensor1D(const tvm::Tensor& input,
                               const tvm::Array<tvm::Expr>& padding,
                               size_t width_axis,
                               const std::string& name) {
  auto pad_left = padding[0];
  auto pad_right = padding[1];

  tvm::Array<tvm::Expr> pad_before(std::vector<tvm::Expr>(input->shape.size(), 0));
  pad_before.Set(width_axis, pad_left);
  tvm::Array<tvm::Expr> pad_after(std::vector<tvm::Expr>(input->shape.size(), 0));
  pad_after.Set(width_axis, pad_right);

  const int64_t* padding_w0 = tvm::as_const_int(pad_left);
  const int64_t* padding_w1 = tvm::as_const_int(pad_right);

  const bool do_pad = ((padding_w0 != nullptr && *padding_w0) ||
                       (padding_w1 != nullptr && *padding_w1));

  return do_pad ? Pad(input, pad_before, pad_after,
                      0, "constant", name + "_input_padded")
                : input;
}

tvm::Tensor Conv1D(const tvm::Tensor& input,
                   const tvm::Tensor& filter,
                   const tvm::Array<tvm::Expr>& out_shape,
                   const tvm::Array<tvm::Expr>& stride,
                   const tvm::Array<tvm::Expr>& padding,
                   const std::string& name) {
  size_t channel_axis = 1;
  size_t width_axis = 2;

  auto stride_width = stride[width_axis - 2];

  auto input_padded = PadTensor1D(input, padding, width_axis, name);
  auto rc = tvm::reduce_axis((tvm::Range(0, filter->shape[1])), "rc");
  auto rx = tvm::reduce_axis((tvm::Range(0, filter->shape[2])), "rx");

  return tvm::compute(
      out_shape,
      [&](const tvm::Array<tvm::Var>& output) {
        tvm::Array<tvm::Expr> indices;
        for (const tvm::Var& var : output) {
          indices.push_back(var);
        }
        indices.Set(channel_axis, rc);
        indices.Set(width_axis, output[width_axis] * stride_width + rx);

        return tvm::sum(input_padded(indices) * filter({output[1], rc, rx}),
                        {rc, rx});
      },
      name);
}

tvm::Tensor Conv2D(const tvm::Tensor& input,
                   const tvm::Tensor& filter,
                   const tvm::Array<tvm::Expr>& output_shape,
                   const tvm::Array<tvm::Expr>& stride,
                   const tvm::Array<tvm::Expr>& padding,
                   const std::string& name) {
  return Conv2D_native(input, filter, output_shape, stride, padding);
}

static tvm::Tensor PadTensor2D(const tvm::Tensor& input,
                               const tvm::Array<tvm::Expr>& padding,
                               size_t height_axis,
                               size_t width_axis,
                               const std::string& name) {
  auto pad_top = padding[0];
  auto pad_left = padding[1];
  auto pad_bottom = padding[2];
  auto pad_right = padding[3];

  tvm::Array<tvm::Expr> pad_before(std::vector<tvm::Expr>(input->shape.size(), 0));
  pad_before.Set(height_axis, pad_top);
  pad_before.Set(width_axis, pad_left);

  tvm::Array<tvm::Expr> pad_after(std::vector<tvm::Expr>(input->shape.size(), 0));
  pad_after.Set(height_axis, pad_bottom);
  pad_after.Set(width_axis, pad_right);

  const int64_t* padding_h0 = tvm::as_const_int(pad_top);
  const int64_t* padding_w0 = tvm::as_const_int(pad_left);
  const int64_t* padding_h1 = tvm::as_const_int(pad_bottom);
  const int64_t* padding_w1 = tvm::as_const_int(pad_right);

  const bool do_pad = ((padding_h0 != nullptr && *padding_h0) ||
                       (padding_w0 != nullptr && *padding_w0)) ||
                      ((padding_h1 != nullptr && *padding_h1) ||
                       (padding_w1 != nullptr && *padding_w1));

  return do_pad ? Pad(input, pad_before, pad_after,
                      0, "constant", name + "_input_padded")
                : input;
}

tvm::Tensor Conv2D_native(const tvm::Tensor& input,
                          const tvm::Tensor& filter,
                          const tvm::Array<tvm::Expr>& out_shape,
                          const tvm::Array<tvm::Expr>& stride,
                          const tvm::Array<tvm::Expr>& padding,
                          const std::string& name) {
  size_t channel_axis = 1;
  size_t height_axis = 2;
  size_t width_axis = 3;

  auto stride_height = stride[height_axis - 2];
  auto stride_width = stride[width_axis - 2];

  auto input_padded = PadTensor2D(input, padding, height_axis, width_axis, name);

  auto rc = tvm::reduce_axis((tvm::Range(0, filter->shape[1])), "rc");
  auto ry = tvm::reduce_axis((tvm::Range(0, filter->shape[2])), "ry");
  auto rx = tvm::reduce_axis((tvm::Range(0, filter->shape[3])), "rx");

  return tvm::compute(
      out_shape,
      [&](const tvm::Array<tvm::Var>& output) {
        tvm::Array<tvm::Expr> indices;
        for (const tvm::Var& var : output) {
          indices.push_back(var);
        }
        indices.Set(channel_axis, rc);
        indices.Set(height_axis, output[height_axis] * stride_height + ry);
        indices.Set(width_axis, output[width_axis] * stride_width + rx);

        return tvm::sum(input_padded(indices) * filter({output[1], rc, ry, rx}),
                        {rc, ry, rx});
      },
      name);
}

tvm::Tensor Conv2D_gemm(const tvm::Tensor& input,
                        const tvm::Tensor& filter,
                        const tvm::Array<tvm::Expr>& out_shape,
                        const tvm::Array<tvm::Expr>& stride,
                        const tvm::Array<tvm::Expr>& padding,
                        const std::string& name) {
  size_t height_axis = 2;
  size_t width_axis = 3;

  auto stride_height = stride[height_axis - 2];
  auto stride_width = stride[width_axis - 2];

  auto input_padded = PadTensor2D(input, padding, height_axis, width_axis, name);

  tvm::Array<tvm::Expr> img_col_tmp(std::vector<tvm::Expr>(6, 0));
  img_col_tmp.Set(0, out_shape[0]);
  img_col_tmp.Set(1, out_shape[2]);
  img_col_tmp.Set(2, out_shape[3]);
  img_col_tmp.Set(3, filter->shape[1]);
  img_col_tmp.Set(4, filter->shape[2]);
  img_col_tmp.Set(5, filter->shape[3]);

  auto img_col = tvm::compute(
      img_col_tmp,
      [&](const tvm::Array<tvm::Var>& output) {
        tvm::Array<tvm::Expr> indices;
        indices.push_back(output[0]);
        indices.push_back(output[3]);
        indices.push_back(output[1] * stride_height + output[4]);
        indices.push_back(output[2] * stride_width + output[5]);
        return input_padded(indices);
      },
      name);

  tvm::Array<tvm::Expr> input_col_shape(std::vector<tvm::Expr>(2, 0));
  input_col_shape.Set(0, img_col_tmp[1] * img_col_tmp[2]);
  input_col_shape.Set(1, img_col_tmp[3] * img_col_tmp[4] * img_col_tmp[5]);
  auto input_col = Reshape(img_col, input_col_shape);

  tvm::Array<tvm::Expr> filter_row_shape(std::vector<tvm::Expr>(2, 0));
  filter_row_shape.Set(0, filter->shape[0]);
  filter_row_shape.Set(1, filter->shape[1] * filter->shape[2] * filter->shape[3]);
  auto filter_row = Reshape(filter, filter_row_shape, name);

  auto Y = MatMul2D(input_col, filter_row, false, true, name);
  auto Y_T = Transpose(Y, /*axes=*/{}, name);
  return Reshape(Y_T, out_shape, name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
