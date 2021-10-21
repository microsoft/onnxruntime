// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"

#include "helper.h"
#include "shaper.h"

namespace onnxruntime {
namespace nnapi {

using std::string;
using std::vector;

std::pair<uint32_t, uint32_t> ComputeConvOutputShape(const uint32_t input_size_y, const uint32_t input_size_x,
                                                     const uint32_t weight_size_y, const uint32_t weight_size_x,
                                                     const vector<int32_t>& onnx_pads,
                                                     const vector<int32_t>& onnx_strides,
                                                     const vector<int32_t>& onnx_dilations) {
  int32_t padding_top = onnx_pads[0];
  int32_t padding_bottom = onnx_pads[2];
  int32_t padding_left = onnx_pads[1];
  int32_t padding_right = onnx_pads[3];
  int32_t stride_y = onnx_strides[0];
  int32_t stride_x = onnx_strides[1];
  int32_t dilation_y = onnx_dilations[0];
  int32_t dilation_x = onnx_dilations[1];

  int64_t output_size_y =
      0 == input_size_y
          ? 0
          : onnxruntime::ComputeOutputShape(input_size_y,
                                            stride_y, weight_size_y, dilation_y,
                                            padding_top, padding_bottom);

  int64_t output_size_x =
      0 == input_size_x
          ? 0
          : onnxruntime::ComputeOutputShape(input_size_x,
                                            stride_x, weight_size_x, dilation_x,
                                            padding_left, padding_right);

  return std::make_pair(static_cast<uint32_t>(output_size_y), static_cast<uint32_t>(output_size_x));
}

#define SHAPER_FUNC(FUNC, ...)                  \
  ORT_RETURN_IF_ERROR(FUNC##Impl(__VA_ARGS__)); \
  shape_ops_.push_back(                         \
      [__VA_ARGS__](Shaper& shaper) {           \
        return shaper.FUNC##Impl(__VA_ARGS__);  \
      });                                       \
  return Status::OK();

Status Shaper::Conv(const std::string& input_name,
                    const std::string& weight_name,
                    const vector<int32_t>& onnx_pads,
                    const vector<int32_t>& onnx_strides,
                    const vector<int32_t>& onnx_dilations,
                    bool nchw,
                    const std::string& output_name) {
  SHAPER_FUNC(Conv,
              input_name, weight_name,
              onnx_pads, onnx_strides, onnx_dilations,
              nchw,
              output_name);
}

Status Shaper::DepthwiseConv(const std::string& input_name,
                             const std::string& weight_name,
                             const std::vector<int32_t>& onnx_pads,
                             const std::vector<int32_t>& onnx_strides,
                             const std::vector<int32_t>& onnx_dilations,
                             bool nchw,
                             const std::string& output_name) {
  SHAPER_FUNC(DepthwiseConv,
              input_name, weight_name,
              onnx_pads, onnx_strides, onnx_dilations,
              nchw,
              output_name);
}

Status Shaper::Pool(const std::string& input_name,
                    const std::vector<int32_t>& onnx_pads,
                    const std::vector<int32_t>& onnx_strides,
                    const std::vector<int32_t>& kernel_shape,
                    bool nchw,
                    const std::string& output_name) {
  SHAPER_FUNC(Pool,
              input_name,
              onnx_pads, onnx_strides, kernel_shape,
              nchw,
              output_name);
}

Status Shaper::Reshape(const std::string& input_name,
                       const std::vector<int32_t>& shape,
                       const std::string& output_name) {
  SHAPER_FUNC(Reshape, input_name, shape, output_name);
}

Status Shaper::Transpose(const std::string& input_name,
                         const std::vector<int32_t>& perm,
                         const std::string& output_name) {
  SHAPER_FUNC(Transpose, input_name, perm, output_name);
}

Status Shaper::Eltwise(const std::string& input1_name,
                       const std::string& input2_name,
                       const std::string& output_name) {
  SHAPER_FUNC(Eltwise, input1_name, input2_name, output_name);
}

Status Shaper::Identity(const std::string& input_name,
                        const std::string& output_name) {
  SHAPER_FUNC(Identity, input_name, output_name);
}

Status Shaper::FC(const std::string& input1_name, const std::string& input2_name,
                  const std::string& output_name) {
  SHAPER_FUNC(FC, input1_name, input2_name, output_name);
}

Status Shaper::Concat(const std::vector<std::string>& input_names,
                      const int32_t axis,
                      const std::string& output_name) {
  SHAPER_FUNC(Concat, input_names, axis, output_name);
}

Status Shaper::Squeeze(const std::string& input_name,
                       const std::vector<int32_t>& axes,
                       const std::string& output_name) {
  SHAPER_FUNC(Squeeze, input_name, axes, output_name);
}

Status Shaper::ResizeUsingScales(const std::string& input_name,
                                 const float scale_h, const float scale_w,
                                 bool nchw,
                                 const std::string& output_name) {
  SHAPER_FUNC(ResizeUsingScales, input_name, scale_h, scale_w, nchw, output_name);
}

Status Shaper::ResizeUsingOutputSizes(const std::string& input_name,
                                      const uint32_t output_h, const uint32_t output_w,
                                      bool nchw,
                                      const std::string& output_name) {
  SHAPER_FUNC(ResizeUsingOutputSizes, input_name, output_h, output_w, nchw, output_name);
}

#undef SHAPER_FUNC

Status Shaper::ConvImpl(const std::string& input_name,
                        const std::string& weight_name,
                        const vector<int32_t>& onnx_pads,
                        const vector<int32_t>& onnx_strides,
                        const vector<int32_t>& onnx_dilations,
                        bool nchw,
                        const std::string& output_name) {
  const Shape& input_dimen = shape_map_.at(input_name);
  const Shape& weight_dimen = shape_map_.at(weight_name);  // num_output, height, width, num_input

  const auto input_size_y = nchw ? input_dimen[2] : input_dimen[1];
  const auto input_size_x = nchw ? input_dimen[3] : input_dimen[2];
  const auto weight_size_y = weight_dimen[1];
  const auto weight_size_x = weight_dimen[2];

  uint32_t output_size_y, output_size_x;
  std::tie(output_size_y, output_size_x) =
      ComputeConvOutputShape(input_size_y, input_size_x,
                             weight_size_y, weight_size_x,
                             onnx_pads, onnx_strides, onnx_dilations);
  Shape output_dimen;
  if (nchw) {
    output_dimen = {input_dimen[0], weight_dimen[0], output_size_y, output_size_x};
  } else {  // nhwc
    output_dimen = {input_dimen[0], output_size_y, output_size_x, weight_dimen[0]};
  }

  shape_map_[output_name] = output_dimen;
  return Status::OK();
}

Status Shaper::DepthwiseConvImpl(const std::string& input_name,
                                 const std::string& weight_name,
                                 const std::vector<int32_t>& onnx_pads,
                                 const std::vector<int32_t>& onnx_strides,
                                 const std::vector<int32_t>& onnx_dilations,
                                 bool nchw,
                                 const std::string& output_name) {
  const Shape& input_dimen = shape_map_.at(input_name);
  const Shape& weight_dimen = shape_map_.at(weight_name);  // 1, height, width, num_output

  const auto input_size_y = nchw ? input_dimen[2] : input_dimen[1];
  const auto input_size_x = nchw ? input_dimen[3] : input_dimen[2];
  const auto weight_size_y = weight_dimen[1];
  const auto weight_size_x = weight_dimen[2];

  uint32_t output_size_y, output_size_x;
  std::tie(output_size_y, output_size_x) =
      ComputeConvOutputShape(input_size_y, input_size_x,
                             weight_size_y, weight_size_x,
                             onnx_pads, onnx_strides, onnx_dilations);

  Shape output_dimen;
  if (nchw) {
    output_dimen = {input_dimen[0], weight_dimen[3], output_size_y, output_size_x};
  } else {  // nhwc
    output_dimen = {input_dimen[0], output_size_y, output_size_x, weight_dimen[3]};
  }
  shape_map_[output_name] = output_dimen;
  return Status::OK();
}

Status Shaper::PoolImpl(const std::string& input_name,
                        const std::vector<int32_t>& onnx_pads,
                        const std::vector<int32_t>& onnx_strides,
                        const std::vector<int32_t>& kernel_shape,
                        bool nchw,
                        const std::string& output_name) {
  const Shape& input_dimen = shape_map_.at(input_name);
  const auto input_size_y = nchw ? input_dimen[2] : input_dimen[1];
  const auto input_size_x = nchw ? input_dimen[3] : input_dimen[2];
  const auto weight_size_y = kernel_shape[0];
  const auto weight_size_x = kernel_shape[1];

  uint32_t output_size_y, output_size_x;
  std::tie(output_size_y, output_size_x) =
      ComputeConvOutputShape(input_size_y, input_size_x,
                             weight_size_y, weight_size_x,
                             onnx_pads, onnx_strides, {1, 1} /* onnx_dilations */);
  Shape output_dimen;
  if (nchw) {
    output_dimen = {input_dimen[0], input_dimen[1], output_size_y, output_size_x};
  } else {  // nhwc
    output_dimen = {input_dimen[0], output_size_y, output_size_x, input_dimen[3]};
  }

  shape_map_[output_name] = output_dimen;
  return Status::OK();
}

Status Shaper::ReshapeImpl(const std::string& input_name,
                           const std::vector<int32_t>& shape,
                           const std::string& output_name) {
  const Shape& input_dimen = shape_map_.at(input_name);
  uint32_t input_size = Product(input_dimen);
  std::vector<uint32_t> output_dimen(shape.size());

  int64_t capacity = 1;
  int unk_dim_idx = -1;
  for (size_t i = 0; i < shape.size(); i++) {
    int32_t dim_i = shape[i];
    ORT_RETURN_IF_NOT(dim_i != 0, "NNAPI does not support 0 reshape dimension");
    if (dim_i == -1) {
      ORT_RETURN_IF_NOT(unk_dim_idx == -1, "Only one input dimension of Attr(shape) can be unknown!");
      unk_dim_idx = i;
    } else {
      capacity *= dim_i;
      output_dimen[i] = static_cast<uint32_t>(dim_i);
    }
  }

  if (unk_dim_idx != -1) {
    if (input_size == 0)
      output_dimen[unk_dim_idx] = 0;
    else
      output_dimen[unk_dim_idx] = input_size / capacity;

    capacity *= output_dimen[unk_dim_idx];
  }

  ORT_RETURN_IF_NOT(capacity == input_size, "Invalid shape is given!");

  shape_map_[output_name] = output_dimen;
  return Status::OK();
}

Status Shaper::TransposeImpl(const std::string& input_name,
                             const std::vector<int32_t>& perm,
                             const std::string& output_name) {
  const Shape& input_dimen = shape_map_.at(input_name);

  ORT_RETURN_IF_NOT(perm.size() == input_dimen.size(), "Invalid perm is given!");

  size_t size = input_dimen.size();
  Shape output_dimen(size);
  for (size_t i = 0; i < size; i++)
    output_dimen[i] = input_dimen[perm[i]];

  shape_map_[output_name] = output_dimen;
  return Status::OK();
}

Status Shaper::EltwiseImpl(const std::string& input1_name,
                           const std::string& input2_name,
                           const std::string& output_name) {
  const Shape& shape1 = shape_map_.at(input1_name);
  const Shape& shape2 = shape_map_.at(input2_name);

  // broadcasting support
  bool shape1IsBigger = shape1.size() >= shape2.size();
  auto max_shape = shape1IsBigger ? shape1 : shape2;
  auto min_shape = shape1IsBigger ? shape2 : shape1;
  for (int i = (int)max_shape.size() - 1,
           j = (int)min_shape.size() - 1;
       i >= 0 && j >= 0;
       i--, j--) {
    int dim_max_shape = max_shape[i];
    int dim_min_shape = min_shape[j];
    if (dim_max_shape != dim_min_shape) {
      ORT_RETURN_IF_NOT(dim_max_shape == 1 || dim_min_shape == 1,
                        "Dimensions are not compatible, dim1: ", std::to_string(dim_max_shape),
                        "dim2: ", std::to_string(dim_min_shape));
    }

    if (dim_max_shape == 0 || dim_min_shape == 0) {
      max_shape[i] = 0;
    } else if (dim_max_shape < dim_min_shape) {
      max_shape[i] = dim_min_shape;
    }
  }

  shape_map_[output_name] = max_shape;
  return Status::OK();
}

Status Shaper::IdentityImpl(const std::string& input_name,
                            const std::string& output_name) {
  shape_map_[output_name] = shape_map_.at(input_name);
  return Status::OK();
}

Status Shaper::FCImpl(const std::string& input1_name, const std::string& input2_name,
                      const std::string& output_name) {
  // Currently we only support A*B'+C
  const Shape& input1_dimen = shape_map_.at(input1_name);
  const Shape& input2_dimen = shape_map_.at(input2_name);  // num_units, input_size
  Shape output_dimen{input1_dimen[0], input2_dimen[0]};
  shape_map_[output_name] = output_dimen;
  return Status::OK();
}

Status Shaper::ConcatImpl(const std::vector<std::string>& input_names,
                          const int32_t axis,
                          const std::string& output_name) {
  std::vector<Shape> dimens;
  for (const auto& input_name : input_names) {
    const Shape& dimen = shape_map_.at(input_name);
    if (!dimens.empty()) {
      for (size_t i = 0; i < dimens[0].size(); i++) {
        if ((int32_t)i == axis)
          continue;
      }
    }

    dimens.push_back(dimen);
  }

  // If one of the inputs has dynamic shape (at axis), we will keep the dimen[axis] as 0 (dynamic)
  auto output_dimen = dimens[0];
  if (output_dimen[axis] != 0) {
    for (size_t i = 1; i < dimens.size(); i++) {
      if (dimens[i][axis] == 0) {
        output_dimen[axis] = 0;
        break;
      }
      output_dimen[axis] += dimens[i][axis];
    }
  }

  shape_map_[output_name] = output_dimen;
  return Status::OK();
}

Status Shaper::SqueezeImpl(const std::string& input_name,
                           const std::vector<int32_t>& axes,
                           const std::string& output_name) {
  const Shape& input_dimen = shape_map_.at(input_name);
  int32_t input_size = input_dimen.size();
  std::unordered_set<int32_t> axes_to_be_squeezed;

  // If the Op is squeezing all by not specifying axes, the axes is pre-populate
  // with axes of all single dimensions by the caller
  for (const auto& axis : axes)
    axes_to_be_squeezed.insert(axis);

  // Make output dimensions
  std::vector<uint32_t> output_dimen;
  output_dimen.reserve(input_size - axes_to_be_squeezed.size());
  for (int32_t i = 0; i < input_size; i++) {
    if (!Contains(axes_to_be_squeezed, i))
      output_dimen.push_back(input_dimen[i]);
  }

  // In case of a tensor has all 1's in dimension such as {1,1,1,1} and gets squeezed all
  // the output shape will be {1}
  if (output_dimen.empty())
    output_dimen.push_back(1);

  shape_map_[output_name] = output_dimen;
  return Status::OK();
}

Status Shaper::ResizeUsingScalesImpl(const std::string& input_name,
                                     const float scale_h, const float scale_w,
                                     bool nchw,
                                     const std::string& output_name) {
  Shape output_dimen = shape_map_.at(input_name);
  if (nchw) {
    output_dimen[2] *= scale_h;
    output_dimen[3] *= scale_w;
  } else {  // nhwc
    output_dimen[1] *= scale_h;
    output_dimen[2] *= scale_w;
  }
  shape_map_[output_name] = output_dimen;
  return Status::OK();
}

Status Shaper::ResizeUsingOutputSizesImpl(const std::string& input_name,
                                          const uint32_t output_h, const uint32_t output_w,
                                          bool nchw,
                                          const std::string& output_name) {
  Shape output_dimen = shape_map_.at(input_name);
  if (nchw) {
    output_dimen[2] = output_h;
    output_dimen[3] = output_w;
  } else {  // nhwc
    output_dimen[1] = output_h;
    output_dimen[2] = output_w;
  }
  shape_map_[output_name] = output_dimen;
  return Status::OK();
}

void Shaper::AddShape(const std::string& name, const Shape& shape) {
  shape_map_[name] = shape;
}

Status Shaper::UpdateShape(const std::string& name, const Shape& new_shape) {
  const Shape& old_shape = shape_map_.at(name);
  if (old_shape != new_shape) {
    ORT_RETURN_IF_NOT(Product(old_shape) == 0 || !old_shape.empty(),
                      "The shape should be same size or old shape has size 0 (dynamic shape)");

    shape_map_[name] = new_shape;
  }

  return Status::OK();
}

Status Shaper::UpdateDynamicDimensions() {
  for (auto& shape_op : shape_ops_)
    ORT_RETURN_IF_ERROR(shape_op(*this));

  return Status::OK();
}

void Shaper::Clear() {
  shape_map_.clear();
  shape_ops_.clear();
}

}  // namespace nnapi
}  // namespace onnxruntime