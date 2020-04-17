#include "shaper.h"
#include <numeric>
#include <algorithm>
#include <functional>

using std::string;
using std::vector;

namespace onnxruntime {

template <typename T>
T Product(const std::vector<T>& v) {
  return static_cast<T>(
      accumulate(v.begin(), v.end(), 1, std::multiplies<T>()));
}

Shaper::len_t Shaper::total(const Shape& shape) {
  return Product(shape);
}

/**
 *  strides: [stride_y, stride_x]
 *  paddings: [top, left, bottom, right]
 */
void Shaper::Conv(const std::string& input_name, const std::string& weight_name,
                  const std::vector<int32_t> paddings,
                  const std::vector<int32_t> strides,
                  const std::string auto_pad,
                  const std::string& output_name) {
  Shaper::Conv(input_name, strides[1], strides[0], 1, 1, paddings[1],
               paddings[3], paddings[0], paddings[2], weight_name, auto_pad,
               output_name);
}

void Shaper::Conv(const std::string& input_name,
                  const std::vector<int32_t> paddings,
                  const std::vector<int32_t> strides,
                  const std::vector<int32_t> dilations,
                  const std::string& weight_name,
                  const std::string auto_pad,
                  const std::string& output_name) {
  Shaper::Conv(input_name, strides[1], strides[0], dilations[1], dilations[0],
               paddings[1], paddings[3], paddings[0], paddings[2],
               weight_name, auto_pad, output_name);
}
void Shaper::Conv(const std::string& input, const std::string& weight,
                  int32_t padding_left, int32_t padding_right,
                  int32_t padding_top, int32_t padding_bottom, int32_t stride_x,
                  int32_t stride_y, const std::string auto_pad,
                  const std::string& output) {
  Conv(input, stride_x, stride_y, 1, 1, padding_left, padding_right,
       padding_top, padding_bottom, weight, auto_pad, output);
}

void Shaper::Conv(const std::string& input_name, int32_t strideX,
                  int32_t strideY, int32_t dilationX, int32_t dilationY,
                  int32_t paddingLeft, int32_t paddingRight, int32_t paddingTop,
                  int32_t paddingBottom, const std::string& weight_name,
                  const std::string auto_pad,
                  const std::string& output_name) {
  Shape weightDimen =
      shape_map_.at(weight_name);  // num_output, height, width, num_input
  // NCHW
  Shape inputDimen = shape_map_.at(input_name);

  if (auto_pad == "VALID") {
    Shape outputDimen{inputDimen[0],
                      weightDimen[0],
                      (inputDimen[2] - ((weightDimen[2] - 1) * dilationY + 1)) / strideY + 1,
                      (inputDimen[3] - ((weightDimen[3] - 1) * dilationX + 1)) / strideX + 1};
    shape_map_[output_name] = outputDimen;
  } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
    int32_t legacy_target_size_Y = (inputDimen[2] + strideY - 1) / strideY;
    int32_t legacy_target_size_X = (inputDimen[3] + strideX - 1) / strideX;
    int32_t pad_needed_Y =
        (legacy_target_size_Y - 1) * strideY + weightDimen[2] - inputDimen[2];
    int32_t pad_needed_X =
        (legacy_target_size_X - 1) * strideX + weightDimen[3] - inputDimen[3];
    Shape outputDimen{inputDimen[0],
                      weightDimen[0],
                      (inputDimen[2] - ((weightDimen[2] - 1) * dilationY + 1) + pad_needed_Y) / strideY + 1,
                      (inputDimen[3] - ((weightDimen[3] - 1) * dilationX + 1) + pad_needed_X) / strideX + 1};
    shape_map_[output_name] = outputDimen;
  } else {  // default: NOTSET
    Shape outputDimen{inputDimen[0],
                      weightDimen[0],
                      (inputDimen[2] - ((weightDimen[2] - 1) * dilationY + 1) + paddingTop + paddingBottom) / strideY + 1,
                      (inputDimen[3] - ((weightDimen[3] - 1) * dilationX + 1) + paddingLeft + paddingRight) / strideX + 1};
    shape_map_[output_name] = outputDimen;
  }
}

void Shaper::DepthwiseConv(const std::string& input_name,
                           const std::vector<int32_t> paddings,
                           const std::vector<int32_t> strides,
                           const std::vector<int32_t> dilations,
                           const std::string& weight_name,
                           const std::string& output_name) {
  Shaper::DepthwiseConv(input_name, strides[1], strides[0], dilations[1],
                        dilations[0], paddings[1], paddings[3], paddings[0],
                        paddings[2], weight_name, output_name);
}

void Shaper::DepthwiseConv(const std::string& input_name,
                           const std::string& weight_name, int32_t padding_left,
                           int32_t padding_right, int32_t padding_top,
                           int32_t padding_bottom, int32_t stride_x,
                           int32_t stride_y, const std::string& output) {
  DepthwiseConv(input_name, stride_x, stride_y, 1, 1, padding_left,
                padding_right, padding_top, padding_bottom, weight_name,
                output);
}

void Shaper::DepthwiseConv(const std::string& input_name, int32_t strideX,
                           int32_t strideY, int32_t dilationX,
                           int32_t dilationY, int32_t paddingLeft,
                           int32_t paddingRight, int32_t paddingTop,
                           int32_t paddingBottom,
                           const std::string& weight_name,
                           const std::string& output_name) {
  Shape weightDimen =
      shape_map_.at(weight_name);  // 1, height, width, num_output
  // NCHW
  Shape inputDimen = shape_map_.at(input_name);
  Shape outputDimen{
      inputDimen[0],
      weightDimen[0],
      (inputDimen[2] - ((weightDimen[2] - 1) * dilationY + 1) +
       paddingTop + paddingBottom) /
              strideY +
          1,
      (inputDimen[3] - ((weightDimen[3] - 1) * dilationX + 1) +
       paddingLeft + paddingRight) /
              strideX +
          1,
  };
  shape_map_[output_name] = outputDimen;
}

void Shaper::DepthwiseConv(const std::string& input_name,
                           const std::string& weight_name,
                           const std::vector<int32_t> paddings,
                           const std::vector<int32_t> strides,
                           const std::string& output_name) {
  DepthwiseConv(input_name, weight_name, paddings[1], paddings[3],
                paddings[0], paddings[2], strides[1], strides[0],
                output_name);
}

void Shaper::Slice(const std::string& input_name,
                   const std::vector<int32_t>& starts,
                   const std::vector<int32_t>& ends,
                   const std::vector<int32_t>& axes,
                   const std::vector<int32_t>& steps,
                   const std::string& output_name) {
  vector<uint32_t> inputDimen = shape_map_.at(input_name);
  vector<uint32_t> outputDimen = inputDimen;
  for (size_t i = 0; i < axes.size(); i++) {
    int32_t axis =
        (axes[i] < 0) ? (axes[i] + (int32_t)inputDimen.size()) : axes[i];
    int32_t dim = outputDimen[axis];
    if (dim > 0) {
      int32_t start = starts[i] < 0 ? (starts[i] + dim) : starts[i];
      int32_t end = ends[i] < 0 ? (ends[i] + dim) : ends[i];
      start = std::max(start, 0);
      end = std::max(end, 0);
      end = std::min(end, dim);
      outputDimen[axis] = end - start;
    }
  }

  shape_map_[output_name] = outputDimen;
}

void Shaper::StridedSlice(const std::string& input_name,
                          const std::vector<int32_t>& starts,
                          const std::vector<int32_t>& ends,
                          const std::vector<int32_t>& strides,
                          int32_t beginMask, int32_t endMask,
                          int32_t shrinkAxisMask,
                          const std::string& output_name) {
  // NHWC
  vector<uint32_t> inputDimen = shape_map_.at(input_name);
  vector<uint32_t> outputDimen;
  for (size_t i = 0; i < inputDimen.size(); i++) {
    if (shrinkAxisMask & (1 << i)) {
      continue;
    }
    int32_t start = starts[i], end = ends[i], stride = strides[i];
    if (beginMask & (1 << i)) {
      start = 0;
    }
    if (endMask & (1 << i)) {
      end = inputDimen[i];
    }
    outputDimen.emplace_back((end - start) / stride);
  }
  shape_map_[output_name] = outputDimen;
}

void Shaper::Gather(const std::string& input_name,
                    const std::string& indices_name,
                    int32_t axis,
                    const std::string& output_name) {
  auto inputDimen = shape_map_.at(input_name);
  auto indicesDimen = shape_map_.at(indices_name);
  int32_t input_rank = indicesDimen.size();
  axis = (axis < 0) ? (axis + input_rank) : axis;

  vector<uint32_t> outputDimen;
  outputDimen.reserve(input_rank - 1 + indicesDimen.size());

  // replace the dimension for axis with the shape from the indices
  for (int32_t i = 0; i < axis; ++i)
    outputDimen.push_back(inputDimen[i]);

  for (const auto dim : indicesDimen)
    outputDimen.push_back(dim);

  for (int32_t i = axis + 1; i < input_rank; ++i)
    outputDimen.push_back(inputDimen[i]);

  shape_map_[output_name] = outputDimen;
}

void Shaper::Pool(const std::string& input_name, int32_t padding_left,
                  int32_t padding_right, int32_t padding_top,
                  int32_t padding_bottom, int32_t stride_x, int32_t stride_y,
                  int32_t width, int32_t height,
                  const std::string& output_name) {
  auto inputDimen = shape_map_.at(input_name);

  // NCHW
  Shape outputDimen;
  if (height == -1 && width == -1) {
    outputDimen = {inputDimen[0], inputDimen[1], 1, 1};
  } else {
    outputDimen = {
        inputDimen[0],
        inputDimen[1],
        (inputDimen[2] - height + padding_top + padding_bottom) / stride_y +
            1,
        (inputDimen[3] - width + padding_left + padding_right) / stride_x +
            1};
  }
  shape_map_[output_name] = outputDimen;
}

/**
 *  kernel_shape: [height, width]
 *  strides: [stride_y, stride_x]
 *  pads: [top, left, bottom, right]
 */
void Shaper::Pool(const std::string& input_name,
                  const std::vector<int32_t> kernel_shape,
                  const std::vector<int32_t> pads,
                  const std::vector<int32_t> strides,
                  const std::string& output_name) {
  Shaper::Pool(input_name, pads[1], pads[3], pads[0], pads[2], strides[1],
               strides[0], kernel_shape[1], kernel_shape[0], output_name);
}

void Shaper::Softmax(const std::string& input_name,
                     const std::string& output_name) {
  shape_map_[output_name] = shape_map_.at(input_name);
}

void Shaper::Relu(const std::string& input_name,
                  const std::string& output_name) {
  shape_map_[output_name] = shape_map_.at(input_name);
}

void Shaper::Concat(const std::vector<std::string>& input_names, uint32_t axis,
                    const std::string& output_name) {
  vector<Shape> dimens;
  for (const auto& input : input_names) {
    auto& dimen = shape_map_.at(input);
    if (!dimens.empty()) {
      for (size_t i = 0; i < dimens[0].size(); i++) {
        if (i == axis) continue;
        if (dimen[i] != dimens[0][i]) {
          throw std::invalid_argument("Wrong input for concat");
        }
      }
    }
    dimens.push_back(shape_map_.at(input));
  }

  auto outputDimen = dimens[0];
  for (size_t i = 1; i < dimens.size(); i++) {
    outputDimen[axis] += dimens[i][axis];
  }
  shape_map_[output_name] = outputDimen;
}

void Shaper::LRN(const std::string& input_name,
                 const std::string& output_name) {
  shape_map_[output_name] = shape_map_.at(input_name);
}

void Shaper::FC(const std::string& input_name, const std::string& weight_name,
                const std::string& output_name) {
  Shape weightDimen = shape_map_.at(weight_name);  // num_units, input_size
  auto input_dimen = shape_map_.at(input_name);
  Shape outputDimen{input_dimen[0], weightDimen[0]};
  shape_map_[output_name] = outputDimen;
}

void Shaper::Eltwise(const std::string& input1_name,
                     const std::string& input2_name,
                     const std::string& output_name) {
  auto shape1 = shape_map_.at(input1_name);
  auto shape2 = shape_map_.at(input2_name);

  // broadcasting support
  auto max_shape = shape1.size() >= shape2.size() ? shape1 : shape2;
  auto min_shape = shape1.size() < shape2.size() ? shape1 : shape2;
  for (int i = (int)max_shape.size() - 1, j = (int)min_shape.size() - 1;
       i >= 0 && j >= 0;
       i--, j--) {
    if (max_shape[i] < min_shape[j])
      max_shape[i] = min_shape[j];
  }

  shape_map_[output_name] = max_shape;
}

void Shaper::Eltwise(const std::string& input1_name,
                     const std::string& output_name) {
  shape_map_[output_name] = shape_map_.at(input1_name);
}

void Shaper::BatchToSpace(const std::string& input_name,
                          const std::vector<int32_t>& block_sizes,
                          const std::string& output_name) {
  auto input_dimen = shape_map_.at(input_name);
  auto output_dimen = {input_dimen[0] / Product(block_sizes),
                       input_dimen[1] * block_sizes[0],
                       input_dimen[2] * block_sizes[1], input_dimen[3]};
  shape_map_[output_name] = output_dimen;
}

void Shaper::SpaceToBatch(const std::string& input_name,
                          const std::vector<int32_t>& block_sizes,
                          const std::vector<int32_t>& pads,
                          const std::string& output_name) {
  auto input_dimen = shape_map_.at(input_name);
  auto output_dimen = {input_dimen[0] * Product(block_sizes),
                       (input_dimen[1] + pads[0] + pads[1]) / block_sizes[0],
                       (input_dimen[2] + pads[2] + pads[3]) / block_sizes[1],
                       input_dimen[3]};
  shape_map_[output_name] = output_dimen;
}

void Shaper::BatchNorm(const std::string& input_name,
                       const std::string& output_name) {
  shape_map_[output_name] = shape_map_.at(input_name);
}

void Shaper::Reshape(const std::string& input_name,
                     const std::vector<int32_t>& shape,
                     const std::string& output_name) {
  auto input_dimen = shape_map_.at(input_name);
  int64_t input_size = std::accumulate(
      input_dimen.begin(), input_dimen.end(), 1, std::multiplies<uint32_t>());
  std::vector<int32_t> output_dimen(shape.size());

  int64_t capacity = 1;
  int unk_dim_idx = -1;
  for (size_t i = 0; i < shape.size(); i++) {
    if (shape[i] == -1) {
      if (unk_dim_idx != -1)
        throw std::invalid_argument(
            "Only one input dimension of Attr(shape) can be unknown!");
      unk_dim_idx = i;
    } else if (shape[i] == 0) {
      if (i >= input_dimen.size())
        throw std::invalid_argument(
            "The index of dimension to copy from input"
            " shape must be less than the size of input shape!");
    } else {
      if (shape[i] < 0)
        throw std::invalid_argument(
            "Each input dimension of Attr(shape) must"
            " not be negtive except one unknown dimension!");
    }
    int32_t output_dim_i = shape[i] ? shape[i] : input_dimen[i];
    output_dimen[i] = output_dim_i;
    capacity *= output_dim_i;
  }

  if (unk_dim_idx != -1) {
    output_dimen[unk_dim_idx] = -input_size / capacity;
    if ((output_dimen[unk_dim_idx] * capacity) != (-input_size))
      throw std::invalid_argument("Invalid shape is given!");
  } else {
    if (capacity != input_size)
      throw std::invalid_argument("Invalid shape is given!");
  }

  Shape final_dimen(shape.size());
  for (size_t i = 0; i < shape.size(); i++) {
    final_dimen[i] = (uint32_t)output_dimen[i];
  }
  shape_map_[output_name] = final_dimen;
}

void Shaper::Transpose(const std::string& input_name,
                       const std::vector<int32_t> perm,
                       const std::string& output_name) {
  auto input_dimen = shape_map_.at(input_name);

  Shape outputDimen(perm.size());

  for (size_t i = 0; i < perm.size(); i++) {
    outputDimen[i] = input_dimen[perm[i]];
  }

  shape_map_[output_name] = outputDimen;
}

void Shaper::Squeeze(const std::string& input_name,
                     const std::vector<int32_t>& axes,
                     const std::string& output_name) {
  vector<uint32_t> inputDimen = shape_map_.at(input_name);
  size_t n_axes = axes.size();
  int cnt_squeezed_dims = 0;
  bool should_squeeze[9] = {false};

  if (n_axes == 0) {
    for (size_t idx = 0; idx < inputDimen.size(); ++idx) {
      if (inputDimen[idx] == 1) {
        should_squeeze[idx] = true;
        ++cnt_squeezed_dims;
      }
    }
  } else {
    for (size_t idx = 0; idx < n_axes; ++idx) {
      int32_t current =
          axes[idx] < 0 ? axes[idx] + inputDimen.size() : axes[idx];
      if (!(should_squeeze[current])) {
        ++cnt_squeezed_dims;
      }
      should_squeeze[current] = true;
    }
  }

  // Make output dimensions
  vector<uint32_t> outputDimen(inputDimen.size() - cnt_squeezed_dims, 0);
  for (size_t in_idx = 0, out_idx = 0; in_idx < inputDimen.size(); ++in_idx) {
    if (!should_squeeze[in_idx]) {
      outputDimen[out_idx++] = inputDimen[in_idx];
    }
  }

  shape_map_[output_name] = outputDimen;
}

void Shaper::Unsqueeze(const std::string& input_name,
                       const std::vector<int32_t>& axes,
                       const std::string& output_name) {
  vector<uint32_t> inputDimen = shape_map_.at(input_name);

  int output_size = inputDimen.size() + axes.size();
  int cur_output_size = inputDimen.size();
  vector<uint32_t> outputDimen(output_size, 0);

  for (int axis : axes) {
    int cur = axis;

    // Move old axis, and insert new axis.
    for (int i = cur_output_size; i >= cur; --i) {
      if (outputDimen[i] == 1) {
        // Move axis
        outputDimen[i + 1] = 1;
        outputDimen[i] = 0;
      }
    }

    outputDimen[cur] = 1;
    // Add the output size.
    cur_output_size++;
  }

  // Make output shape
  for (int in_idx = 0, out_idx = 0; out_idx < output_size; ++out_idx) {
    if (outputDimen[out_idx] == 0) {
      outputDimen[out_idx] = inputDimen[in_idx++];
    }
  }

  shape_map_[output_name] = outputDimen;
}

void Shaper::Affine(const std::string& input_name,
                    const std::string& output_name) {
  shape_map_[output_name] = shape_map_.at(input_name);
}
void Shaper::Affine(const std::string& input_name, const std::string& a,
                    const std::string& b, const std::string& output_name) {
  (void)a;
  (void)b;
  Shaper::Affine(input_name, output_name);
}

void Shaper::Identity(const std::string& input_name,
                      const std::string& output_name) {
  shape_map_[output_name] = shape_map_.at(input_name);
}

void Shaper::AddShape(const std::string& name, const Shape& shape) {
  shape_map_[name] = shape;
}

size_t Shaper::GetSize(const std::string& name) {
  return static_cast<size_t>(Product(shape_map_.at(name)));
}

void Shaper::Clear() {
  shape_map_.clear();
}

std::ostream& operator<<(std::ostream& os, const Shaper& shaper) {
  for (const auto& p : shaper.shape_map_) {
    os << (p.first + ": ") /*<< p.second*/ << std::endl;
  }
  return os;
}

}  // namespace onnxruntime