#include "helper.h"
#include "Shaper.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/NeuralNetworksWrapper.h"

using std::string;
using std::vector;

Shaper::len_t Shaper::total(const Shape& shape) {
  return Product(shape);
}

// /**
//  *  strides: [stride_y, stride_x]
//  *  paddings: [top, left, bottom, right]
//  */
// void Shaper::Conv(const std::string &input_name, const std::string
// &weight_name,
//                   const std::vector<int32_t> paddings,
//                   const std::vector<int32_t> strides,
//                   const std::string &output_name) {
//     Shaper::Conv(input_name, strides[1], strides[0], 1, 1, paddings[1],
//                  paddings[3], paddings[0], paddings[2], weight_name,
//                  output_name);
// }
//
// void Shaper::Conv(const std::string &input_name,
//                   const std::vector<int32_t> paddings,
//                   const std::vector<int32_t> strides,
//                   const std::vector<int32_t> dilations,
//                   const std::string &weight_name,
//                   const std::string &output_name) {
//     Shaper::Conv(input_name, strides[1], strides[0], dilations[1],
//     dilations[0],
//                  paddings[1], paddings[3], paddings[0], paddings[2],
//                  weight_name, output_name);
// }

void Shaper::Conv(const std::string& input_name,
                  const std::string& weight_name,
                  int32_t padding_left,
                  int32_t padding_right,
                  int32_t padding_top,
                  int32_t padding_bottom,
                  int32_t stride_x,
                  int32_t stride_y,
                  int32_t dilation_x,
                  int32_t dilation_y,
                  bool nchw,
                  const std::string& output_name) {
  Shape weightDimen =
      shape_map_.at(weight_name);  // num_output, height, width, num_input
  // NHWC
  Shape inputDimen = shape_map_.at(input_name);
  Shape outputDimen;
  if (nchw) {
    outputDimen =
        {
            inputDimen[0],
            weightDimen[0],
            inputDimen[2] == 0
                ? 0
                : (inputDimen[2] - ((weightDimen[1] - 1) * dilation_y + 1) +
                   padding_top + padding_bottom) /
                          stride_y +
                      1,
            inputDimen[3] == 0
                ? 0
                : (inputDimen[3] - ((weightDimen[2] - 1) * dilation_x + 1) +
                   padding_left + padding_right) /
                          stride_x +
                      1,
        };
  } else {  // nhwc
    outputDimen =
        {
            inputDimen[0],
            inputDimen[1] == 0
                ? 0
                : (inputDimen[1] - ((weightDimen[1] - 1) * dilation_y + 1) +
                   padding_top + padding_bottom) /
                          stride_y +
                      1,
            inputDimen[2] == 0
                ? 0
                : (inputDimen[2] - ((weightDimen[2] - 1) * dilation_x + 1) +
                   padding_left + padding_right) /
                          stride_x +
                      1,
            weightDimen[0],
        };
  }

  shape_map_[output_name] = outputDimen;
}

void Shaper::DepthwiseConv(const std::string& input_name,
                           const std::string& weight_name,
                           int32_t padding_left,
                           int32_t padding_right,
                           int32_t padding_top,
                           int32_t padding_bottom,
                           int32_t stride_x,
                           int32_t stride_y,
                           int32_t dilation_x,
                           int32_t dilation_y,
                           bool nchw,
                           const std::string& output_name) {
  Shape weightDimen =
      shape_map_.at(weight_name);  // 1, height, width, num_output
  // NHWC
  Shape inputDimen = shape_map_.at(input_name);
  Shape outputDimen;
  if (nchw) {
    outputDimen =
        {
            inputDimen[0],
            weightDimen[3],
            inputDimen[2] == 0
                ? 0
                : (inputDimen[2] - ((weightDimen[1] - 1) * dilation_y + 1) +
                   padding_top + padding_bottom) /
                          stride_y +
                      1,
            inputDimen[3] == 0
                ? 0
                : (inputDimen[3] - ((weightDimen[2] - 1) * dilation_x + 1) +
                   padding_left + padding_right) /
                          stride_x +
                      1,
        };
  } else {  // nhwc
    outputDimen =
        {
            inputDimen[0],
            inputDimen[1]
                ? 0
                : (inputDimen[1] - ((weightDimen[1] - 1) * dilation_y + 1) +
                   padding_top + padding_bottom) /
                          stride_y +
                      1,
            inputDimen[2]
                ? 0
                : (inputDimen[2] - ((weightDimen[2] - 1) * dilation_x + 1) +
                   padding_left + padding_right) /
                          stride_x +
                      1,
            weightDimen[3],
        };
  }
  shape_map_[output_name] = outputDimen;
}

void Shaper::Reshape(const std::string& input_name,
                     const std::vector<int32_t>& shape,
                     const std::string& output_name) {
  auto input_dimen = shape_map_.at(input_name);
  int64_t input_size = Product(input_dimen);
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
    } else if (shape[i] < 0) {
      throw std::invalid_argument(
          "Each input dimension of Attr(shape) must"
          " not be negtive except one unknown dimension!");
    }

    int32_t output_dim_i = shape[i] ? shape[i] : input_dimen[i];
    output_dimen[i] = output_dim_i;
    capacity *= output_dim_i;
  }

  if (unk_dim_idx != -1) {
    if (input_size == 0)
      output_dimen[unk_dim_idx] = 0;
    else
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

void Shaper::Pool(const std::string& input_name,
                  int32_t padding_left,
                  int32_t padding_right,
                  int32_t padding_top,
                  int32_t padding_bottom,
                  int32_t stride_x,
                  int32_t stride_y,
                  bool nchw,
                  int32_t width,
                  int32_t height,
                  const std::string& output_name) {
  auto inputDimen = shape_map_.at(input_name);

  Shape outputDimen;
  if (nchw) {
    outputDimen = {
        inputDimen[0],
        inputDimen[1],
        inputDimen[2] == 0
            ? 0
            : (inputDimen[2] - height + padding_top + padding_bottom) / stride_y + 1,
        inputDimen[3] == 0
            ? 0
            : (inputDimen[3] - width + padding_left + padding_right) / stride_x + 1,
    };
  } else {
    outputDimen = {
        inputDimen[0],
        inputDimen[1] == 0
            ? 0
            : (inputDimen[1] - height + padding_top + padding_bottom) / stride_y + 1,
        inputDimen[2] == 0
            ? 0
            : (inputDimen[2] - width + padding_left + padding_right) / stride_x + 1,
        inputDimen[3]};
  }

  shape_map_[output_name] = outputDimen;
}

void Shaper::Softmax(const std::string& input_name,
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
  auto& shape1 = shape_map_.at(input1_name);
  auto& shape2 = shape_map_.at(input2_name);

  // broadcasting support
  bool shape1IsBigger = shape1.size() >= shape2.size();
  auto max_shape = shape1IsBigger ? shape1 : shape2;
  auto min_shape = shape1IsBigger ? shape2 : shape1;
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

// std::ostream& operator<<(std::ostream& os, const Shaper& shaper) {
//   for (const auto& p : shaper.shape_map_) {
//     os << (p.first + ": ") << p.second << std::endl;
//   }
//   return os;
// }
