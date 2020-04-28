#pragma once

#include <string>
#include <vector>
#include <map>

#include <iostream>

namespace onnxruntime {
namespace rknpu {

class Shaper {
 public:
  using len_t = uint32_t;
  using Shape = std::vector<len_t>;

  static len_t total(const Shape& shape);

  void Conv(const std::string& input_name, const std::string& weight_name,
            const std::vector<int32_t> strides,
            const std::vector<int32_t> paddings,
            const std::string auto_pad,
            const std::string& output_name);
  void Conv(const std::string& input_name, const std::vector<int32_t> strides,
            const std::vector<int32_t> dilations,
            const std::vector<int32_t> paddings,
            const std::string& weight_name, const std::string auto_pad,
            const std::string& output_name);
  void Conv(const std::string& input, const std::string& weight,
            int32_t padding_left, int32_t padding_right, int32_t padding_top,
            int32_t padding_bottom, int32_t stride_x, int32_t stride_y,
            const std::string auto_pad,
            const std::string& output);
  void Conv(const std::string& input_name, int32_t strideX, int32_t strideY,
            int32_t dilationX, int32_t dilationY, int32_t paddingLeft,
            int32_t paddingRight, int32_t paddingTop, int32_t paddingBottom,
            const std::string& weight_name, const std::string auto_pad,
            const std::string& output_name);
  void DepthwiseConv(const std::string& input_name,
                     const std::vector<int32_t> strides,
                     const std::vector<int32_t> dilations,
                     const std::vector<int32_t> paddings,
                     const std::string& weight_name,
                     const std::string& output_name);
  void DepthwiseConv(const std::string& input_name,
                     const std::string& weight_name, int32_t padding_left,
                     int32_t padding_right, int32_t padding_top,
                     int32_t padding_bottom, int32_t stride_x,
                     int32_t stride_y, const std::string& output);
  void DepthwiseConv(const std::string& input_name, int32_t strideX,
                     int32_t strideY, int32_t dilationX, int32_t dilationY,
                     int32_t paddingLeft, int32_t paddingRight,
                     int32_t paddingTop, int32_t paddingBottom,
                     const std::string& weight_name,
                     const std::string& output_name);
  void DepthwiseConv(const std::string& input_name,
                     const std::string& weight_name,
                     const std::vector<int32_t> paddings,
                     const std::vector<int32_t> strides,
                     const std::string& output_name);
  void Slice(const std::string& input_name,
             const std::vector<int32_t>& starts,
             const std::vector<int32_t>& ends,
             const std::vector<int32_t>& axes,
             const std::vector<int32_t>& steps,
             const std::string& output_name);
  void StridedSlice(const std::string& input_name,
                    const std::vector<int32_t>& starts,
                    const std::vector<int32_t>& ends,
                    const std::vector<int32_t>& strides, int32_t beginMask,
                    int32_t endMask, int32_t shrinkAxisMask,
                    const std::string& output_name);
  void Gather(const std::string& input_name,
              const std::string& indices_name,
              const int32_t axis,
              const std::string& output_name);
  void Pool(const std::string& input_name, int32_t padding_left,
            int32_t padding_right, int32_t padding_top,
            int32_t padding_bottom, int32_t stride_x, int32_t stride_y,
            int32_t width, int32_t height, const std::string& output_name);
  void Pool(const std::string& input_name,
            const std::vector<int32_t> kernel_shape,
            const std::vector<int32_t> pads,
            const std::vector<int32_t> strides,
            const std::string& output_name);
  void Softmax(const std::string& input_name, const std::string& output_name);
  void Relu(const std::string& input_name, const std::string& output_name);
  void Concat(const std::vector<std::string>& input_names, uint32_t axis,
              const std::string& output_name);
  void LRN(const std::string& input_name, const std::string& output_name);
  void FC(const std::string& input_name, const std::string& weight_name,
          const std::string& output_name);
  void Eltwise(const std::string& input1_name, const std::string& input2_name,
               const std::string& output_name);
  void Eltwise(const std::string& input1_name,
               const std::string& output_name);
  void Affine(const std::string& input_name, const std::string& output_name);
  void Affine(const std::string& input_name, const std::string& a,
              const std::string& b, const std::string& output_name);
  void Identity(const std::string& input_name,
                const std::string& output_name);
  void BatchToSpace(const std::string& input_name,
                    const std::vector<int32_t>& block_sizes,
                    const std::string& output_name);
  void SpaceToBatch(const std::string& input_name,
                    const std::vector<int32_t>& block_sizes,
                    const std::vector<int32_t>& pads,
                    const std::string& output_name);
  void BatchNorm(const std::string& input_name,
                 const std::string& output_name);
  void Reshape(const std::string& input_name,
               const std::vector<int32_t>& shape,
               const std::string& output_name);
  void Transpose(const std::string& input_name, const std::vector<int32_t> perm,
                 const std::string& output_name);
  void Squeeze(const std::string& input_name, const std::vector<int32_t>& axes,
               const std::string& output_name);
  void Unsqueeze(const std::string& input_name,
                 const std::vector<int32_t>& axes,
                 const std::string& output_name);
  void AddShape(const std::string& name, const Shape& shape);
  size_t GetSize(const std::string& name);
  void Clear();

  inline const Shape& operator[](const std::string& key) {
    return shape_map_.at(key);
  }
  friend std::ostream& operator<<(std::ostream& os, const Shaper& shaper);

 private:
  std::map<std::string, Shape> shape_map_;
};

}  // namespace rknpu
}  // namespace onnxruntime
