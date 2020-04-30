#pragma once

#include <string>
#include <vector>
#include <map>

#include <iostream>

namespace onnxruntime {
namespace rknpu {

/**
 * Help to caculate the output shape of nodes.
 */
class Shaper {
 public:
  using len_t = uint32_t;
  using Shape = std::vector<len_t>;

  static len_t total(const Shape& shape);

  void Conv(const std::string& input,
            const std::string& weight,
            const std::vector<int32_t>& strides,
            const std::vector<int32_t>& paddings,
            const std::string& auto_pad,
            const std::string& output);
  void Conv(const std::string& input,
            const std::vector<int32_t>& strides,
            const std::vector<int32_t>& dilations,
            const std::vector<int32_t>& paddings,
            const std::string& weight,
            const std::string& auto_pad,
            const std::string& output);
  void Conv(const std::string& input,
            const std::string& weight,
            const int32_t padding_left,
            const int32_t padding_right,
            const int32_t padding_top,
            const int32_t padding_bottom,
            const int32_t stride_x,
            const int32_t stride_y,
            const std::string& auto_pad,
            const std::string& output);
  void Conv(const std::string& input,
            const int32_t strideX,
            const int32_t strideY,
            const int32_t dilationX,
            const int32_t dilationY,
            const int32_t paddingLeft,
            const int32_t paddingRight,
            const int32_t paddingTop,
            const int32_t paddingBottom,
            const std::string& weight,
            const std::string& auto_pad,
            const std::string& output);
  void DepthwiseConv(const std::string& input,
                     const std::vector<int32_t>& strides,
                     const std::vector<int32_t>& dilations,
                     const std::vector<int32_t>& paddings,
                     const std::string& weight,
                     const std::string& output);
  void DepthwiseConv(const std::string& input,
                     const std::string& weight,
                     const int32_t padding_left,
                     const int32_t padding_right,
                     const int32_t padding_top,
                     const int32_t padding_bottom,
                     const int32_t stride_x,
                     const int32_t stride_y,
                     const std::string& output);
  void DepthwiseConv(const std::string& input,
                     const int32_t strideX,
                     const int32_t strideY,
                     const int32_t dilationX,
                     const int32_t dilationY,
                     const int32_t paddingLeft,
                     const int32_t paddingRight,
                     const int32_t paddingTop,
                     const int32_t paddingBottom,
                     const std::string& weight,
                     const std::string& output);
  void DepthwiseConv(const std::string& input,
                     const std::string& weight,
                     const std::vector<int32_t>& paddings,
                     const std::vector<int32_t>& strides,
                     const std::string& output);
  void Slice(const std::string& input,
             const std::vector<int32_t>& starts,
             const std::vector<int32_t>& ends,
             const std::vector<int32_t>& axes,
             const std::vector<int32_t>& steps,
             const std::string& output);
  void StridedSlice(const std::string& input,
                    const std::vector<int32_t>& starts,
                    const std::vector<int32_t>& ends,
                    const std::vector<int32_t>& strides,
                    const int32_t beginMask,
                    const int32_t endMask,
                    const int32_t shrinkAxisMask,
                    const std::string& output);
  void Gather(const std::string& input,
              const std::string& indices,
              const int32_t axis,
              const std::string& output);
  void Pool(const std::string& input,
            const int32_t padding_left,
            const int32_t padding_right,
            const int32_t padding_top,
            const int32_t padding_bottom,
            const int32_t stride_x,
            const int32_t stride_y,
            const int32_t width,
            const int32_t height,
            const std::string& output);
  void Pool(const std::string& input,
            const std::vector<int32_t>& kernel_shape,
            const std::vector<int32_t>& pads,
            const std::vector<int32_t>& strides,
            const std::string& output);
  void Softmax(const std::string& input,
               const std::string& output);
  void Relu(const std::string& input,
            const std::string& output);
  void Concat(const std::vector<std::string>& inputs,
              const int32_t axis,
              const std::string& output);
  void LRN(const std::string& input,
           const std::string& output);
  void FC(const std::string& input,
          const std::string& weight,
          const std::string& output);
  void Eltwise(const std::string& input1,
               const std::string& input2,
               const std::string& output);
  void Eltwise(const std::string& input1,
               const std::string& output);
  void Affine(const std::string& input,
              const std::string& output);
  void Affine(const std::string& input,
              const std::string& a,
              const std::string& b,
              const std::string& output);
  void Identity(const std::string& input,
                const std::string& output);
  void BatchToSpace(const std::string& input,
                    const std::vector<int32_t>& block_sizes,
                    const std::string& output);
  void SpaceToBatch(const std::string& input,
                    const std::vector<int32_t>& block_sizes,
                    const std::vector<int32_t>& pads,
                    const std::string& output);
  void BatchNorm(const std::string& input,
                 const std::string& output);
  void Reshape(const std::string& input,
               const std::vector<int32_t>& shape,
               const std::string& output);
  void Transpose(const std::string& input,
                 const std::vector<int32_t>& perm,
                 const std::string& output);
  void Squeeze(const std::string& input,
               const std::vector<int32_t>& axes,
               const std::string& output);
  void Unsqueeze(const std::string& input,
                 const std::vector<int32_t>& axes,
                 const std::string& output);
  void AddShape(const std::string& name,
                const Shape& shape);
  size_t GetSize(const std::string& name);
  void Clear();

  inline const Shape& operator[](const std::string& key) {
    return shape_map_[key];
  }
  friend std::ostream& operator<<(std::ostream& os,
                                  const Shaper& shaper);

 private:
  std::map<std::string, Shape> shape_map_;
};

}  // namespace rknpu
}  // namespace onnxruntime
