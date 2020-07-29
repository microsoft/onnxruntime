#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace onnxruntime {
namespace nnapi {

class Shaper {
 public:
  using Shape = std::vector<uint32_t>;

  void AddShape(const std::string& name, const Shape& shape);
  inline const Shape& operator[](const std::string& key) const {
    return shape_map_.at(key);
  }

  void Conv(const std::string& input_name,
            const std::string& weight_name,
            const std::vector<int32_t>& onnx_pads,
            const std::vector<int32_t>& onnx_strides,
            const std::vector<int32_t>& onnx_dilations,
            bool nchw,
            const std::string& output_name);

  void DepthwiseConv(const std::string& input_name,
                     const std::string& weight_name,
                     const std::vector<int32_t>& onnx_pads,
                     const std::vector<int32_t>& onnx_strides,
                     const std::vector<int32_t>& onnx_dilations,
                     bool nchw,
                     const std::string& output_name);

  void Pool(const std::string& input_name,
            const std::vector<int32_t>& onnx_pads,
            const std::vector<int32_t>& onnx_strides,
            const std::vector<int32_t>& kernel_shape,
            bool nchw,
            const std::string& output_name);

  void Reshape(const std::string& input_name, const std::vector<int32_t>& shape, const std::string& output_name);

  void Transpose(const std::string& input_name, const std::vector<int32_t>& perm, const std::string& output_name);

  void Eltwise(const std::string& input1_name, const std::string& input2_name, const std::string& output_name);

  void Identity(const std::string& input_name, const std::string& output_name);

  void FC(const std::string& input1_name, const std::string& input2_name, const std::string& output_name);

  void Concat(const std::vector<std::string>& input_names, const int32_t axis, const std::string& output_name);

  void Squeeze(const std::string& input, const std::vector<int32_t>& axes, const std::string& output);

  // If the shape of certain input is dynamic
  // Use the following 2 functions to update the particular shape
  // and calculate the new output shape
  // Only perform this when the NNAPI model is finalized!
  Status UpdateShape(const std::string& name, const Shape& new_shape);
  void UpdateDynamicDimensions();

  void Clear();

 private:
  void ConvImpl(const std::string& input_name,
                const std::string& weight_name,
                const std::vector<int32_t>& onnx_pads,
                const std::vector<int32_t>& onnx_strides,
                const std::vector<int32_t>& onnx_dilations,
                bool nchw,
                const std::string& output_name);

  void DepthwiseConvImpl(const std::string& input_name,
                         const std::string& weight_name,
                         const std::vector<int32_t>& onnx_pads,
                         const std::vector<int32_t>& onnx_strides,
                         const std::vector<int32_t>& onnx_dilations,
                         bool nchw,
                         const std::string& output_name);

  void PoolImpl(const std::string& input_name,
                const std::vector<int32_t>& onnx_pads,
                const std::vector<int32_t>& onnx_strides,
                const std::vector<int32_t>& kernel_shape,
                bool nchw,
                const std::string& output_name);

  void ReshapeImpl(const std::string& input_name, const std::vector<int32_t>& shape, const std::string& output_name);
  void TransposeImpl(const std::string& input_name, const std::vector<int32_t>& perm, const std::string& output_name);
  void EltwiseImpl(const std::string& input1_name, const std::string& input2_name, const std::string& output_name);
  void IdentityImpl(const std::string& input_name, const std::string& output_name);
  void FCImpl(const std::string& input1_name, const std::string& input2_name, const std::string& output_name);
  void ConcatImpl(const std::vector<std::string>& input_names, const int32_t axis, const std::string& output_name);
  void SqueezeImpl(const std::string& input, const std::vector<int32_t>& axes, const std::string& output);

  std::unordered_map<std::string, Shape> shape_map_;
  std::vector<std::function<void(Shaper&)>> shape_ops_;
};

std::string Shape2String(const Shaper::Shape& shape);

}  // namespace nnapi
}  // namespace onnxruntime