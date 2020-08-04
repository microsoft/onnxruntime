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

  Status Conv(const std::string& input_name,
              const std::string& weight_name,
              const std::vector<int32_t>& onnx_pads,
              const std::vector<int32_t>& onnx_strides,
              const std::vector<int32_t>& onnx_dilations,
              bool nchw,
              const std::string& output_name);

  Status DepthwiseConv(const std::string& input_name,
                       const std::string& weight_name,
                       const std::vector<int32_t>& onnx_pads,
                       const std::vector<int32_t>& onnx_strides,
                       const std::vector<int32_t>& onnx_dilations,
                       bool nchw,
                       const std::string& output_name);

  Status Pool(const std::string& input_name,
              const std::vector<int32_t>& onnx_pads,
              const std::vector<int32_t>& onnx_strides,
              const std::vector<int32_t>& kernel_shape,
              bool nchw,
              const std::string& output_name);

  Status Reshape(const std::string& input_name, const std::vector<int32_t>& shape, const std::string& output_name);

  Status Transpose(const std::string& input_name, const std::vector<int32_t>& perm, const std::string& output_name);

  Status Eltwise(const std::string& input1_name, const std::string& input2_name, const std::string& output_name);

  Status Identity(const std::string& input_name, const std::string& output_name);

  Status FC(const std::string& input1_name, const std::string& input2_name, const std::string& output_name);

  Status Concat(const std::vector<std::string>& input_names, const int32_t axis, const std::string& output_name);

  Status Squeeze(const std::string& input, const std::vector<int32_t>& axes, const std::string& output);

  // If the shape of certain input is dynamic
  // Use the following 2 functions to update the particular shape
  // and calculate the new output shape
  // Only perform this when the NNAPI model is finalized!
  Status UpdateShape(const std::string& name, const Shape& new_shape);
  Status UpdateDynamicDimensions();

  void Clear();

 private:
  Status ConvImpl(const std::string& input_name,
                  const std::string& weight_name,
                  const std::vector<int32_t>& onnx_pads,
                  const std::vector<int32_t>& onnx_strides,
                  const std::vector<int32_t>& onnx_dilations,
                  bool nchw,
                  const std::string& output_name);

  Status DepthwiseConvImpl(const std::string& input_name,
                           const std::string& weight_name,
                           const std::vector<int32_t>& onnx_pads,
                           const std::vector<int32_t>& onnx_strides,
                           const std::vector<int32_t>& onnx_dilations,
                           bool nchw,
                           const std::string& output_name);

  Status PoolImpl(const std::string& input_name,
                  const std::vector<int32_t>& onnx_pads,
                  const std::vector<int32_t>& onnx_strides,
                  const std::vector<int32_t>& kernel_shape,
                  bool nchw,
                  const std::string& output_name);

  Status ReshapeImpl(const std::string& input_name, const std::vector<int32_t>& shape, const std::string& output_name);
  Status TransposeImpl(const std::string& input_name, const std::vector<int32_t>& perm, const std::string& output_name);
  Status EltwiseImpl(const std::string& input1_name, const std::string& input2_name, const std::string& output_name);
  Status IdentityImpl(const std::string& input_name, const std::string& output_name);
  Status FCImpl(const std::string& input1_name, const std::string& input2_name, const std::string& output_name);
  Status ConcatImpl(const std::vector<std::string>& input_names, const int32_t axis, const std::string& output_name);
  Status SqueezeImpl(const std::string& input, const std::vector<int32_t>& axes, const std::string& output);

  std::unordered_map<std::string, Shape> shape_map_;
  std::vector<std::function<Status(Shaper&)>> shape_ops_;
};

std::string Shape2String(const Shaper::Shape& shape);

}  // namespace nnapi
}  // namespace onnxruntime