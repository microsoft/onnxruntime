#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <core/session/onnxruntime_c_api.h>

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
              const std::string& output_name) ORT_MUST_USE_RESULT;

  Status DepthwiseConv(const std::string& input_name,
                       const std::string& weight_name,
                       const std::vector<int32_t>& onnx_pads,
                       const std::vector<int32_t>& onnx_strides,
                       const std::vector<int32_t>& onnx_dilations,
                       bool nchw,
                       const std::string& output_name) ORT_MUST_USE_RESULT;

  Status Pool(const std::string& input_name,
              const std::vector<int32_t>& onnx_pads,
              const std::vector<int32_t>& onnx_strides,
              const std::vector<int32_t>& kernel_shape,
              bool nchw,
              const std::string& output_name) ORT_MUST_USE_RESULT;

  Status Reshape(const std::string& input_name, const std::vector<int32_t>& shape, const std::string& output_name)
      ORT_MUST_USE_RESULT;

  Status Transpose(const std::string& input_name, const std::vector<int32_t>& perm, const std::string& output_name)
      ORT_MUST_USE_RESULT;

  Status Eltwise(const std::string& input1_name, const std::string& input2_name, const std::string& output_name)
      ORT_MUST_USE_RESULT;

  Status Identity(const std::string& input_name, const std::string& output_name) ORT_MUST_USE_RESULT;

  Status FC(const std::string& input1_name, const std::string& input2_name, const std::string& output_name)
      ORT_MUST_USE_RESULT;

  Status Concat(const std::vector<std::string>& input_names, const int32_t axis, const std::string& output_name)
      ORT_MUST_USE_RESULT;

  Status Squeeze(const std::string& input, const std::vector<int32_t>& axes, const std::string& output)
      ORT_MUST_USE_RESULT;

  // If the shape of certain input is dynamic
  // Use the following 2 functions to update the particular shape
  // and calculate the new output shape
  // Only perform this when the NNAPI model is finalized!
  Status UpdateShape(const std::string& name, const Shape& new_shape) ORT_MUST_USE_RESULT;
  Status UpdateDynamicDimensions() ORT_MUST_USE_RESULT;

  void Clear();

 private:
  Status ConvImpl(const std::string& input_name,
                  const std::string& weight_name,
                  const std::vector<int32_t>& onnx_pads,
                  const std::vector<int32_t>& onnx_strides,
                  const std::vector<int32_t>& onnx_dilations,
                  bool nchw,
                  const std::string& output_name) ORT_MUST_USE_RESULT;

  Status DepthwiseConvImpl(const std::string& input_name,
                           const std::string& weight_name,
                           const std::vector<int32_t>& onnx_pads,
                           const std::vector<int32_t>& onnx_strides,
                           const std::vector<int32_t>& onnx_dilations,
                           bool nchw,
                           const std::string& output_name) ORT_MUST_USE_RESULT;

  Status PoolImpl(const std::string& input_name,
                  const std::vector<int32_t>& onnx_pads,
                  const std::vector<int32_t>& onnx_strides,
                  const std::vector<int32_t>& kernel_shape,
                  bool nchw,
                  const std::string& output_name) ORT_MUST_USE_RESULT;

  Status ReshapeImpl(const std::string& input_name, const std::vector<int32_t>& shape, const std::string& output_name)
      ORT_MUST_USE_RESULT;
  Status TransposeImpl(const std::string& input_name, const std::vector<int32_t>& perm, const std::string& output_name)
      ORT_MUST_USE_RESULT;
  Status EltwiseImpl(const std::string& input1_name, const std::string& input2_name, const std::string& output_name)
      ORT_MUST_USE_RESULT;
  Status IdentityImpl(const std::string& input_name, const std::string& output_name) ORT_MUST_USE_RESULT;
  Status FCImpl(const std::string& input1_name, const std::string& input2_name, const std::string& output_name)
      ORT_MUST_USE_RESULT;
  Status ConcatImpl(const std::vector<std::string>& input_names, const int32_t axis, const std::string& output_name)
      ORT_MUST_USE_RESULT;
  Status SqueezeImpl(const std::string& input, const std::vector<int32_t>& axes, const std::string& output)
      ORT_MUST_USE_RESULT;

  std::unordered_map<std::string, Shape> shape_map_;
  std::vector<std::function<Status(Shaper&)>> shape_ops_;
};

std::string Shape2String(const Shaper::Shape& shape);

}  // namespace nnapi
}  // namespace onnxruntime