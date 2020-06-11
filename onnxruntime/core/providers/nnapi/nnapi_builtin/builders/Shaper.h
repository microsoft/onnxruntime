#pragma once

#include <string>
#include <vector>
#include <unordered_map>

class Shaper {
 public:
  using len_t = uint32_t;
  using Shape = std::vector<len_t>;

  static len_t total(const Shape& shape);
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

  void Reshape(const std::string& input_name,
               const std::vector<int32_t>& shape,
               const std::string& output_name);
  void Transpose(const std::string& input_name,
                 const std::vector<int32_t>& perm,
                 const std::string& output_name);
  void Eltwise(const std::string& input1_name, const std::string& input2_name,
               const std::string& output_name);
  void Identity(const std::string& input_name,
                const std::string& output_name);

  void AddShape(const std::string& name, const Shape& shape);
  size_t GetSize(const std::string& name);
  void Clear();

  inline const Shape& operator[](const std::string& key) {
    return shape_map_.at(key);
  }

 private:
  std::unordered_map<std::string, Shape> shape_map_;
};
