#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <map>

class Shaper {
 public:
  using len_t = uint32_t;
  using Shape = std::vector<len_t>;

  static len_t total(const Shape& shape);
  void Conv(const std::string& input_name,
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
            const std::string& output_name);
  void DepthwiseConv(const std::string& input_name,
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
                     const std::string& output_name);

  void Pool(const std::string& input_name,
            int32_t padding_left,
            int32_t padding_right,
            int32_t padding_top,
            int32_t padding_bottom,
            int32_t stride_x,
            int32_t stride_y,
            int32_t width,
            int32_t height,
            bool nchw,
            const std::string& output_name);
  void Reshape(const std::string& input_name,
               const std::vector<int32_t>& shape,
               const std::string& output_name);
  void Transpose(const std::string& input_name,
                 const std::vector<uint32_t>& perm,
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
  std::map<std::string, Shape> shape_map_;
};
