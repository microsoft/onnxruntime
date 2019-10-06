// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <vector>
#include <cmath>

#include "core/common/common.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/util/math.h"

namespace onnxruntime {

class Initializer final {
 public:
  static bool IsSupportedDataType(const ONNX_NAMESPACE::TensorProto* tensor_proto) {
    return !(tensor_proto == nullptr ||
             (tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
              tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 &&
              tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_DOUBLE));
  }

  Initializer(ONNX_NAMESPACE::TensorProto_DataType data_type,
              const std::string& name,
              const std::vector<int64_t>& dims) : size_(0) {
    data_type_ = data_type;
    name_ = name;
    dims_.assign(dims.cbegin(), dims.cend());

    size_ = std::accumulate(dims_.begin(), dims_.end(), static_cast<int64_t>(1), std::multiplies<int64_t>{});

    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
        float16_data_.assign(static_cast<size_t>(size_), 0);
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        float_data_.assign(static_cast<size_t>(size_), 0.0f);
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        double_data_.assign(static_cast<size_t>(size_), 0.0);
        break;
      }
      default:
        ORT_NOT_IMPLEMENTED(__FUNCTION__, "data type is not supported");
        break;
    }
  }

  Initializer(const ONNX_NAMESPACE::TensorProto* tensor_proto) : size_(0) {
    data_type_ = tensor_proto->data_type();
    if (utils::HasName(*tensor_proto)) {
      name_ = tensor_proto->name();
    }
    dims_.reserve(tensor_proto->dims_size());
    for (int i = 0; i < tensor_proto->dims_size(); i++) {
      dims_.push_back(tensor_proto->dims(i));
    }

    size_ = std::accumulate(dims_.begin(), dims_.end(), static_cast<int64_t>(1), std::multiplies<int64_t>{});

    if (utils::HasRawData(*tensor_proto)) {
      raw_data_ = tensor_proto->raw_data();
    } else {
      switch (data_type_) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
          int64_t size = tensor_proto->int32_data_size();
          ORT_ENFORCE(size_ == size, "size is different");
          for (int i = 0; i < size_; i++) {
            float16_data_.push_back(static_cast<uint16_t>(tensor_proto->int32_data(i)));
          }
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
          int64_t size = tensor_proto->float_data_size();
          ORT_ENFORCE(size_ == size, "size is different");
          for (int i = 0; i < size_; i++) {
            float_data_.push_back(tensor_proto->float_data(i));
          }
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
          int64_t size = tensor_proto->double_data_size();
          ORT_ENFORCE(size_ == size, "size is different");
          for (int i = 0; i < size_; i++) {
            double_data_.push_back(tensor_proto->double_data(i));
          }
          break;
        }
        default:
          ORT_NOT_IMPLEMENTED(__FUNCTION__, "data type is not supported");
          break;
      }
    }
  }

  ~Initializer() = default;

  void ToProto(ONNX_NAMESPACE::TensorProto* tensor_proto) {
    tensor_proto->clear_name();
    if (!name_.empty()) {
      tensor_proto->set_name(name_);
    }

    tensor_proto->clear_dims();
    for (auto d : dims_) {
      tensor_proto->add_dims(d);
    }

    tensor_proto->clear_data_type();
    tensor_proto->set_data_type(data_type_);

    if (!raw_data_.empty()) {
      tensor_proto->clear_raw_data();
      tensor_proto->set_raw_data(raw_data_);
    } else {
      switch (data_type_) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
          tensor_proto->clear_int32_data();
          for (int i = 0; i < size_; i++) {
            tensor_proto->add_int32_data(float16_data_[i]);
          }
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
          tensor_proto->clear_float_data();
          for (int i = 0; i < size_; i++) {
            tensor_proto->add_float_data(float_data_[i]);
          }
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
          tensor_proto->clear_double_data();
          for (int i = 0; i < size_; i++) {
            tensor_proto->add_double_data(double_data_[i]);
          }
          break;
        }
        default:
          ORT_NOT_IMPLEMENTED(__FUNCTION__, "data type is not supported");
          break;
      }
    }
  }

  int data_type() const {
    return data_type_;
  }

  int& data_type() {
    return data_type_;
  }

  const std::string& name() {
    return name_;
  }

  template <typename T>
  T* data() {
    if (!raw_data_.empty()) {
      return (T*)&raw_data_[0];
    }
      switch (data_type_) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
          return (T*)float16_data_.data();
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
          return (T*)float_data_.data();
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
          return (T*)double_data_.data();
          break;
        }
        default:
          break;
      }

    return nullptr;
  }

  template <typename T>
  const T* data() const {
    if (!raw_data_.empty()) {
      return (T*)&raw_data_[0];
    }
      switch (data_type_) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
          return (T*)float16_data_.data();
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
          return (T*)float_data_.data();
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
          return (T*)double_data_.data();
          break;
        }
        default:
          break;
      }

    return nullptr;
  }

  const std::vector<int64_t>& dims() const {
    return dims_;
  }

  const std::vector<int64_t>& dims() {
    return dims_;
  }

  int64_t size() const { return size_; }

  Initializer& add(float value) {
    int64_t n = size();
    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
        uint16_t* dst = data<uint16_t>();
        for (int i = 0; i < n; i++) {
          dst[i] = math::floatToHalf(math::halfToFloat(dst[i]) + value);
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        float* dst = data<float>();
        for (int i = 0; i < n; i++) {
          dst[i] += value;
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        double* dst = data<double>();
        for (int i = 0; i < n; i++) {
          dst[i] += value;
        }
        break;
      }
      default:
        break;
    }
    return *this;
  }

  Initializer& add(const Initializer& other) {
    int64_t n = size();
    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
        uint16_t* dst = data<uint16_t>();
        const uint16_t* src = other.data<uint16_t>();
        for (int i = 0; i < n; i++) {
          dst[i] = math::floatToHalf(math::halfToFloat(dst[i]) + math::halfToFloat(src[i]));
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        float* dst = data<float>();
        const float* src = other.data<float>();
        for (int i = 0; i < n; i++) {
          dst[i] += src[i];
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        double* dst = data<double>();
        const double* src = other.data<double>();
        for (int i = 0; i < n; i++) {
          dst[i] += src[i];
        }
        break;
      }
      default:
        break;
    }
    return *this;
  }
  Initializer& sub(const Initializer& other) {
    int64_t n = size();
    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
        uint16_t* dst = data<uint16_t>();
        const uint16_t* src = other.data<uint16_t>();
        for (int i = 0; i < n; i++) {
          dst[i] = math::floatToHalf(math::halfToFloat(dst[i]) - math::halfToFloat(src[i]));
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        float* dst = data<float>();
        const float* src = other.data<float>();
        for (int i = 0; i < n; i++) {
          dst[i] -= src[i];
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        double* dst = data<double>();
        const double* src = other.data<double>();
        for (int i = 0; i < n; i++) {
          dst[i] -= src[i];
        }
        break;
      }
      default:
        break;
    }
    return *this;
  }

  Initializer& mul(const Initializer& other) {
    int64_t n = size();
    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
        uint16_t* dst = data<uint16_t>();
        const uint16_t* src = other.data<uint16_t>();
        for (int i = 0; i < n; i++) {
          dst[i] = math::floatToHalf(math::halfToFloat(dst[i]) * math::halfToFloat(src[i]));
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        float* dst = data<float>();
        const float* src = other.data<float>();
        for (int i = 0; i < n; i++) {
          dst[i] *= src[i];
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        double* dst = data<double>();
        const double* src = other.data<double>();
        for (int i = 0; i < n; i++) {
          dst[i] *= src[i];
        }
        break;
      }
      default:
        break;
    }
    return *this;
  }
  Initializer& div(const Initializer& other) {
    int64_t n = size();
    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
        uint16_t* dst = data<uint16_t>();
        const uint16_t* src = other.data<uint16_t>();
        for (int i = 0; i < n; i++) {
          dst[i] = math::floatToHalf(math::halfToFloat(dst[i]) / math::halfToFloat(src[i]));
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        float* dst = data<float>();
        const float* src = other.data<float>();
        for (int i = 0; i < n; i++) {
          dst[i] /= src[i];
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        double* dst = data<double>();
        const double* src = other.data<double>();
        for (int i = 0; i < n; i++) {
          dst[i] /= src[i];
        }
        break;
      }
      default:
        break;
    }
    return *this;
  }

  Initializer& sqrt() {
    int64_t n = size();
    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
        uint16_t* dst = data<uint16_t>();
        for (int i = 0; i < n; i++) {
          dst[i] = math::floatToHalf(std::sqrt(math::halfToFloat(dst[i])));
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        float* dst = data<float>();
        for (int i = 0; i < n; i++) {
          dst[i] = std::sqrt(dst[i]);
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        double* dst = data<double>();
        for (int i = 0; i < n; i++) {
          dst[i] = std::sqrt(dst[i]);
        }
        break;
      }
      default:
        break;
    }
    return *this;
  }

  inline void scale_by_axis(const Initializer& other, int axis) {
    int64_t num = 1;
    for (size_t k = axis; k < dims_.size(); k++) {
      num *= dims_[k];
    }

    int64_t n = size() / num;
    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
        uint16_t* dst = data<uint16_t>();
        const uint16_t* src = other.data<uint16_t>();
        for (int i = 0; i < n; i++) {
          int index = other.size() == 1 ? 0 : i;
          for (int64_t j = 0; j < num; j++) {
            auto k = i * num + j;
            dst[k] = math::floatToHalf(math::halfToFloat(dst[k]) * math::halfToFloat(src[index]));
          }
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        float* dst = data<float>();
        const float* src = other.data<float>();
        for (int i = 0; i < n; i++) {
          int index = other.size() == 1 ? 0 : i;
          for (int64_t j = 0; j < num; j++) {
            dst[i * num + j] *= src[index];
          }
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        double* dst = data<double>();
        const double* src = other.data<double>();
        for (int i = 0; i < n; i++) {
          int index = other.size() == 1 ? 0 : i;
          for (int64_t j = 0; j < num; j++) {
            dst[i * num + j] *= src[index];
          }
        }
        break;
      }
      default:
        break;
    }
  }

 private:
  int data_type_;
  std::string name_;
  std::vector<int64_t> dims_;
  int64_t size_;

  std::string raw_data_;
  std::vector<float> float_data_;
  std::vector<uint16_t> float16_data_;
  std::vector<double> double_data_;
};

}  // namespace onnxruntime
