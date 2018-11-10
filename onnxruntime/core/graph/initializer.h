// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <vector>
#include <cmath>

#include "core/common/common.h"
#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {

class Initializer final {
 public:
  static bool IsSupportedDataType(const ONNX_NAMESPACE::TensorProto* tensor_proto) {
    if (tensor_proto == nullptr ||
        (tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
         tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_DOUBLE)) {
      return false;
    }
    return true;
  }

  Initializer(const ONNX_NAMESPACE::TensorProto* tensor_proto) : size_(0),
                                                                 data_(nullptr),
                                                                 is_raw_data_(false) {
    data_type_ = tensor_proto->data_type();
    dims_.reserve(tensor_proto->dims_size());
    for (int i = 0; i < tensor_proto->dims_size(); i++) {
      dims_.push_back(tensor_proto->dims(i));
    }

    size_ = std::accumulate(dims_.begin(), dims_.end(), (int64_t)1, std::multiplies<int64_t>{});

    if (tensor_proto->has_raw_data()) {
      is_raw_data_ = true;
      raw_data_ = std::move(tensor_proto->raw_data());
    } else {
      switch (data_type_) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
          int64_t size = tensor_proto->float_data_size();
          ONNXRUNTIME_ENFORCE(size_ == size, "size is different");
          float* float_data = static_cast<float*>(malloc(sizeof(float) * size_));
          ONNXRUNTIME_ENFORCE(float_data != nullptr, "failed to allocate memory");
          for (int i = 0; i < size_; i++) {
            float_data[i] = tensor_proto->float_data(i);
          }
          data_ = float_data;
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
          int64_t size = tensor_proto->double_data_size();
          ONNXRUNTIME_ENFORCE(size_ == size, "size is different");
          double* double_data = static_cast<double*>(malloc(sizeof(double) * size_));
          ONNXRUNTIME_ENFORCE(double_data != nullptr, "failed to allocate memory");
          for (int i = 0; i < size_; i++) {
            double_data[i] = tensor_proto->double_data(i);
          }
          data_ = double_data;
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
        case ONNX_NAMESPACE::TensorProto_DataType_INT8:
        case ONNX_NAMESPACE::TensorProto_DataType_INT16:
        case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
        case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
          int64_t size = tensor_proto->int32_data_size();
          ONNXRUNTIME_ENFORCE(size_ == size, "size is different");
          int32_t* int32_data = static_cast<int32_t*>(malloc(sizeof(int32_t) * size_));
          ONNXRUNTIME_ENFORCE(int32_data != nullptr, "failed to allocate memory");
          for (int i = 0; i < size_; i++) {
            int32_data[i] = tensor_proto->int32_data(i);
          }
          data_ = int32_data;
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
          int64_t size = tensor_proto->int64_data_size();
          ONNXRUNTIME_ENFORCE(size_ == size, "size is different");
          int64_t* int64_data = static_cast<int64_t*>(malloc(sizeof(int64_t) * size_));
          ONNXRUNTIME_ENFORCE(int64_data != nullptr, "failed to allocate memory");
          for (int i = 0; i < size_; i++) {
            int64_data[i] = tensor_proto->int64_data(i);
          }
          data_ = int64_data;
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
        case ONNX_NAMESPACE::TensorProto_DataType_UINT64: {
          int64_t size = tensor_proto->uint64_data_size();
          ONNXRUNTIME_ENFORCE(size_ == size, "size is different");
          uint64_t* uint64_data = static_cast<uint64_t*>(malloc(sizeof(uint64_t) * size_));
          ONNXRUNTIME_ENFORCE(uint64_data != nullptr, "failed to allocate memory");
          for (int i = 0; i < size_; i++) {
            uint64_data[i] = tensor_proto->uint64_data(i);
          }
          data_ = uint64_data;
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
        case ONNX_NAMESPACE::TensorProto_DataType_STRING:
        case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
        case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64:
        case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128:
        case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
          ONNXRUNTIME_NOT_IMPLEMENTED(__FUNCTION__, "data type is not supported");
          break;
      }
    }
  }

  ~Initializer() { free(data_); }

  void ToProto(ONNX_NAMESPACE::TensorProto* tensor_proto) {
    // update data type
    tensor_proto->set_data_type(data_type_);

    if (is_raw_data_) {
      tensor_proto->clear_raw_data();
      tensor_proto->set_raw_data(raw_data_);
    } else {
      switch (data_type_) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
          float* float_data = static_cast<float*>(data_);
          tensor_proto->clear_float_data();
          for (int i = 0; i < size_; i++) {
            tensor_proto->add_float_data(float_data[i]);
          }
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
          double* double_data = static_cast<double*>(data_);
          tensor_proto->clear_double_data();
          for (int i = 0; i < size_; i++) {
            tensor_proto->add_double_data(double_data[i]);
          }
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
        case ONNX_NAMESPACE::TensorProto_DataType_INT8:
        case ONNX_NAMESPACE::TensorProto_DataType_INT16:
        case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
        case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
          int32_t* int32_data = static_cast<int32_t*>(data_);
          tensor_proto->clear_int32_data();
          for (int i = 0; i < size_; i++) {
            tensor_proto->add_int32_data(int32_data[i]);
          }
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
          int64_t* int64_data = static_cast<int64_t*>(data_);
          tensor_proto->clear_int64_data();
          for (int i = 0; i < size_; i++) {
            tensor_proto->add_int64_data(int64_data[i]);
          }
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
        case ONNX_NAMESPACE::TensorProto_DataType_UINT64: {
          uint64_t* uint64_data = static_cast<uint64_t*>(data_);
          tensor_proto->clear_uint64_data();
          for (int i = 0; i < size_; i++) {
            tensor_proto->add_uint64_data(uint64_data[i]);
          }
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
        case ONNX_NAMESPACE::TensorProto_DataType_STRING:
        case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
        case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64:
        case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128:
        case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
          break;
      }
    }
  }

  ONNX_NAMESPACE::TensorProto_DataType data_type() const {
    return data_type_;
  }

  ONNX_NAMESPACE::TensorProto_DataType& data_type() {
    return data_type_;
  }

  template <typename T>
  T* data() {
    if (is_raw_data_) {
      return (T*)&raw_data_.data()[0];
    } else {
      return (T*)data_;
    }
  }

  template <typename T>
  const T* data() const {
    if (is_raw_data_) {
      return (T*)&raw_data_.data()[0];
    } else {
      return (T*)data_;
    }
  }

  std::vector<int64_t> dims() const {
    return dims_;
  }

  std::vector<int64_t>& dims() {
    return dims_;
  }

  size_t size() const { return size_; }

  Initializer& add(float value) {
    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        for (int i = 0; i < size_; i++) {
          data<float>()[i] += value;
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        for (int i = 0; i < size_; i++) {
          data<double>()[i] += value;
        }
        break;
      }
      default:
        break;
    }
    return *this;
  }

  Initializer& add(const Initializer& other) {
    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        for (int i = 0; i < size_; i++) {
          data<float>()[i] += other.data<float>()[i];
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        for (int i = 0; i < size_; i++) {
          data<double>()[i] += other.data<double>()[i];
        }
        break;
      }
      default:
        break;
    }
    return *this;
  }
  Initializer& sub(const Initializer& other) {
    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        for (int i = 0; i < size_; i++) {
          data<float>()[i] -= other.data<float>()[i];
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        for (int i = 0; i < size_; i++) {
          data<double>()[i] -= other.data<double>()[i];
        }
        break;
      }
      default:
        break;
    }
    return *this;
  }

  Initializer& mul(const Initializer& other) {
    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        for (int i = 0; i < size_; i++) {
          data<float>()[i] *= other.data<float>()[i];
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        for (int i = 0; i < size_; i++) {
          data<double>()[i] *= other.data<double>()[i];
        }
        break;
      }
      default:
        break;
    }
    return *this;
  }
  Initializer& div(const Initializer& other) {
    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        for (int i = 0; i < size_; i++) {
          data<float>()[i] /= other.data<float>()[i];
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        for (int i = 0; i < size_; i++) {
          data<double>()[i] /= other.data<double>()[i];
        }
        break;
      }
      default:
        break;
    }
    return *this;
  }

  Initializer& sqrt() {
    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        for (int i = 0; i < size_; i++) {
          data<float>()[i] = std::sqrt(data<float>()[i]);
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        for (int i = 0; i < size_; i++) {
          data<double>()[i] = std::sqrt(data<double>()[i]);
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

    switch (data_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        for (int64_t i = 0; i < dims_[0]; i++) {
          for (int64_t j = 0; j < num; j++) {
            data<float>()[i * num + j] *= other.data<float>()[i];
          }
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        for (int64_t i = 0; i < dims_[0]; i++) {
          for (int64_t j = 0; j < num; j++) {
            data<double>()[i * num + j] *= other.data<double>()[i];
          }
        }
        break;
      }
      default:
        break;
    }
  }

 private:
  ONNX_NAMESPACE::TensorProto_DataType data_type_;
  std::vector<int64_t> dims_;
  int64_t size_;
  void* data_;

  bool is_raw_data_;
  std::string raw_data_;
};

}  // namespace onnxruntime
