// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/autopad_type.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

enum class PoolType : uint8_t {
  kMaxPool,
  kAveragePool,
  kLpPool
};

class LpPool;

class PoolProcessContext {
 private:
  int64_t p_;

 public:
  friend class LpPool;
  PoolProcessContext() {}
  void init(const OpKernelInfo& info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("p", &p_).IsOK());
  }
};

class AveragePool {
 public:
  static float Initialize() {
    return 0.0;
  }

  template <typename T>
  static void Process(const T& x_data, T& y_data, const PoolProcessContext& /*cxt*/) {
    y_data += x_data;
  }

  template <typename T>
  static void Finalize(const int64_t size, T& y_data, const PoolProcessContext& /*cxt*/) {
    y_data /= size;
  }

  static const PoolType type = PoolType::kAveragePool;
};

template <int VERSION>
class MaxPool;

template <>
class MaxPool<1 /*VERSION*/> {
 public:
  static float Initialize() {
    return std::numeric_limits<float>::lowest();
  }

  template <typename T>
  static void Process(const T& x_data, T& y_data, const PoolProcessContext& /*cxt*/) {
    if (x_data > y_data) {
      y_data = x_data;
    }
  }

  template <typename T>
  static void Finalize(const int64_t /*size*/, T& /*y_data*/, const PoolProcessContext& /*cxt*/) {}

  static const PoolType type = PoolType::kMaxPool;
};

template <>
class MaxPool<8 /*VERSION*/> {
 public:
  static const PoolType type = PoolType::kMaxPool;
};

class LpPool {
 public:
  static float Initialize() {
    return 0.0f;
  }

  template <typename T>
  static void Process(const T& x_data, T& y_data, const PoolProcessContext& cxt) {
    y_data += static_cast<T>(std::pow(std::abs(x_data), cxt.p_));
  }

  template <typename T>
  static void Finalize(const int64_t /*size*/, T& y_data, const PoolProcessContext& cxt) {
    y_data = static_cast<T>(std::pow(y_data, 1.0f / cxt.p_));
  }
  static const PoolType type = PoolType::kLpPool;
};

class PoolBase {
 protected:
  PoolBase(const OpKernelInfo& info) {
    op_name_ = info.GetKernelDef().OpName();
    global_pooling_ = (op_name_ == "GlobalAveragePool" || op_name_ == "GlobalMaxPool" || op_name_ == "GlobalLpPool");

    if (!global_pooling_) {
      ORT_ENFORCE(info.GetAttrs<int64_t>("kernel_shape", kernel_shape_).IsOK(),
                  "No kernel shape is set.");

      std::string auto_padding;
      ORT_ENFORCE(info.GetAttr<std::string>("auto_pad", &auto_padding).IsOK());
      auto_pad_ = StringToAutoPadType(auto_padding);

      if (!info.GetAttrs<int64_t>("pads", pads_).IsOK() || pads_.empty()) {
        pads_.resize(kernel_shape_.size() * 2, 0);
      }

      if (!info.GetAttrs<int64_t>("strides", strides_).IsOK() || strides_.empty()) {
        strides_.resize(kernel_shape_.size(), 1);
      }

      if (!info.GetAttr<int64_t>("ceil_mode", &ceil_mode_).IsOK()) {
        ceil_mode_ = 0;
      }

      if (!info.GetAttrs<int64_t>("dilations", dilations_).IsOK() || dilations_.empty()) {
        dilations_.resize(kernel_shape_.size(), 1);
      }

      if (op_name_ == "AveragePool") {
        int64_t temp;
        ORT_ENFORCE(info.GetAttr<int64_t>("count_include_pad", &temp).IsOK());
        count_include_pad_ = (temp != 0);
      }

      if (op_name_ == "MaxPool") {
        int start, end;
        info.GetKernelDef().SinceVersion(&start, &end);
        if (start == 8) {
          storage_order_ = info.GetAttrOrDefault<int64_t>("storage_order", 0 /*default_value*/);
        }
      }

      for (size_t dim = 0; dim < kernel_shape_.size(); ++dim) {
        ORT_ENFORCE(kernel_shape_[dim] > 0);
        ORT_ENFORCE(pads_[dim] < kernel_shape_[dim] && pads_[dim + kernel_shape_.size()] < kernel_shape_[dim],
                    "Pad should be smaller than kernel.");
      }

      ORT_ENFORCE(strides_.size() == kernel_shape_.size());
      ORT_ENFORCE(dilations_.size() == kernel_shape_.size(),
                  "Dilations dimensions should match kernel shape");
    }
  }

  ~PoolBase(){};

  std::vector<int64_t> SetOutputSize(const TensorShape& input_shape,
                                     int64_t output_channel,
                                     std::vector<int64_t>* pads,
                                     const std::vector<int64_t>& dilations,
                                     int64_t ceil_mode) const {
    ORT_ENFORCE(input_shape.Size() > 0);
    std::vector<int64_t> output_dims;
    int64_t N = input_shape[0];
    InferOutputSize(input_shape.GetDims(), &output_dims, pads, dilations, ceil_mode);

    output_dims.insert(output_dims.begin(), {N, output_channel});

    return output_dims;
  }

  inline void InferOutputSize(const std::vector<int64_t>& input_dims,
                              std::vector<int64_t>* output_dims,
                              std::vector<int64_t>* pads,
                              const std::vector<int64_t>& dilations,
                              int64_t ceil_mode) const {
    ORT_ENFORCE(input_dims.size() >= 2);
    if (global_pooling_) {
      output_dims->assign(input_dims.size() - 2, 1);
    } else {
      for (size_t dim = 0; dim < input_dims.size() - 2; ++dim) {
        int64_t dim_size = 0;
        ComputeSizePadDilations(static_cast<int>(input_dims[dim + 2]),
                                strides_[dim],
                                kernel_shape_[dim],
                                &pads->at(dim),
                                &pads->at(input_dims.size() + dim - 2),
                                dilations[dim],
                                ceil_mode,
                                &dim_size);
        output_dims->push_back(dim_size);
      }
    }
  }

  inline void ComputeSizePadDilations(const int64_t in_size,
                                      const int64_t stride,
                                      const int64_t kernel,
                                      int64_t* pad_head,
                                      int64_t* pad_tail,
                                      int64_t dilation,
                                      int64_t ceil_mode,
                                      int64_t* out_size) const {
    if (auto_pad_ != AutoPadType::NOTSET) {
      switch (auto_pad_) {
        case AutoPadType::VALID:
          *pad_head = 0;
          *pad_tail = 0;
          *out_size = ComputeOutputSize(in_size, stride, kernel, 0, dilation, ceil_mode);
          break;
        case AutoPadType::SAME_LOWER: {
          int64_t legacy_target_size = (in_size + stride - 1) / stride;
          int64_t pad_needed = (legacy_target_size - 1) * stride + kernel - in_size;
          *pad_head = (pad_needed + 1) / 2;
          *pad_tail = pad_needed - *pad_head;
          *out_size = ComputeOutputSize(in_size, stride, kernel, pad_needed, dilation, ceil_mode);
          break;
        }
        case AutoPadType::SAME_UPPER: {
          int64_t legacy_target_size = (in_size + stride - 1) / stride;
          int64_t pad_needed = (legacy_target_size - 1) * stride + kernel - in_size;
          *pad_head = pad_needed / 2;
          *pad_tail = pad_needed - *pad_head;
          *out_size = ComputeOutputSize(in_size, stride, kernel, pad_needed, dilation, ceil_mode);
          break;
        }
        default: {
          ORT_THROW("Unsupported AutoPad Type.");
        }
      }
    } else {
      *out_size = ComputeOutputSize(in_size, stride, kernel, *pad_head + *pad_tail, dilation, ceil_mode);
    }
  }

  inline int64_t ComputeOutputSize(int64_t in_size,
                                   int64_t stride,
                                   int64_t kernel,
                                   int64_t pad_needed,
                                   int64_t dilation,
                                   int64_t ceil_mode) const {
    if (ceil_mode == 0) {
      return static_cast<int64_t>(static_cast<float>(in_size + pad_needed - dilation * (kernel - 1) - 1) / stride + 1);
    } else {
      return static_cast<int64_t>(ceil(static_cast<float>(in_size + pad_needed - dilation * (kernel - 1) - 1) / stride + 1));
    }
  }

  Status Compute(OpKernelContext* context, MLAS_POOLING_KIND kind) const;

 protected:
  std::string op_name_;
  bool global_pooling_{};
  bool count_include_pad_{};
  int64_t storage_order_{0};  // MaxPool_8 only. 0 is row major, and 1 is column major. Default is 0.
  int64_t ceil_mode_{0};      // Introduced in MaxPool_10
  std::vector<int64_t> kernel_shape_;
  std::vector<int64_t> pads_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> dilations_;  // Introduced in MaxPool_10

  AutoPadType auto_pad_;

  inline int64_t stride_h() const {
    return global_pooling_ ? 1 : strides_[0];
  }

  inline int64_t stride_w() const {
    return global_pooling_ ? 1 : strides_[1];
  }

  inline int64_t stride_d() const {
    return global_pooling_ ? 1 : strides_[2];
  }
};

}  // namespace onnxruntime
