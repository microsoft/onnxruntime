// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "../telum_common.h"
#include "../utils/tensor_converter.h"

namespace onnxruntime {
namespace telum {

/**
 * @brief Base class for all Telum kernels
 *
 * Provides common functionality for converting between ORT and zDNN tensors,
 * error handling, and resource management.
 */
class TelumKernel : public OpKernel {
 public:
  explicit TelumKernel(const OpKernelInfo& info) : OpKernel(info) {}

  virtual ~TelumKernel() = default;

 protected:
  /**
   * @brief Convert ORT tensor to zDNN ztensor
   *
   * @param ort_tensor Input ORT tensor
   * @param ztensor Output zDNN ztensor
   * @param layout Desired zDNN data layout
   * @return Status indicating success or failure
   */
  Status ConvertToZTensor(const Tensor& ort_tensor,
                         zdnn_ztensor& ztensor,
                         zdnn_data_layouts layout) const {
    return TensorConverter::ConvertToZTensor(ort_tensor, ztensor, layout);
  }

  /**
   * @brief Convert zDNN ztensor back to ORT tensor
   *
   * @param ztensor Input zDNN ztensor
   * @param ort_tensor Output ORT tensor
   * @return Status indicating success or failure
   */
  Status ConvertFromZTensor(const zdnn_ztensor& ztensor,
                           Tensor& ort_tensor) const {
    return TensorConverter::ConvertFromZTensor(ztensor, ort_tensor);
  }

  /**
   * @brief Initialize zDNN ztensor for output
   *
   * @param ort_tensor ORT tensor to base initialization on
   * @param ztensor Output zDNN ztensor to initialize
   * @param layout Desired zDNN data layout
   * @return Status indicating success or failure
   */
  Status InitZTensorForOutput(const Tensor& ort_tensor,
                              zdnn_ztensor& ztensor,
                              zdnn_data_layouts layout) const {
    return TensorConverter::InitZTensorForOutput(ort_tensor, ztensor, layout);
  }

  /**
   * @brief Validate zDNN status and convert to ORT status
   *
   * @param status zDNN status code
   * @param operation Name of the operation for error reporting
   * @return ORT Status
   */
  Status CheckStatus(zdnn_status status, const char* operation) const {
    return CheckZDNNStatus(status, operation);
  }

  /**
   * @brief RAII wrapper for zDNN ztensor to ensure cleanup
   */
  class ZTensorGuard {
   public:
    explicit ZTensorGuard(zdnn_ztensor* ztensor) : ztensor_(ztensor) {}

    ~ZTensorGuard() {
      if (ztensor_ != nullptr) {
        if (ztensor_->buffer != nullptr) {
          zdnn_free_ztensor_buffer(ztensor_);
          ztensor_->buffer = nullptr;
        }
        // TensorConverter allocates the descriptors on the heap. Clean them up here so callers
        // can treat a zdnn_ztensor as an RAII-managed object via this guard.
        delete ztensor_->pre_transformed_desc;
        delete ztensor_->transformed_desc;
        ztensor_->pre_transformed_desc = nullptr;
        ztensor_->transformed_desc = nullptr;
      }
    }

    // Disable copy
    ZTensorGuard(const ZTensorGuard&) = delete;
    ZTensorGuard& operator=(const ZTensorGuard&) = delete;

    // Enable move
    ZTensorGuard(ZTensorGuard&& other) noexcept : ztensor_(other.ztensor_) {
      other.ztensor_ = nullptr;
    }

    ZTensorGuard& operator=(ZTensorGuard&& other) noexcept {
      if (this != &other) {
        if (ztensor_ != nullptr) {
          if (ztensor_->buffer != nullptr) {
            zdnn_free_ztensor_buffer(ztensor_);
            ztensor_->buffer = nullptr;
          }
          delete ztensor_->pre_transformed_desc;
          delete ztensor_->transformed_desc;
          ztensor_->pre_transformed_desc = nullptr;
          ztensor_->transformed_desc = nullptr;
        }
        ztensor_ = other.ztensor_;
        other.ztensor_ = nullptr;
      }
      return *this;
    }

   private:
    zdnn_ztensor* ztensor_;
  };
};

}  // namespace telum
}  // namespace onnxruntime

// Made with Bob
