// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <sstream>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/providers/snpe/snpe_lib.h"
#include "core/providers/snpe/snpe_execution_provider.h"

namespace onnxruntime {
namespace contrib {
namespace snpe {

class SnpeKernel : public OpKernel {
 public:
  explicit SnpeKernel(const OpKernelInfo& info) : OpKernel(info) {
    input_count_ = info.GetInputCount();
    output_count_ = info.GetOutputCount();
    std::vector<int64_t> input_sizes(input_count_);
    std::vector<std::string> output_names;
    output_dims_.resize(output_count_);
    for (uint32_t output_i = 0; output_i < output_count_; ++output_i) {
      auto output = info.node().OutputDefs().at(output_i);
      auto output_shape = output->Shape();
      for (int i = 0; i < output_shape->dim_size(); ++i) {
        output_dims_.at(output_i).push_back(output_shape->dim(i).dim_value());
      }
      output_names.push_back(output->Name());
      output_names_index_.emplace(output->Name(), static_cast<size_t>(output_i));
    }
    for (uint32_t input_i = 0; input_i < input_count_; ++input_i) {
      auto input = info.node().InputDefs().at(input_i);
      input_names_.push_back(input->Name());
      auto input_shape = input->Shape();
      int64_t input_size = 1;
      for (int j = 0; j < input_shape->dim_size(); ++j) {
        input_size *= input_shape->dim(j).dim_value();
      }
      input_sizes.at(input_i) = input_size;
    }

    const auto dlc_payload = info.GetAttrOrDefault<std::string>("DLC", "");
    ORT_ENFORCE((dlc_payload.length() > 0), "dlc model payload is empty!");
    const auto snpe_ep = static_cast<const SNPEExecutionProvider*>(info.GetExecutionProvider());
    snpe_rt_ = SnpeLibFactory(reinterpret_cast<const unsigned char*>(dlc_payload.c_str()),
                              dlc_payload.length(),
                              snpe_ep->GetRuntimeOptions(),
                              output_names,
                              input_names_,
                              input_sizes,
                              buffer_type_);
  }

  Status Compute(OpKernelContext* context) const override {
    std::vector<const unsigned char*> input_data_array;
    std::vector<size_t> input_size_array;
    for (uint32_t input_i = 0; input_i < input_count_; ++input_i) {
      const Tensor* input_tensor = context->Input<Tensor>(input_i);
      const auto input_data = input_tensor->DataRaw();
      const size_t input_size = input_tensor->Shape().Size();
      const size_t input_element_byte_size = input_tensor->DataType()->Size();
      input_data_array.push_back(static_cast<const unsigned char*>(input_data));
      input_size_array.push_back(input_size * input_element_byte_size);
    }

    std::vector<unsigned char*> output_data_array;
    std::vector<size_t> output_size_array;
    for (uint32_t output_i = 0; output_i < output_count_; ++output_i) {
      TensorShape output_shape = TensorShape(output_dims_.at(output_i));
      auto output_tensor = context->Output(output_i, output_shape);
      auto output_data = output_tensor->MutableDataRaw();
      const size_t output_size = output_tensor->Shape().Size();
      const size_t output_element_byte_size = output_tensor->DataType()->Size();
      output_data_array.push_back(static_cast<unsigned char*>(output_data));
      output_size_array.push_back(output_size * output_element_byte_size);
    }

    if (buffer_type_ != BufferType::ITENSOR && buffer_type_ != BufferType::UNKNOWN) {
      // process with user buffer
      ORT_RETURN_IF_ERROR(snpe_rt_->SnpeProcessWithUserBuffer(input_names_,
                                                              input_data_array.data(),
                                                              input_count_,
                                                              output_data_array.data(),
                                                              output_names_index_));
    } else if (input_count_ == 1 && output_count_ == 1) {
      ORT_RETURN_IF_ERROR(snpe_rt_->SnpeProcess(input_data_array.at(0),
                                                input_size_array.at(0),
                                                output_data_array.at(0),
                                                output_size_array.at(0),
                                                output_names_index_));
    } else if (input_count_ == 1 && output_count_ > 1) {
      ORT_RETURN_IF_ERROR(snpe_rt_->SnpeProcessMultipleOutput(input_data_array.at(0),
                                                              input_size_array.at(0),
                                                              output_count_,
                                                              output_data_array.data(),
                                                              output_size_array.data(),
                                                              output_names_index_));
    } else if (input_count_ > 1 && output_count_ >= 1) {
      ORT_RETURN_IF_ERROR(snpe_rt_->SnpeProcessMultiInputsMultiOutputs(input_data_array.data(),
                                                                       input_size_array.data(),
                                                                       input_count_,
                                                                       output_data_array.data(),
                                                                       output_size_array.data(),
                                                                       output_count_,
                                                                       output_names_index_));
    }

    return Status::OK();
  }

 private:
  std::vector<std::vector<int64_t>> output_dims_;
  std::unique_ptr<SnpeLib> snpe_rt_;
  uint32_t input_count_;
  uint32_t output_count_;
  std::unordered_map<std::string, size_t> output_names_index_;
  std::vector<std::string> input_names_;
  BufferType buffer_type_;
};
}  // namespace snpe
}  // namespace contrib
}  // namespace onnxruntime
