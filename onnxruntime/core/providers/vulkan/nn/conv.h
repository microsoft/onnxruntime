// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/vulkan/vulkan_kernel.h"

namespace onnxruntime {
class Node;
class VulkanExecutionProvider;
namespace vulkan {

class ConvKernel : VulkanKernel {
 public:
  static bool IsSupported(const GraphViewer&, const onnxruntime::Node&, const logging::Logger&) {
    // TODO: check attribs/inputs against NCNN requirements.
    return true;
  }

  static std::unique_ptr<VulkanKernel> Create(const VulkanExecutionProvider& vulkan_ep,
                                              const onnxruntime::Node& node) {
    return std::unique_ptr<VulkanKernel>(new ConvKernel(vulkan_ep, node));
  }

  std::string_view GetNcnnLayerName() const override {
    switch (op_type_) {
      case Convolution:
        return "Convolution";
      case Convolution1D:
        return "Convolution1D";
      case Convolution3D:
        return "Convolution3D";
      case ConvolutionDepthWise:
        return "ConvolutionDepthWise";
      case ConvolutionDepthWise1D:
        return "ConvolutionDepthWise1D";
      case ConvolutionDepthWise3D:
        return "ConvolutionDepthWise3D";
      default:
        ORT_THROW("ConvType has not been initialized correctly: ", op_type_);
    }
  }

  Status SetupParamDict(const GraphViewer& graph_viewer, ncnn::ParamDict& params) override;

  Status SetupConstantInitializers(const GraphViewer& graph_viewer, ncnn::Layer& layer) override;

 private:
  ConvKernel(const VulkanExecutionProvider& vulkan_ep, const onnxruntime::Node& node)
      : VulkanKernel{vulkan_ep, node} {
    // TODO: set op_type_ based on node if possible. If we need a GraphViewer we can set it up in CreateNcnnKernel.
    // - Convolution
    // - ConvolutionDepthWise
    // - Convolution1D
  }

  enum ConvType {
    Unset,
    Convolution,
    Convolution1D,
    Convolution3D,
    ConvolutionDepthWise,
    ConvolutionDepthWise1D,
    ConvolutionDepthWise3D,
  };

  // set in CreateNcnnKernel where we have access to the GraphViewer. not needed externally until CreateNcnnKernel
  // calls VulkanKernel::SetupNcnnLayer, which calls GetNcnnLayerName.
  ConvType op_type_{Unset};

  enum ConvParams {
    /*
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    dilation_w = pd.get(2, 1);
    dilation_h = pd.get(12, dilation_w);
    stride_w = pd.get(3, 1);
    stride_h = pd.get(13, stride_w);
    pad_left = pd.get(4, 0);
    pad_right = pd.get(15, pad_left);
    pad_top = pd.get(14, pad_left);
    pad_bottom = pd.get(16, pad_top);
    pad_value = pd.get(18, 0.f);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    int8_scale_term = pd.get(8, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());

    dynamic_weight = pd.get(19, 0);
    */
  };

  enum Conv1DParams {
    /*
        num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    dilation_w = pd.get(2, 1);
    stride_w = pd.get(3, 1);
    pad_left = pd.get(4, 0);
    pad_right = pd.get(15, pad_left);
    pad_value = pd.get(18, 0.f);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());

    dynamic_weight = pd.get(19, 0);
*/
  };

  enum ConvDepthwiseParams {
    /*
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    dilation_w = pd.get(2, 1);
    dilation_h = pd.get(12, dilation_w);
    stride_w = pd.get(3, 1);
    stride_h = pd.get(13, stride_w);
    pad_left = pd.get(4, 0);
    pad_right = pd.get(15, pad_left);
    pad_top = pd.get(14, pad_left);
    pad_bottom = pd.get(16, pad_top);
    pad_value = pd.get(18, 0.f);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    group = pd.get(7, 1);
    int8_scale_term = pd.get(8, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());

    dynamic_weight = pd.get(19, 0);
    */
  };
};

}  // namespace vulkan
}  // namespace onnxruntime
