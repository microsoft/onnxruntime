// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/nn/conv.h"
#include "include/ncnn/layer/convolution.h"
#include "include/ncnn/layer/convolution1d.h"
#include "include/ncnn/layer/convolution3d.h"
#include "include/ncnn/layer/convolutiondepthwise.h"
#include "include/ncnn/layer/convolutiondepthwise1d.h"
#include "include/ncnn/layer/convolutiondepthwise3d.h"

namespace onnxruntime {
namespace vulkan {
Status ConvKernel::SetupParamDict(const GraphViewer& /*graph_viewer*/, ncnn::ParamDict& /*params*/) {
  // const auto& node = Node();
  // const auto& input_defs = node.InputDefs();

  // figure out op_type

  // populate params based on op_type

  return Status::OK();
}

Status ConvKernel::SetupConstantInitializers(const GraphViewer& /*graph_viewer*/, ncnn::Layer& /*layer*/) {
  // update the ncnn::Mat values in the base NCNN layer for any constant initializers
  switch (op_type_) {
    case ConvType::Convolution: {
      // auto& conv_layer = static_cast<ncnn::Convolution&>(layer);
      // Populate these fields
      //
      // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid  TODO: This requires fusion with the following node
      // int activation_type;
      // Mat activation_params;
      //
      // int dynamic_weight;
      // Mat weight_data;
      // Mat bias_data;
      break;
    }
    case ConvType::Convolution1D: {
      // auto& conv_layer = static_cast<ncnn::Convolution1D&>(layer);
      break;
    }
    case ConvType::Convolution3D: {
      // auto& conv_layer = static_cast<ncnn::Convolution3D&>(layer);
      break;
    }
    case ConvType::ConvolutionDepthWise: {
      // auto& conv_layer = static_cast<ncnn::ConvolutionDepthWise&>(layer);
      break;
    }
    case ConvType::ConvolutionDepthWise1D: {
      // auto& conv_layer = static_cast<ncnn::ConvolutionDepthWise1D&>(layer);
      break;
    }
    case ConvType::ConvolutionDepthWise3D: {
      // auto& conv_layer = static_cast<ncnn::ConvolutionDepthWise3D&>(layer);
      break;
    }
    default:
      ORT_THROW("ConvType has not been initialized correctly: ", op_type_);
  }
  return Status::OK();
}

}  // namespace vulkan
}  // namespace onnxruntime
