// Copyright(C) 2018 Intel Corporation
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License

#ifdef _WIN32
#pragma warning(disable : 4244)
#endif

#include "core/providers/armnn/armnn_common.h"

namespace onnxruntime {
namespace armnn_ep {

armnn::TensorShape ArmNNTensorShape(const TensorShape& tensorShape, unsigned int extDim) {
  std::vector<unsigned int> dims;
  unsigned int inDim = tensorShape.NumDimensions();
  unsigned int outDim = (extDim > inDim) ? extDim : inDim;

  for (unsigned int i = 0; i < inDim; ++i)
	  dims.push_back(tensorShape.GetDims()[i]);

  // extend dimensions
  for (unsigned int i = 0; i < outDim - inDim; i++)
          dims.push_back(1);

  return armnn::TensorShape{static_cast<unsigned int>(dims.size()), dims.data()};
}

}  // namespace armnn_ep
}  // namespace onnxruntime
