// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

// class DnnlSubgraphPrimitive;
// class DnnlNode;

namespace onnxruntime {
namespace ort_dnnl {

class DnnlConvGrad {
 public:
  enum InputTensors : int {
    IN_DY = 0,
    IN_X = 1,
    IN_W = 2
  };

  enum OutputTensors : int {
    OUT_DX = 0,
    OUT_DW = 1,
    OUT_DB = 2
  };

  enum ConvShape : size_t {
    SHAPE_UNKNOWN = 0,
    SHAPE_1D = 1,
    SHAPE_2D = 2,
    SHAPE_3D = 3
  };

  DnnlConvGrad();
  void CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node);

 private:
  std::vector<int64_t> GetKernelShape(DnnlNode& node);
  /* Get the 'pads' attribute */
  dnnl::memory::dims GetPads(DnnlNode& node, ConvShape shape);
  /* Get the padding left values from the infered pads */
  dnnl::memory::dims GetPaddingLeft(const std::vector<int64_t>& onnx_padding, ConvShape shape);
  /* Get the padding right values from the infered pads */
  dnnl::memory::dims GetPaddingRight(const std::vector<int64_t>& onnx_padding, ConvShape shape);
  /*
   * Get the 'dilations' attribute.
   *  Note dilations in OneDNN and Onnx differ:
   *    - For Onnx a non-dilated kernel would be all 1s
   *    - For OneDNN a non-dilated kernel would be all 0s
   *
   * The memory dimentions returned is in the form expected for OneDNN each dilation dimention
   * will be 1 less than the dilated dimention expected by Onnx specification. Be aware of this
   * fact as 'dilations' are used in any calcuations since this could result in an off-by-one
   * error.
   */
  dnnl::memory::dims GetDilations(DnnlNode& node, ConvShape shape);
  /* Get the 'strides' attribute */
  dnnl::memory::dims GetStrides(DnnlNode& node, ConvShape shape);
  /* Get the 'group' attributes */
  int64_t GetGroup(DnnlNode& node);
};

}  // namespace ort_dnnl
}  // namespace onnxruntime