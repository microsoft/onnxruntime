// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlConv {
 public:
  enum InputTensors : int {
    IN_X = 0,
    IN_W = 1,
    IN_B = 2
  };

  enum OutputTensors : int {
    OUT_Y = 0
  };

  enum ConvShape : size_t {
    SHAPE_UNKNOWN = 0,
    SHAPE_1D = 1,
    SHAPE_2D = 2,
    SHAPE_3D = 3
  };

  DnnlConv();
  void CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node);

 private:
  /*
   * Return the infered padding.
   *
   * The padding will be based on the specified padding or will infered based on the
   * Onnx 'auto_pad' attributes.
   *
   * This will return the padding in the format specified in the Onnx specification.
   * > Format should be as follows [x1_begin, x2_begin...x1_end, x2_end,...],
   * > where xi_begin the number of pixels added at the beginning of axis `i`
   * > and xi_end, the number of pixels added at the end of axis `i`.
   */
  std::vector<int64_t> GetInferedPads(DnnlNode& node,
                                      const dnnl::memory::dims& src_dims,
                                      const dnnl::memory::dims& dilations,
                                      const std::vector<int64_t>& kernel_shape,
                                      const dnnl::memory::dims& strides);
  /* Get the padding left values from the infered pads */
  dnnl::memory::dims GetPaddingLeft(const std::vector<int64_t>& onnx_padding, ConvShape shape);
  /* Get the padding right values from the infered pads */
  dnnl::memory::dims GetPaddingRight(const std::vector<int64_t>& onnx_padding, ConvShape shape);

  /*
   * Collection of functions to get OnnxRuntime attributes. Note, if the attribute is used directly by
   * OneDNN the return type is converted to the format expected by OneDNN not the type expected by
   * OnnxRuntime. Typically this means returning `dnnl::memory::dims` instead of `vector<int64_t>`.
   */
  /* Get the 'auto_pad' attribute */
  AutoPadType GetAutoPad(DnnlNode& node);

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

  /* Get the 'group' attributes */
  int64_t GetGroup(DnnlNode& node);

  /* Get the 'kernel_shape' attribute */
  std::vector<int64_t> GetKernelShape(DnnlNode& node);

  /* Get the 'pads' attribute */
  std::vector<int64_t> GetPads(DnnlNode& node);

  /* Get the 'strides' attribute */
  dnnl::memory::dims GetStrides(DnnlNode& node, ConvShape shape);

  /*
   * ComputePad is copy/paste of a the ComputePad found in core/providers/common.h
   * With some minor modifications. i.e. return bool instead of status.
   * ComputePad is not exposed to the shared library so this copy is used instead.
   *
   * Returns true if pads successfully computed.
   */
  bool ComputePad(const int64_t in_dim,
                  const int64_t stride,
                  const int64_t kernel,
                  const int64_t dilation,
                  AutoPadType pad_type,
                  int64_t& pad_head, /* output param */
                  int64_t& pad_tail, /* output param */
                  bool force_symmetric_auto_padding = false);

  /*
   * Use input shapes and attributes to figure out the output shape that will
   * result from the convolution.
   */
  dnnl::memory::dims InferOutputShape(DnnlNode& node,
                                      const dnnl::memory::dims& x_shape,
                                      const dnnl::memory::dims& w_shape,
                                      const std::vector<int64_t>& kernel_shape,
                                      const dnnl::memory::dims& strides,
                                      const dnnl::memory::dims& dilations,
                                      const std::vector<int64_t>& pads);
};

}  // namespace ort_dnnl
}  // namespace onnxruntime