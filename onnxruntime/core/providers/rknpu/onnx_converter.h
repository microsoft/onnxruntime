// Copyright 2020 rock-chips.com Inc.

#pragma once

#include <onnx/onnx_pb.h>

#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>
#include <string>

#include "shaper.h"

#include "rknpu/rknpu_pub.h"

namespace onnxruntime {
namespace rknpu {

/**
 *  For convert from onnx::ModelProto to rk::nn::Graph.
 */
class OnnxConverter {
 public:
  OnnxConverter() {}
  ~OnnxConverter() { Clear(); }

  /** Get the supported subgraph.
  */
  std::vector<std::vector<int>> GetSupportedNodes(
      const ONNX_NAMESPACE::ModelProto& model_proto);

  /** Convert onnx::ModelProto to rk::nn::Graph.
   *  Because some attributes of rk::nn::Tensor are used as input in onnx and these attributes
   *  can't be found in onnx::ModelProto, so additional input-related information is required.
  */
  void Convert(const ONNX_NAMESPACE::ModelProto& model,
               rk::nn::Graph* graph,
               const std::vector<const void*>& input_bufs,
               const std::unordered_map<std::string, int>& input_maps);

  std::string m(const std::string& str) const;

 private:
  Shaper shaper_;

  enum class FuseCode { FUSED_NONE,
                        FUSED_RELU,
                        FUSED_RELU1,
                        FUSED_RELU6 };

  std::map<std::string, std::string> name_map_;

  ONNX_NAMESPACE::ModelProto model_proto_;
  rk::nn::Graph* graph_;
  std::vector<int> skipped_act_;
  std::vector<std::string> dequantize_after_;

  std::vector<std::string> operands_;
  std::map<std::string, std::shared_ptr<rk::nn::Tensor>> rk_tensors_;

  // for GetSupportedNodes
  std::map<std::string, std::vector<uint32_t>> tensor_dims_;

  std::vector<void*> free_list_;  // remember free

  std::pair<std::pair<int, ONNX_NAMESPACE::NodeProto>, FuseCode>
  FindActivation(const ONNX_NAMESPACE::ModelProto& model_proto,
                 const std::string& output);

  std::shared_ptr<rk::nn::Tensor>
  CreateRknnTensor(const std::string& name,
                   const std::vector<uint32_t>& dims,
                   const void* data = NULL,
                   const rk::nn::TensorRole role = rk::nn::TensorRole::VAR,
                   const rk::nn::PrecisionType precision = rk::nn::PrecisionType::FLOAT32,
                   const rk::nn::DataLayoutType layout = rk::nn::DataLayoutType::NCHW,
                   const rk::nn::QuantizationType qntType = rk::nn::QuantizationType::NONE,
                   const uint8_t bits = 8,
                   const float scale = 1.0,
                   const uint32_t zero_point = 0,
                   const int8_t fl = 0);

  void HandleInitializer();
  std::vector<std::shared_ptr<rk::nn::Tensor>> GetInputOfOnnxModel(
      const std::vector<const void*>& input_bufs,
      const std::unordered_map<std::string, int>& input_maps);
  std::vector<std::shared_ptr<rk::nn::Tensor>> GetOutputOfOnnxModel();

  std::pair<bool, std::string> IsNodeSupported(
      const ONNX_NAMESPACE::ModelProto& model_proto,
      const ONNX_NAMESPACE::NodeProto& node_proto) const;

  void AddConv(const std::string& input,
               const std::vector<int>& strides,
               const std::vector<int>& pads,
               const std::vector<int>& dilations,
               const int32_t group,
               const std::string& ori_weight,
               const std::string& bias,
               const std::string& auto_pad,
               const std::string& output);
  void AddQLinearConv(const std::string& input,
                      const std::string& input_scale,
                      const std::string& input_zp,
                      const std::vector<int>& strides,
                      const std::vector<int>& pads,
                      const std::vector<int>& dilations,
                      const int group,
                      const std::string& auto_pad,
                      const std::string& weight,
                      const std::string& weight_scale,
                      const std::string& weight_zp,
                      const std::string& bias,
                      const std::string& output,
                      const std::string& output_scale,
                      const std::string& output_zp);
  void AddLayerPool(const std::string& op,
                    const std::string& input,
                    const std::vector<int>& kernel_shape,
                    const std::vector<int>& pads,
                    const std::vector<int>& strides,
                    const int32_t ceil_mode,
                    const std::string& output);
  void SetIdentity(const std::string& input,
                   const std::string& output);
  void AddLayerConvImpl(const std::string& input,
                        const std::string& weight,
                        const std::string& bias,
                        const std::vector<int32_t>& pads,
                        const std::vector<int32_t>& strides,
                        const int32_t group,
                        const std::string& auto_pad,
                        const std::string& output);
  void AddLayerQLinearConvImpl(const std::string& input,
                               const std::string& input_scale,
                               const std::string& input_zp,
                               const std::string& weight,
                               const std::string& weight_scale,
                               const std::string& weight_zp,
                               const std::string& bias,
                               const std::vector<int>& pads,
                               const std::vector<int>& strides,
                               const int group,
                               const std::string& auto_pad,
                               const std::string& output,
                               const std::string& output_scale,
                               const std::string& output_zp);
  void AddLayerAvePoolImpl(const std::string& input,
                           const std::vector<int32_t>& kernel_shape,
                           const std::vector<int32_t>& pads,
                           const std::vector<int32_t>& strides,
                           const int32_t ceil_mode,
                           const std::string& output);
  void AddLayerMaxPoolImpl(const std::string& input,
                           const std::vector<int32_t>& kernel_shape,
                           const std::vector<int32_t>& pads,
                           const std::vector<int32_t>& strides,
                           const int32_t ceil_mode,
                           const std::string& output);
  void AddLayerReLU(const std::string& input,
                    const std::string& output);
  void AddLayerSoftmax(const std::string& input,
                       const std::string& output);
  void AddLayerFC(const std::string& input,
                  const std::string& weight,
                  const std::string& bias,
                  const std::string& output);
  void AddLayerAdd(const std::string& input1,
                   const std::string& input2,
                   const std::string& output);
  void AddLayerSub(const std::string& input1,
                   const std::string& input2,
                   const std::string& output);
  void AddLayerConcat(const std::vector<std::string>& inputs,
                      const int32_t axis,
                      const std::string& output);
  void AddLayerDepthwiseConvImpl(const std::string& input,
                                 const std::string& weight,
                                 const std::string& bias,
                                 const std::vector<int32_t>& pads,
                                 const std::vector<int32_t>& strides,
                                 const int32_t depth_multiplier,
                                 const int32_t group,
                                 const std::string& output);
  void AddLayerBatchToSpaceND(const std::string& input,
                              const std::vector<int32_t>& block_sizes,
                              const std::string& output);
  void AddLayerSpaceToBatchND(const std::string& input,
                              const std::vector<int32_t>& block_sizes,
                              const std::vector<int32_t>& pads,
                              const std::string& output);
  void AddLayerSlice(const std::string& input,
                     const std::string& starts,
                     const std::string& ends,
                     const std::string& axes,
                     const std::string& steps,
                     const std::string& output);
  void AddLayerStridedSlice(const std::string& input,
                            const std::vector<int32_t>& starts,
                            const std::vector<int32_t>& ends,
                            const std::vector<int32_t>& strides,
                            const int32_t begin_mask,
                            const int32_t end_mask,
                            const int32_t shrink_axis_mask,
                            const std::string& output);
  void AddLayerMul(const std::string& input1,
                   const std::string& input2,
                   const std::string& output);
  void AddLayerAdd(const std::string& input,
                   const float scalar,
                   const std::string& output);
  void AddLayerMul(const std::string& input,
                   const float scalar,
                   const std::string& output);
  void AddLayerDequantize(const std::string& input,
                          const std::string& output);
  void AddLayerLRN(const std::string& input,
                   const int32_t radius,
                   const float bias,
                   const float alpha,
                   const float beta,
                   const std::string& output);
  void AddLayerTanh(const std::string& input,
                    const std::string& output);
  void AddLayerFloor(const std::string& input,
                     const std::string& output);
  void AddLayerLogistic(const std::string& input,
                        const std::string& output);
  void AddLayerBatchNorm(const std::string& input,
                         const std::string& scale,
                         const std::string& bias,
                         const std::string& mean,
                         const std::string& var,
                         const float eps,
                         const std::string& output);
  void AddLayerReshape(const std::string& input,
                       const std::string& shape,
                       const std::string& output);
  void AddLayerFlatten(const std::string& input,
                       const int32_t axis,
                       const std::string& output);
  void AddLayerTranspose(const std::string& input,
                         const std::vector<int32_t>& perm,
                         const std::string& output);
  void AddLayerSqueeze(const std::string& input,
                       const std::vector<int32_t>& axes,
                       const std::string& output);
  void AddLayerUnsqueeze(const std::string& input,
                         const std::vector<int32_t>& axes,
                         const std::string& output);
  void AddLayerGather(const std::string& input,
                      const std::string& indices,
                      const int32_t axis,
                      const std::string& output);
  void AddLayerLeakyRelu(const std::string& input,
                         const float alpha,
                         const std::string& output);
  void AddLayerClip(const std::string& input,
                    const int32_t min,
                    const int32_t max,
                    const std::string& output);
  void AddLayerDequantizeLinear(const std::string& input,
                                const std::string& input_scale,
                                const std::string& input_zp,
                                const std::string& output);
  void AddLayerQuantizeLinear(const std::string& input,
                              const std::string& output_scale,
                              const std::string& output_zp,
                              const std::string& output);

  void Clear();

  OnnxConverter(const OnnxConverter&);
  OnnxConverter& operator=(const OnnxConverter&);
};

}  // namespace rknpu
}  // namespace onnxruntime
