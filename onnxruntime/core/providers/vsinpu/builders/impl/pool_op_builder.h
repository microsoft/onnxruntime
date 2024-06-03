/****************************************************************************
 *
 *    Copyright (c) 2023 Vivante Corporation
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a
 *    copy of this software and associated documentation files (the "Software"),
 *    to deal in the Software without restriction, including without limitation
 *    the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *    and/or sell copies of the Software, and to permit persons to whom the
 *    Software is furnished to do so, subject to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in
 *    all copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *    DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/
#include "core/providers/vsinpu/builders/impl/base_op_builder.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace vsi {
namespace npu {
class BasePoolOpBuilder : public BaseOpBuilder {
 public:
  BasePoolOpBuilder(tim::vx::PoolType pool_type) : pool_type_(pool_type) {}

 protected:
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer, const Node* node) const override {
    auto shape = vsi::npu::util::GetTensorShape(*node->InputDefs()[0]);
    if (shape.NumDimensions() == 5) {
      LOGS_DEFAULT(WARNING) << "3DPool is not supported yet.";
      return false;
    }

    NodeAttrHelper helper(*node);
    if (helper.HasAttr("dilations")) {
      LOGS_DEFAULT(WARNING) << "NonMaxPool with Dilation parameter is not supported.";
      return false;
    }
    return true;
  }
  bool CreatePoolingOp(vsi::npu::GraphEP* graph_ep,
                       std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                       std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                       const std::array<uint32_t, 2>& kernel_size,
                       const std::array<uint32_t, 2>& strides,
                       const std::array<uint32_t, 4>& pads,
                       bool is_global,
                       const tim::vx::RoundType ceil_mode) {
    const bool is_1d_pool = inputs[0]->GetShape().size() == 3;
    std::shared_ptr<tim::vx::Operation> op;

    // Create the appropriate pooling operation
    if (is_global) {
      if (is_1d_pool) {
        op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Pool1d>(pool_type_, inputs[0]->GetShape()[0], ceil_mode);
      } else {
        std::array<uint32_t, 2> input_size = {inputs[0]->GetShape()[0], inputs[0]->GetShape()[1]};
        op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Pool2d>(pool_type_, input_size, ceil_mode);
      }

    } else {
      if (is_1d_pool) {
        op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Pool1d>(pool_type_, std::array<uint32_t, 2>{pads[2], pads[0]}, kernel_size[1], strides[1], ceil_mode);
      } else {
        op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Pool2d>(pool_type_, pads, kernel_size, strides, ceil_mode);
      }
    }

    (*op).BindInputs(inputs).BindOutputs(outputs);
    graph_ep->GetOps().push_back(std::move(op));
    return true;
  }
  tim::vx::PoolType pool_type_;
};

class TraditionalPoolOpBuilder : public BasePoolOpBuilder {
 public:
  TraditionalPoolOpBuilder() : BasePoolOpBuilder(tim::vx::PoolType::MAX) {}

 protected:
  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const NodeUnit& node_unit) override {
    NodeAttrHelper helper(node_unit.GetNode());
    auto ksize = helper.Get("kernel_shape", std::vector<uint32_t>{1U, 1U});
    auto strides = helper.Get("strides", std::vector<uint32_t>{1U, 1U});
    auto pads = helper.Get("pads", std::vector<uint32_t>{0U, 0U, 0U, 0U});
    tim::vx::RoundType ceil_mode = helper.Get("ceil_mode", 0U) == 0 ? tim::vx::RoundType::FLOOR : tim::vx::RoundType::CEILING;
    return CreatePoolingOp(graph_ep, inputs, outputs,
                           {ksize[1], ksize[0]}, {strides[1], strides[0]}, {pads[1], pads[3], pads[0], pads[2]}, false, ceil_mode);
  }
};

class GlobalPoolOpBuilder : public BasePoolOpBuilder {
 public:
  GlobalPoolOpBuilder() : BasePoolOpBuilder(tim::vx::PoolType::MAX) {}

 protected:
  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const NodeUnit& node_unit) override {
    NodeAttrHelper helper(node_unit.GetNode());
    tim::vx::RoundType ceil_mode = helper.Get("ceil_mode", 0U) == 0 ? tim::vx::RoundType::FLOOR : tim::vx::RoundType::CEILING;
    return CreatePoolingOp(graph_ep, inputs, outputs, {}, {}, {}, true, ceil_mode);
  }
};

class GlobalAveragePoolOpBuilder : public GlobalPoolOpBuilder {
 public:
  GlobalAveragePoolOpBuilder() { pool_type_ = tim::vx::PoolType::AVG; }
};

class GlobalMaxPoolOpBuilder : public GlobalPoolOpBuilder {
 public:
  GlobalMaxPoolOpBuilder() { pool_type_ = tim::vx::PoolType::MAX; }
};

class AveragePoolOpBuilder : public TraditionalPoolOpBuilder {
 public:
  AveragePoolOpBuilder() { pool_type_ = tim::vx::PoolType::AVG; }
};

class MaxPoolOpBuilder : public TraditionalPoolOpBuilder {
 public:
  MaxPoolOpBuilder() { pool_type_ = tim::vx::PoolType::MAX; }
};

}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
