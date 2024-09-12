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
#pragma once
#include "core/graph/graph_viewer.h"
#include "core/framework/node_unit.h"

namespace onnxruntime {
namespace vsi {
namespace npu {
class GraphEP;

class IOpBuilder {
 public:
  IOpBuilder() {}
  virtual ~IOpBuilder() {}
  virtual bool IsSupported(const onnxruntime::GraphViewer& graph_viewer,
                           const NodeUnit& node_unit) const {
    return true;
  }
  virtual bool BuildOp(GraphEP* graph_ep,
                       const onnxruntime::GraphViewer& graph_viewer,
                       const NodeUnit& node_unit) = 0;
};
}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
