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
#include <string>
#include "core/providers/vsinpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace vsi {
namespace npu {
bool BaseOpBuilder::IsSupported(const onnxruntime::GraphViewer& graph_viewer,
                                const NodeUnit& node_unit) const {
  if (!HasSupportedOpSet(node_unit)) {
    return false;
  }
  if (!HasSupportedInputOutputs(graph_viewer, node_unit)) {
    return false;
  }
  return IsOpSupported(graph_viewer, &node_unit.GetNode());
}

bool BaseOpBuilder::HasSupportedInputOutputs(const GraphViewer& graph_viewer,
                                             const NodeUnit& node_unit) const {
  // We do not support unknown(null) input shape
  auto has_supported_shape = [](const NodeArg& node_arg, const std::string& name, const std::string& op_type) {
    const auto* shape_proto = node_arg.Shape();
    if (!shape_proto) {
      LOGS_DEFAULT(WARNING) << "Node [" << name << "] type [" << op_type
                            << "] Input [" << node_arg.Name() << "] has no shape";
      return false;
    }

    // We do not support dynamic shape input yet, but resize op's second input can be empty
    for (const auto& dim : shape_proto->dim()) {
      if (!dim.has_dim_value()) {
        LOGS_DEFAULT(WARNING) << "Dynamic shape is not supported for now, for input:" << node_arg.Name();
        return false;
      }
      if (dim.dim_value() == 0 && op_type != "Resize") {
        LOGS_DEFAULT(WARNING) << "Zero in shape is not supported for now, for input:" << node_arg.Name();
        return false;
      }
    }
    return true;
  };

  auto has_initialized_quant_param = [](const NodeArg& arg, const InitializersNames& initializers) {
    const bool found = initializers.contains(arg.Name());
    if (!found) {
      LOGS_DEFAULT(WARNING) << "The quantization param must be an initializer tensor";
    }
    return found;
  };

  for (const auto& input : node_unit.Inputs()) {
    if (!input.node_arg.Exists()) {
      continue;
    }
    if (!has_supported_shape(input.node_arg, node_unit.Name(), node_unit.OpType()))
      return false;

    if (input.quant_param.has_value()) {
      if (!has_supported_shape(input.quant_param->scale, node_unit.Name(), node_unit.OpType()))
        return false;

      if (!has_initialized_quant_param(input.quant_param->scale, graph_viewer.GetAllInitializersNames()))
        return false;
      // zero point is optional
      if (input.quant_param->zero_point) {
        if (!has_supported_shape(*input.quant_param->zero_point, node_unit.Name(), node_unit.OpType()))
          return false;
        if (!has_initialized_quant_param(*input.quant_param->zero_point, graph_viewer.GetAllInitializersNames()))
          return false;
        if (input.quant_param->zero_point->Type() != input.node_arg.Type()) {
          LOGS_DEFAULT(ERROR) << "Invalid input type because the data type mismatch with its' quant param type.";
          return false;
        }
      }
    }
  }
  for (const auto& output : node_unit.Outputs()) {
    for (const auto& dim : output.node_arg.Shape()->dim()) {
      if (!dim.has_dim_value()) {
        LOGS_DEFAULT(WARNING) << "Dynamic shape is not supported for now, for output:" << output.node_arg.Name();
        return false;
      }
      if (dim.dim_value() == 0 && output.node_arg.Shape()->dim_size() > 1) {
        LOGS_DEFAULT(WARNING) << "Zero in shape is not supported for now, for output:" << output.node_arg.Name();
        return false;
      }
    }
    if (output.quant_param.has_value()) {
      if (!has_supported_shape(output.quant_param->scale, node_unit.Name(), node_unit.OpType()))
        return false;

      if (!has_initialized_quant_param(output.quant_param->scale, graph_viewer.GetAllInitializersNames()))
        return false;
      // zero point is optional
      if (output.quant_param->zero_point) {
        if (!has_supported_shape(*output.quant_param->zero_point, node_unit.Name(), node_unit.OpType()))
          return false;
        if (!has_initialized_quant_param(*output.quant_param->zero_point, graph_viewer.GetAllInitializersNames()))
          return false;
      }
    }
  }
  return HasSupportedInputOutputsImpl(initializers, node_unit);
}

bool BaseOpBuilder::HasSupportedInputOutputsImpl(
    const InitializedTensorSet& /* initializers */, const NodeUnit& node_unit) const {
  // Check input/output data type, int64 is generally unsupported
  // specific op builder can override this if the int64 input corresponds to VSINPU param
  for (const auto& input : node_unit.Inputs()) {
    auto input_type = input.node_arg.Type();
    if (*input_type == "tensor(int64)" || !util::IsTypeSupported(&input.node_arg)) {
      LOGS_DEFAULT(WARNING) << node_unit.OpType() << " has unsupported input type : "
                            << *input_type;
      return false;
    }
  }
  for (const auto& output : node_unit.Outputs()) {
    auto output_type = output.node_arg.Type();
    if (*output_type == "tensor(int64)" || !util::IsTypeSupported(&output.node_arg)) {
      LOGS_DEFAULT(WARNING) << node_unit.OpType() << " has unsupported output type : "
                            << *output_type;
      return false;
    }
  }
  return true;
}

bool BaseOpBuilder::HasSupportedOpSet(const NodeUnit& node_unit) const {
  auto since_version = node_unit.SinceVersion();
  if (since_version < GetMinSupportedOpSet(node_unit) || since_version > GetMaxSupportedOpSet(node_unit)) {
    LOGS_DEFAULT(VERBOSE) << node_unit.OpType() << " opset [" << since_version
                          << "] is only supported for opset ["
                          << GetMinSupportedOpSet(node_unit) << ", "
                          << GetMaxSupportedOpSet(node_unit) << "]";
    return false;
  }

  return true;
}

bool BaseOpBuilder::BuildOp(vsi::npu::GraphEP* graph_ep,
                            const onnxruntime::GraphViewer& graph_viewer,
                            const NodeUnit& node_unit) {
  std::vector<std::shared_ptr<tim::vx::Tensor>> inputs;
  std::vector<NodeUnitIODef> input_defs = node_unit.Inputs();
  std::vector<NodeUnitIODef> output_defs = node_unit.Outputs();

  for (const auto input_def : input_defs) {
    auto it = std::find_if(
        graph_ep->GetGraphInputs().begin(), graph_ep->GetGraphInputs().end(),
        [input_def](const std::shared_ptr<GraphIOInfo>& info) {
          return info->name == input_def.node_arg.Name();
        });
    tim::vx::TensorAttribute attr;
    if (graph_viewer.IsConstantInitializer(input_def.node_arg.Name(), true)) {
      attr = tim::vx::TensorAttribute::CONSTANT;
    } else if (it == graph_ep->GetGraphInputs().end()) {
      attr = tim::vx::TensorAttribute::TRANSIENT;
    } else {
      attr = tim::vx::TensorAttribute::INPUT;
    }

    auto tensor = graph_ep->MapTIMVXTensor(graph_ep->GetGraph(), input_def, node_unit,
                                           &graph_viewer, attr);
    inputs.push_back(tensor);
  }

  std::vector<std::shared_ptr<tim::vx::Tensor>> outputs;

  for (auto output_def : output_defs) {
    auto it = std::find_if(
        graph_ep->GetGraphOutputs().begin(), graph_ep->GetGraphOutputs().end(),
        [output_def](const std::shared_ptr<GraphIOInfo>& info) {
          return info->name == output_def.node_arg.Name();
        });
    tim::vx::TensorAttribute attribute =
        it == graph_ep->GetGraphOutputs().end()
            ? tim::vx::TensorAttribute::TRANSIENT
            : tim::vx::TensorAttribute::OUTPUT;
    auto tensor = graph_ep->MapTIMVXTensor(graph_ep->GetGraph(), output_def, node_unit,
                                           &graph_viewer, attribute);
    outputs.push_back(tensor);
  }
  return HandleBuildOp(graph_ep, inputs, outputs, node_unit);
}
}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
