// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/slice_helper.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class SliceOpBuilder : public BaseOpBuilder {
  // Add operator related.
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
  // TODO: Support Slice opset < 10, which uses attributes for starts and ends.
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 10; }
};

// Add operator related.

void SliceOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // Skip all initializer except the first inputs(data).
  for (size_t i = 1; i < node.InputDefs().size(); i++) {
    model_builder.AddInitializerToSkip(node.InputDefs()[i]->Name());
  }
}

Status SliceOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                             const Node& node,
                                             const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");
  auto rank = input_shape.size();
  NodeAttrHelper helper(node);

  emscripten::val inputs = model_builder.GetOperand(input_defs[0]->Name());
  std::vector<int32_t> starts(rank);
  std::vector<int32_t> sizes(rank);

  // Copy the data from the starts/ends/axes/steps initializers.
  std::vector<int64_t> input_starts;
  std::vector<int64_t> input_ends;
  std::vector<int64_t> input_axes;
  std::vector<int64_t> input_steps;
  SliceOp::PrepareForComputeMetadata compute_metadata(input_shape);
  const auto CopyInputData = [&input_defs, &model_builder, &logger](size_t input_idx, std::vector<int64_t>& data,
                                                                    bool is_required = false) {
    data.clear();
    std::string input_name;
    // This is an optional input, return empty vector.
    if (!is_required) {
      if (input_defs.size() <= input_idx)
        return Status::OK();
      input_name = input_defs[input_idx]->Name();
      if (input_name.empty())
        return Status::OK();
    }
    input_name = input_defs[input_idx]->Name();
    const auto& initializers(model_builder.GetInitializerTensors());
    const auto& tensor = *initializers.at(input_name);
    if (!ReadIntArrayFrom1DTensor(tensor, data, logger)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Data type for starts and ends inputs is not supported in this build.");
    }

    return Status::OK();
  };
  ORT_RETURN_IF_ERROR(CopyInputData(1, input_starts, true));
  ORT_RETURN_IF_ERROR(CopyInputData(2, input_ends, true));
  ORT_RETURN_IF_ERROR(CopyInputData(3, input_axes));
  ORT_RETURN_IF_ERROR(CopyInputData(4, input_steps));
  ORT_RETURN_IF_ERROR(
      SliceOp::PrepareForComputeHelper(input_starts, input_ends, input_axes, input_steps, compute_metadata));

  std::transform(compute_metadata.starts_.cbegin(), compute_metadata.starts_.cend(),
                 starts.begin(),
                 [](int64_t i) { return SafeInt<uint32_t>(i); });
  std::transform(compute_metadata.ends_.cbegin(), compute_metadata.ends_.cend(), compute_metadata.starts_.cbegin(),
                 sizes.begin(),
                 [](int64_t i, int64_t j) { return SafeInt<uint32_t>(i - j); });

  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>("slice", inputs,
                                                                            emscripten::val::array(starts),
                                                                            emscripten::val::array(sizes));

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

bool SliceOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers,
                                       const Node& node,
                                       const WebnnDeviceType /* device_type */,
                                       const logging::Logger& logger) const {
  const auto& name = node.Name();
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    return false;
  }

  if (input_defs.size() < 3) {
    LOGS(logger, VERBOSE) << op_type << " [" << name << "] requires at least 3 inputs (data starts and ends) but got "
                          << input_defs.size();
    return false;
  }

  // Inputs: starts, ends, axes and steps if present must be constant initializers.
  for (size_t i = 1; i < input_defs.size(); i++) {
    if (!Contains(initializers, input_defs[i]->Name())) {
      LOGS(logger, VERBOSE) << "Input [" << input_defs[i]->Name() << "] of " << op_type
                            << " [" << name << "] must be known as initializer";
      return false;
    }
  }

  if (input_defs.size() == 5) {  // Check steps.
    const auto& steps_tensor = *initializers.at(input_defs[4]->Name());
    std::vector<uint8_t> unpacked_tensor;
    auto status = onnxruntime::utils::UnpackInitializerData(steps_tensor, unpacked_tensor);
    if (!status.IsOK()) {
      LOGS(logger, ERROR) << "Error while unpacking steps_tensor: " << status.ErrorMessage();
      return false;
    }
    const auto data_type = steps_tensor.data_type();
    // WebNN doesn't support steps other than 1.
    if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      if (!std::all_of(reinterpret_cast<int64_t*>(unpacked_tensor.data()),
                       reinterpret_cast<int64_t*>(unpacked_tensor.data() + unpacked_tensor.size()),
                       [](int64_t i) { return i == 1; })) {
        return false;
      }
    } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
      if (!std::all_of(reinterpret_cast<int32_t*>(unpacked_tensor.data()),
                       reinterpret_cast<int32_t*>(unpacked_tensor.data()) +
                           unpacked_tensor.size() / sizeof(int32_t),
                       [](int32_t i) { return i == 1; })) {
        return false;
      }
    }
  }

  return true;
}

void CreateSliceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SliceOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
