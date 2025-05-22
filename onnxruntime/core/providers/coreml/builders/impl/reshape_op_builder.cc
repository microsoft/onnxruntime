// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

class ReshapeOpBuilder : public BaseOpBuilder {
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  // Reshape opset 4- uses attributes for new shape which we do not support for now
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 5; }

  bool SupportsMLProgram() const override { return true; }
};

void ReshapeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // Skip the second input which is the new shape as we always have to create a new version as the CoreML rules
  // are different from ONNX.
  model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());
}

// modeled after onnxruntime/core/providers/cpu/tensor/reshape_helper.h
// but allowing multiple symbolic dimensions in the input_shape
// which requires us to manually calculate the input_shape Size (number of elements)
void ReshapeHelper(const TensorShape& input_shape, TensorShapeVector& requested_shape) {
    const auto input_dims = input_shape.NumDimensions();
    int64_t input_size = 1; // number of elements in the input tensor, ignoring any -1 dimensions
    int num_negative_one = 0;
    for (size_t i = 0; i < input_dims; ++i) {
      if (input_shape[i] == -1) {
        ++num_negative_one;
      } else {
        input_size *= input_shape[i];
      }
    }

    auto nDims = requested_shape.size();
    ptrdiff_t unknown_dim = -1;
    int64_t size = 1;
    for (size_t i = 0; i < nDims; ++i) {
      ORT_ENFORCE(requested_shape[i] >= -1, "A dimension cannot be less than -1, got ", requested_shape[i]);
      if (requested_shape[i] == -1) {
        ORT_ENFORCE(unknown_dim == -1, "At most one dimension can be -1.");
        unknown_dim = i;
      } else {
        if (requested_shape[i] == 0) {
          ORT_ENFORCE(i < input_shape.NumDimensions(),
                      "The dimension with value zero exceeds"
                      " the dimension size of the input tensor.");
          requested_shape[i] = input_shape[i];
        }
        size *= requested_shape[i];
      }
    }

    if (unknown_dim != -1) {
      // calculate unknown dimension
      ORT_ENFORCE(size != 0 && (input_size % size) == 0,
                  "The input tensor cannot be reshaped to the requested shape. Input shape:", input_shape,
                  ", requested shape:", TensorShape(requested_shape));
      requested_shape[unknown_dim] = input_size / size;
    } else {
      // check if the output shape is valid.
      ORT_ENFORCE(input_size == size,
                  "The input tensor cannot be reshaped to the requested shape. Input shape:", input_shape,
                  ", requested shape:", TensorShape(requested_shape));
    }
}

Status ReshapeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                               const Node& node,
                                               const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape of data");

  const auto& data_name = input_defs[0]->Name();
  const auto& new_shape_name = input_defs[1]->Name();
  const auto* shape_constant = model_builder.GetConstantInitializer(new_shape_name);
  TensorShapeVector new_shape;
  if (shape_constant) {
    Initializer unpacked_tensor(*shape_constant);
    new_shape = ToShapeVector(unpacked_tensor.DataAsSpan<int64_t>());
  }

  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;

    // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.tensor_transformation.reshape
    std::unique_ptr<Operation> reshape_op = model_builder.CreateOperation(node, "reshape");

    AddOperationInput(*reshape_op, "x", data_name);
    if (shape_constant) {
      AddOperationInput(*reshape_op, "shape",
                        model_builder.AddConstant(reshape_op->type(), "shape", ToConstSpan(new_shape)));
    } else {
      AddOperationInput(*reshape_op, "shape", new_shape_name);
    }

    AddOperationOutput(*reshape_op, *node.OutputDefs()[0]);

    model_builder.AddOperation(std::move(reshape_op));
  } else {
    std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = model_builder.CreateNNLayer(node);

    *layer->mutable_input()->Add() = data_name;
    *layer->mutable_reshapestatic()->mutable_targetshape() = {new_shape.cbegin(), new_shape.cend()};
    *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

    model_builder.AddLayer(std::move(layer));
  }
  return Status::OK();
}

bool AllPositiveShape(gsl::span<const int64_t> shape) {
  return std::all_of(shape.begin(), shape.end(), [](int64_t dim) { return dim > 0; });
}

// Checks that all values in new_shape (except -1) appear in input_shape
// and that the number of 0's in new_shape accounts for unaccounted dimensions
bool ValidateShapeValues(gsl::span<const int64_t> input_shape, gsl::span<const int64_t> new_shape) {
  // First, gather all unique values from input_shape
  std::unordered_set<int64_t> input_values;
  int input_values_seen = 0;
  for (const auto& dim : input_shape) {
    input_values.insert(dim);
    input_values_seen++;
  }

  // Now check that all values in new_shape (except -1 and 0) appear in input_shape
  int new_shape_unknown = 0;
  for (const auto& dim : new_shape) {
    if (dim == -1) {
      new_shape_unknown++;
    } else if (dim == 0) {
      continue;
    } else if (input_values.find(dim) == input_values.end()) {
      // Found a value that doesn't appear in input_shape
      return false;
    } else {
      input_values_seen--;
    }
  }

  // If we have unknown dimensions in input_shape, we need at least as many
  // placeholders (0 or -1) in new_shape to account for them
  // return new_shape_zeros == input_values_seen;
  return input_values_seen < 2 && new_shape_unknown <= 1;
}

bool LegalNegativeOneInNewShape(gsl::span<const int64_t> input_shape, gsl::span<const int64_t> new_shape) {
  // Count how many -1 dimensions exist in input_shape
  auto input_negative_one_count = std::count(input_shape.begin(), input_shape.end(), -1);
  // Count how many -1 dimensions exist in new_shape
  auto negative_one_count = std::count(new_shape.begin(), new_shape.end(), -1);
  // Count how many 0 dimensions exist in new_shape
  auto zero_count = std::count(new_shape.begin(), new_shape.end(), 0);

  // Case 1: If new_shape has no -1 dimensions, it's always valid
  if (negative_one_count == 0 && zero_count == 0) {
    return true;
  }

  // Case 2: If new_shape has exactly one -1 dimension, check if input_shape has at least one -1
  if (negative_one_count == 1 && ValidateShapeValues(input_shape, new_shape)) {
  // This isn't an explicit requirement of the spec, but to ensure that we can calculate the unknown
  // dimension in reshape, check that the number of -1 (unknown) dimensions in the input shape is
  // less than or equal to the number of -1 and 0's in the new shape.
    return input_negative_one_count >= 1 && input_negative_one_count <= (negative_one_count + zero_count);
  }

  // Case 3: If new_shape has more than one -1 dimension, it's invalid
  return false;
}

bool ReshapeOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                         const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& new_shape_name = input_defs[1]->Name();
  const auto* new_shape_tensor = input_params.graph_viewer.GetConstantInitializer(new_shape_name);

  NodeAttrHelper helper(node);
  const bool allow_zero = helper.Get("allowzero", 0) == 1;

  if (!new_shape_tensor && !allow_zero && input_defs[1]->Shape() && input_params.create_mlprogram) {
    // If the new shape is not a constant but the rank is known and zeroes are not
    // allowed, then we can assume that the new shape is valid
    return true;
  }

  if (!new_shape_tensor) {
    // ONNX has different rules around how -1 and 0 values are used/combined, and
    // we can't check if those can be translated to CoreML if the shape is unknown.
    LOGS(logger, VERBOSE) << "New shape of reshape must be a constant initializer";
    return false;
  }

  Initializer unpacked_tensor(*new_shape_tensor);
  auto new_shape = unpacked_tensor.DataAsSpan<int64_t>();

  if (new_shape.empty()) {
    LOGS(logger, VERBOSE) << "New shape of reshape cannot be empty";
    return false;
  }

  // CoreML reshape does not support 0 as a dimension
  if (allow_zero) {
    if (std::find(new_shape.begin(), new_shape.end(), int64_t{0}) != new_shape.end()) {
      LOGS(logger, VERBOSE) << "Reshape does not support new shape with 0 as dimension when allowzero is enabled. "
                            << "New shape: " << Shape2String(new_shape);
      return false;
    }
  }

  std::vector<int64_t> input_shape;
  if (!input_params.create_mlprogram) {
    // if we are using NeuralNetwork, input shape must be static
    if (!GetStaticShape(*input_defs[0], input_shape, logger)) {
      LOGS(logger, VERBOSE) << "Unable to get shape of input -- input must have static shape for NeuralNetwork.";
      return false;
    }
  }

  // first input must be fixed rank OR (first input has variadic rank AND shape only contains positive integers)
  // as per docs, 0 is considered an illegal shape element if the input is variadic
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    LOGS(logger, VERBOSE) << "Unable to get shape of input -- input must have fixed rank for reshape.";
    return false;
  }

  if (input_shape.empty() && !AllPositiveShape(new_shape)) {
    // unknown rank & fails the positive shape check
    LOGS(logger, VERBOSE) << "Reshape does not support empty input shape unless the shape input contains all positive integers. "
                             "Input shape: "
                          << Shape2String(input_shape) << ", new shape: " << Shape2String(new_shape);
    return false;
  }

  if (new_shape.size() > 5) {
    LOGS(logger, VERBOSE) << "Reshape does not support new shape with more than 5 dimensions. "
                             "Input shape: "
                          << Shape2String(input_shape) << ", new shape: " << Shape2String(new_shape);
    return false;
  }

  return true;
}

void CreateReshapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ReshapeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
