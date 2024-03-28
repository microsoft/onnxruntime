// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/reshape_helper.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class EinsumOpBuilder : public BaseOpBuilder {
  // Add operator related.

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
};

// Helper functions, thanks for DML EP's OperatorHelper.
enum class RecognizedOperatorType {
  None,
  Identity,
  ReduceSum,
  Transpose,
  Diagonal,  // TODO: support diagonal, but now wait for webnn triangular op
  Multiply,
  Pairwise,
  Total,
};

struct RecognizedOperatorInfo {
  RecognizedOperatorType recognized_operator_type;
  std::initializer_list<uint32_t> component_ranks;
  std::initializer_list<uint32_t> label_indices;
};

struct Component {
  uint32_t label_index_begin;
  uint32_t label_index_end;

  uint32_t GetDimensionCount() const noexcept {
    return label_index_end - label_index_begin;
  }
  gsl::span<const uint32_t> GetLabels(gsl::span<const uint32_t> labels) const {
    return labels.subspan(label_index_begin, label_index_end - label_index_begin);
  }
};

bool ParseEquationComponents(const InitializedTensorSet& initializers,
                             const Node& node,
                             const std::string& equation,
                             std::vector<uint32_t>& label_indices,
                             std::vector<Component>& components,
                             std::vector<uint32_t>& output_dimensions,
                             uint32_t& num_labels,
                             const logging::Logger& logger) {
  // Parse an equation like 'ij,jk->ik' into components {ij, jk, ik} mapping letters to
  // numeric indices {(0,1}, {1,2}, {0,2}}. The last component is the output.
  // Read first to last character in equation, looking for letters, commas, and one arrow.
  // TODO: support for parsing ellipsis...
  std::map<char, uint32_t> label_maps;
  std::set<char> repeated_labels;

  num_labels = 0;
  Component current_component = {};
  bool at_output = false;
  bool end_flag = false;

  for (const char* it = equation.data(); !end_flag; ++it) {
    // std::string.data() promises the end of the string is '\0'
    char ch = *it;

    if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')) {
      const auto [i, inserted] = label_maps.insert({ch, num_labels});
      if (inserted) {
        if (at_output) {
          LOGS(logger, VERBOSE) << "Found label in equation output not matching any label from inputs.";
          return false;
        }
        ++num_labels;
      } else if (!at_output) {
        repeated_labels.insert(ch);
      }
      label_indices.push_back(i->second);
    } else if (ch == ' ') {
      continue;
    } else {
      current_component.label_index_end = static_cast<uint32_t>(label_indices.size());
      components.push_back(current_component);
      current_component.label_index_begin = current_component.label_index_end;

      switch (ch) {
        case ',':
          break;

        case '-':
          ++it;
          if (*it != '>') {
            LOGS(logger, VERBOSE) << "Expected '->' for output.";
            return false;
          }
          if (at_output) {
            LOGS(logger, VERBOSE) << "Only one output arrow '->' is valid.";
            return false;
          }
          at_output = true;
          break;

        case '.':
          // Ellipsis is unsupported
          LOGS(logger, VERBOSE) << "Ellipsis is unsupported.";
          return false;

        case '\0':
          end_flag = true;
          break;  // End of string.

        default:
          LOGS(logger, VERBOSE) << "Unsupported character in equation string.";
          return false;
      }
    }
  }

  // If no explicit output was given, generate an implicit output by ordering all the
  // labels in alphabetic order (by ASCII value consistent with numpy, so Z < a).
  // Exclude any labels that occurred more than once, as these cancel out.
  if (!at_output) {
    for (auto i : label_maps) {
      if (repeated_labels.count(i.first) == 0) {
        label_indices.push_back(i.second);
      }
    }

    current_component.label_index_end = static_cast<uint32_t>(label_indices.size());
    components.push_back(current_component);
  }
  return true;
}

// For two inputs A,B and one output C
Status PairwiseOperandProcess(ModelBuilder& model_builder,
                              const Node& node,
                              const std::vector<uint32_t>& label_indices,
                              const std::vector<Component>& components,
                              const std::vector<uint32_t>& output_dimensions,
                              uint32_t num_labels,
                              emscripten::val& output,
                              const logging::Logger& logger) {
  auto input_a_labels = components[0].GetLabels(label_indices);
  auto input_b_labels = components[1].GetLabels(label_indices);
  auto output_labels = components[2].GetLabels(label_indices);

  /*
  Step 1. Transpose and Reshape

  (0/1,0/1,0/1) means dim i whether appears in (A,B,C)
  For new A, it has three segements [...a_1..., a_2, a_3], a_1 has multiple dims, a_2 and a_3 only have one dim respectively
  For new B, it has three segements [...b_1..., b_2, b_3], b_1 has multiple dims, b_2 and b_3 only have one dim respectively
  a_1 and b_1 are batch dims, and [a_2,a_3], [b_2,b_3] are for matmul

  case (1,0,0) and (0,1,0): reduce, here we treat it as batch dimension, and reduceSum at the end.
            add additional dim for B/A
  case (1,1,1): batch dimension, put it in the front.
  case (1,0,1): gemm dim for A, put it in a_2
  case (0,1,1): gemm dim for B, put it in b_3
  case (1,1,0): summation dim / gemm dim for both A and B, put it in a_3 and b_2

  Attention:
    # of (1,1,0) maybe > 1, flatten / reshape a_3 and b_2
    # of (1,1,0) maybe = 0, add one additional dim for a_3 and b_2
  */

  // The index in input/output of the dim index
  std::map<uint32_t, int32_t> input_a_axes_map, input_b_axes_map, output_axes_map;

  for (uint32_t i = 0; i <= num_labels + 1; ++i) {
    input_a_axes_map[i] = input_b_axes_map[i] = output_axes_map[i] = -1;
  }
  int32_t index = 0;
  for (auto axis : input_a_labels) {
    input_a_axes_map[axis] = index++;
  }
  index = 0;
  for (auto axis : input_b_labels) {
    input_b_axes_map[axis] = index++;
  }
  index = 0;
  for (auto axis : output_labels) {
    output_axes_map[axis] = index++;
  }

  // Inputs Reshape
  // a_0 = [a_1,a_2,a_3], b_0 = [b_1,b_2,b_3]
  std::vector<uint32_t> a_0, a_1, a_2, a_3, b_0, b_1, b_2, b_3;
  uint32_t a_idx = input_a_labels.size();
  uint32_t b_idx = input_b_labels.size();
  bool a_flag = false;  // whether a_2 has element
  bool b_flag = false;  // whether b_3 has element

  for (uint32_t i = 0; i < num_labels; ++i) {
    if (input_a_axes_map[i] != -1) {
      if (input_b_axes_map[i] != -1) {
        if (output_axes_map[i] != -1) {
          // The index in input/output of the dim index
          a_1.push_back(i);
          b_1.push_back(i);
        } else {
          // (1,1,0) push back in the middle for b and end for a
          a_3.push_back(i);
          b_2.push_back(i);
        }
      } else {
        // (1,0,x) push back in the middle for a. If more than one, push back in the front for a, b.
        if (a_flag) {
          a_1.push_back(i);
          b_1.push_back(i);
          input_b_axes_map[i] = b_idx++;
        } else {
          a_2.push_back(i);
          a_flag = true;
        }
      }
    } else {
      // (0,1,x) push back in the end for b. If more than one, push back in the front for a, b.
      if (input_b_axes_map[i] != -1) {
        if (b_flag) {
          a_1.push_back(i);
          b_1.push_back(i);
          input_a_axes_map[i] = a_idx++;
        } else {
          b_3.push_back(i);
          b_flag = true;
        }
      }
    }
  }

  // Matrix multiplication can be formatted in (...,i,j) * (...,j,k) ==> (...,i,k)
  // Even inner and outter product can be reformatted as this.
  // Inner product (1,i) * (i,1) ==> (1,1)
  // Outter product (i,1) * (1,j) ==> (i,j)
  // i.e., in our expression, (a_2,a_3) * (b_2,b_3) ==> (a_2,b_3)

  if (!a_flag) {
    // Lack of a_2 element, add a new a_2, whose dim value = 1
    a_2.push_back(num_labels + 1);
    input_a_axes_map[num_labels + 1] = a_idx++;
  }
  if (!b_flag) {
    // Lack of b_3 element, add a new b_3, whose dim value = 1
    b_3.push_back(num_labels + 2);
    input_b_axes_map[num_labels + 2] = b_idx++;
    b_idx++;
  }

  if (a_3.empty()) {
    // Lack of a_3 and b_2 elements, add a new a_3 for A and a new b_2 for B, whose dim value = 1
    a_3.push_back(num_labels);
    b_2.push_back(num_labels);
    input_a_axes_map[num_labels] = a_idx;
    input_b_axes_map[num_labels] = b_idx;
  }

  a_0 = a_1;
  b_0 = b_1;
  a_0.insert(a_0.end(), a_2.begin(), a_2.end());
  a_0.insert(a_0.end(), a_3.begin(), a_3.end());
  b_0.insert(b_0.end(), b_2.begin(), b_2.end());
  b_0.insert(b_0.end(), b_3.begin(), b_3.end());

  std::vector<uint32_t> permutation_a, permutation_b;
  for (uint32_t i = 0; i < a_0.size(); ++i) {
    permutation_a.push_back(static_cast<uint32_t>(input_a_axes_map[a_0[i]]));
    permutation_b.push_back(static_cast<uint32_t>(input_b_axes_map[b_0[i]]));
  }

  const auto& input_defs = node.InputDefs();
  emscripten::val input_a = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val input_b = model_builder.GetOperand(input_defs[1]->Name());
  std::vector<uint32_t> new_a_shape, new_b_shape;
  if (input_a_labels.size() < a_0.size()) {
    std::vector<int64_t> input_a_shape;
    ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_a_shape, logger), "Cannot get shape");
    std::transform(input_a_shape.begin(), input_a_shape.end(), std::back_inserter(new_a_shape),
                   [](int64_t i) { return static_cast<uint32_t>(i); });
    for (uint32_t i = 0; i < a_0.size() - input_a_labels.size(); ++i) {
      new_a_shape.push_back(SafeInt<int32_t>(1));
    }
    input_a = model_builder.GetBuilder().call<emscripten::val>("reshape", input_a, emscripten::val::array(new_a_shape));
  }
  if (input_b_labels.size() < b_0.size()) {
    std::vector<int64_t> input_b_shape;
    ORT_RETURN_IF_NOT(GetShape(*input_defs[1], input_b_shape, logger), "Cannot get shape");
    std::transform(input_b_shape.begin(), input_b_shape.end(), std::back_inserter(new_b_shape),
                   [](int64_t i) { return static_cast<uint32_t>(i); });
    for (uint32_t i = 0; i < b_0.size() - input_b_labels.size(); ++i) {
      new_b_shape.push_back(SafeInt<int32_t>(1));
    }
    input_b = model_builder.GetBuilder().call<emscripten::val>("reshape", input_b, emscripten::val::array(new_b_shape));
  }

  // Inputs Transpose
  std::vector<uint32_t> sequence(permutation_a.size());
  std::iota(sequence.begin(), sequence.end(), 0);
  if (permutation_a != sequence) {
    emscripten::val options = emscripten::val::object();
    options.set("permutation", emscripten::val::array(permutation_a));
    input_a = model_builder.GetBuilder().call<emscripten::val>("transpose", input_a, options);
  }
  if (permutation_b != sequence) {
    emscripten::val options = emscripten::val::object();
    options.set("permutation", emscripten::val::array(permutation_b));
    input_b = model_builder.GetBuilder().call<emscripten::val>("transpose", input_b, options);
  }

  // Input Reshape: if the number of (1,1,0) > 1, flatten the b_2 and a_3 dims.
  if (a_3.size() > 1) {
    if (new_a_shape.empty()) {
      std::vector<int64_t> input_a_shape;
      ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_a_shape, logger), "Cannot get shape");
      std::transform(input_a_shape.begin(), input_a_shape.end(), std::back_inserter(new_a_shape),
                     [](int64_t i) { return static_cast<uint32_t>(i); });
    }
    if (new_b_shape.empty()) {
      std::vector<int64_t> input_b_shape;
      ORT_RETURN_IF_NOT(GetShape(*input_defs[1], input_b_shape, logger), "Cannot get shape");
      std::transform(input_b_shape.begin(), input_b_shape.end(), std::back_inserter(new_b_shape),
                     [](int64_t i) { return static_cast<uint32_t>(i); });
    }
    std::vector<uint32_t> new_new_a_shape, new_new_b_shape;
    uint32_t a_dim = 1, b_dim = 1;
    for (auto idx : a_1) {
      new_new_a_shape.push_back(new_a_shape[idx]);
    }
    for (auto idx : a_2) {
      new_new_a_shape.push_back(new_a_shape[idx]);
    }
    for (auto idx : a_3) {
      a_dim *= new_a_shape[idx];
    }
    new_new_a_shape.push_back(a_dim);
    for (auto idx : b_1) {
      new_new_b_shape.push_back(new_b_shape[idx]);
    }
    for (auto idx : b_2) {
      b_dim *= new_b_shape[idx];
    }
    new_new_b_shape.push_back(b_dim);
    for (auto idx : b_3) {
      new_new_b_shape.push_back(new_b_shape[idx]);
    }

    input_a = model_builder.GetBuilder().call<emscripten::val>("reshape", input_a,
                                                               emscripten::val::array(new_new_a_shape));
    input_b = model_builder.GetBuilder().call<emscripten::val>("reshape", input_b,
                                                               emscripten::val::array(new_b_shape));
  }

  // Step 2. Matmul
  output = model_builder.GetBuilder().call<emscripten::val>("matmul", input_a, input_b);
  std::vector<uint32_t> output_indices = a_1;
  output_indices.push_back(a_2.back());
  output_indices.push_back(b_3.back());

  /*
    Step 3. Output Transpose:
    Use the following fast permutation calculation algorithm
    to calculate the permutation of transpose.
    sequence X[] -> sequence Y[] : permutation P[]
    X[S[i]] = i, Y[T[i]] = i, P[S[i]] = T[i]
    output_indices is X and target_output_indices is Y
  */
  std::vector<uint32_t> target_output_indices(output_labels.begin(), output_labels.end());

  // map output dim labels to 0 ~ n-1
  std::vector<uint32_t> arr(output_indices.begin(), output_indices.end());
  std::map<uint32_t, uint32_t> mapping;
  std::sort(arr.begin(), arr.end());
  for (size_t i = 0; i < arr.size(); i++) {
    mapping[arr[i]] = i;
  }

  for (size_t i = 0; i < output_indices.size(); i++) {
    output_indices[i] = mapping[output_indices[i]];
    if (i < target_output_indices.size()) {
      target_output_indices[i] = mapping[target_output_indices[i]];
    }
  }

  uint32_t p = target_output_indices.size();
  std::vector<int64_t> s(output_indices.size(), -1), t(output_indices.size(), -1);
  std::vector<uint32_t> v(output_indices.size(), 0);
  for (uint32_t i = 0; i < output_indices.size(); ++i) {
    s[output_indices[i]] = i;
    if (i < target_output_indices.size()) {
      t[target_output_indices[i]] = i;
    }
  }
  for (uint32_t i = 0; i < output_indices.size(); ++i) {
    if (t[i] == -1) {
      t[i] = p++;
    }
    v[static_cast<uint32_t>(s[i])] = static_cast<uint32_t>(t[i]);
  }

  std::vector<uint32_t> sequence_o(output_indices.size());
  std::iota(sequence_o.begin(), sequence_o.end(), 0);
  if (v != sequence_o) {
    emscripten::val options = emscripten::val::object();
    options.set("permutation", emscripten::val::array(v));
    output = model_builder.GetBuilder().call<emscripten::val>("transpose", output, options);
  }

  // Step 4. Output ReduceSum
  if (output_labels.size() < output_indices.size()) {
    std::vector<int32_t> axes_data;
    for (uint32_t i = output_labels.size(); i < output_indices.size(); ++i) {
      axes_data.push_back(SafeInt<int32_t>(i));
    }
    emscripten::val options_reduce = emscripten::val::object();
    options_reduce.set("axes", emscripten::val::array(axes_data));
    output = model_builder.GetBuilder().call<emscripten::val>("reduceSum", output, options_reduce);
  }
  return Status::OK();
}

RecognizedOperatorType DetermineRecognizedOperatorType(const std::vector<uint32_t>& label_indices,
                                                       const std::vector<Component>& components,
                                                       const std::vector<uint32_t>& output_dimensions) {
  if (components.empty()) return RecognizedOperatorType::None;

  auto equals = [](gsl::span<const uint32_t> a, gsl::span<const uint32_t> b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end());
  };

  std::array<uint32_t, 3> component_ranks;
  if (components.size() > component_ranks.size()) {
    // So far, not support for more than two inputs and one output.
    return RecognizedOperatorType::None;
  } else if (components.size() == 2) {  // one input
    auto input_labels = components[0].GetLabels(label_indices);
    auto output_labels = components[1].GetLabels(label_indices);
    if (input_labels.size() == output_labels.size()) {
      if (equals(input_labels, output_labels)) {  // identity
        return RecognizedOperatorType::Identity;
      } else {
        return RecognizedOperatorType::Transpose;
      }
    } else if (output_labels.empty()) {  // scalar output, reduce
      return RecognizedOperatorType::ReduceSum;
    }
    return RecognizedOperatorType::None;

  } else if (components.size() == 3) {  // two inputs
    auto input_A_labels = components[0].GetLabels(label_indices);
    auto input_B_labels = components[1].GetLabels(label_indices);
    auto output_labels = components[2].GetLabels(label_indices);
    if (equals(input_A_labels, output_labels) && equals(input_B_labels, output_labels)) {  // element-wise product
      return RecognizedOperatorType::Multiply;
    }
  }

  return RecognizedOperatorType::Pairwise;
}

// Add operator related.

Status EinsumOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                              const Node& node,
                                              const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& initializers(model_builder.GetInitializerTensors());
  emscripten::val output = emscripten::val::object();

  NodeAttrHelper helper(node);
  const auto equation = helper.Get("equation", std::string(" "));

  std::vector<uint32_t> label_indices;
  std::vector<Component> components;
  std::vector<uint32_t> output_dimensions;
  uint32_t num_labels;
  ORT_RETURN_IF_NOT(ParseEquationComponents(initializers, node, equation, label_indices,
                                            components, output_dimensions, num_labels, logger),
                    "Error parsing equation components.");

  RecognizedOperatorType recognized_operator_type = DetermineRecognizedOperatorType(label_indices, components,
                                                                                    output_dimensions);

  switch (recognized_operator_type) {
    case RecognizedOperatorType::Multiply: {
      const size_t a_idx = 0, b_idx = 1;
      emscripten::val a = model_builder.GetOperand(node.InputDefs()[a_idx]->Name());
      emscripten::val b = model_builder.GetOperand(node.InputDefs()[b_idx]->Name());
      output = model_builder.GetBuilder().call<emscripten::val>("mul", a, b);
    } break;

    case RecognizedOperatorType::ReduceSum: {
      auto kept_axes = components.back().GetLabels(label_indices);
      assert(kept_axes.size() <= 1);
      std::vector<uint32_t> reduced_axes;
      uint32_t kept_axes_mask = 0;
      for (auto axis : kept_axes) {
        kept_axes_mask |= (1 << axis);
      }
      std::vector<int64_t> input_shape;
      ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
      for (uint32_t axis = 0, axis_count = static_cast<uint32_t>(input_shape.size()); axis < axis_count; ++axis) {
        if (~kept_axes_mask & (1 << axis)) {
          reduced_axes.push_back(axis);
        }
      }

      emscripten::val input = model_builder.GetOperand(node.InputDefs()[0]->Name());
      emscripten::val options = emscripten::val::object();
      options.set("keepDimensions", false);
      options.set("axes", emscripten::val::array(reduced_axes));

      output = model_builder.GetBuilder().call<emscripten::val>("reduceSum", input, options);
    } break;

    case RecognizedOperatorType::Transpose: {
      emscripten::val input = model_builder.GetOperand(node.InputDefs()[0]->Name());
      assert(components.front().GetDimensionCount() == components.back().GetDimensionCount());
      // Remap transposed strides using the component labels from input to output.
      auto output_labels = components.back().GetLabels(label_indices);

      std::vector<uint32_t> permutation{output_labels.begin(), output_labels.end()};
      emscripten::val options = emscripten::val::object();
      options.set("permutation", emscripten::val::array(permutation));
      output = model_builder.GetBuilder().call<emscripten::val>("transpose", input, options);
    } break;

    case RecognizedOperatorType::Identity: {
      // identity has not been supported by XNNPack backend, but it will be coming soon.
      emscripten::val input = model_builder.GetOperand(node.InputDefs()[0]->Name());
      output = input;
    } break;

    case RecognizedOperatorType::Pairwise: {
      ORT_RETURN_IF_ERROR(PairwiseOperandProcess(model_builder, node, label_indices, components,
                                                 output_dimensions, num_labels, output, logger));
    } break;

    default:
      break;
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

bool EinsumOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers,
                                        const Node& node,
                                        const WebnnDeviceType device_type,
                                        const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  NodeAttrHelper helper(node);
  const auto equation = helper.Get("equation", std::string(" "));
  std::vector<uint32_t> label_indices;
  std::vector<Component> components;
  std::vector<uint32_t> output_dimensions;
  uint32_t num_labels;

  if (!ParseEquationComponents(initializers, node, equation, label_indices,
                               components, output_dimensions, num_labels, logger)) {
    LOGS(logger, VERBOSE) << "EinSum input equation is illegal.";
    return false;
  }

  if (static_cast<uint32_t>(input_defs.size()) + 1 != components.size()) {
    LOGS(logger, VERBOSE) << "EinSum input tensor count is inconsistent with the equation component count.";
    return false;
  }

  RecognizedOperatorType recognized_operator_type = DetermineRecognizedOperatorType(label_indices, components,
                                                                                    output_dimensions);
  if (recognized_operator_type == RecognizedOperatorType::None) {
    LOGS(logger, VERBOSE) << "The equation is not supported in Einsum.";
    return false;
  }

  if (recognized_operator_type == RecognizedOperatorType::ReduceSum && device_type == WebnnDeviceType::CPU) {
    LOGS(logger, VERBOSE) << "Einsum is not supported for cpu in WebNN EP. ReduceSum is not supported in XNNPACK.";
    return false;
  }

  return true;
}

void CreateEinsumOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<EinsumOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
