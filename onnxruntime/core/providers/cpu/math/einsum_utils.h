//TODO: Add copyright

#pragma once

#include "einsum_auxiliary_ops.h"

namespace onnxruntime {

namespace EinsumOp {

constexpr size_t num_of_letters = 26;

}  // namespace EinsumOp

// Prologue:
// In the sample Einsum string: 'ij, jk'
// Subscripts are 'ij' and 'jk'
// Subscript labels are 'i', 'j', and 'k'

// This is a pre-processor class that maps subscript labels to a dimension value, etc.
template <typename T>
class EinsumComputePreprocessor final {
 public:
  EinsumComputePreprocessor(const std::string& einsum_equation,
                            const std::vector<const Tensor*>& inputs,
                            const AllocatorPtr& allocator);

  // Get the output dims of the op's output
  const std::vector<int64_t>& GetOutputDims() const;

  // Process all input tensors so that they are all of the same rank and easy to perfrom multiplication and reduction
  // along the way
  std::vector<Tensor>& GetPreprocessedTensors();

  // For each subscript index, get the last input the axis correspodning to the subscript index was seen in
  const std::vector<int64_t>& GetMappedSubscriptIndicesToLastInputIndex() const;

  // For each subscript index, get the index corresponding to the output's shape
  const std::vector<int64_t>& GetMappedSubscriptIndicesToOutputindices() const;

  // Get the number of subscript labels in the einsum equation
  const int64_t GetNumSubscriptLabels() const;

 private:
  void CollectMetadata();

  // A function to process bradcasted dims (ellipsis) of inputs that they occur in
  void PostProcessBroadcastedDims();

  // Check if the Einsum equation has an explicit form (equation string contains "->")
  // If it is of explicit form, parse the output subscript (substring following "->")
  // If it is of implicit form (equation string does not contain "->"), compose the output subscript
  // If the output subscript is an empty string, the result is a scalar
  void ParseOrCreateOutputSubscript();

  void CalculateOutputShape();

  void PreprocessInputs();

  // private members

  // Einsum equation
  std::string einsum_equation_;

  // In explicit form, holds the left side of the einsum equation
  // (e.g.) Einsum equation = 'i,j->i', then left_equation_ = 'i,j'
  // In implicit form, holds the entire einsum equation
  // (e.g.) Einsum equation = 'i,j', then left_equation_ = 'i,j'
  std::string left_equation_;

  // Holds constructed or parsed output subscript
  std::string right_equation_;

  // Flag indicating if the op is being used in explicit form
  bool is_explicit_ = false;

  // Flag indicating if einsum equation has an ellipsis (requests broadcasting support if so)
  bool has_ellipses_ = false;

  // The number of dims that encompasses an "ellipsis"
  size_t num_of_ellipsis_dims_ = 0;

  // All original inputs to the op
  const std::vector<const Tensor*>& inputs_;

  // All preprocessed inputs
  std::vector<Tensor> preprocessed_inputs_;

  // Count of unique subscript labels
  // E.g. 1 : With equation -> 'ij, jk -> ik'
  // num_subscript_labels_ = 3 (i, j, k)
  // E.g. 2 : With equation -> '...ij', 'jk' -> '...ik'
  // num_subscript_labels_ = 3 (i, j, k) + number of dims specified by an ellipsis (across all inputs)
  int64_t num_subscript_labels_ = 0;

  // Hold the count corresponding to the letter seen
  // `0` means the corresponding letter wasn't seen at all
  std::array<int64_t, EinsumOp::num_of_letters> letter_to_count_;

  // Hold the assigned index corresponding to the letter seen
  // `-1` means the corresponding letter wasn't seen at all
  std::array<int64_t, EinsumOp::num_of_letters> letter_to_index_;

  // TODO: Reserve appropriately for the following vectors
  // Holds the input index of the last input to have the index correpesponding to the subscript label
  // If the value is `-1`, then the subscript label is never seen (or) it appears in the output
  std::vector<int64_t> index_to_last_input_;

  // Hold the dim value of the index correpesponding to the subscript label
  // `-1` means the corresponding label wasn't seen at all
  std::vector<int64_t> index_to_dim_value_;

  // Holds the final calculated output dimensions
  std::vector<int64_t> output_dims_;

  // TODO: Fill in description
  std::vector<std::vector<int64_t>> input_dim_indices_to_subscript_indices_;

  // TODO: Fill in description
  std::vector<int64_t> index_to_output_indices_;

  // Allocator to use for ad-hoc tensor buffer allocation
  const AllocatorPtr& allocator_;
};

// This method does the heavy-lifting compute portion of Einsum
template <typename T>
Status EinsumTypedProcessor(OpKernelContext* ctx, const std::string& equation);

}  // namespace onnxruntime
