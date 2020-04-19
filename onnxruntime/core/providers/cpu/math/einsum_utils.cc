#include "einsum_utils.h"

namespace onnxruntime {

namespace EinsumOp {

// This helps decide if we need to apply (and pay the cost) of a Transpose
static bool IsTransposeRequired(size_t input_rank, const std::vector<size_t>& permutation) {
  ORT_ENFORCE(input_rank == permutation.size(), "The rank of the input must match permutation size for Transpose");
  
  // No transpose required for scalars
  if (input_rank == 0) {
    return false; 
  }

  // Weeds out cases where permutation is something like [0, 1, 2] for a 3D input and so on
  bool transpose_required = false;
  for (size_t i = 0; i < input_rank; ++i) {
    if (permutation[i] != i) {
      transpose_required = true;
      break;
    }
  }

  return transpose_required;
}
     // We have an the result in an output "candidate". Now we have to copy the contents in its buffer
// into the buffer of the actual output given to us by the execution frame
// We need to do this because the buffer owned by the output tensor of the op could be user provided buffer
static void CopyOutputCandidateIntoOpOutout(Tensor& output, Tensor& candidate) {
  ORT_ENFORCE(output.SizeInBytes() == candidate.SizeInBytes(),
              "Einsum op: The candidate output does not match the actual output's shape");
  // There are no string tensors - so safely use memcpy
  memcpy(output.MutableDataRaw(), candidate.DataRaw(), candidate.SizeInBytes());
}
// Here we take a "candidate output"(candidate output is a tensor that is a permutation and / or a reshape away from the final output),
// and after a few operations to get it to the required output structure, copy it to the op's output
template <typename T>
static void FinalizeOutput(Tensor& candidate_output, const std::vector<int64_t>& subscript_indices_in_candidate,
                           const std::vector<int64_t>& subscript_indices_to_output_indices,
                           Tensor& output, const std::vector<int64_t>& output_dims, const AllocatorPtr& allocator) {
  auto output_rank = output_dims.size();
  ORT_ENFORCE(output_rank == subscript_indices_in_candidate.size());

  // Identtify the permutation required by the op's output
  std::vector<size_t> output_permutation;
  output_permutation.resize(output_rank, 0);
  for (size_t iter = 0; iter < subscript_indices_in_candidate.size(); ++iter) {
    auto output_index = subscript_indices_to_output_indices[subscript_indices_in_candidate[iter]];
    ORT_ENFORCE(output_index != -1);
    output_permutation[output_index] = iter;
  }

  // Transpose to the required final output order
  // (Identify no-op transposes and prevent triggering the transpose)
  if (IsTransposeRequired(candidate_output.Shape().GetDims().size(), output_permutation)) {
    candidate_output = Transpose(candidate_output, output_permutation, allocator);  
  }

  // Change the shape to the required final output shape
  CreateReshapedView(candidate_output, output_dims);

  // Copy the output candidate into the op's output
  CopyOutputCandidateIntoOpOutout(output, candidate_output);
}

// Processes Einsum operands in a pair-wise fashion
// Employs Transpose, ReduceSum, and MatMul under the hood
// to achieve MatMul(a, b) and reduces (by summing) along specified axes
template <typename T>
static Tensor PairwiseOperandProcess(Tensor& left, Tensor& right,
                                     const std::vector<int64_t>& reduce_dims,
                                     concurrency::ThreadPool* tp,
                                     const AllocatorPtr& allocator,
                                     const EinsumComputePreprocessor& einsum_compute_preprocessor,
                                     bool is_final_pair, Tensor& final_output) {
  // Make copies as we may mutate the tensor objects downstream
  std::vector<int64_t> left_dims = left.Shape().GetDims();
  std::vector<int64_t> right_dims = right.Shape().GetDims();

  int64_t left_rank = static_cast<int64_t>(left_dims.size());
  int64_t right_rank = static_cast<int64_t>(right_dims.size());

  // If the following error condition is hit, it is most likely a pre-processing bug
  ORT_ENFORCE(left_rank == right_rank, "Ranks of pair-wise operands must be equal");

  // Following vectors hold:
  // lro: dim indices that are present in left, right, and reduce_dims
  // lo: dim indices that are present in left and reduce_dims
  // ro: dim indices that are present in right and reduce_dims
  std::vector<size_t> lro;
  std::vector<size_t> lo;
  std::vector<size_t> ro;

  // Maintain sizes to create reshaped "views"
  int64_t lro_size = 1;
  int64_t lo_size = 1;
  int64_t ro_size = 1;
  int64_t reduced_size = 1;

  size_t reduce_dims_iter = 0;
  size_t reduce_dims_size = reduce_dims.size();

  for (int64_t i = 0; i < left_rank; ++i) {
    int64_t left_dim = left_dims[i];
    int64_t right_dim = right_dims[i];

    bool has_left_dim = left_dim > 1;    // non-trivial dimension (dim_value != 1)
    bool has_right_dim = right_dim > 1;  // non-trivial dimension (dim_value != 1)

    if (reduce_dims_iter < reduce_dims_size && reduce_dims[reduce_dims_iter] == i) {  // This dimension is to be reduced after this pair-wise operation
      ++reduce_dims_iter;
      if (has_left_dim && has_right_dim) {  // Both inputs have non-trivial dim values along this dimension
        // Both the left and right operands have non-trivial dimension value along this axis
        // They must be equal
        ORT_ENFORCE(left_dim == right_dim, "Einsum op: Input dimensions must be equal along an axis to be reduced across all inputs");
        reduced_size *= left_dim;
      } else if (has_left_dim) {  // if it is only in one of left and right, we can reduce right away
        left = ReduceSum<T>(left, i, allocator, tp);
      } else if (has_right_dim) {
        right = ReduceSum<T>(right, i, allocator, tp);
      }
    } else {  // This dimension is not reduced (i.e.) it appears in the output after processing these 2 operands
      // Both the left and right operands have non-trivial dimension value along this axis
      // They must be equal
      if (has_left_dim && has_right_dim) {
        ORT_ENFORCE(left_dim == right_dim, "Einsum op: Input shapes do not align");
        lro.push_back(i);
        lro_size *= left_dim;
      } else if (has_left_dim) {
        // The left operand has non-trivial dimension value
        lo.push_back(i);
        lo_size *= left_dim;
      } else {
        // The right operand may or may not have non-trivial dim value
        // If it has trivial dim value (1),
        // it will just form a trailing dimension for the right operand
        ro.push_back(i);
        ro_size *= right_dim;
      }
    }
  }

  // Permutate the left operand so that the axes order go like this: [lro, lo, reduce_dims, ro]
  std::vector<size_t> left_permutation;
  left_permutation.reserve(lro.size() + lo.size() + reduce_dims.size() + ro.size());
  left_permutation.insert(left_permutation.end(), lro.begin(), lro.end());
  left_permutation.insert(left_permutation.end(), lo.begin(), lo.end());
  left_permutation.insert(left_permutation.end(), reduce_dims.begin(), reduce_dims.end());
  left_permutation.insert(left_permutation.end(), ro.begin(), ro.end());
  if (IsTransposeRequired(left.Shape().GetDims().size(), left_permutation)) {
    left = Transpose(left, left_permutation, allocator);
  }
  CreateReshapedView(left, {lro_size, lo_size, reduced_size});

  // Permutate the right operand so that the axes order go like this: [lro, reduce_dims, ro, lo]
  std::vector<size_t> right_permutation;
  right_permutation.reserve(lro.size() + lo.size() + reduce_dims.size() + ro.size());
  right_permutation.insert(right_permutation.end(), lro.begin(), lro.end());
  right_permutation.insert(right_permutation.end(), reduce_dims.begin(), reduce_dims.end());
  right_permutation.insert(right_permutation.end(), ro.begin(), ro.end());
  right_permutation.insert(right_permutation.end(), lo.begin(), lo.end());
  if (IsTransposeRequired(right.Shape().GetDims().size(), right_permutation)) {
    right = Transpose(right, right_permutation, allocator);
  }
  CreateReshapedView(right, {lro_size, reduced_size, ro_size});

  // Calculate output size
  // Output shape will be, by rules of matmul:
  // because we are multiplying two tensors of shapes [lro, lo, reduce_dims] , [lro, reduce_dims, ro]
  // [dim_value of `lro` dims,
  //  dim_value of `lo` dims,
  // `1` for each of the `reduce_dims`,
  // dim_value of `ro` dims]
  std::vector<int64_t> output_dims;
  output_dims.reserve(lro.size() + lo.size() + reduce_dims.size() + ro.size());
  for (size_t i = 0; i < lro.size(); ++i) {
    output_dims.push_back(left_dims[lro[i]]);
  }
  for (size_t i = 0; i < lo.size(); ++i) {
    output_dims.push_back(left_dims[lo[i]]);
  }

  if (!is_final_pair) {
    for (size_t i = 0; i < reduce_dims.size(); ++i) {
      output_dims.push_back(1);  // reduced dimensions will have a value 1 in it
    }
  }

  for (size_t i = 0; i < ro.size(); ++i) {
    output_dims.push_back(right_dims[ro[i]]);
  }

  std::vector<int64_t> current_subscript_order;

  // Calculate output permutation
  // After the MatMul op, the because the two operands have been permutated,
  // the output is permutated as well with respect to the original ordering of the axes.
  // The permutated order will be the dims in: [lro, lo, reduced_dims, ro]
  // Hence invert the permutation by a permutation that puts the axes in the same ordering
  std::vector<size_t> output_permutation;
  if (!is_final_pair) {  // If this is not the final pair, we need to permutate the result to match the pre-fixed order for the next iteration
    output_permutation.resize(lro.size() + lo.size() + reduce_dims.size() + ro.size(), 0);
    size_t iter = 0;
    for (size_t i = 0; i < lro.size(); ++i) {
      output_permutation[lro[i]] = iter++;
    }
    for (size_t i = 0; i < lo.size(); ++i) {
      output_permutation[lo[i]] = iter++;
    }
    for (size_t i = 0; i < reduce_dims.size(); ++i) {
      output_permutation[reduce_dims[i]] = iter++;
    }
    for (size_t i = 0; i < ro.size(); ++i) {
      output_permutation[ro[i]] = iter++;
    }
  } else {
    current_subscript_order.reserve(lro.size() + lo.size() + ro.size());
    current_subscript_order.insert(current_subscript_order.end(), lro.begin(), lro.end());
    current_subscript_order.insert(current_subscript_order.end(), lo.begin(), lo.end());
    current_subscript_order.insert(current_subscript_order.end(), ro.begin(), ro.end());
  }

  // Multiply the mutated inputs
  auto output = MatMul<T>(left, right, allocator, tp);

  // Bring the output back to the original rank (MatMul operated on 3D inputs)
  CreateReshapedView(output, output_dims);

  if (!is_final_pair) {  // This is not the final pair - so bring the axes order to what the inputs conformed to
    if (IsTransposeRequired(output.Shape().GetDims().size(), output_permutation)) {
      output = Transpose(output, output_permutation, allocator);
    }
  } else {  // This is the final pair - Transpose directly to the output ordering required and copy the contents to the op's output
    FinalizeOutput<T>(output, current_subscript_order,
                      einsum_compute_preprocessor.GetMappedSubscriptIndicesToOutputindices(), final_output,
                      einsum_compute_preprocessor.GetOutputDims(), allocator);
  }

  return output;
}

}  // namespace EinsumOp

EinsumComputePreprocessor::EinsumComputePreprocessor(EinsumEquationPreprocessor& einsum_equation_preprocessor,
                                                     const std::vector<const Tensor*>& inputs,
                                                     AllocatorPtr allocator)
    : einsum_equation_preprocessor_(einsum_equation_preprocessor), inputs_(inputs), allocator_(allocator) {
  letter_to_index_.fill(-1);

  letter_to_count_.fill(0);

  ProcessSubscripts();

  PostProcessBroadcastedDims();

  ParseOrCreateOutputSubscript();

  CalculateOutputShape();

  PreprocessInputs();
}

const std::vector<int64_t>& EinsumComputePreprocessor::GetOutputDims() const {
  return output_dims_;
}

std::vector<Tensor>& EinsumComputePreprocessor::GetPreprocessedTensors() {
  return preprocessed_inputs_;
}

const std::vector<int64_t>& EinsumComputePreprocessor::GetMappedSubscriptIndicesToLastInputIndex() const {
  return subscript_indices_to_last_input_;
}

const std::vector<int64_t>& EinsumComputePreprocessor::GetMappedSubscriptIndicesToOutputindices() const {
  return subscript_indices_to_output_indices_;
}

int64_t EinsumComputePreprocessor::GetNumSubscriptIndices() const {
  return num_subscript_indices_;
}

void EinsumComputePreprocessor::ProcessSubscripts() {
  const auto& left_equation_split = einsum_equation_preprocessor_.left_equation_split_;
  ORT_ENFORCE(left_equation_split.size() == inputs_.size(), "Number of subscripts in the input equation does not match number of input tensors");

  int64_t input_index = 0;

  // Holds mapping between input indices to its corresponding subscript labels for each input
  input_subscript_indices_.reserve(inputs_.size());

  // We arbitrarily reserve space for 10 values as we don't expect to see any input with rank >10
  // which would make num_subscript_indices_ > 10
  subscript_indices_to_last_input_.reserve(10);
  subscript_indices_to_dim_value_.reserve(10);

  for (const auto& subscript : left_equation_split) {
    const auto& shape = inputs_[input_index]->Shape();
    const auto& dims = shape.GetDims();
    size_t rank = dims.size();
    size_t dim_counter = 0;

    std::vector<int64_t> current_subscript_indices;
    current_subscript_indices.reserve(rank);

    // Temp variables to deal with "ellipsis" in the input
    bool is_in_middle_of_ellipsis = false;
    int64_t ellipsis_char_count = 0;

    // Iterate through all subscript labels in the subscript
    for (auto subscript_label : subscript) {
      // Broadcasting based dims
      if (subscript_label == '.') {
        is_in_middle_of_ellipsis = true;
        // Make sure there aren't more than 3 '.'s in the current subscript
        ORT_ENFORCE(++ellipsis_char_count <= 3, "Found a '.' not part of an ellipsis in input: ", input_index);

        // We have seen all 3 '.'s. We can safely process the ellipsis now.
        if (ellipsis_char_count == 3) {
          is_in_middle_of_ellipsis = false;

          // Example for the following line of code
          // Subscript "...ij" for an input of rank 6
          // num_of_ellipsis_dims = 6 - 5 + 3 = 4
          int64_t current_num_of_ellipsis_dims = rank - subscript.length() + 3;
          ORT_ENFORCE(current_num_of_ellipsis_dims >= 0,
                      "Einsum subscripts string contains too many subscript labels when compared to the rank of the input ",
                      input_index);

          // Theoretically, current_num_of_ellipsis_dims could be 0
          // Example: For an input of rank 2 paired with a subscript "...ij"
          if (current_num_of_ellipsis_dims != 0) {
            // We have seen a ellipsis before - make sure ranks align as per the ONNX spec -
            // "Ellipsis must indicate a fixed number of dimensions."
            if (num_of_ellipsis_dims_ != 0) {
              ORT_ENFORCE(num_of_ellipsis_dims_ == static_cast<size_t>(current_num_of_ellipsis_dims),
                          "Ellipsis must indicate a fixed number of dimensions across all inputs");
            } else {
              num_of_ellipsis_dims_ = static_cast<size_t>(current_num_of_ellipsis_dims);
            }

            // We reserve '26' for broadcasted dims as we only allow 'a' - 'z' (0 - 25) for non-broadcasted dims
            // We will assign appropriate indices (based o number of dimensions the ellipsis corresponds to) during broadcasting related post-processing
            for (size_t i = 0; i < num_of_ellipsis_dims_; ++i) {
              current_subscript_indices.push_back(26);
            }

            // Offset 'dim_counter' by number of dimensions the ellipsis corresponds to
            dim_counter += num_of_ellipsis_dims_;
          }
        }
      } else {  // regular letter based dimension -> 'i', 'j', etc.
        ORT_ENFORCE(!is_in_middle_of_ellipsis, "Found '.' not part of an ellipsis in input: ", input_index);

        ORT_ENFORCE(subscript_label >= 'a' && subscript_label <= 'z',
                    "The only subscript labels allowed are lowercase letters (a-z)");

        auto letter_index = static_cast<int64_t>(subscript_label - 'a');
        auto dim_value = dims[dim_counter];

        // Subscript label not found in global subscript label array
        // Hence add it to both local and global subscript arrays
        if (letter_to_count_[letter_index] == 0) {
          letter_to_index_[letter_index] = num_subscript_indices_++;
          subscript_indices_to_dim_value_.push_back(dim_value);
          subscript_indices_to_last_input_.push_back(input_index);
        } else {  // This subscript label has been seen in atleast one other operand's subscript
          // It must be equal unless one of them is a 1 (Numpy allows this)
          auto mapped_index = letter_to_index_[letter_index];

          subscript_indices_to_last_input_[mapped_index] = input_index;

          if (subscript_indices_to_dim_value_[mapped_index] != dim_value) {
            // Set the value to the new dim value if the value is 1 in the map
            if (subscript_indices_to_dim_value_[mapped_index] == 1) {
              subscript_indices_to_dim_value_[mapped_index] = dim_value;
            } else {
              ORT_ENFORCE(dim_value == 1,
                          "Einsum operands could not be broadcast together. "
                          "Please check input shapes/equation provided."
                          "Input shape of operand ",
                          input_index, " is incompatible. The shape is ", shape);
            }
          }
        }

        ++letter_to_count_[letter_index];

        current_subscript_indices.push_back(letter_to_index_[letter_index]);

        ORT_ENFORCE(++dim_counter <= rank,
                    "Einsum subscripts string contains too many subscript labels when compared to the rank of the input ",
                    input_index);
      }
    }

    // If no broadcasting is requested, the number of subscript labels (dim_counter) should match input rank
    if (num_of_ellipsis_dims_ == 0) {
      ORT_ENFORCE(dim_counter == rank,
                  "Einsum subscripts does not contain enough subscript labels and there is no ellipsis for input ", input_index);
    }

    input_subscript_indices_.push_back(std::move(current_subscript_indices));
    ++input_index;
  }
}

void EinsumComputePreprocessor::PostProcessBroadcastedDims() {
  // Pay the cost of this function only if we saw an ellipsis in any of the inputs
  if (num_of_ellipsis_dims_ > 0) {
    // extend the number of subscript labels to include each ellipsis dim as
    // theoretically each ellipsis dim does correspond to a "virtual" subscript label
    num_subscript_indices_ += num_of_ellipsis_dims_;

    // We are going to assign the broadcasted dims outermost subscript indices (i.e.) 0 -> num_of_ellipsis_dims_ - 1
    // as most likely bradcasted dims will be batch dimensions (i.e.) outermost dimensions and hence we don't have to pay
    // transposing while "homogenizing" the input

    // Hence offset all subscript indices by num_of_ellipsis_dims_
    for (size_t i = 0; i < EinsumOp::num_of_letters; ++i) {
      if (letter_to_index_[i] != -1) {
        letter_to_index_[i] += num_of_ellipsis_dims_;
      }
    }

    std::vector<int64_t> temp_index_to_last_input(num_subscript_indices_, -1);
    for (size_t i = 0; i < subscript_indices_to_last_input_.size(); ++i) {
      temp_index_to_last_input[i + num_of_ellipsis_dims_] = subscript_indices_to_last_input_[i];
    }
    subscript_indices_to_last_input_ = std::move(temp_index_to_last_input);

    std::vector<int64_t> temp_index_to_dim_value(num_subscript_indices_, -1);
    for (size_t i = 0; i < subscript_indices_to_dim_value_.size(); ++i) {
      temp_index_to_dim_value[i + num_of_ellipsis_dims_] = subscript_indices_to_dim_value_[i];
    }
    subscript_indices_to_dim_value_ = std::move(temp_index_to_dim_value);

    for (size_t i = 0; i < input_subscript_indices_.size(); ++i) {
      auto& current_input_dim_indices_to_subscript_indices = input_subscript_indices_[i];
      std::vector<int64_t> temp_current_input_dim_indices_to_subscript_indices;
      temp_current_input_dim_indices_to_subscript_indices.reserve(current_input_dim_indices_to_subscript_indices.size());

      const auto& dims = inputs_[i]->Shape().GetDims();
      auto rank = dims.size();

      size_t dim_iter = 0;
      size_t num_broadcasted_indices = 0;
      while (dim_iter < current_input_dim_indices_to_subscript_indices.size()) {
        auto value = current_input_dim_indices_to_subscript_indices[dim_iter];
        if (value == 26) {  //This is a broadcasted dim
          ORT_ENFORCE(num_broadcasted_indices < num_of_ellipsis_dims_);
          temp_current_input_dim_indices_to_subscript_indices.push_back(static_cast<int64_t>(num_broadcasted_indices));
          subscript_indices_to_last_input_[num_broadcasted_indices] = i;

          // This is the first time we are seeing this broadcasted dim
          if (subscript_indices_to_dim_value_[num_broadcasted_indices] == -1) {
            subscript_indices_to_dim_value_[num_broadcasted_indices] = dims[dim_iter];
          } else {  // We have seen this broadcasted dim before
            // Check if the previous value is equal to the current value
            if (subscript_indices_to_dim_value_[num_broadcasted_indices] != dims[dim_iter]) {
              // If they are not equal, one of them needs to be 1
              if (subscript_indices_to_dim_value_[num_broadcasted_indices] == 1) {
                subscript_indices_to_dim_value_[num_broadcasted_indices] = dims[dim_iter];
              } else {
                ORT_ENFORCE(dims[dim_iter] == 1, "Given inputs are not broadcastable");
              }
            }
          }
          ++num_broadcasted_indices;
        } else {  // This is a regular dim - offset it by number of broadcasted dims
          temp_current_input_dim_indices_to_subscript_indices.push_back(value + static_cast<int64_t>(num_of_ellipsis_dims_));
        }
        ++dim_iter;
      }
      // Shouldn't hit this error - just a sanity check
      ORT_ENFORCE(dim_iter == rank);
      current_input_dim_indices_to_subscript_indices = std::move(temp_current_input_dim_indices_to_subscript_indices);
    }
  }
}

void EinsumComputePreprocessor::ParseOrCreateOutputSubscript() {
  // Explicit form - no op as the output would have been parsed while parsing the input
  if (einsum_equation_preprocessor_.is_explicit_) {
    // Make sure that the given explicit equation contains an ellipsis if the input contains ellipses in them
    if (num_of_ellipsis_dims_ > 0) {
      ORT_ENFORCE(einsum_equation_preprocessor_.right_equation_.find("...") != std::string::npos,
                  "Inputs have ellipses in them but the provided output subscript does not contain an ellipsis");
    }
    return;
  }

  // Implicit form - construct the output subscript
  std::stringstream output_equation;

  // If the an ellipsis was seen in the input, add it
  if (num_of_ellipsis_dims_ > 0) {
    output_equation << "...";
  }

  // In sorted order of letters, add those letters that were seen only once in the input
  size_t iter = 0;
  for (const auto& count : letter_to_count_) {
    if (count == 1) {
      output_equation << static_cast<char>('a' + iter);
    }
    ++iter;
  }

  einsum_equation_preprocessor_.right_equation_ = output_equation.str();
}

void EinsumComputePreprocessor::CalculateOutputShape() {
  // Iterate through all subscript labels in the output subscript
  bool is_in_middle_of_ellipsis = false;
  int64_t ellipsis_char_count = 0;

  subscript_indices_to_output_indices_.resize(num_subscript_indices_, -1);

  std::array<int64_t, EinsumOp::num_of_letters> output_letter_to_count;
  output_letter_to_count.fill(0);

  // Arbitrarily reserve some space as we don't expect rank of output to be > 10 (pay re-allocation cost if it is)
  output_dims_.reserve(10);

  int64_t output_dim_counter = 0;
  for (auto subscript_label : einsum_equation_preprocessor_.right_equation_) {
    if (subscript_label == '.') {
      is_in_middle_of_ellipsis = true;
      // Make sure there aren't more than 3 '.'s in the current subscript
      ORT_ENFORCE(++ellipsis_char_count <= 3, "Found a '.' not part of an ellipsis in the output");
      if (ellipsis_char_count == 3) {  // Ellipsis is complete. Process it.
        is_in_middle_of_ellipsis = false;
        for (size_t i = 0; i < num_of_ellipsis_dims_; ++i) {
          output_dims_.push_back(subscript_indices_to_dim_value_[i]);
          // The ellipsis is seen in the output and hence the corresponding dims are to not be reduced
          subscript_indices_to_last_input_[i] = -1;
          subscript_indices_to_output_indices_[i] = output_dim_counter++;
        }
      }
    } else {
      ORT_ENFORCE(!is_in_middle_of_ellipsis, "Found '.' not part of an ellipsis in the output");

      ORT_ENFORCE(subscript_label >= 'a' && subscript_label <= 'z',
                  "The only subscript labels allowed are lowercase letters (a-z)");

      auto letter_index = static_cast<int64_t>(subscript_label - 'a');

      ORT_ENFORCE(output_letter_to_count[letter_index] == 0,
                  "Output subscript contains repeated letters");
      ++output_letter_to_count[letter_index];

      auto mapped_index = letter_to_index_[letter_index];
      ORT_ENFORCE(mapped_index != -1,
                  "Output subscript contains letters not seen in the inputs");

      output_dims_.push_back(subscript_indices_to_dim_value_[mapped_index]);

      // Reset the last input index for this subscript label
      // given that it is seen in the output and hence can't be reduced
      subscript_indices_to_last_input_[mapped_index] = -1;

      subscript_indices_to_output_indices_[mapped_index] = output_dim_counter++;
    }
  }
}

void EinsumComputePreprocessor::PreprocessInputs() {
  preprocessed_inputs_.reserve(inputs_.size());
  // As part of input preprocessing we "homogenize" them by
  // 1) Making them all of the same rank
  // 2) The axes order in all the inputs are to be made the same
  int64_t input_iter = 0;
  for (const auto* input : inputs_) {
    // We need to make a copy of the op's inputs because they will be mutated along the op's compute steps
    Tensor preprocessed = input->Clone(allocator_);
    const auto& input_dims = input->Shape().GetDims();
    const auto& current_subscript_indices = input_subscript_indices_[input_iter];

    // If all has gone well, we will have a subscript index (subscript label) for each dim of the input
    ORT_ENFORCE(input_dims.size() == current_subscript_indices.size(),
                "Rank of the input must match number of subscript labels corresponding to the input");

    std::vector<int64_t> subscript_indices_to_input_index(num_subscript_indices_, -1);

    // This is the input dims after re-ordering so that all inputs have same axes order
    std::vector<int64_t> homogenized_input_dims(num_subscript_indices_, 1);

    // Preprocessed dim rank may not be the same as original input rank if we need to parse diagonals along the way
    // (which reduces rank in the preprocessed input by 1 for each diagonal we parse)
    int64_t dim_index_in_preprocessed_input = 0;
    int64_t dim_index_in_original_input = 0;

    // iterate through all subscript inidices in this input
    for (const auto& subscript_index : current_subscript_indices) {
      if (subscript_indices_to_input_index[subscript_index] == -1) {  // This is the first time we are seeing this subscript label in this input
        subscript_indices_to_input_index[subscript_index] = dim_index_in_preprocessed_input++;
        homogenized_input_dims[subscript_index] = input_dims[dim_index_in_original_input];
      } else {  // Diagonal needs to be parsed along the repeated axes
        preprocessed = EinsumOp::Diagonal(preprocessed, subscript_indices_to_input_index[subscript_index], dim_index_in_preprocessed_input,
                                          allocator_);
      }
      ++dim_index_in_original_input;
    }

    std::vector<size_t> permutation;
    permutation.reserve(input_dims.size());
    for (auto& d : subscript_indices_to_input_index) {
      if (d != -1) {
        permutation.push_back(static_cast<size_t>(d));
      }
    }

    // (Identify no-op transpose and prevent triggering the transpose)
    if (EinsumOp::IsTransposeRequired(preprocessed.Shape().GetDims().size(), permutation)) {
      preprocessed = EinsumOp::Transpose(preprocessed, permutation, allocator_);
    }

    EinsumOp::CreateReshapedView(preprocessed, homogenized_input_dims);

    preprocessed_inputs_.push_back(std::move(preprocessed));

    ++input_iter;
  }
}

// Templated core Einsum logic
template <typename T>
Status EinsumTypedComputeProcessor(OpKernelContext* context,
                                   AllocatorPtr allocator,
                                   EinsumComputePreprocessor& einsum_compute_preprocessor) {
  const auto& mapped_indices_to_last_input_index = einsum_compute_preprocessor.GetMappedSubscriptIndicesToLastInputIndex();

  auto& preprocessed_inputs = einsum_compute_preprocessor.GetPreprocessedTensors();

  auto num_subscript_labels = einsum_compute_preprocessor.GetNumSubscriptIndices();

  const auto& output_dims = einsum_compute_preprocessor.GetOutputDims();

  auto* output = context->Output(0, output_dims);

  auto num_inputs = context->InputCount();

  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();

  // Preprocess the first input so as to reduce any dims that only it has
  Tensor result;

  {
    std::vector<int64_t> reduced_dims;
    std::vector<int64_t> preserved_dims;            // dims which were not reduced
    std::vector<int64_t> preserved_shape;           // shape pertaining to only the dims that were preserved (not reduced)
    reduced_dims.reserve(num_subscript_labels);     // num_subscript_labels is the upper bound. No harm in over-reserving.
    preserved_dims.reserve(num_subscript_labels);   // num_subscript_labels is the upper bound. No harm in over-reserving.
    preserved_shape.reserve(num_subscript_labels);  // num_subscript_labels is the upper bound. No harm in over-reserving.

    auto& first_input = preprocessed_inputs[0];
    const auto& dims = first_input.Shape().GetDims();
    for (int64_t i = 0; i < num_subscript_labels; ++i) {
      if (mapped_indices_to_last_input_index[i] == 0) {
        reduced_dims.push_back(i);
      } else {
        preserved_dims.push_back(i);
        preserved_shape.push_back(dims[i]);
      }
    }

    // Reduce the dims that are last seen in the first input alone
    if (reduced_dims.size() != 0) {
      first_input = EinsumOp::ReduceSum<T>(first_input, reduced_dims, allocator, tp);
    }

    // Finalize the output at this stage if num_inputs == 1
    if (num_inputs == 1) {
      // Create reshaped view to squeeze out the reduced dims
      EinsumOp::CreateReshapedView(first_input, preserved_shape);

      // Finalize the output by applying any transpose required to get it to the required output ordering and move it to the op's output
      EinsumOp::FinalizeOutput<T>(first_input, preserved_dims, einsum_compute_preprocessor.GetMappedSubscriptIndicesToOutputindices(), *output, output_dims, allocator);

      return Status::OK();
    } else {  // Assign the first tensor to result to proceed further
      result = std::move(first_input);
    }
  }

  // Process the operands in a pair-wise fashion
  {
    bool is_final_pair = false;
    // Keep processing each input pair-wise
    for (int input = 1; input < num_inputs; ++input) {
      std::vector<int64_t> reduced_dims;
      reduced_dims.reserve(num_subscript_labels);  // num_subscript_labels is the upper bound. No harm in over-reserving by a small margin.
      for (int64_t dim = 0; dim < num_subscript_labels; ++dim) {
        if (mapped_indices_to_last_input_index[dim] == input) {
          // This is the last input we are seeing this dimension (and it doesn't occur in the output), so reduce along the dimension
          reduced_dims.push_back(dim);
        }
      }
      if (input == num_inputs - 1) {
        is_final_pair = true;
      }
      result = EinsumOp::PairwiseOperandProcess<float>(result, preprocessed_inputs[input], reduced_dims, tp, allocator, einsum_compute_preprocessor, is_final_pair, *output);
    }
  }
  return Status::OK();
}

// Explicit template instantiation
template Status EinsumTypedComputeProcessor<float>(OpKernelContext* context, AllocatorPtr allocator, EinsumComputePreprocessor& einsum_compute_preprocessor);

}  // namespace onnxruntime
