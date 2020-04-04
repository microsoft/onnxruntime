#include "einsum_utils.h"

namespace onnxruntime {

namespace EinsumOp {

// Here we take a "candidate output"(candidate output is a tensor that is a permutation and / or a reshape away from the final output),
// and after a few operations, move it to the op's output
template <typename T>
static void FinalizeOutput(Tensor& candidate_output, const std::vector<int64_t>& subscript_indices_in_candidate,
                           const std::vector<int64_t>& subscript_indices_to_output_indices,
                           Tensor& output, const std::vector<int64_t>& output_dims, const AllocatorPtr& allocator) {
  auto output_rank = output_dims.size();
  ORT_ENFORCE(output_rank == subscript_indices_in_candidate.size());

  std::vector<size_t> output_permutation;
  output_permutation.resize(output_rank, 0);

  for (size_t iter = 0; iter < subscript_indices_in_candidate.size(); ++iter) {
    auto output_index = subscript_indices_to_output_indices[subscript_indices_in_candidate[iter]];
    ORT_ENFORCE(output_index != -1);
    output_permutation[output_index] = iter;
  }

  // Transpose to the required final output order
  // TODO: Identify no-op transposes and prevent triggering the transpose
  auto transposed = Transpose(candidate_output, output_permutation, allocator);

  // Change the shape to the required final output shape
  CreateReshapedView(transposed, output_dims);

  // Move the transposed and reshaped output to the final output
  output = std::move(transposed);
}

// Processes Einsum operands in a pair-wise fashion
// Employs Transpose, ReduceSum, and MatMul under the hood
// to achieve MatMul(a, b) and reduces (by summing) along specified axes
template <typename T>
static Tensor PairwiseOperandProcess(Tensor& left, Tensor& right,
                                     const std::vector<int64_t>& reduce_dims,
                                     concurrency::ThreadPool* tp,
                                     const AllocatorPtr& allocator,
                                     const EinsumComputePreprocessor<T>& einsum_preprocessor,
                                     bool is_final_pair, Tensor& final_output) {
  // Make copies as we may mutate the tensor objects downstream
  std::vector<int64_t> left_dims = left.Shape().GetDims();
  std::vector<int64_t> right_dims = right.Shape().GetDims();

  int64_t left_rank = static_cast<int64_t>(left_dims.size());
  int64_t right_rank = static_cast<int64_t>(right_dims.size());

  // If the following error condition is hit, it is most likely a pre-processing bug
  ORT_ENFORCE(left_rank == right_rank, "Ranks of pair-wise operands must be equal");

  // Follwing vectors hold:
  // lro: dim indices that are present in left, right, and reduce_dims
  // lo: dim indices that are present in left and reduce_dims
  // ro: dim indices that are present in right and reduce_dims
  std::vector<size_t> lro;
  std::vector<size_t> lo;
  std::vector<size_t> ro;

  // Maintain sizes to create reshaped "views" after permutating later
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

    if (reduce_dims_iter < reduce_dims_size && reduce_dims[reduce_dims_iter] == i) {  // reduce_dims will hold the dims to be reduced in a sorted fashion
      ++reduce_dims_iter;
      if (has_left_dim && has_right_dim) {
        // Both the left and right operands have non-trivial dimension value along this axis
        // They must be equal
        ORT_ENFORCE(left_dim == right_dim, "Einsum op: Input dimensions must be equal along an axis to be reduced across all inputs");
        reduced_size *= left_dim;
      } else if (has_left_dim) {  // if it is only in one of left and right, we can sum right away
        left = ReduceSum<T>(left, i, allocator);
      } else if (has_right_dim) {
        right = ReduceSum<T>(right, i, allocator);
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
  left = Transpose(left, left_permutation, allocator);
  CreateReshapedView(left, {lro_size, lo_size, reduced_size});

  // Permutate the right operand so that the axes order go like this: [lro, reduce_dims, ro, lo]
  std::vector<size_t> right_permutation;
  right_permutation.reserve(lro.size() + lo.size() + reduce_dims.size() + ro.size());
  right_permutation.insert(right_permutation.end(), lro.begin(), lro.end());
  right_permutation.insert(right_permutation.end(), reduce_dims.begin(), reduce_dims.end());
  right_permutation.insert(right_permutation.end(), ro.begin(), ro.end());
  right_permutation.insert(right_permutation.end(), lo.begin(), lo.end());
  right = Transpose(right, right_permutation, allocator);
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
    output = Transpose(output, output_permutation, allocator);
  } else {  // This is the final pair - Transpose directly to the output ordering required
    FinalizeOutput<T>(output, current_subscript_order,
                      einsum_preprocessor.GetMappedSubscriptIndicesToOutputindices(), final_output,
                      einsum_preprocessor.GetOutputDims(), allocator);
  }

  return output;
}

}  // namespace EinsumOp

template <typename T>
EinsumComputePreprocessor<T>::EinsumComputePreprocessor(const std::string& einsum_equation,
                                                        const std::vector<const Tensor*>& inputs,
                                                        const AllocatorPtr& allocator)
    : inputs_(inputs), einsum_equation_(einsum_equation), allocator_(allocator) {
  // Remove space characters
  einsum_equation_.erase(std::remove(einsum_equation_.begin(), einsum_equation_.end(), ' '),
                         einsum_equation_.end());

  auto mid_index = einsum_equation_.find("->");
  if (mid_index != std::string::npos) {
    // Separate right and left hand sides of the equation
    left_equation_ = einsum_equation_.substr(0, mid_index);
    right_equation_ = einsum_equation_.substr(mid_index + 2);
    is_explicit_ = true;
  } else {
    left_equation_ = einsum_equation_;
  };

  letter_to_index_.fill(-1);

  letter_to_count_.fill(0);

  CollectMetadata();

  PostProcessBroadcastedDims();

  ParseOrCreateOutputSubscript();

  CalculateOutputShape();

  PreprocessInputs();
}

template <typename T>
const std::vector<int64_t>& EinsumComputePreprocessor<T>::GetOutputDims() const {
  return output_dims_;
}

template <typename T>
std::vector<Tensor>& EinsumComputePreprocessor<T>::GetPreprocessedTensors() {
  return preprocessed_inputs_;
}

template <typename T>
const std::vector<int64_t>& EinsumComputePreprocessor<T>::GetMappedSubscriptIndicesToLastInputIndex() const {
  return index_to_last_input_;
}

template <typename T>
const std::vector<int64_t>& EinsumComputePreprocessor<T>::GetMappedSubscriptIndicesToOutputindices() const {
  return index_to_output_indices_;
}

template <typename T>
const int64_t EinsumComputePreprocessor<T>::GetNumSubscriptLabels() const {
  return num_subscript_labels_;
}

template <typename T>
void EinsumComputePreprocessor<T>::CollectMetadata() {
  std::stringstream str(left_equation_);
  std::string subscript;

  // Holds mapping between input indices to its corresponding subscript labels
  int64_t input_index = 0;

  input_dim_indices_to_subscript_indices_.reserve(inputs_.size());

  while (std::getline(str, subscript, ',')) {
    const auto& shape = inputs_[input_index]->Shape();
    const auto& dims = shape.GetDims();
    size_t rank = dims.size();
    size_t dim_counter = 0;

    std::vector<int64_t> current_input_dim_indices_to_subscript_indices_;
    current_input_dim_indices_to_subscript_indices_.reserve(rank);

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
          size_t current_num_of_ellipsis_dims = rank - subscript.length() + 3;
          ORT_ENFORCE(current_num_of_ellipsis_dims >= 0,
                      "Einsum subscripts string contains too many subscript labels when compared to the rank of the input ",
                      input_index);

          // Theoretically, current_num_of_ellipsis_dims could be 0
          // Example: For an input of rank 2 paired with a subscript "...ij"
          if (current_num_of_ellipsis_dims != 0) {
            // We have seen a ellipsis before - make sure ranks align as per the ONNX spec -
            // "Ellipsis must indicate a fixed number of dimensions."
            if (num_of_ellipsis_dims_ != 0) {
              ORT_ENFORCE(num_of_ellipsis_dims_ == current_num_of_ellipsis_dims,
                          "Ellipsis must indicate a fixed number of dimensions across all inputs");
            } else {
              num_of_ellipsis_dims_ = current_num_of_ellipsis_dims;
            }

            // We reserve '26' for broadcasted dims as we only allow 'a' - 'z' (0 - 25) for non-broadcasted dims
            // We will assign an appropriate indices during broadcasting related post-processing
            for (size_t i = 0; i < num_of_ellipsis_dims_; ++i) {
              current_input_dim_indices_to_subscript_indices_.push_back(26);
            }
          }
        }
      } else {  // regular letter based dimension -> 'i', 'j', etc.
        ORT_ENFORCE(!is_in_middle_of_ellipsis, "Found '.' not part of an ellipsis in input: ", input_index);

        ORT_ENFORCE(subscript_label >= 'a' && subscript_label <= 'z',
                    "The only subscript labels allowed are lowercase letters (a-z)");

        auto letter_index = subscript_label - 'a';
        auto dim_value = dims[dim_counter];

        // Subscript label not found in global subscript label array
        // Hence add it to both local and global subscript arrays
        if (letter_to_count_[letter_index] == 0) {
          letter_to_index_[letter_index] = num_subscript_labels_++;
          index_to_dim_value_.push_back(dim_value);
          index_to_last_input_.push_back(input_index);
        } else {  // This subscript label has been seen in atleast one other operand's subscript
          // It must be equal unless one of them is a 1 (Numpy allows this)
          auto mapped_index = letter_to_index_[letter_index];

          index_to_last_input_[mapped_index] = input_index;

          if (index_to_dim_value_[mapped_index] != dim_value) {
            // Set the value to the new dim value if the value is 1 in the map
            if (index_to_dim_value_[mapped_index] == 1) {
              index_to_dim_value_[mapped_index] = dim_value;
            } else {
              ORT_ENFORCE(dim_value == 1,
                          "Einsum operands could not be broadcast together. "
                          "Please check input shapes/equation provided.");
            }
          }
        }

        ++letter_to_count_[letter_index];

        current_input_dim_indices_to_subscript_indices_.push_back(letter_to_index_[letter_index]);

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

    input_dim_indices_to_subscript_indices_.push_back(std::move(current_input_dim_indices_to_subscript_indices_));
    ++input_index;
  }
}

template <typename T>
void EinsumComputePreprocessor<T>::PostProcessBroadcastedDims() {
  // Pay the cost of this function only if we saw an ellipsis in any of the inputs
  if (num_of_ellipsis_dims_ > 0) {
    // extend the number of subscript labels to include each ellipsis dim as
    // theoretically each ellipsis dim does correspond to a "virtual" subscript label
    num_subscript_labels_ += num_of_ellipsis_dims_;

    for (size_t i = 0; i < EinsumOp::num_of_letters; ++i) {
      if (letter_to_index_[i] != -1) {
        letter_to_index_[i] += num_of_ellipsis_dims_;
      }
    }

    std::vector<int64_t> temp_index_to_last_input(num_subscript_labels_, -1);
    for (size_t i = 0; i < index_to_last_input_.size(); ++i) {
      temp_index_to_last_input[i + num_of_ellipsis_dims_] = index_to_last_input_[i];
    }
    index_to_last_input_ = std::move(temp_index_to_last_input);

    std::vector<int64_t> temp_index_to_dim_value(num_subscript_labels_, -1);
    for (size_t i = 0; i < index_to_dim_value_.size(); ++i) {
      temp_index_to_dim_value[i + num_of_ellipsis_dims_] = index_to_dim_value_[i];
    }
    index_to_dim_value_ = std::move(temp_index_to_dim_value);

    for (size_t i = 0; i < input_dim_indices_to_subscript_indices_.size(); ++i) {
      auto& current_input_dim_indices_to_subscript_indices = input_dim_indices_to_subscript_indices_[i];
      std::vector<int64_t> temp_current_input_dim_indices_to_subscript_indices;
      temp_current_input_dim_indices_to_subscript_indices.reserve(current_input_dim_indices_to_subscript_indices.size());

      const auto& dims = inputs_[i]->Shape().GetDims();
      auto rank = dims.size();

      size_t dim_iter = 0;
      size_t num_broadcasted_indices = 0;
      while (dim_iter < current_input_dim_indices_to_subscript_indices.size()) {
        auto value = current_input_dim_indices_to_subscript_indices[dim_iter];
        if (value == 26) {  //This is a broadcasted dim
          // Shouldn't hit this error - just a sanity check
          ORT_ENFORCE(num_broadcasted_indices < num_of_ellipsis_dims_);
          temp_current_input_dim_indices_to_subscript_indices.push_back(static_cast<int64_t>(num_broadcasted_indices));
          index_to_last_input_[num_broadcasted_indices] = i;

          // This is the first time we are seeing this broadcasted dim
          if (index_to_dim_value_[num_broadcasted_indices] == -1) {
            index_to_dim_value_[num_broadcasted_indices] = dims[dim_iter];
          } else {  // We have seen this broadcasted dim before
            // Check if the previous value is equal to the current value
            if (index_to_dim_value_[num_broadcasted_indices] != dims[dim_iter]) {
              // If they are not equal, one of them needs to be 1
              if (index_to_dim_value_[num_broadcasted_indices] == 1) {
                index_to_dim_value_[num_broadcasted_indices] = dims[dim_iter];
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

template <typename T>
void EinsumComputePreprocessor<T>::ParseOrCreateOutputSubscript() {
  // Implicit form - construct the output subscript
  // Explicit form - no op as the output would have been parsed while parsing the input
  if (!is_explicit_) {
    return;
  }

  //TODO: Implement
}

template <typename T>
void EinsumComputePreprocessor<T>::CalculateOutputShape() {
  // Iterate through all subscript labels in the output subscript
  bool is_in_middle_of_ellipsis = false;
  int64_t ellipsis_char_count = 0;

  index_to_output_indices_.resize(num_subscript_labels_, -1);

  std::array<int64_t, EinsumOp::num_of_letters>
      output_letter_to_count;
  output_letter_to_count.fill(0);

  int64_t output_dim_counter = 0;
  for (auto subscript_label : right_equation_) {
    if (subscript_label == '.') {
      is_in_middle_of_ellipsis = true;
      // Make sure there aren't more than 3 '.'s in the current subscript
      ORT_ENFORCE(++ellipsis_char_count <= 3, "Found a '.' not part of an ellipsis in the output");
      if (ellipsis_char_count == 3) {  // Ellipsis is complete. Process it.
        is_in_middle_of_ellipsis = false;
        for (size_t i = 0; i < num_of_ellipsis_dims_; ++i) {
          output_dims_.push_back(index_to_dim_value_[i]);
          // The ellipsis is seen in the output and hence the corresponding dims are to not be reduced
          index_to_last_input_[i] = -1;
          index_to_output_indices_[i] = output_dim_counter++;
        }
      }
    } else {
      ORT_ENFORCE(!is_in_middle_of_ellipsis, "Found '.' not part of an ellipsis in the output");

      ORT_ENFORCE(subscript_label >= 'a' && subscript_label <= 'z',
                  "The only subscript labels allowed are lowercase letters (a-z)");

      auto letter_index = subscript_label - 'a';

      ORT_ENFORCE(output_letter_to_count[letter_index] == 0,
                  "Output subscript contains repeated letters");
      ++output_letter_to_count[letter_index];

      auto mapped_index = letter_to_index_[letter_index];
      ORT_ENFORCE(mapped_index != -1,
                  "Output subscript contains letters not seen in the inputs");

      output_dims_.push_back(index_to_dim_value_[mapped_index]);

      // Reset the last input index for this subscript label
      // given that it is seen in the output and hence can't be reduced
      index_to_last_input_[mapped_index] = -1;

      index_to_output_indices_[mapped_index] = output_dim_counter++;
    }
  }
}

template <typename T>
void EinsumComputePreprocessor<T>::PreprocessInputs() {
  preprocessed_inputs_.reserve(inputs_.size());
  // TODO: Write comments
  int64_t iter = 0;
  for (const auto* input : inputs_) {
    // We need to make a copy of the op's inputs because they will be mutated along the op's compute steps
    Tensor preprocessed = input->Clone(allocator_);

    const auto& input_dims = preprocessed.Shape().GetDims();

    std::vector<int64_t> subscript_label_to_input_index(num_subscript_labels_, -1);
    std::vector<int64_t> homogenized_input_dims(num_subscript_labels_, 1);

    auto current_input_dim_indices_to_subscript_indices_ = input_dim_indices_to_subscript_indices_[iter];
    for (size_t i = 0; i < current_input_dim_indices_to_subscript_indices_.size(); ++i) {
      auto temp_index = current_input_dim_indices_to_subscript_indices_[i];
      if (subscript_label_to_input_index[temp_index] == -1) {  // This is the first time we are seeing this subscript label in this input
        subscript_label_to_input_index[temp_index] = i;
        homogenized_input_dims[temp_index] = input_dims[i];
      } else {  // Diagonal needs to be parsed along the repeated axes
        preprocessed = EinsumOp::Diagonal<T>(preprocessed, subscript_label_to_input_index[temp_index], i, allocator_);
      }
    }

    std::vector<size_t> permutation;
    permutation.reserve(input_dims.size());

    for (auto& d : subscript_label_to_input_index) {
      if (d != -1) {
        permutation.push_back(static_cast<size_t>(d));
      }
    }

    // TODO: Identify no-op transpose and prevent triggering the transpose
    preprocessed = EinsumOp::Transpose(preprocessed, permutation, allocator_);

    EinsumOp::CreateReshapedView(preprocessed, homogenized_input_dims);

    preprocessed_inputs_.push_back(std::move(preprocessed));

    ++iter;
  }
}

template <typename T>
Status EinsumTypedProcessor<T>(OpKernelContext* context, const std::string& equation) {
  int num_inputs = context->InputCount();
  if (num_inputs == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Einsum op: There must be atleast one input");
  }

  std::vector<const Tensor*> inputs;
  inputs.reserve(num_inputs);

  // Hold the inputs
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(context->Input<Tensor>(i));
  }

  AllocatorPtr allocator;
  auto status = context->GetTempSpaceAllocator(&allocator);
  if (!status.IsOK()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, "There was a problem acquiring temporary space allocator in Einsum op");
  }

  auto einsum_preprocessor = EinsumComputePreprocessor<T>(equation, inputs, allocator);

  const auto& mapped_indices_to_last_input_index = einsum_preprocessor.GetMappedSubscriptIndicesToLastInputIndex();

  auto& preprocessed_inputs = einsum_preprocessor.GetPreprocessedTensors();

  auto num_subscript_labels = einsum_preprocessor.GetNumSubscriptLabels();

  const auto& output_dims = einsum_preprocessor.GetOutputDims();

  auto* output = context->Output(0, output_dims);

  // Preprocess the first input so as to reduce any dims that only it has
  Tensor result;

  {
    std::vector<int64_t> reduced_dims;
    std::vector<int64_t> preserved_dims;           // dims which were not reduced
    std::vector<int64_t> preserved_shape;          // shape pertaining to only the dims that were preserved (not reduced)
    reduced_dims.reserve(num_subscript_labels);    // num_subscript_labels is the upper bound. No harm in over-reserving.
    preserved_dims.reserve(num_subscript_labels);  // num_subscript_labels is the upper bound. No harm in over-reserving.
    preserved_shape.reserve(num_subscript_labels);

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
      first_input = EinsumOp::ReduceSum<T>(first_input, reduced_dims, allocator);
    }

    // Finalize the output at this stage if num_inputs == 1
    if (num_inputs == 1) {
      // Create reshaped view to squeeze out the reduced dims
      EinsumOp::CreateReshapedView(first_input, preserved_shape);

      // Finalize the output by applying any transpose required to get it to the required output ordering and move it to the op's output
      EinsumOp::FinalizeOutput<T>(first_input, preserved_dims, einsum_preprocessor.GetMappedSubscriptIndicesToOutputindices(), *output, output_dims, allocator);

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
      result = EinsumOp::PairwiseOperandProcess(result, preprocessed_inputs[input], reduced_dims, context->GetOperatorThreadPool(), allocator, einsum_preprocessor, is_final_pair, *output);
    }
  }
  return Status::OK();
}

// Explicit template instantiation
template Status EinsumTypedProcessor<float>(OpKernelContext* ctx, const std::string& equation);

}  // namespace onnxruntime
