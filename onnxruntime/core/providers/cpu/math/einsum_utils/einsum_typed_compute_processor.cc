// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "einsum_typed_compute_processor.h"

namespace onnxruntime {

template <typename T>
void EinsumTypedComputeProcessor<T>::FinalizeOutput(const Tensor& candidate_output,
                                                    const std::vector<int64_t>& ordered_subscript_indices_in_candidate) {
  const std::vector<int64_t>& subscript_indices_to_output_indices =
      einsum_compute_preprocessor_.GetMappedSubscriptIndicesToOutputindices();
  const auto& output_dims = einsum_compute_preprocessor_.GetOutputDims();
  TensorShape output_shape = TensorShape(output_dims);
  const auto output_rank = output_dims.size();
  Tensor& output = *context_->Output(0, output_dims);

  ORT_ENFORCE(candidate_output.Shape().Size() == output_shape.Size(),
              "Einsum op: The candidate output cannot be reshaped into the op's output");

  const auto& candidate_output_dims = candidate_output.Shape().GetDims();
  const auto candidate_output_rank = candidate_output_dims.size();

  // This vector holds the shape of the candidate_output after removing the dims that have
  // been reduced in the final output
  std::vector<int64_t> candidate_output_shape_without_reduced_dims;
  candidate_output_shape_without_reduced_dims.reserve(candidate_output_rank);  // reserve upper bound

  // Identify the permutation required by the op's output
  std::vector<size_t> output_permutation;
  output_permutation.resize(output_rank, 0);
  size_t output_iter = 0;

  for (size_t iter = 0, end = ordered_subscript_indices_in_candidate.size(); iter < end; ++iter) {
    auto output_index = subscript_indices_to_output_indices[ordered_subscript_indices_in_candidate[iter]];

    // If output_index is -1, then this dimension does not show up in the op's output and has been reduced along the way
    if (output_index != -1) {
      output_permutation[output_index] = output_iter++;
      candidate_output_shape_without_reduced_dims.push_back(candidate_output_dims[iter]);
    } else {
      // This dim doesn't show up in the op's output and hence we check if the dim has been reduced in the candidate output
      ORT_ENFORCE(candidate_output_dims[iter] == 1,
                  "Not all dimensions to be reduced have been reduced in the candidate output. "
                  "Candidate output dims: ",
                  candidate_output.Shape());
    }
  }

  // Transpose to the required final output order
  // (Identify no-op transposes and prevent triggering the transpose)
  if (EinsumOp::IsTransposeRequired(candidate_output_shape_without_reduced_dims.size(), output_permutation)) {
    auto candidate_output_transposed = EinsumOp::Transpose(candidate_output, candidate_output_shape_without_reduced_dims,
                                                           output_permutation,
                                                           allocator_, einsum_ep_assets_, device_transpose_func_);

    // We have the result in an output "candidate". Now we have to copy the contents in its buffer
    // into the buffer of the actual output given to us by the execution frame
    // We need to do this because the buffer owned by the output tensor of the op could be user provided buffer

    auto status = device_data_copy_func_(*candidate_output_transposed, output, einsum_ep_assets_);
    ORT_ENFORCE(status.IsOK(), "Einsum op: Could not copy the intermediate output's buffer into the op's output buffer. Error: ",
                status.ErrorMessage());

  } else {
    // Copy the output candidate into the op's output
    auto status = device_data_copy_func_(candidate_output, output, einsum_ep_assets_);
    ORT_ENFORCE(status.IsOK(), "Einsum op: Could not copy the intermediate output's buffer into the op's output buffer. Error: ",
                status.ErrorMessage());
  }
}

static bool IsTransposeReshapeForEinsum(const std::vector<size_t>& perm,
                                        const std::vector<int64_t>& input_dims,
                                        std::vector<int64_t>& new_shape) {
  // As long as the dims with values > 1 stay in the same order, it's a reshape.
  // Example: Shape=(1,1,1024,4096) -> perm=(2,0,3,1).
  size_t last_permuted_axis = 0;
  for (size_t i = 0; i < perm.size(); ++i) {
    if (input_dims[perm[i]] == 1)
      continue;
    if (perm[i] < last_permuted_axis)
      return false;
    last_permuted_axis = perm[i];
  }
  new_shape = input_dims;
  for (size_t i = 0; i < perm.size(); ++i) {
    new_shape[i] = input_dims[perm[i]];
  }
  return true;
}

template <typename T>
std::unique_ptr<Tensor> EinsumTypedComputeProcessor<T>::PairwiseOperandProcess(const Tensor& left,
                                                                               const TensorShape& left_shape_override,
                                                                               const Tensor& right,
                                                                               const TensorShape& right_shape_override,
                                                                               const std::vector<int64_t>& reduce_dims,
                                                                               bool is_final_pair) {
  // Use the provided dim overrides instead of the actual shapes of the operands
  ORT_ENFORCE(left.Shape().Size() == left_shape_override.Size(),
              "The override dims are not compatible with given tensor's shape. ",
              "Left shape: ", left.Shape(), " Left shape override: ", left_shape_override.Size());
  ORT_ENFORCE(right.Shape().Size() == right_shape_override.Size(),
              "Right shape: ", right.Shape(), " Right shape override: ", right_shape_override.Size());

  // Make copy as this may be overridden downstream
  const auto& left_dims = left_shape_override.GetDims();
  const auto& right_dims = right_shape_override.GetDims();

  int64_t left_rank = static_cast<int64_t>(left_dims.size());
  int64_t right_rank = static_cast<int64_t>(right_dims.size());

  std::unique_ptr<Tensor> current_left;
  std::unique_ptr<Tensor> current_right;

  // If the following error condition is hit, it is most likely a pre-processing bug
  ORT_ENFORCE(left_rank == right_rank,
              "Ranks of pair-wise operands must be equal. ",
              "Left shape: ", left.Shape(), " Right shape: ", right.Shape());

  // Following vectors hold:
  // lro: dim indices that are present in left, right, and reduce_dims
  // lo: dim indices that are present in left and reduce_dims
  // ro: dim indices that are present in right and reduce_dims
  std::vector<size_t> lro;
  lro.reserve(8);  // Reserve an arbitrary amount of space for this vector (not bound to see a tensor of rank > 8)

  std::vector<size_t> lo;
  lo.reserve(8);  // Reserve an arbitrary amount of space for this vector (not bound to see a tensor of rank > 8)

  std::vector<size_t> ro;
  ro.reserve(8);  // Reserve an arbitrary amount of space for this vector (not bound to see a tensor of rank > 8)

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

    if (reduce_dims_iter < reduce_dims_size && reduce_dims[reduce_dims_iter] == i) {
      // This dimension is to be reduced after this pair-wise operation
      ++reduce_dims_iter;
      if (has_left_dim && has_right_dim) {  // Both inputs have non-trivial dim values along this dimension
        // Both the left and right operands have non-trivial dimension value along this axis
        // They must be equal
        ORT_ENFORCE(left_dim == right_dim,
                    "Einsum op: Input dimensions must be equal along an axis to be reduced across all inputs");
        reduced_size *= left_dim;
      } else if (has_left_dim) {  // if the dim to be reduced is only in one of left and right, we can reduce right away
        const Tensor& tensor_to_be_reduced = current_left ? *current_left : left;
        const std::vector<int64_t>& tensor_to_be_reduced_dims =
            current_left ? current_left->Shape().GetDims() : left_dims;

        current_left = EinsumOp::ReduceSum<T>(
            tensor_to_be_reduced, tensor_to_be_reduced_dims, {i}, allocator_, tp_, einsum_ep_assets_, device_reduce_sum_func_);
      } else if (has_right_dim) {
        const Tensor& tensor_to_be_reduced = current_right ? *current_right : right;
        const std::vector<int64_t>& tensor_to_be_reduced_dims =
            current_right ? current_right->Shape().GetDims() : right_dims;

        current_right = EinsumOp::ReduceSum<T>(
            tensor_to_be_reduced, tensor_to_be_reduced_dims, {i}, allocator_, tp_, einsum_ep_assets_, device_reduce_sum_func_);
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
  std::vector<int64_t> reshaped_dims;
  std::vector<size_t> left_permutation;
  left_permutation.reserve(lro.size() + lo.size() + reduce_dims.size() + ro.size());
  left_permutation.insert(left_permutation.end(), lro.begin(), lro.end());
  left_permutation.insert(left_permutation.end(), lo.begin(), lo.end());
  left_permutation.insert(left_permutation.end(), reduce_dims.begin(), reduce_dims.end());
  left_permutation.insert(left_permutation.end(), ro.begin(), ro.end());
  if (EinsumOp::IsTransposeRequired(current_left ? current_left->Shape().GetDims().size() : left_dims.size(),
                                    left_permutation)) {
    if (current_left && IsTransposeReshapeForEinsum(left_permutation,
                                                    current_left->Shape().GetDims(),
                                                    reshaped_dims)) {
      // This can be done because curent_* tensors (if they exist) and output tensors are
      // intermediate tensors and cannot be input tensors to the Einsum node itself
      // (which are immutable).
      // Covered by ExplicitEinsumAsTensorContractionReshapeLeft.
      current_left->Reshape(reshaped_dims);
    } else {
      // Covered by ExplicitEinsumAsTensorContraction, DiagonalWithMatmul, ...
      current_left = EinsumOp::Transpose(current_left ? *current_left : left,
                                         current_left ? current_left->Shape().GetDims() : left_dims,
                                         left_permutation, allocator_, einsum_ep_assets_,
                                         device_transpose_func_);
    }
  }

  // Permutate the right operand so that the axes order go like this: [lro, reduce_dims, ro, lo]
  std::vector<size_t> right_permutation;
  right_permutation.reserve(lro.size() + lo.size() + reduce_dims.size() + ro.size());
  right_permutation.insert(right_permutation.end(), lro.begin(), lro.end());
  right_permutation.insert(right_permutation.end(), reduce_dims.begin(), reduce_dims.end());
  right_permutation.insert(right_permutation.end(), ro.begin(), ro.end());
  right_permutation.insert(right_permutation.end(), lo.begin(), lo.end());
  if (EinsumOp::IsTransposeRequired(current_right ? current_right->Shape().GetDims().size() : right_dims.size(),
                                    right_permutation)) {
    if (current_right && IsTransposeReshapeForEinsum(right_permutation,
                                                     current_right->Shape().GetDims(),
                                                     reshaped_dims)) {
      // See note following the previous call of function IsTransposeReshapeForEinsum.
      // Covered by ExplicitEinsumAsBatchedMatmulWithBroadcasting_1, ExplicitEinsumAsMatmul_2, ...
      current_right->Reshape(reshaped_dims);
    } else {
      // Covered by DiagonalWithMatmul, ExplicitEinsumAsBatchedMatmul, ...
      current_right = EinsumOp::Transpose(current_right ? *current_right : right,
                                          current_right ? current_right->Shape().GetDims() : right_dims,
                                          right_permutation, allocator_, einsum_ep_assets_,
                                          device_transpose_func_);
    }
  }

  // Calculate output size
  // Output shape will be determined by rules of MatMul:
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

  for (size_t i = 0; i < reduce_dims.size(); ++i) {
    output_dims.push_back(1);  // reduced dimensions will have a value 1 in it
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
    current_subscript_order.reserve(lro.size() + lo.size() + reduce_dims.size() + ro.size());
    current_subscript_order.insert(current_subscript_order.end(), lro.begin(), lro.end());
    current_subscript_order.insert(current_subscript_order.end(), lo.begin(), lo.end());
    current_subscript_order.insert(current_subscript_order.end(), reduce_dims.begin(), reduce_dims.end());
    current_subscript_order.insert(current_subscript_order.end(), ro.begin(), ro.end());
  }

  // Multiply the mutated inputs
  auto output = EinsumOp::MatMul<T>(current_left ? *current_left : left, {lro_size, lo_size, reduced_size},
                                    current_right ? *current_right : right, {lro_size, reduced_size, ro_size},
                                    allocator_, tp_, einsum_ep_assets_, device_matmul_func_);

  output->Reshape(output_dims);

  if (!is_final_pair) {  // This is not the final pair - so bring the axes order to what the inputs conformed to
    if (EinsumOp::IsTransposeRequired(output_dims.size(), output_permutation)) {
      if (IsTransposeReshapeForEinsum(output_permutation,
                                      output_dims,
                                      reshaped_dims)) {
        // See note following the previous call of function IsTransposeReshapeForEinsum.
        // Covered by ExplicitEinsumAsTensorContractionReshapeFinal.
        output->Reshape(reshaped_dims);
      } else {
        output = EinsumOp::Transpose(*output, output_dims, output_permutation, allocator_,
                                     einsum_ep_assets_, device_transpose_func_);
      }
    }
  } else {  // This is the final pair - Transpose directly to the output ordering required and copy the contents to the op's output
    FinalizeOutput(*output, current_subscript_order);
  }

  return output;
}

template <typename T>
void EinsumTypedComputeProcessor<T>::SetDeviceHelpers(const EinsumOp::DeviceHelpers::Transpose& device_transpose_func,
                                                      const EinsumOp::DeviceHelpers::MatMul<T>& device_matmul_func,
                                                      const EinsumOp::DeviceHelpers::ReduceSum<T>& device_reduce_sum_func,
                                                      const EinsumOp::DeviceHelpers::DataCopy& device_data_copy_func) {
  device_transpose_func_ = device_transpose_func;
  device_matmul_func_ = device_matmul_func;
  device_reduce_sum_func_ = device_reduce_sum_func;
  device_data_copy_func_ = device_data_copy_func;
}

template <typename T>
Status EinsumTypedComputeProcessor<T>::Run() {
  const auto& mapped_indices_to_last_input_index = einsum_compute_preprocessor_.GetMappedSubscriptIndicesToLastInputIndex();

  auto& preprocessed_inputs = einsum_compute_preprocessor_.GetPreprocessedInputTensors();

  const auto& raw_inputs = einsum_compute_preprocessor_.GetRawInputTensors();

  const auto& homogenized_input_dims = einsum_compute_preprocessor_.GetHomogenizedInputDims();

  auto num_subscript_labels = einsum_compute_preprocessor_.GetNumSubscriptIndices();

  auto num_inputs = context_->InputCount();

  // Pre-process the first input so as to reduce any dims that only it has
  std::unique_ptr<const Tensor> result;

  {
    std::vector<int64_t> reduced_dims;
    std::vector<int64_t> preserved_dims;           // dims which were not reduced
    std::vector<int64_t> preserved_shape;          // shape pertaining to only the dims that were preserved (not reduced)
    reduced_dims.reserve(num_subscript_labels);    // num_subscript_labels is the upper bound. No harm in over-reserving.
    preserved_dims.reserve(num_subscript_labels);  // num_subscript_labels is the upper bound. No harm in over-reserving.

    for (int64_t i = 0; i < num_subscript_labels; ++i) {
      if (mapped_indices_to_last_input_index[i] == 0) {
        reduced_dims.push_back(i);
      } else {
        preserved_dims.push_back(i);
      }
    }

    // Reduce the dims that are last seen in the first input alone
    if (reduced_dims.size() != 0) {
      result = EinsumOp::ReduceSum<T>(preprocessed_inputs[0] ? *preprocessed_inputs[0] : *raw_inputs[0],
                                      homogenized_input_dims[0].GetDims(), reduced_dims, allocator_, tp_,
                                      einsum_ep_assets_, device_reduce_sum_func_);
    } else {
      // Check if there is a pre-processed version of this input
      // If so assign it to result
      if (preprocessed_inputs[0]) {
        result = std::move(preprocessed_inputs[0]);
      }
    }

    // Finalize the output at this stage if num_inputs == 1
    if (num_inputs == 1) {
      // Finalize the output by applying any transpose required to get
      // it to the required output ordering and move it to the op's output
      FinalizeOutput(result ? *result : *raw_inputs[0], preserved_dims);

      return Status::OK();
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
      // Use either the preprocessed inputs (if it is available) or the corresponding raw inputs
      result = PairwiseOperandProcess(result ? *result : *raw_inputs[0],
                                      result ? result->Shape() : homogenized_input_dims[0],
                                      preprocessed_inputs[input] ? *preprocessed_inputs[input] : *raw_inputs[input],
                                      homogenized_input_dims[input],
                                      reduced_dims, is_final_pair);
    }
  }

  return Status::OK();
}

// Explicit class instantiation
template class EinsumTypedComputeProcessor<float>;
template class EinsumTypedComputeProcessor<int32_t>;
template class EinsumTypedComputeProcessor<double>;
template class EinsumTypedComputeProcessor<int64_t>;
template class EinsumTypedComputeProcessor<MLFloat16>;

}  // namespace onnxruntime
