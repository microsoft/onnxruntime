// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "einsum_compute_preprocessor.h"

namespace onnxruntime {

EinsumComputePreprocessor::EinsumComputePreprocessor(EinsumEquationPreprocessor& einsum_equation_preprocessor,
                                                     const std::vector<const Tensor*>& inputs,
                                                     AllocatorPtr allocator,
                                                     void* einsum_cuda_assets)
    : einsum_equation_preprocessor_(einsum_equation_preprocessor),
      inputs_(inputs),
      allocator_(allocator),
      einsum_ep_assets_(einsum_cuda_assets) {
  letter_to_index_.fill(-1);

  letter_to_count_.fill(0);
}

Status EinsumComputePreprocessor::Run() {
  ORT_RETURN_IF_ERROR(ProcessSubscripts());

  ORT_RETURN_IF_ERROR(PostProcessBroadcastedDims());

  ORT_RETURN_IF_ERROR(ParseOrCreateOutputSubscript());

  ORT_RETURN_IF_ERROR(CalculateOutputShape());

  ORT_RETURN_IF_ERROR(PreprocessInputs());

  return Status::OK();
}

const std::vector<int64_t>& EinsumComputePreprocessor::GetOutputDims() const {
  return output_dims_;
}

std::vector<std::unique_ptr<Tensor>>& EinsumComputePreprocessor::GetPreprocessedInputTensors() {
  return preprocessed_inputs_;
}

const std::vector<const Tensor*>& EinsumComputePreprocessor::GetRawInputTensors() {
  return inputs_;
}

const std::vector<TensorShape>& EinsumComputePreprocessor::GetHomogenizedInputDims() {
  return homogenized_input_dims_;
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

void EinsumComputePreprocessor::SetDeviceHelpers(const EinsumOp::DeviceHelpers::Diagonal& device_diagonal_func,
                                                 const EinsumOp::DeviceHelpers::Transpose& device_transpose_func) {
  device_diagonal_func_ = device_diagonal_func;
  device_transpose_func_ = device_transpose_func;
}

Status EinsumComputePreprocessor::ProcessSubscripts() {
  const auto& left_equation_split = einsum_equation_preprocessor_.left_equation_split_;
  if (left_equation_split.size() != inputs_.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Number of subscripts in the input equation does not match number of input tensors");
  }

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
        if (++ellipsis_char_count > 3) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Found a '.' not part of an ellipsis in input: ", input_index);
        }

        // We have seen all 3 '.'s. We can safely process the ellipsis now.
        if (ellipsis_char_count == 3) {
          is_in_middle_of_ellipsis = false;

          // Example for the following line of code
          // Subscript "...ij" for an input of rank 6
          // num_of_ellipsis_dims = 6 - 5 + 3 = 4
          int64_t current_num_of_ellipsis_dims = rank - subscript.length() + 3;
          if (current_num_of_ellipsis_dims < 0) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                   "Einsum subscripts string contains too many subscript labels when compared to the rank of the input");
          }

          // Theoretically, current_num_of_ellipsis_dims could be 0
          // Example: For an input of rank 2 paired with a subscript "...ij"
          if (current_num_of_ellipsis_dims != 0) {
            // We have seen a ellipsis before - make sure ranks align as per the ONNX spec -
            // "Ellipsis must indicate a fixed number of dimensions."
            if (num_of_ellipsis_dims_ != 0) {
              if (num_of_ellipsis_dims_ != static_cast<size_t>(current_num_of_ellipsis_dims)) {
                return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                       "Ellipsis must indicate a fixed number of dimensions across all inputs");
              }
            } else {
              num_of_ellipsis_dims_ = static_cast<size_t>(current_num_of_ellipsis_dims);
            }

            // We reserve 'EinsumOp::num_of_letters' for broadcasted dims as we only allow 'a' - 'z'
            // and 'A' - 'Z' (0 - 51) for non-broadcasted dims.
            // We will assign appropriate indices (based on number of dimensions the ellipsis corresponds to)
            // during broadcasting related post-processing.
            for (size_t i = 0; i < num_of_ellipsis_dims_; ++i) {
              current_subscript_indices.push_back(EinsumOp::num_of_letters);
            }

            // Offset 'dim_counter' by number of dimensions the ellipsis corresponds to
            dim_counter += num_of_ellipsis_dims_;
          }
        }
      } else {  // regular letter based dimension -> 'i', 'j', etc.
        if (is_in_middle_of_ellipsis) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Found '.' not part of an ellipsis in input: ", input_index);
        }

        auto letter_index = EinsumOp::LetterToIndex(subscript_label);
        if (letter_index == -1) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "The only subscript labels allowed are lower-cased letters (a-z) and "
                                 "upper-cased letters (A-Z)");
        }

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
              if (dim_value != 1) {
                return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                       "Einsum operands could not be broadcast together. "
                                       "Please check input shapes/equation provided."
                                       "Input shape of operand ",
                                       input_index, " is incompatible in the dimension ", dim_counter,
                                       ". The shape is: ", shape,
                                       "Another operand has a dim value of ", subscript_indices_to_dim_value_[mapped_index],
                                       " in the same dimension");
              }
            }
          }
        }

        ++letter_to_count_[letter_index];

        current_subscript_indices.push_back(letter_to_index_[letter_index]);
        if (++dim_counter > rank) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Einsum subscripts string contains too many subscript labels when compared to the rank of the input ",
                                 input_index);
        }
      }
    }

    // If no broadcasting is requested, the number of subscript labels (dim_counter) should match input rank
    if (num_of_ellipsis_dims_ == 0) {
      if (dim_counter != rank) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Einsum subscripts does not contain enough subscript labels and there is no ellipsis for input ",
                               input_index);
      }
    }

    input_subscript_indices_.push_back(std::move(current_subscript_indices));
    ++input_index;
  }

  return Status::OK();
}

Status EinsumComputePreprocessor::PostProcessBroadcastedDims() {
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
        if (value == EinsumOp::num_of_letters) {  //This is a broadcasted dim
          // Shouldn't hit this error - just a sanity check
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
                if (dims[dim_iter] != 1) {
                  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                         "The broadcasted dimensions of the inputs are incompatible");
                }
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

  return Status::OK();
}

Status EinsumComputePreprocessor::ParseOrCreateOutputSubscript() {
  // Explicit form - no op as the output would have been parsed while parsing the input
  if (einsum_equation_preprocessor_.is_explicit_) {
    // Make sure that the given explicit equation contains an ellipsis if the input contains ellipses in them
    if (num_of_ellipsis_dims_ > 0) {
      if (einsum_equation_preprocessor_.right_equation_.find("...") == std::string::npos) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Inputs have ellipses in them but the provided output subscript does not contain an ellipsis");
      }
    }
    return Status::OK();
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
  return Status::OK();
}

Status EinsumComputePreprocessor::CalculateOutputShape() {
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
      if (++ellipsis_char_count > 3) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Found a '.' not part of an ellipsis in the output subscript provided");
      }

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
      if (is_in_middle_of_ellipsis) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Found '.' not part of an ellipsis in the output subscript provided");
      }

      auto letter_index = EinsumOp::LetterToIndex(subscript_label);
      if (letter_index == -1) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                              "The only subscript labels allowed are lower-cased letters (a-z) and "
                              "upper-cased letters (A-Z)");
      }

      if (output_letter_to_count[letter_index] != 0) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Output subscript contains repeated letters");
      }
      ++output_letter_to_count[letter_index];

      auto mapped_index = letter_to_index_[letter_index];
      if (mapped_index == -1) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Output subscript contains letters not seen in the inputs");
      }

      output_dims_.push_back(subscript_indices_to_dim_value_[mapped_index]);

      // Reset the last input index for this subscript label
      // given that it is seen in the output and hence can't be reduced
      subscript_indices_to_last_input_[mapped_index] = -1;

      subscript_indices_to_output_indices_[mapped_index] = output_dim_counter++;
    }
  }

  return Status::OK();
}

Status EinsumComputePreprocessor::PreprocessInputs() {
  preprocessed_inputs_.reserve(inputs_.size());
  homogenized_input_dims_.reserve(inputs_.size());
  // As part of input preprocessing we "homogenize" them by
  // 1) Making them all of the same rank
  // 2) The axes order in all the inputs are to be made the same
  int64_t input_iter = 0;
  for (const auto* input : inputs_) {
    // Eventually will hold the "preprocessed" version of the original input
    std::unique_ptr<Tensor> preprocessed;

    const auto& input_dims = input->Shape().GetDims();
    const auto& current_subscript_indices = input_subscript_indices_[input_iter];

    // If all has gone well, we will have a subscript index (subscript label) for each dim of the input
    if (input_dims.size() != current_subscript_indices.size()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Rank of the input must match number of subscript labels corresponding to the input");
    }

    std::vector<int64_t> subscript_indices_to_input_index(num_subscript_indices_, -1);

    // This is the input dims after re-ordering so that all inputs have same axes order
    std::vector<int64_t> homogenized_input_dims(num_subscript_indices_, 1);

    // Preprocessed dim rank may not be the same as original input rank if we need to parse diagonals along the way
    // (which reduces rank in the preprocessed input by 1 for each diagonal we parse)
    int64_t dim_index_in_preprocessed_input = 0;
    int64_t dim_index_in_original_input = 0;

    // iterate through all subscript indices in this input
    for (const auto& subscript_index : current_subscript_indices) {
      if (subscript_indices_to_input_index[subscript_index] == -1) {  // This is the first time we are seeing this subscript label in this input
        subscript_indices_to_input_index[subscript_index] = dim_index_in_preprocessed_input++;
        homogenized_input_dims[subscript_index] = input_dims[dim_index_in_original_input];
      } else {  // Diagonal needs to be parsed along the repeated axes
        preprocessed = device_diagonal_func_(preprocessed ? *preprocessed : *inputs_[input_iter],
                                             subscript_indices_to_input_index[subscript_index],
                                             dim_index_in_preprocessed_input,
                                             allocator_, einsum_ep_assets_);
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
    if (EinsumOp::IsTransposeRequired(preprocessed ? preprocessed->Shape().GetDims().size() : inputs_[input_iter]->Shape().GetDims().size(),
                                      permutation)) {
      preprocessed = EinsumOp::Transpose(preprocessed ? *preprocessed : *inputs_[input_iter],
                                         preprocessed ? preprocessed->Shape().GetDims() : inputs_[input_iter]->Shape().GetDims(),
                                         permutation, allocator_, einsum_ep_assets_, device_transpose_func_);
    }

    // pre-processed may be null if the input didn't have need diagonals parsed and didn't need transposing
    // If the pre-processed inputs are null, we will use raw inputs in conjunction with "homogenized_input_dims" for
    // downstream compute
    if (preprocessed) {  // If the pre-processed version of the operand exists, reshape it to homogenized_input_dims
      preprocessed->Reshape(homogenized_input_dims);
    }
    preprocessed_inputs_.push_back(std::move(preprocessed));
    homogenized_input_dims_.emplace_back(homogenized_input_dims);

    ++input_iter;
  }

  return Status::OK();
}

}  // namespace onnxruntime
