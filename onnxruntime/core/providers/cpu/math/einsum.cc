#include "core/framework/tensor.h"
#include "core/providers/cpu/tensor/utils.h"
#include "einsum.h"
#include "core/common/common.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/util/math.h"
#include "core/providers/cpu/reduction/reduction_ops.h"

#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace onnxruntime {

// Prologue:
// In the sample Einsum string: 'ij, jk'
// Subscripts are 'ij' and 'jk'
// Subscript labels are 'i', 'j', and 'k'
// Creative credit: Implementation heavily influenced by PyTorch implementation at the time of writing

ONNX_CPU_OPERATOR_KERNEL(
    Einsum,
    12,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllNumericTensorTypes()),
    Einsum);

namespace EinsumOp {

constexpr size_t num_of_letters = 26;

// Thin wrapper over the Transpose op
static Tensor Transpose(const Tensor& input, const std::vector<size_t>& permutation,
                        const AllocatorPtr& allocator) {
  const auto& input_dims = input.Shape().GetDims();
  auto input_rank = input_dims.size();
  // Should not hit this error
  ORT_ENFORCE(input_rank == permutation.size(), "Length of permutations must match ");
  std::vector<int64_t> output_dims;
  output_dims.reserve(input_rank);
  for (const auto& dim : permutation) {
    // Use at() for range check rather than [] operator
    // if invoked correctly, this will not fail at the range check
    // TODO: Do we want to introduce added overhead of checking for dupes in permutation ?
    output_dims.push_back(input_dims.at(dim));
  }

  // Pass in allocator as that will be used as an allocator deleter by the framework
  // and it will de-allocate the memory for this intermediate tensor
  // when it goes out of scope
  Tensor output(input.DataType(), output_dims, allocator);
  TransposeBase::DoTranspose(permutation, input, output);
  return output;
}

// Creates a "reshaped view" for the same tensor (i.e.)
// mutates the shape for the same tensor.
// We will use it to introduce some "unsqueezed" dims (i.e.) extra dims with dim value as 1
inline static void CreateReshapedView(Tensor& input, const std::vector<int64_t>& new_dims) {
  input.Reshape(new_dims);
}

// Thin wrapper over the MatMul math utility
// Not using the MatMulHelper to compute output dim as it adds a lot of checking overhead
// In our case, we have a more simplistic version which doesn't need to have those checks

static Tensor MatMul(const Tensor& input_1, const Tensor& input_2,
                     const AllocatorPtr& allocator) {
  const auto& input1_dims = input_1.Shape().GetDims();
  const auto& input2_dims = input_2.Shape().GetDims();

  // Should not hit any of the following error conditions
  ORT_ENFORCE(input_1.DataType() == input_2.DataType(), "Data types of the inputs must match");
  ORT_ENFORCE(input1_dims.size() == 3 && input2_dims.size() == 3, "Only 1 batch dimension is allowed");
  ORT_ENFORCE(input1_dims[0] == input2_dims[0], "Batch dimension should match");
  ORT_ENFORCE(input1_dims[2] == input2_dims[1], "Incompatible matrix dimensions for multiplication");

  size_t batches = static_cast<size_t>(input1_dims[0]);
  size_t M = static_cast<size_t>(input1_dims[1]);
  size_t K = static_cast<size_t>(input1_dims[2]);
  size_t N = static_cast<size_t>(input2_dims[2]);

  size_t left_offset = M * K;
  size_t right_offset = K * N;
  size_t output_offset = M * N;

  std::vector<int64_t> output_dims;
  output_dims.reserve(3);
  output_dims.push_back(static_cast<int64_t>(batches));
  output_dims.push_back(static_cast<int64_t>(M));
  output_dims.push_back(static_cast<int64_t>(N));

  // Pass in allocator as that will be used as an allocator deleter by the framework
  // and it will de-allocate the memory for this intermediate tensor
  // when it goes out of scope
  Tensor output(input_1.DataType(), output_dims, allocator);

  // TODO: Switch on data types
  const float* input_1_data = input_1.template Data<float>();
  const float* input_2_data = input_2.template Data<float>();
  float* output_data = output.template MutableData<float>();

  // Process each batch
  for (size_t i = 0; i < batches; ++i) {
    math::MatMul<float>(
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),
        input_1_data + i * left_offset,
        input_2_data + i * right_offset,
        output_data + i * output_offset, nullptr);  // TODO: Add Threadpool
  }

  return output;
}

static Tensor ReduceSum(const Tensor& input, const std::vector<int64_t>& axes, OpKernelContext* ctx,
                        const AllocatorPtr& allocator) {
  // TODO: Switch on type
  return onnxruntime::foo(input, axes, ctx, allocator);
}

static Tensor ReduceSum(const Tensor& input, int64_t axis, OpKernelContext* ctx, const AllocatorPtr& allocator) {
  std::vector<int64_t> axes(1, axis);
  return ReduceSum(input, axes, ctx, allocator);
}

}  // namespace EinsumOp

// Processes Einsum operands in a pair-wise fashion
// Employs Transpose, ReduceSum, and MatMul under the hood
// to achieve MatMul(a, b) and reduces (by summing) along specified axes
static Tensor PairwiseOperandProcess(Tensor& left, Tensor& right,
                                     const std::vector<int64_t>& reduce_dims,
                                     OpKernelContext* ctx,
                                     const AllocatorPtr& allocator) {
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
        ORT_ENFORCE(left_dim == right_dim, "TODO");
        reduced_size *= left_dim;
      } else if (has_left_dim) {  // if it is only in one of left and right, we can sum right away
        left = EinsumOp::ReduceSum(left, i, ctx, allocator);
      } else if (has_right_dim) {
        right = EinsumOp::ReduceSum(right, i, ctx, allocator);
      }
    } else {  // This dimension is not reduced (i.e.) it appears in the output after processing these 2 operands
      // Both the left and right operands have non-trivial dimension value along this axis
      // They must be equal
      if (has_left_dim && has_right_dim) {
        ORT_ENFORCE(left_dim == right_dim, "TODO");
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
  left = EinsumOp::Transpose(left, left_permutation, allocator);
  EinsumOp::CreateReshapedView(left, {lro_size, lo_size, reduced_size});

  // Permutate the right operand so that the axes order go like this: [lro, reduce_dims, ro, lo]
  std::vector<size_t> right_permutation;
  right_permutation.reserve(lro.size() + lo.size() + reduce_dims.size() + ro.size());
  right_permutation.insert(right_permutation.end(), lro.begin(), lro.end());
  right_permutation.insert(right_permutation.end(), reduce_dims.begin(), reduce_dims.end());
  right_permutation.insert(right_permutation.end(), ro.begin(), ro.end());
  right_permutation.insert(right_permutation.end(), lo.begin(), lo.end());
  right = EinsumOp::Transpose(right, right_permutation, allocator);
  EinsumOp::CreateReshapedView(right, {lro_size, reduced_size, ro_size});

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
  for (size_t i = 0; i < reduce_dims.size(); ++i) {
    output_dims.push_back(1);  // reduced dimensions will have a value 1 in it
  }
  for (size_t i = 0; i < ro.size(); ++i) {
    output_dims.push_back(right_dims[ro[i]]);
  }

  // Calculate output permutation
  // After the MatMul op, the because the two operands have been permutated,
  // the output is permutated as well with respect to the original ordering of the axes.
  // The permutated order will be the dims in: [lro, lo, reduced_dims, ro]
  // Hence invert the permutation by a permutation that puts the axes in the same ordering
  std::vector<size_t> output_permutation;
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

  auto output = EinsumOp::MatMul(left, right, allocator);

  EinsumOp::Transpose(output, output_permutation, allocator);

  EinsumOp::CreateReshapedView(output, output_dims);

  return output;
}

class EinsumPreprocessor final {
 public:
  EinsumPreprocessor(const std::string& einsum_equation,
                     const std::vector<const Tensor*>& inputs,
                     const AllocatorPtr& allocator)
      : inputs_(inputs), einsum_equation_(einsum_equation), allocator_(allocator) {
    // Remove space characters
    einsum_equation_.erase(std::remove(einsum_equation_.begin(), einsum_equation_.end(), ' '),
                           einsum_equation_.end());

    // TODO: Throw on mal-formed '->' and multiple '->'s

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

    ParseOrCreateOutputSubscript();

    CalculateOutputShape();

    PreprocessInputs();
  }

  const std::vector<int64_t>& GetOutputDims() const {
    return output_dims_;
  }

  std::vector<Tensor>& GetPreprocessedTensors() {
    return preprocessed_inputs_;
  }

  const std::vector<int64_t>& GetMappedSubscriptIndicesToLastInputIndex() const {
    return index_to_last_input_;
  }

  const int64_t GetNumSubscriptLabels() const {
    return num_subscript_labels_;
  }

 private:
  void CollectMetadata() {
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

      // Keep track of the subscript labels seen in this specific subscript alone
      std::array<int64_t, EinsumOp::num_of_letters> current_letters_seen;
      current_letters_seen.fill(0);

      // Temp variables to deal with "ellipsis" in the input
      bool is_in_middle_of_ellipsis = false;
      int64_t ellipsis_char_count = 0;

      // Iterate through all subscript labels in the subscript
      for (auto subscript_label : subscript) {
        // Broadcasting based dims
        if (subscript_label == '.') {
          is_in_middle_of_ellipsis = true;
          // Make sure there aren't more than 3 '.'s in the current subscript
          ORT_ENFORCE(++ellipsis_char_count <= 3, "Found a . not part of an ellipsis in input: ", input_index);

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
                ORT_ENFORCE(num_of_ellipsis_dims_ == current_num_of_ellipsis_dims,
                            "Ellipsis must indicate a fixed number of dimensions across all inputs");
              } else {
                num_of_ellipsis_dims_ = current_num_of_ellipsis_dims;
              }

              // We reserve '26' for broadcasted dims as we only allow 'a' - 'z' (0 - 25) for non-broadcasted dims
              // We will assign an appropriate indices during broadcasting related post-processing
              for (int64_t i = 0; i < num_of_ellipsis_dims_; ++i) {
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
            ++letter_to_count_[letter_index];
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

            // This subscript label hasn't been seen in this subscript before
            if (current_letters_seen[letter_index] == 0) {
              current_letters_seen[letter_index] = 1;
            } else {
              ++current_letters_seen[letter_index];
              ORT_ENFORCE(false, "Diagonal parsing supported yet");
            }
          }

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

  // A function to process bradcasted dims (ellipsis) of inputs that they occur in
  void PostProcessingEllipses() {
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
          if (value == 26) { //This is a broadcasted dim
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
          } else { // This is a regular dim - offset it by number of broadcasted dims
            temp_current_input_dim_indices_to_subscript_indices.push_back(value + num_of_ellipsis_dims_);
          }
          ++dim_iter;
        }
        // Shouldn't hit this error - just a sanity check
        ORT_ENFORCE(dim_iter == rank);
        current_input_dim_indices_to_subscript_indices = std::move(temp_current_input_dim_indices_to_subscript_indices);
      }
    }
  }

  // Check if the Einsum equation has an explicit form (equation string contains "->")
  // If it is of explicit form, parse the output subscript (substring following "->")
  // If it is of implicit form (equation string does not contain "->"), compose the output subscript
  // If the output subscript is an empty string, the result is a scalar
  void ParseOrCreateOutputSubscript() {
    // Implicit form - construct the output subscript
    // Explicit form - no op as the output would have been parsed while parsing the input
    if (!is_explicit_) {
      return;
    }

    //TODO: Implement
  }

  void CalculateOutputShape() {
    // TODO: Account for broadcasting

    output_dims_.reserve(right_equation_.length());

    //std::vector<int64_t> temp_mapped_indices_to_output_dim_indices_(num_subscript_labels_, -1);

    //int64_t iter = 0;
    // Iterate through all subscript labels in the output subscript
    for (auto subscript_label : right_equation_) {
      ORT_ENFORCE(subscript_label >= 'a' && subscript_label <= 'z',
                  "The only subscript labels allowed are lowercase letters (a-z)");

      auto letter_index = subscript_label - 'a';
      auto mapped_index = letter_to_index_[letter_index];
      ORT_ENFORCE(mapped_index != -1,
                  "Output subscript contains letters not seen in the inputs");

      output_dims_.push_back(index_to_dim_value_[mapped_index]);

      // Reset the last input index for this subscript label
      // given that it is seen in the output and hence can't be reduced
      index_to_last_input_[mapped_index] = -1;

      //temp_mapped_indices_to_output_dim_indices_[subscript_labels_to_index_[index]] = iter++;
    }

    // Move this constructed vector
    // mapped_indices_to_output_dim_indices_ = std::move(temp_mapped_indices_to_output_dim_indices_);
  }

  void PreprocessInputs() {
    preprocessed_inputs_.reserve(inputs_.size());
    // TODO: Write comments
    int64_t iter = 0;
    for (const auto* input : inputs_) {
      const auto& input_dims = input->Shape().GetDims();

      std::vector<int64_t> subscript_label_to_input_index(num_subscript_labels_, -1);
      std::vector<int64_t> homogenized_input_dims(num_subscript_labels_, 1);

      auto current_input_dim_indices_to_subscript_indices_ = input_dim_indices_to_subscript_indices_[iter];
      for (size_t i = 0; i < current_input_dim_indices_to_subscript_indices_.size(); ++i) {
        auto temp_index = current_input_dim_indices_to_subscript_indices_[i];
        subscript_label_to_input_index[temp_index] = i;
        homogenized_input_dims[temp_index] = input_dims[i];
      }

      std::vector<size_t> permutation;
      permutation.reserve(input_dims.size());

      for (auto& d : subscript_label_to_input_index) {
        if (d != -1) {
          permutation.push_back(static_cast<size_t>(d));
        }
      }

      // If the permutation is a no-op - for example - permitation - [0,1] for a 2D tensor
      // we still do it because we have to mutate the resulting tensor downstream
      // Perhaps an optimization for the future could be to use memcpy to copy over the buffer
      // without calling Transpose in case of a no-op Transpose
      auto preprocessed = EinsumOp::Transpose(*input, permutation, allocator_);
      EinsumOp::CreateReshapedView(preprocessed, homogenized_input_dims);

      preprocessed_inputs_.push_back(std::move(preprocessed));

      ++iter;
    }
  }

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
  int64_t num_of_ellipsis_dims_ = 0;

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

  // TODO: Unfinalized
  // For each input, holds its corresponding input_dims which have been "homogenized"
  // For example, if the input_dim is [2,3] and the subscript for the input is "ij"
  // and the num_subscript_labels_ is 3 (Order: i, j, k), the homogenized dims
  // will be [2, 3, 1]
  // std::vector<std::vector<int64_t>> homogenized_input_dims_;

  const AllocatorPtr& allocator_;
};

Status Einsum::Compute(OpKernelContext* context) const {
  int num_inputs = context->InputCount();
  ORT_ENFORCE(num_inputs > 0, "Einsum op: There must be atleast one input");

  std::vector<const Tensor*> inputs;
  inputs.reserve(num_inputs);

  // Hold the inputs
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(context->Input<Tensor>(i));
  }

  AllocatorPtr allocator;
  auto status = context->GetTempSpaceAllocator(&allocator);
  if (!status.IsOK()) {
    ORT_THROW("There was a problem acquiring temporary space allocator in Einsum op");
  }

  auto einsum_preprocessor = EinsumPreprocessor(equation_, inputs, allocator);

  const auto& mapped_indices_to_last_input_index = einsum_preprocessor.GetMappedSubscriptIndicesToLastInputIndex();

  auto& preprocessed_tensors = einsum_preprocessor.GetPreprocessedTensors();

  auto num_subscript_labels = einsum_preprocessor.GetNumSubscriptLabels();

  const auto& output_dims = einsum_preprocessor.GetOutputDims();

  auto* output = context->Output(0, output_dims);

  // Preprocess the first input so as to reduce any dims that only it has
  Tensor result;
  {
    std::vector<int64_t> reduced_dims;
    reduced_dims.reserve(num_subscript_labels);  // num_subscript_labels is the upper bound. No harm in over-reserving.
    for (int64_t i = 0; i < num_subscript_labels; ++i) {
      if (mapped_indices_to_last_input_index[i] == 0) {
        reduced_dims.push_back(i);
      }
    }
    if (reduced_dims.size() != 0) {
      preprocessed_tensors[0] = EinsumOp::ReduceSum(preprocessed_tensors[0], reduced_dims, context, allocator);
    }
    result = std::move(preprocessed_tensors[0]);
  }

  // Keep processing each input pair-wise
  if (num_inputs > 1) {
    for (int input = 1; input < num_inputs; ++input) {
      std::vector<int64_t> reduced_dims;
      reduced_dims.reserve(num_subscript_labels);  // num_subscript_labels is the upper bound. No harm in over-reserving.
      for (int64_t dim = 0; dim < num_subscript_labels; ++dim) {
        if (mapped_indices_to_last_input_index[dim] == input) {
          // This is the last input we are seeing this dimension (and it doesn't occur in the output), so reduce along the dimension
          reduced_dims.push_back(dim);
        }
      }
      result = PairwiseOperandProcess(result, preprocessed_tensors[input], reduced_dims, context, allocator);
    }
  }

  // Create a reshaped view of the final result (based on the required output shape for this op)
  EinsumOp::CreateReshapedView(result, output_dims);
  *output = std::move(result);
  return Status::OK();
}

}  // namespace onnxruntime
