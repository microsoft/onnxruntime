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
  Tensor output(input.DataType(), output_dims, allocator, 0);
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

  std::vector<int64_t> output_dims;
  output_dims.reserve(3);
  output_dims.push_back(input1_dims[0]);
  output_dims.push_back(input1_dims[1]);
  output_dims.push_back(input2_dims[2]);

  // Pass in allocator as that will be used as an allocator deleter by the framework
  // and it will de-allocate the memory for this intermediate tensor
  // when it goes out of scope
  Tensor output(input_1.DataType(), output_dims, allocator, 0);
  int64_t M = static_cast<size_t>(input1_dims[1]);
  int64_t K = static_cast<size_t>(input1_dims[2]);
  int64_t N = static_cast<size_t>(input2_dims[2]);

  size_t left_offset = M * K;
  size_t right_offset = K * N;
  size_t output_offset = M * N;

  // TODO: Switch on data types
  // Process each batch
  for (size_t i = 0; i < static_cast<size_t>(input1_dims[0]); i++) {
    math::MatMul<float>(
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),
        input_1.template Data<float>() + i * left_offset,
        input_2.template Data<float>() + i * right_offset,
        output.template MutableData<float>() + i * output_offset, nullptr);
  }
}

static Tensor ReduceSum(const Tensor& input, const std::vector<int64_t>& axes, const AllocatorPtr& allocator) {
  // TODO: Switch on type
  return onnxruntime::ReduceSum<float>::ComputeOutput(input, axes, allocator);
}

static Tensor ReduceSum(const Tensor& input, int64_t axis, const AllocatorPtr& allocator) {
  std::vector<int64_t> axes(1, axis);
  ReduceSum(input, axes, allocator);
}

}  // namespace EinsumOp

// Processes Einsum operands in a pair-wise fashion
// Employs Transpose, ReduceSum, and MatMul under the hood
// to achieve MatMul(a, b) and reduces (by summing) along specified axes
static Tensor PairwiseOperandProcess(Tensor& left, Tensor& right,
                                     const std::vector<int64_t>& reduce_dims,
                                     const AllocatorPtr& allocator) {
  // Make copies as we may mutate the tensor objects downstream
  std::vector<int64_t> left_dims = left.Shape().GetDims();
  std::vector<int64_t> right_dims = right.Shape().GetDims();

  auto left_rank = static_cast<int64_t>(left_dims.size());
  auto right_rank = static_cast<int64_t>(right_dims.size());

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

  for (int64_t i = 0; i < left_rank; ++i) {
    auto left_dim = left_dims[i] > 1;    // non-trivial dimension (dim_value != 1)
    auto right_dim = right_dims[i] > 1;  // non-trivial dimension (dim_value != 1)

    if (reduce_dims[i] == 1) {
      if (left_dim && right_dim) {
        // Both the left and right operands have non-trivial dimension value along this axis
        // They must be equal
        ORT_ENFORCE(left_dim == right_dim, "TODO");
        reduced_size *= left_dim;
      } else if (left_dim) {  // if it is only in one of left and right, we can sum right away
        left = EinsumOp::ReduceSum(left, left_dim);
      } else if (right_dim) {
        right = EinsumOp::ReduceSum(right, right_dim);
      }
    } else {  // This dimension is not reduced (i.e.) it appears in the output
      // Both the left and right operands have non-trivial dimension value along this axis
      // They must be equal
      if (left_dim && right_dim) {
        ORT_ENFORCE(left_dim == right_dim, "TODO");
        lro.push_back(i);
        lro_size *= left_dim;
      } else if (left_dim) {
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
  right_permutation.insert(left_permutation.end(), lro.begin(), lro.end());
  right_permutation.insert(left_permutation.end(), reduce_dims.begin(), reduce_dims.end());
  right_permutation.insert(left_permutation.end(), ro.begin(), ro.end());
  right_permutation.insert(left_permutation.end(), lo.begin(), lo.end());
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
  std::vector<size_t> output_permutation(lro.size() + lo.size() + reduce_dims.size() + ro.size(), -1);
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
      output_subscript_ = einsum_equation_.substr(mid_index + 2);
      is_explicit_ = true;
    } else {
      left_equation_ = einsum_equation_;
    }

    subscript_labels_to_dim_value_.fill(-1);

    subscript_labels_to_index_.fill(-1);

    CollectMetadata();

    ParseOrCreateOutputSubscript();

    CalculateOutputShape();

    PreprocessInputs();
  }

  const std::vector<int64_t>& GetOutputDims() const {
    return output_dims_;
  }

 private:
  void CollectMetadata() {
    std::stringstream str(left_equation_);
    std::string subscript;

    // Holds mapping between input indices to its corresponding subscript labels
    int64_t input_index = 0;

    std::array<int64_t, EinsumOp::num_of_letters> subscript_labels_to_last_input_index_;
    subscript_labels_to_last_input_index_.fill(-1);

    mapped_indices_to_input_dim_indices_.reserve(inputs_.size());

    while (std::getline(str, subscript, ',')) {
      const auto& shape = inputs_[input_index]->Shape();
      const auto& dims = shape.GetDims();
      size_t rank = dims.size();
      size_t dim_counter = 0;

      std::unordered_map<int64_t, int64_t> mapped_indices_to_input_dim_index;

      // Keep track of the subscript labels seen in this specific subscript alone
      std::array<int64_t, EinsumOp::num_of_letters> local_subscript_labels_seen;
      local_subscript_labels_seen.fill(-1);

      // Iterate through all subscript labels in the subscript
      int64_t iter = 0;
      for (auto subscript_label : subscript) {
        ORT_ENFORCE(subscript_label >= 'a' && subscript_label <= 'z',
                    "The only subscript labels allowed are lowercase letters (a-z)");

        auto index = subscript_label - 'a';
        auto dim_value = dims[dim_counter];

        // Update the last input index for this subscript label
        subscript_labels_to_last_input_index_[index] = input_index;

        // Subscript label not found in global subscript label array
        // Hence add it to both local and global subscript arrays
        if (subscript_labels_to_dim_value_[dim_value] == -1) {
          subscript_labels_to_dim_value_[index] = dim_value;
          subscript_labels_to_index_[index] = num_subscript_labels_++;
        } else {  // This subscript label has been seen in atleast one other operand's subscript
          // It must be equal unless one of them is a 1 (Numpy allows this)
          if (subscript_labels_to_dim_value_[index] != dim_value) {
            // Set the value to the new dim value if the value is 1 in the map
            if (subscript_labels_to_dim_value_[index] == 1) {
              subscript_labels_to_dim_value_[index] = dim_value;
            } else {
              ORT_ENFORCE(dim_value == 1,
                          "Einsum operands could not be broadcast together. "
                          "Please check input shapes/equation provided.");
            }
          }

          // This subscript label hasn't been seen in this subscript before
          if (local_subscript_labels_seen[index] == -1) {
            local_subscript_labels_seen[index] = 1;
          } else {
            ++local_subscript_labels_seen[index];
            ORT_ENFORCE(false, "Diagonal parsing supported yet");
          }
        }

        mapped_indices_to_input_dim_index[subscript_labels_to_index_[index]] = iter++;

        ORT_ENFORCE(++dim_counter <= rank,
                    "Einsum subscripts string contains too many subscript labels for input ", input_index);
      }

      // TODO: Account for broadcasting
      ORT_ENFORCE(dim_counter == rank,
                  "Einsum subscripts does not contain enough subscript labels and there is no ellipsis for input ", input_index);

      mapped_indices_to_input_dim_indices_.push_back(std::move(mapped_indices_to_input_dim_index));
      ++input_index;
    }

    // Remap the subscript_labels_to_last_input_index_ based on the remapped indices
    mapped_indices_to_last_input_index_.reserve(num_subscript_labels_);

    int64_t temp_counter = 0;
    for (size_t i = 0; i < EinsumOp::num_of_letters; ++i) {
      auto val = subscript_labels_to_last_input_index_[i];
      if (val != -1) {
        ++temp_counter;
        mapped_indices_to_last_input_index_[subscript_labels_to_index_[i]] = val;
      }
    }
    ORT_ENFORCE(temp_counter == num_subscript_labels_);
  }

  // Check if the Einsum equation has an explicit form (equation string contains "->")
  // If it is of explicit form, parse the output subscript (substring following "->")
  // If it is of implicit form (equation string does not contain "->"), compose the output subscript
  // If the output subscript is an empty string, the result is a scalar
  void ParseOrCreateOutputSubscript() {
    // Implicit form - construct the output subscript.
    // Explicit form - no op as the output would have been parsed while parsing the input
    if (!is_explicit_) {
      return;
    }

    //TODO: Implement
  }

  void CalculateOutputShape() {
    // TODO: Account for broadcasting

    output_dims_.reserve(output_subscript_.length());

    //std::vector<int64_t> temp_mapped_indices_to_output_dim_indices_(num_subscript_labels_, -1);

    int64_t iter = 0;
    // Iterate through all subscript labels in the output subscript
    for (auto subscript_label : output_subscript_) {
      ORT_ENFORCE(subscript_label >= 'a' && subscript_label <= 'z',
                  "The only subscript labels allowed are lowercase letters (a-z)");

      auto index = subscript_label - 'a';

      ORT_ENFORCE(subscript_labels_to_index_[index] != -1,
                  "Output subscript contains letters not seen in the inputs");

      output_dims_.push_back(subscript_labels_to_dim_value_[index]);

      // Reset the last input index for this subscript label
      // given that it is seen in the output and hence can't be reduced
      mapped_indices_to_last_input_index_[subscript_labels_to_index_[index]] = -1;

      //temp_mapped_indices_to_output_dim_indices_[subscript_labels_to_index_[index]] = iter++;
    }

    // Move this constructed vector
    // mapped_indices_to_output_dim_indices_ = std::move(temp_mapped_indices_to_output_dim_indices_);
  }

  void PreprocessInputs() {
    preprocessed_inputs_.reserve(inputs_.size());

    // Pre-process inputs by permutating the dims and homogenizing input dims
    // Example: For an input with dim [2, 3], with subscript label "kj", and
    // subscript_labels_to_index_ - {i -> 0, j->1, k->2}, since "j" comes ahead of "k"
    // in the order, we would have to permutate the input with permutation [1, 0].
    // After this the shape is [3, 2], we would have to account for the unseen subscript label
    // and unsqueeze the corresponding dim to make the final shape: [1, 3, 2]

    int64_t iter = 0;
    for (const auto* input : inputs_) {
      const auto& input_dims = input->Shape().GetDims();

      std::vector<int64_t> subscript_label_to_input_index(num_subscript_labels_, -1);
      std::vector<int64_t> homogenized_input_dims(num_subscript_labels_, 1);

      auto mapped_index_to_input_dim_indices_ = mapped_indices_to_input_dim_indices_[iter];
      for (auto pair : mapped_index_to_input_dim_indices_) {
        homogenized_input_dims[pair.first] = input_dims[pair.second];
        subscript_label_to_input_index[pair.first] = pair.second;
      }

      std::vector<size_t> permutation;
      permutation.resize(input_dims.size());

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

  // Flag indicating if the op is being used in explicit form
  bool is_explicit_ = false;

  // Holds constructed or parsed output subscript
  std::string output_subscript_;

  // All original inputs to the op
  const std::vector<const Tensor*>& inputs_;

  // All preprocessed inputs
  std::vector<Tensor> preprocessed_inputs_;

  // Hold the dim value corresponding to the subscript labels seen
  // `-1` means the corresponding label wasn't seen at all
  // The size is fixed at 26 (one for each letter of the alphabet - 'a' to 'z')
  std::array<int64_t, EinsumOp::num_of_letters> subscript_labels_to_dim_value_;

  // Hold the assigned index corresponding to the subscript labels seen
  // `-1` means the corresponding label wasn't seen at all
  std::array<int64_t, EinsumOp::num_of_letters> subscript_labels_to_index_;

  // Holds the input index of the last operand to have the correpesponding subscript label
  // (based on the index which is given by the subscript_labels_to_index_)
  // If the value is `-1`, then the subscript label is never seen (or)
  // it appears in the output
  std::vector<int64_t> mapped_indices_to_last_input_index_;

  // Count of unique subscript labels
  // E.g.: With equation -> 'ij, jk -> ik'
  // num_subscript_labels_ = 3 (i, j, k)
  int64_t num_subscript_labels_ = 0;

  // Holds the final calculated output dimensions
  std::vector<int64_t> output_dims_;

  // Holds the actual index of the output dims in the index corresponding to a subscript label
  // (based on the index which is given by the subscript_labels_to_index_)
  // A value of `-1` means that the subscript label is not present in the output
  //std::vector<int64_t> mapped_indices_to_output_dim_indices_;

  // For each input, Holds the actual index of the input dim (value)
  // in the index corresponding to a subscript label (key)
  // (The key is based on the index which is given by the subscript_labels_to_index_)
  // A value of `-1` means that the subscript label is not present in the input
  std::vector<std::unordered_map<int64_t, int64_t>> mapped_indices_to_input_dim_indices_;

  // For each input, holds its corresponding input_dims which have been "homogenized"
  // For example, if the input_dim is [2,3] and the subscript for the input is "ij"
  // and the num_subscript_labels_ is 3 (Order: i, j, k), the homogenized dims
  // will be [2, 3, 1]
  std::vector<std::vector<int64_t>> homogenized_input_dims_;

  AllocatorPtr allocator_;
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

  return Status::OK();
}

}  // namespace onnxruntime