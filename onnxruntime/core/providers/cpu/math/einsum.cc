#include "core/framework/tensor.h"
#include "einsum.h"

#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace onnxruntime {

static void ComputeStrides(const std::vector<int64_t>& dims, std::vector<int64_t>& strides) {
  // TODO: Implement this
  return;
}

// Utility class that deals with all apsects of einsum equation string
// Covers everything from validating it to generating the output subscripts if needed
// From the computed output subscript, the output shape is easily calculated
class EinsumEquationParser final {
 public:
  EinsumEquationParser(const std::string& einsum_equation,
                       const std::vector<const TensorShape*>& input_dims)
      : input_dims_(input_dims), einsum_equation_(einsum_equation) {
    // TODO: Validate input string so that it only contains letters, spaces, and optionally one '->'
    // TODO: Throw on mal-formed '->'

    SubscriptLabelToDimValueMapping();

    ParseOrCreateOutputSubscript();

    CalculateOutputShape();
  }

  const std::unordered_map<char, int64_t>& GetSubscriptLabelToDimValue() const {
    return subscript_label_to_dim_value_;
  }

  const std::unordered_map<int64_t, char>& GetIndexToSubscriptLabel() const {
    return index_to_subscript_label_;
  }

  const std::vector<std::unique_ptr<const std::unordered_set<char>>>& GetInputIndicesToSubscriptLabels() const {
    return input_indices_to_subscript_labels_;
  }

  const std::unordered_set<char>& GetOutputSubscriptLabels() const {
    return output_subscript_labels_;
  }

  const std::vector<int64_t>& GetOutputDims() const {
    return output_dims_;
  }

 private:
  // In the sample Einsum string: 'ij, jk'
  // Subscripts are 'ij' and 'jk'
  // Subscript labels are 'i', 'j', and 'k'
  void SubscriptLabelToDimValueMapping() {
    if (input_subscripts_.size() != input_dims_.size()) {
      if (input_subscripts_.size() < input_dims_.size())
        ORT_THROW("More inputs provided than the number of subscripts in the equation");
      else
        ORT_THROW("Fewer inputs provided than the number of subscripts in the equation");
    }

    size_t input_count = input_subscripts_.size();

    // Holds mapping between input indices to its corresponding subscript labels
    input_indices_to_subscript_labels_.reserve(input_count);

    int64_t global_index_counter = 0;

    for (size_t i = 0; i < input_count; ++i) {
      auto* subscript = input_subscripts_[i];
      auto* shape = input_dims_[i];
      const auto& dims = shape->GetDims();
      auto rank = dims.size();
      int64_t dim_counter = 0;

      std::unordered_map<char, int64_t> local_subscript_label_to_dim_value;
      std::unordered_set<char>* local_subscript_labels = new std::unordered_set<char>();

      // Iterate through all subscript labels in the subscript
      for (auto subscript_label : *subscript) {
        // Subscript labels may contain spaces and are to be ignored
        if (subscript_label != ' ') {
          local_subscript_labels->insert(subscript_label);

          auto dim_value = dims[dim_counter];

          auto dim_value_iterator = subscript_label_to_dim_value_.find(subscript_label);

          // Subscript label not found in global map
          // Hence add it to both local and global maps
          if (dim_value_iterator == subscript_label_to_dim_value_.end()) {
            local_subscript_label_to_dim_value[subscript_label] = dim_value;
            subscript_label_to_dim_value_[subscript_label] = dim_value;
            index_to_subscript_label_[global_index_counter] = subscript_label;
            subscript_label_to_index_[subscript_label] = global_index_counter;
            ++global_index_counter;
          } else {
            // Subscript label found in global map

            // Check if the subscript label is found in the local map
            auto local_dim_value_iterator = local_subscript_label_to_dim_value.find(subscript_label);

            // If found in the local map
            if (local_dim_value_iterator != local_subscript_label_to_dim_value.end()) {
              // Value must match any dim value found in this input's shape
              ORT_ENFORCE(local_dim_value_iterator->second == dim_value, "TODO");
            } else {
              // Add it to local map
              local_subscript_label_to_dim_value[subscript_label] = dim_value;
            }

            // Value must match any dim value seen in any other input's shape (unless one of them is 1)
            auto global_dim_value_iterator = subscript_label_to_dim_value_.find(subscript_label);
            // It must be equal unless one of them is a 1 (Numpy allows this)
            if (global_dim_value_iterator->second != dim_value) {
              // Set the value to the new dim value if the value is 1 in the map
              if (global_dim_value_iterator->second == 1) {
                global_dim_value_iterator->second = dim_value;
              } else {
                ORT_ENFORCE(dim_value == 1,
                            "Einsum operands could not be broadcast together. "
                            "Please check input shapes/equation provided.");
              }
            }
          }

          ORT_ENFORCE(++dim_counter <= rank,
                      "Einsum subscripts string contains too many subscript labels for input ", i);
        }
      }

      input_indices_to_subscript_labels_.push_back(onnxruntime::make_unique<const std::unordered_set<char>>(local_subscript_labels));
    }
  }

  // Check if the Einsum equation has an explicit form (equation string contains "->")
  // If it is of explicit form, parse the output subscript (substring following "->")
  // If it is of implicit form (equation string does not contain "->"), compose the output subscript
  // If the output subscript is an empty string, the result is a scalar
  void ParseOrCreateOutputSubscript() {
    std::string delimiter = "->";
    size_t pos = einsum_equation_.find(delimiter);
    // Explicit form
    if (pos != std::string::npos) {
      output_subscript_ = einsum_equation_.substr(pos + 2);
      // TODO: Output subscript needs to be validated so that:
      // 1) It only contains letters
      // 2) It only contains letters seen in the input(s)
      ValidateAndCleanOutputSubscript();
    }

    // Implicit form - construct the output subscript and return it
  }

  // In case the output subscript is user-provided (Einsum in explicit form),
  // it must be validated so that it only contains letters (spaces are allowed)
  // and only letters that have been seen in the inputs.
  void ValidateAndCleanOutputSubscript() const {
    // TODO: Complete
    return;
  }

  void CalculateOutputShape() {
     // TODO: Reserve ??
      // Iterate through all subscript labels in the subscript
      for (auto subscript_label : output_subscript_) {
        // Subscript labels may contain spaces and are to be ignored
        if (subscript_label != ' ') {
          // The output_subscript_ has already been validated to make sure that it only contains
          // known letters from inputs => won't look up a key not in the map
          output_dims_.push_back(subscript_label_to_dim_value_.find(subscript_label)->second);
          output_subscript_labels_.insert(subscript_label);
        }
      }
  }

  // private members

  // Einsum equation
  const std::string& einsum_equation_;

  // Input dim of each input to the op
  const std::vector<const TensorShape*>& input_dims_;

  // Holds the calculated output dimensions
  std::vector<int64_t> output_dims_;

  // Holds the parsed subscripts for each input
  std::vector<const std::string*> input_subscripts_;

  // Holds constructed or parsed output subscript
  std::string output_subscript_;

  // Used to hold mapping between subscript labels to dimension values across all inputs (eg: i->6, j->3)
  std::unordered_map<char, int64_t> subscript_label_to_dim_value_;

  // Used to hold mapping between an assigned index to its corresponding subscript label (eg: 0->i, 1->j)
  std::unordered_map<int64_t, char> index_to_subscript_label_;

  // Used to hold mapping between asubscript label to its assigned index (eg: i->0, j->1)
  std::unordered_map<char, int64_t> subscript_label_to_index_;

  // Used to hold each input's subscript labels
  std::vector<std::unique_ptr<const std::unordered_set<char>>> input_indices_to_subscript_labels_;

  // Used to hold output's subscript labels
  std::unordered_set<char> output_subscript_labels_;
};

template <typename T>
class EinsumProcessor final {
 public:
  EinsumProcessor(const std::string& einsum_equation,
                  const std::vector<const TensorShape*>& input_dims) {
    // Parse the einsum equation and do the necessary pre-processing
    auto equation_parser = EinsumEquationParser(einsum_equation, input_dims);
  }
  void Compute() {}
};

namespace EinsumOp {
// A class to walk through all possible combinations
// of the different dimension letters
// For example, if the einsum equation is: 'ij, jk'
// for inputs of shape (2,3) and (3,4) -> i = 2, j = 3, k = 4.
// Then this class walks over all combinations of (i, j, k)
// from (0, 0, 0) to (1, 2, 3)

class DimensionWalker {
  DimensionWalker(const std::unordered_map<char, int64_t>& subscript_label_to_dim_value,
                  const std::unordered_map<char, int64_t>& index_to_subscript_label)
      : subscript_label_to_dim_value_(subscript_label_to_dim_value),
        index_to_subscript_label_(index_to_subscript_label) {
    num_of_dims_ = index_to_subscript_label_.size();

    // Zero initialize all current dimensions
    current_dims_.reserve(num_of_dims_);
    for (size_t i = 0; i < num_of_dims_; ++i) {
      current_dims_.push_back(0);
    }

    // Set maximum of each dimension in limit_dims
    limit_dims_.resize(num_of_dims_);  // make sure we don't just reserve but have enough indices that can get filled below
    max_num_of_clocks_ = 1;
    for (size_t i = 0; i < num_of_dims_; ++i) {
      // The following lookups cannot be empty
      auto subscript_label = index_to_subscript_label_.find(static_cast<int64_t>(i))->second;

      const auto dim_value = subscript_label_to_dim_value_.find(subscript_label)->second;

      limit_dims_[i] = dim_value;

      // Set max number of clocks along the way
      max_num_of_clocks_ *= dim_value;
    }

    // Set current num of clocks
    num_of_clocks_ = 0;
  }

  // Returns true if there is a valid current_dims to be processed
  // Returns false if all combinations hve been exhausted
  std::pair<bool, std::unordered_set<char>> Clock() const {
    std::pair<bool, std::unordered_set<char>> pair;

    if (num_of_clocks_ >= max_num_of_clocks_) {
      pair.first = false;
      return pair;
    }

    size_t dim = num_of_dims_ - 1;
    while (dim >= 0) {
      pair.second.insert(index_to_subscript_label_.find(static_cast<int64_t>(dim))->second);

      if (++current_dims_[dim] != limit_dims_[dim]) {
        break;
      }
      current_dims_[dim] = 0;
      --dim;
    }

    pair.first = true;
    return pair;
  }

  size_t num_of_dims_;

  size_t num_of_clocks_;
  size_t max_num_of_clocks_;

  mutable std::vector<int64_t> current_dims_;
  std::vector<int64_t> limit_dims_;

  const std::unordered_map<char, int64_t>& subscript_label_to_dim_value_;
  const std::unordered_map<char, int64_t>& index_to_subscript_label_;
};

template <typename T>
class Input {
 public:
  Input(const T* buffer, const std::vector<int64>& dims, 
       const std::unordered_set<char>& subscript_labels, 
       const std::vector<int64_t>& dim_indices_based_on_subscript_label_order)
      : buffer_(buffer), dims_(dims) : subscript_labels_(subscript_labels) {
    ComputeStrides(dims_, strides_);
  }

  T operator*(const std::vector<int64_t>& current_combination,
              const std::unordered_set<char>& change_from_prev) {
    if (offset == -1 || RecalcNeeded(change_from_prev)) {
      // TODO: Implement
    }
    return buffer[offset_];
  }

  bool RecalcNeeded(const std::unordered_set<char>& change_from_prev) {
    for (auto it = subscript_labels_.begin(); it != subscript_labels_.end(); ++it) {
      auto find_iter = change_from_prev.find(*it);
      if (find_iter != change_from_prev.end()) {
        return true;
      }
    }
    return false;
  }

 protected:
  size_t offset_ = -1;
  const std::vector<int64_t>& dims_;
  const std::unordered_set<char>& subscript_labels_;
  std::vector<int64_t> strides_;

 private:
  const T* buffer_;
};

template <typename T>
class Output : protected Input {
 public:
  Output(T* buffer, const std::vector<int64>& dims,
        const std::unordered_set<char>& subscript_labels,
        const std::vector<int64_t>& dim_indices_based_on_subscript_label_order)
      : buffer_(buffer), dims_(dims) : subscript_labels_(subscript_labels) {
    ComputeStrides(dims_, strides_);
  }

 private:
  T* buffer_;
};

}  // namespace EinsumOp

Status Einsum::Compute(OpKernelContext* context) const {
  int num_inputs = context->InputCount();
  std::vector<const Tensor*> inputs(num_inputs);
  std::vector<const TensorShape*> input_dims(num_inputs);

  // Hold the inputs and their dimensions
  for (int i = 0; i < num_inputs; ++i) {
    const auto* input = context->Input<Tensor>(i);
    inputs.push_back(input);
    input_dims.push_back(&input->Shape());
  }

  EinsumProcessor<float>(equation_, input_dims);
  return Status::OK();
}

}  // namespace onnxruntime
