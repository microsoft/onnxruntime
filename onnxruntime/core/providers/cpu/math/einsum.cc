#include <vector>
#include <unordered_map>

#include "core/framework/tensor.h"

namespace onnxruntime {

// Utility class that deals with all apsects of einsum equation string
// Covers everything from validating it to generating the output subscripts if needed
// From the computed output subscript, the output shape is easily calculated
class EinsumEquationParser final {
 public:
  EinsumEquationParser(const std::string& einsum_equation,
                       const std::vector<TensorShape*>& input_dims)
      : input_dims_(input_dims), einsum_equation_(einsum_equation) {
    // TODO: Validate input string so that it only contains letters, spaces, and optionally one '->'
    // TODO: Throw on mal-formed '->'
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

    // Used to hold mapping between subscript labels to dimension values across all inputs
    size_t count = input_subscripts_.size();
    int64_t global_index_counter = 0;

    for (size_t i = 0; i < count; ++i) {
      auto* subscript = input_subscripts_[i];
      auto* shape = input_dims_[i];
      const auto& dims = shape->GetDims();
      auto rank = dims.size();
      int64_t dim_counter = 0;

      // Used to hold mapping between subscript labels to dimension values in this specific input
      std::unordered_map<char, int64_t> local_subscript_label_to_dim_value;

      // Iterate through all subscript labels in the subscript
      for (auto subscript_label : *subscript) {
        // Subscript labels may contain spaces and are to be ignored
        if (subscript_label != ' ') {
          auto dim_value = dims[dim_counter];

          auto global_dim_value_iterator = global_subscript_label_to_dim_value_.find(subscript_label);

          // Subscript label not found in global map
          // Hence add it to both local and global maps
          if (global_dim_value_iterator == global_subscript_label_to_dim_value_.end()) {
            local_subscript_label_to_dim_value[subscript_label] = dim_value;
            global_subscript_label_to_dim_value_[subscript_label] = dim_value;
            global_subscript_label_to_index_[subscript_label] = global_index_counter++;
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
            auto global_dim_value_iterator = global_subscript_label_to_dim_value_.find(subscript_label);
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
    }

    // Implicit form - construct the output subscript and return it
  }

  // In case the output subscript is user-provided (Einsum in explicit form),
  // it must be validated so that it only contains letters (spaces are allowed)
  // and only letters that have been seen in the inputs. Along the way, it removes spaces
  void ValidateAndCleanOutputSubscript() const {
  }

  void CalculateOutputShape() {
    auto output_rank = output_subscript_.length();
    if (output_rank > 0) {
      output_dims_.resize(output_rank);

      // Iterate through all subscript labels in the subscript
      for (auto subscript_label : output_subscript_) {
        // The output_subscript_ has already been validated to make sure that it only contains
        // known letters from inputs => won't look up a key not in the map
        output_dims_.push_back(global_subscript_label_to_dim_value_.find(subscript_label)->second);
      }
    }
  }

  // private members
  const std::string& einsum_equation_;
  const std::vector<TensorShape*>& input_dims_;

  std::vector<int64_t> output_dims_;

  std::vector<std::string*> input_subscripts_;
  std::string output_subscript_;

  std::unordered_map<char, int64_t> global_subscript_label_to_dim_value_;
  std::unordered_map<char, int64_t> global_subscript_label_to_index_;
};

}  // namespace onnxruntime
