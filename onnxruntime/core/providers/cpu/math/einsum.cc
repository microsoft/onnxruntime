#include <vector>
#include <unordered_map>

#include "core/framework/tensor.h"

namespace onnxruntime {

static void IncrementInnerAxis(std::vector<int64_t>& current_dims,
                               const std::vector<int64_t>& dims) {
  if (dims.size() != current_dims.size() || dims.size() < 1) {
  }

  size_t dimension = current_dims.size() - 1;

  while (dimension > 0) {
    if (++current_dims[dimension] != dims[dimension])
      break;
    current_dims[dimension] = 0;
    --dimension;
  }
}

// utility class that deals with all apsects of einsum equation string
// Covers everything from validating it to generating the output subscripts if needed
class EinsumEquationParser final {
 public:
  EinsumEquationParser(const std::string& einsum_equation) {
  }
 
 private:
  // Check if the Einsum equation has an explicit form (equation string contains "->")
  // If it is of explicit form, parse the output subscript (substring following "->")
  // If it is of implicit form (equation string does not contain "->"), compose the output subscript
  // If the output subscript is an empty string, the result is a scalar
  std::string GetOutputSubscript(const std::string& einsum_equation) {
    std::string delimiter = "->";
    std::string output_subscript;
    size_t pos = einsum_equation.find(delimiter);

    // Explicit form
    if (pos != std::string::npos) {
      output_subscript = einsum_equation.substr(pos + 2);
      // TODO: Output subscript needs to be validated so that:
      // 1) It only contains letters
      // 2) It only contains letters seen in the input(s)
      return output_subscript;
    }

    // Implicit form - construct the output subscript and return it

    return output_subscript;
  }

// In the sample Einsum string: ('ij, jk', a, b)
  // Subscripts are 'ij' and 'jk'
  // Subscript labels are 'i', 'j', and 'k'
  // The operands (inputs) are tensors 'a' and 'b'
  static std::unordered_map<char, int64_t> SubscriptLabelToDimValueMapping(
      const std::vector<std::string*>& subscripts,
      const std::vector<TensorShape*>& shapes) {
    if (subscripts.size() != shapes.size()) {
      if (subscripts.size() < shapes.size())
        ORT_THROW("More inputs provided than the number of subscripts in the equation");
      else
        ORT_THROW("Fewer inputs provided than the number of subscripts in the equation");
    }

    // Used to hold mapping between subscript labels to dimension values across all inputs
    std::unordered_map<char, int64_t> global_subscript_label_to_dim_value;
    size_t count = subscripts.size();

    for (size_t i = 0; i < count; ++i) {
      auto* subscript = subscripts[i];
      auto* shape = shapes[i];
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

          auto local_dim_value_iterator = local_subscript_label_to_dim_value.find(subscript_label);

          // Subscript label not found in local map
          // Hence add it to both local and global maps
          if (local_dim_value_iterator == local_subscript_label_to_dim_value.end()) {
            local_subscript_label_to_dim_value[subscript_label] = dim_value;
            global_subscript_label_to_dim_value[subscript_label] = dim_value;
          } else {
            // Subscript label found in local (and global) map

            // Value must match any dim value found in this input's shape
            ORT_ENFORCE(local_dim_value_iterator->second == dim_value, "TODO");

            // Value must match any dim value seen in any other input's shape (unless one of them is 1)
            auto global_dim_value_iterator = global_subscript_label_to_dim_value.find(subscript_label);
            // It must be equal unless one of them is a 1 (Numpy allows this)
            if (global_dim_value_iterator->second != dim_value) {
              // Set the value to the new dim value if the value is 1 in the map
              if (global_dim_value_iterator->second == 1) {
                global_dim_value_iterator->second = dim_value;
              } else {
                ORT_ENFORCE(dim_value == 1,
                            "Einsum operands could not be broadcast together. Please check input shapes/equation provided.");
              }
            }
          }

          ORT_ENFORCE(++dim_counter <= rank,
                      "Einsum subscripts string contains too many subscript labels for input ", i);
        }
      }
    }

    return global_subscript_label_to_dim_value;
  }
};


}  // namespace onnxruntime
