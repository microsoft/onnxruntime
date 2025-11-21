// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "squeeze_op_data_propagation.h"
#include "core/common/common.h"
#include "core/graph/node_arg.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/common.h"
#include "core/common/inlined_containers.h"

namespace onnxruntime {

Status SqueezeOpDataPropagation::infer() {
  const auto* input_0 = node_.InputDefs()[0];

  // Return and do nothing if input doesn't exist
  if (!input_0 || !input_0->Exists()) {
    return Status::OK();
  }

  if (input_0->GetInferredShapeValues().has_value()) {
    const auto& tensor_shape_proto = input_0->GetInferredShapeValues().value();

    // The TensorShapeProto (inferred shape values) should have rank > 0 and
    // all the dimensions have values (not symbolic)
    if (tensor_shape_proto.dim_size() > 0) {
      for (const auto& dim : tensor_shape_proto.dim()) {
        if (!dim.has_dim_value()) {
          return Status::OK();
        }
      }
    }

    if (tensor_shape_proto.dim_size() == 1) {
      output_def_.SetInferredShapeScalarValue(tensor_shape_proto.dim(0).dim_value());
    } else if (tensor_shape_proto.dim_size() > 1) {
      // Get axes value
      TensorShapeVector axes;
      InlinedHashSet<int64_t> axes_set;

      // Note: Starting from opset 13, "axes" is provided as a second input to the Squeeze operator.
      //       In opset 11 and earlier, "axes" is defined as a node attribute instead.
      if (node_.InputDefs().size() > 1) {
        const auto* input_1 = node_.InputDefs()[1];
        ORT_TRY {
          ORT_RETURN_IF_ERROR(get_initialized_input_values_func_(input_1->Name(), axes));
        }
        ORT_CATCH(const std::exception& ex) {
          ORT_HANDLE_EXCEPTION([&]() {
            LOGS(logger_, ERROR) << ex.what();
            LOGS(logger_, INFO) << "Skip Squeeze op custom data propagation.";
          });
          return Status::OK();
        }
      } else {
        const auto& attrs = node_.GetAttributes();
        auto it = attrs.find("axes");
        if (it != attrs.end()) {
          const auto& axes_attr = it->second;
          for (const auto& i : axes_attr.ints()) {
            axes.push_back(i);
          }
        }
      }

      ORT_TRY {
        for (size_t i = 0; i < axes.size(); ++i) {
          // Negative value means counting dimensions from the back.
          axes_set.insert(HandleNegativeAxis(axes[i], tensor_shape_proto.dim_size()));
        }
      }
      ORT_CATCH(const std::exception& ex) {
        ORT_HANDLE_EXCEPTION([&]() {
          LOGS(logger_, ERROR) << ex.what();
          LOGS(logger_, INFO) << "Skip Squeeze op custom data propagation.";
        });
        return Status::OK();
      }

      auto& inferred_shape_values = output_def_.GetMutableInferredShapeValues();

      if (!inferred_shape_values.has_value()) {
        inferred_shape_values.emplace();
      }
      inferred_shape_values->clear_dim();

      int64_t dim_index = 0;
      for (const auto& dim : tensor_shape_proto.dim()) {
        auto value = dim.dim_value();
        if (axes_set.size() > 0) {
          if (axes_set.find(dim_index) == axes_set.end()) {
            inferred_shape_values->add_dim()->set_dim_value(value);
          }
        } else {
          if (value != 1) {
            inferred_shape_values->add_dim()->set_dim_value(value);
          }
        }

        dim_index++;
      }
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
