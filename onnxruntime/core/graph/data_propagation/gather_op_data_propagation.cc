// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gather_op_data_propagation.h"
#include "core/common/common.h"
#include "core/graph/node_arg.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/common.h"
#include "core/graph/data_propagation/data_propagation_value_utils.h"

namespace onnxruntime {

Status GatherOpDataPropagation::infer() {
  if (output_from_onnx_op_data_propagation_.has_tensor_type() &&
      output_from_onnx_op_data_propagation_.tensor_type().has_shape()) {
    int dim_size = output_from_onnx_op_data_propagation_.tensor_type().shape().dim_size();
    // Check there is no result from Gather's PartialDataPropagationFunction(),
    // so that it can run custom data propagation below.
    // Otherwise, this infer() function won't be called as the result from Gather's PartialDataPropagationFunction()
    // will be used in Graph::SaveShapeValuesFromDataPropagation().
    if (dim_size != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "ORT shouldn't run Gather's custom data propagation here as Gather's"
                             "PartialDataPropagationFunction() already infers shape values in the output.");
    }
  }

  // Following code extracts an element from a 1D array if all conditions are met.
  // e.g.
  // shape data is [1, 3, 64, 64] -> gets 64 if the index is 2.
  // shape data is [1, 3, 64, 64] -> gets 3 if the index is 1.

  // Get "data" input
  // Note: The "data" input should be an one dimension array in this case.
  const auto* input_0 = node_.InputDefs()[0];

  // Get "indices" input
  // Note: The "indices" input could be one of the three cases:
  //       1. A tensor with rank > 0 and all tensor values are known.
  //       2. A tensor with rank > 0 but not all tensor values are known.
  //       3. A scalar.
  //
  //       If it's case #1, ONNX operator's PartialDataPropagationFunction()
  //       should have inferred the output shape value.
  //       If it's case #2, neither ONNX operator's PartialDataPropagationFunction()
  //       nor Gather's custom data propagation can handle it.
  //       This Gather's custom data propagation handles case #3.
  const auto* input_1 = node_.InputDefs()[1];

  // Return and do nothing if input doesn't exist
  if (!input_0 || !input_1 || !input_0->Exists() || !input_1->Exists()) {
    return Status::OK();
  }

  // If input's inferred shape values is present, we then perfrom the gather operation on the shape values
  // and saves the result in output's node_arg.
  if (input_0->GetInferredShapeValues().has_value()) {
    const auto& tensor_shape_proto = input_0->GetInferredShapeValues().value();

    ORT_TRY {
      TensorShapeVector indices;
      int indices_num_dims = -1;
      ORT_RETURN_IF_ERROR(get_initialized_input_values_func_(input_1->Name(), indices, indices_num_dims));
      if (indices.size() == 1) {
        // Note: Index value is expected to be within bounds [-s, s-1] along axis of size s
        auto index = static_cast<int32_t>(
            HandleNegativeAxis(indices[0], tensor_shape_proto.dim_size()));

        auto& dim = tensor_shape_proto.dim(index);
        if (dim.has_dim_value()) {
          // Gather output rank = data_rank - 1 + indices_rank. The "data" input here is the
          // 1-D Shape output, so the output rank equals the indices rank. Route by that rank:
          //   * a 0-D scalar index   -> a scalar output value,
          //   * a 1-D index          -> a rank-1 [1] output value (so consumers that require a
          //     1-D tensor, e.g. TopK's K input, still observe the correct rank),
          //   * a rank >= 2 index (or an unknown rank) -> decline, because the single-value
          //     channel cannot represent a rank >= 2 Gather output; falling back to ONNX data
          //     propagation is safer than fabricating a rank-1 value.
          //
          // The index rank is sourced canonically from the same constant initializer the index
          // value came from (indices_num_dims). If that is unavailable (sentinel < 0), fall back
          // to the indices NodeArg shape; an unknown shape then routes to decline.
          int effective_num_dims = indices_num_dims;
          if (effective_num_dims < 0) {
            const ONNX_NAMESPACE::TensorShapeProto* indices_shape = input_1->Shape();
            effective_num_dims = (indices_shape != nullptr) ? indices_shape->dim_size() : -1;
          }

          if (effective_num_dims == 0) {
            // 0-D scalar index -> a scalar output value.
            SetSinglePropagatedShapeValue(output_def_, dim.dim_value(), /*is_rank1=*/false);
          } else if (effective_num_dims == 1) {
            // 1-D index -> a rank-1 [1] output value.
            SetSinglePropagatedShapeValue(output_def_, dim.dim_value(), /*is_rank1=*/true);
          }
          // rank >= 2 or unknown index rank (effective_num_dims < 0): leave the output unset and
          // let ONNX data propagation handle it. The single-value channel cannot represent a
          // rank >= 2 Gather output, so emitting a rank-1 value would fabricate a misleading rank.
        }
      }
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        LOGS(logger_, ERROR) << ex.what();
        LOGS(logger_, INFO) << "Skip Gather op custom data propagation.";
      });
      return Status::OK();
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
