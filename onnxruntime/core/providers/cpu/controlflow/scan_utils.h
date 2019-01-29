// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "core/common/common.h"
#include "core/framework/allocator.h"
#include "core/framework/ml_value.h"
#include "core/framework/mlvalue_tensor_slicer.h"
#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {
class GraphViewer;
class OpKernelContextInternal;
namespace scan {
namespace detail {

enum class ScanDirection { kForward = 0,
                           kReverse = 1 };

/**
Class to provide input/output MLValue instances for a loop state variable.
The MLValue flips between two internal temporary buffers to minimize copies.
*/
class LoopStateVariable {
 public:
  LoopStateVariable(const MLValue& original_value, MLValue& final_value, const int64_t sequence_len,
                    AllocatorPtr& allocator);

  // get current Input MLValue
  const MLValue& Input() const;

  // get current Output MLValue
  MLValue& Output();

  // move to next usage of the loop state variable. call after each iteration of the subgraph.
  void Next();

 private:
  int64_t iteration_num_{0};
  const int64_t sequence_len_;

  // copy original and final value from temporary MLValue provided by iterator
  const MLValue original_value_;
  MLValue final_value_;

  /* we use original_value and final_value once, 
     and alternate between a_ and b_ as input/output for each iteration to avoid copies

    Iteration   Input             Output
    0           original_value    a_
    1           a_                b_
    2           b_                a_
    ...
    seq len - 1 <previous output> final_value
    */
  MLValue a_;
  MLValue b_;
};

/*
Class that co-ordinates writing to slices of the overall Scan output buffer returned by OpKernelContext.Output(i). 
If the subgraph has a symbolic dimension in an output it will use a temporary MLValue for the first execution
in order to discover the output shape. Once the shape is known, it will switch to using the overall output buffer 
to avoid copies.
If 'temporary' is true it will use a temporary MLValue for the overall output as well. Set this to true if the output
needs to be transposed before being returned by the Scan operator. The data_type also needs to be provided if 
'temporary' is true to do the allocation.
*/
class OutputIterator {
 public:
  static Status Create(OpKernelContextInternal& context,
                       int output_index,
                       bool is_loop_state_var,
                       bool is_v8,
                       TensorShape final_shape,
                       std::unique_ptr<OutputIterator>& iterator,
                       ScanDirection direction = ScanDirection::kForward,
                       bool temporary = false,
                       MLDataType data_type = nullptr) {
    iterator.reset(new OutputIterator(context, output_index, is_loop_state_var, is_v8, final_shape,
                                      direction, temporary, data_type));
    return iterator->Initialize();
  }

  MLValue& operator*();
  OutputIterator& operator++();

  bool FinalOutputAllocated() const { return is_concrete_shape_; }

  // custom fetch allocator that can be used when the final shape is not concrete.
  // when the subgraph requests the allocation of the subgraph output, we forward the request to this instance,
  // allocate the overall output (taking into account the sequence length dimension),
  // and use a slicer to return the chunk for the subgraph output for this iteration.
  Status AllocateSubgraphOutput(const TensorShape& shape, MLValue& mlvalue);

  // set the output for the current iteration to zeros. used for short sequence lengths
  void ZeroOutCurrent() {
    auto* tensor = (**this).GetMutable<Tensor>();
    memset(tensor->MutableDataRaw(), 0, tensor->Size());
  }

  const MLValue& GetOutput() const {
    ORT_ENFORCE(final_output_mlvalue_, "Attempt to retrieve final output before it was set.");
    return *final_output_mlvalue_;
  }

 private:
  OutputIterator(OpKernelContextInternal& context,
                 int output_index,
                 bool is_loop_state_var,
                 bool is_v8,
                 TensorShape final_shape,
                 ScanDirection direction,
                 bool temporary,
                 MLDataType data_type);

  Status Initialize();
  Status AllocateFinalBuffer();

  OpKernelContextInternal& context_;
  bool is_v8_;
  const int output_index_;
  ONNX_NAMESPACE::TensorShapeProto per_iteration_shape_;
  TensorShape final_shape_;
  bool is_loop_state_var_;
  ScanDirection direction_;
  int64_t num_iterations_;
  int64_t cur_iteration_;

  // is the final shape concrete, or does it have symbolic dimensions
  bool is_concrete_shape_;

  // one or more slicers for writing to the output
  std::vector<MLValueTensorSlicer<MLValue>::Iterator> slicer_iterators_;
  std::vector<MLValueTensorSlicer<MLValue>::Iterator>::iterator cur_slicer_iterator_;

  // if true allocate temporary_final_output_mlvalue_ with data_type_ using the temporary allocator
  // and point final_output_value_ at that.
  // if false, final_output_value_ is an output from the Scan operator and allocated using the context_.
  bool temporary_;
  MLDataType data_type_;
  MLValue temporary_final_output_mlvalue_;

  MLValue* final_output_mlvalue_;
};

void ReadDirections(const OpKernelInfo& info, const std::string& attr_name,
                    std::vector<int64_t>& directions, int64_t num_entries);

Status AllocateOutput(OpKernelContextInternal& context, const GraphViewer& subgraph,
                      int output_index, bool is_loop_state_var, int64_t batch_size, int64_t sequence_len,
                      std::unique_ptr<OutputIterator>& output_iterator,
                      ScanDirection direction = ScanDirection::kForward,
                      bool temporary = false);

Status IterateSequence(OpKernelContextInternal& context,
                       const SessionState& session_state,
                       const GraphViewer& subgraph,
                       std::vector<LoopStateVariable>& loop_state_variables,
                       std::vector<MLValueTensorSlicer<const MLValue>::Iterator>& scan_input_stream_iterators,
                       int64_t seq_length,
                       int num_loop_state_variables,
                       int num_variadic_inputs,
                       int num_variadic_outputs,
                       std::unordered_map<std::string, const MLValue*>& implicit_inputs,
                       std::vector<std::string>& subgraph_output_names,
                       std::vector<std::unique_ptr<OutputIterator>>& output_iterators);

MLValue AllocateTensorInMLValue(const MLDataType data_type, const TensorShape& shape, AllocatorPtr& allocator);

/**
Calculate the transpose permutations and shape by shifting the chosen axis TO the first dimension.
The other dimension indexes or values are pushed in order after the chosen axis.

e.g. if shape is {2, 3, 4} and axis 1 is chosen the permutations will be {1, 0, 2} and output shape will be {3, 2, 4}
     if axis 2 is chosen the permutations will be {2, 0, 1} and the output shape will be {4, 2, 3}
*/
void CalculateTransposedShapeForInput(const TensorShape& original_shape, int64_t axis,
                                      std::vector<int64_t>& permutations, std::vector<int64_t>& transposed_shape);

/**
Calculate the transpose permutations and shape by shifting the chosen axis FROM the first dimension.

e.g. if shape is {4, 2, 3} and axis 2 is chosen, dimension 0 will move to dimension 2, 
     the permutations will be {1, 2, 0} and output shape will be {2, 3, 4}
*/
void CalculateTransposedShapeForOutput(const TensorShape& original_shape, int64_t axis,
                                       std::vector<int64_t>& permutations, std::vector<int64_t>& transposed_shape);

}  // namespace detail
}  // namespace scan
}  // namespace onnxruntime
