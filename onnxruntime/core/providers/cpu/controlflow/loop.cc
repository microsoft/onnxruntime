// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include "core/providers/cpu/controlflow/loop.h"
#include "core/providers/cpu/controlflow/utils.h"

#include "core/framework/framework_common.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/sequential_executor.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/providers/cpu/tensor/utils.h"

#include "gsl/gsl_algorithm"
#include "gsl/span"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
/*
ONNX_OPERATOR_SET_SCHEMA(
    Loop,
    1,
    OpSchema()
        .SetDoc(Loop_ver1_doc)
        .Input(
            0,
            "M",
            "A maximum trip-count for the loop specified at runtime. Optional."
            " Pass empty string to skip.",
            "I",
            OpSchema::Optional)
        .Input(
            1,
            "cond",
            "A boolean termination condition. Optional. Pass empty string to skip.",
            "B",
            OpSchema::Optional)
        .Input(
            2,
            "v_initial",
            "The initial values of any loop-carried dependencies (values that "
            "change across loop iterations)",
            "V",
            OpSchema::Variadic)
        .Output(
            0,
            "v_final_and_scan_outputs",
            "Final N loop carried dependency values then K scan_outputs",
            "V",
            OpSchema::Variadic)
        .Attr(
            "body",
            "The graph run each iteration. It has 2+N inputs: (iteration_num, "
            "condition, loop carried dependencies...). It has 1+N+K outputs: "
            "(condition, loop carried dependencies..., scan_outputs...). Each "
            "scan_output is created by concatenating the value of the specified "
            "output value at the end of each iteration of the loop. It is an error"
            " if the dimensions or data type of these scan_outputs change across loop"
            " iterations.",
            AttributeProto::GRAPH)
        .TypeConstraint("V", OpSchema::all_tensor_types(), "All Tensor types")
        .TypeConstraint("I", {"int64"}, "Only int64")
        .TypeConstraint("B", {"bool"}, "Only bool")
        .TypeAndShapeInferenceFunction(LoopInferenceFunction));
*/

ONNX_CPU_OPERATOR_KERNEL(Loop,
                         1,
                         KernelDefBuilder()
                             .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                             .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                             .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
                         Loop);

class LoopImpl {
 public:
  LoopImpl(OpKernelContextInternal& context,
           const SessionState& session_state);

  // Initialize by validating all the inputs, and allocating the output tensors
  Status Initialize();

  Status CreateFeedsFetchesManager(std::unique_ptr<FeedsFetchesManager>& ffm);

  // Execute the batch, by iterating the sequence in each batch entry
  // and calling the subgraph with each item in the sequence.
  Status Execute(FeedsFetchesManager* ffm, const FeedsFetchesManager* cached_ffm);

 private:
  void CreateInitialFeeds(std::vector<MLValue>& feeds);
  void UpdateFeeds(const std::vector<MLValue>& last_outputs, std::vector<MLValue>& next_inputs);

  // create the single Loop output from a collection of per-iteration outputs
  Status ConcatenateLoopOutput(std::vector<MLValue>& per_iteration_output, int output_index);

  OpKernelContextInternal& context_;
  const SessionState& session_state_;
  const GraphViewer& subgraph_;

  int64_t max_trip_count_;
  bool condition_;

  int num_loop_carried_vars_;
  int num_subgraph_inputs_;
  int num_outputs_;

  std::unordered_map<std::string, const MLValue*> implicit_inputs_;

  MLValue iter_num_mlvalue_;
  MLValue condition_mlvalue_;

  std::vector<std::string> subgraph_input_names_;
  std::vector<std::string> subgraph_output_names_;

  // collection of MLValue outputs from each loop iteration for the loop outputs.
  // the order from the subgraph matches the order from the loop output
  std::vector<std::vector<MLValue>> loop_output_tensors_;
};

Status Loop::Compute(OpKernelContext* ctx) const {
  auto ctx_internal = static_cast<OpKernelContextInternal*>(ctx);
  auto* session_state = ctx_internal->SubgraphSessionState("body");
  ORT_ENFORCE(session_state, "Subgraph SessionState was not found for 'body' attribute.");

  LoopImpl loop_impl{*ctx_internal, *session_state};

  auto status = loop_impl.Initialize();
  ORT_RETURN_IF_ERROR(status);

  // create FeedsFetchesManager if needed and call LoopImpl::Execute
  status = controlflow::detail::SubgraphExecuteHelper(cached_feeds_fetches_manager_, loop_impl);

  return status;
}

LoopImpl::LoopImpl(OpKernelContextInternal& context,
                   const SessionState& session_state)
    : context_{context},
      session_state_{session_state},
      subgraph_{*session_state.GetGraphViewer()},
      implicit_inputs_{context_.GetImplicitInputs()} {
  auto* max_trip_count_tensor = context.Input<Tensor>(0);
  max_trip_count_ = max_trip_count_tensor ? *max_trip_count_tensor->Data<int64_t>() : INT64_MAX;

  auto cond_tensor = context.Input<Tensor>(1);
  condition_ = cond_tensor ? *cond_tensor->Data<bool>() : true;

  num_loop_carried_vars_ = context.InputCount() - 2;  // skip 'M' and 'cond'
  num_subgraph_inputs_ = num_loop_carried_vars_ + 2;  // iter_num, cond, loop carried vars
  num_outputs_ = context_.OutputCount();
}

template <typename T>
static MLValue MakeScalarMLValue(AllocatorPtr& allocator, T value) {
  auto* data_type = DataTypeImpl::GetType<T>();
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(data_type,
                                                              TensorShape({1}),
                                                              allocator);

  *p_tensor->MutableData<T>() = value;

  return MLValue{p_tensor.release(),
                 DataTypeImpl::GetType<Tensor>(),
                 DataTypeImpl::GetType<Tensor>()->GetDeleteFunc()};
}

Status LoopImpl::Initialize() {
  auto status = Status::OK();

  auto& subgraph_inputs = subgraph_.GetInputs();

  // we know how many inputs we are going to call the subgraph with based on the Loop inputs,
  // and that value is in num_subgraph_inputs_.
  // validate that the subgraph has that many inputs.
  if (static_cast<size_t>(num_subgraph_inputs_) != subgraph_inputs.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Graph in 'body' attribute of Loop should have ",
                           num_subgraph_inputs_, " inputs. Found:", subgraph_.GetInputs().size());
  }

  auto& subgraph_outputs = subgraph_.GetOutputs();
  auto num_subgraph_outputs = subgraph_outputs.size();

  // check num outputs are correct. the 'cond' output from the subgraph is not a Loop output, so diff is 1
  if (num_subgraph_outputs - 1 != static_cast<size_t>(num_outputs_)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "'Loop' node has ", num_outputs_,
                           " outputs so the subgraph requires ", num_outputs_ + 1,
                           " but has ", num_subgraph_outputs);
  }

  AllocatorPtr allocator;
  status = context_.GetTempSpaceAllocator(&allocator);
  ORT_RETURN_IF_ERROR(status);

  condition_mlvalue_ = MakeScalarMLValue<bool>(allocator, condition_);
  iter_num_mlvalue_ = MakeScalarMLValue<int64_t>(allocator, 0);

  subgraph_input_names_.reserve(num_subgraph_inputs_);
  for (int i = 0; i < num_subgraph_inputs_; ++i) {
    subgraph_input_names_.push_back(subgraph_inputs[i]->Name());
  }

  subgraph_output_names_.reserve(num_subgraph_outputs);
  loop_output_tensors_.resize(num_outputs_ - num_loop_carried_vars_);

  // save list of subgraph output names in their provided order to use when fetching the results
  // from each subgraph execution. the Loop outputs will match this order.
  for (size_t i = 0; i < num_subgraph_outputs; ++i) {
    auto& output = subgraph_outputs[i];
    subgraph_output_names_.push_back(output->Name());
  }

  return status;
}

Status LoopImpl::CreateFeedsFetchesManager(std::unique_ptr<FeedsFetchesManager>& ffm) {
  auto num_implicit_inputs = implicit_inputs_.size();
  std::vector<std::string> feed_names;
  feed_names.reserve(num_subgraph_inputs_ + num_implicit_inputs);

  std::copy(subgraph_input_names_.cbegin(), subgraph_input_names_.cend(), std::back_inserter(feed_names));
  for (auto& entry : implicit_inputs_) {
    feed_names.push_back(entry.first);
  }

  FeedsFetchesInfo ffi(feed_names, subgraph_output_names_);
  auto status = FeedsFetchesManager::Create(feed_names, subgraph_output_names_, session_state_.GetMLValueNameIdxMap(),
                                            ffm);

  return status;
}

void LoopImpl::CreateInitialFeeds(std::vector<MLValue>& feeds) {
  auto num_implicit_inputs = implicit_inputs_.size();
  feeds.reserve(num_subgraph_inputs_ + num_implicit_inputs);

  // This ordering is the same as used in CreateFeedsFetchesManager
  feeds.push_back(iter_num_mlvalue_);
  feeds.push_back(condition_mlvalue_);

  // populate loop carried var inputs which conveniently start at slot 2 in both the Loop and subgraph inputs
  for (int i = 2; i < num_subgraph_inputs_; ++i) {
    feeds.push_back(*context_.GetInputMLValue(i));
  }

  // pass in implicit inputs as feeds.
  for (auto& entry : implicit_inputs_) {
    ORT_ENFORCE(entry.second, "All implicit inputs should have MLValue instances by now. ",
                entry.first, " did not.");
    feeds.push_back(*entry.second);
  }
}

void LoopImpl::UpdateFeeds(const std::vector<MLValue>& last_outputs, std::vector<MLValue>& next_inputs) {
  // last_output: cond, loop vars..., loop output...
  // next_input: iter_num, cond, loop_vars. iter_num is re-used

  // simple copy for cond and loop carried vars. start at 1 to skip iter_num in input
  for (int i = 1; i < num_subgraph_inputs_; ++i) {
    next_inputs[i] = last_outputs[i - 1];
  }

  // save loop outputs as we have to concatenate at the end
  for (int j = num_loop_carried_vars_; j < num_outputs_; ++j) {
    loop_output_tensors_[j - num_loop_carried_vars_].push_back(last_outputs[j + 1]);  // skip 'cond' in output
  }
}

Status LoopImpl::ConcatenateLoopOutput(std::vector<MLValue>& per_iteration_output, int output_index) {
  const auto& first_output = per_iteration_output.front().Get<Tensor>();
  size_t bytes_per_iteration = first_output.Size();
  const auto& per_iteration_shape = first_output.Shape();
  const auto& per_iteration_dims = per_iteration_shape.GetDims();

  // prepend number of iterations to the dimensions
  int64_t num_iterations = gsl::narrow_cast<int64_t>(per_iteration_output.size());
  std::vector<int64_t> dims{num_iterations};
  std::copy(per_iteration_dims.cbegin(), per_iteration_dims.cend(), std::back_inserter(dims));
  TensorShape output_shape{dims};

  Tensor* output = context_.Output(output_index, output_shape);

  // we can't easily use a C++ template for the tensor element type,
  // so use a span for some protection but work in bytes
  gsl::span<gsl::byte> output_span = gsl::make_span<gsl::byte>(static_cast<gsl::byte*>(output->MutableDataRaw()),
                                                               output->Size());

  for (int64_t i = 0; i < num_iterations; ++i) {
    auto& mlvalue = per_iteration_output[i];
    auto& iteration_data = mlvalue.Get<Tensor>();

    // sanity check
    if (bytes_per_iteration != iteration_data.Size()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Inconsistent shape in loop output for output ", output_index,
                             " Expected:", per_iteration_shape, " Got:", iteration_data.Shape());
    }

    auto num_bytes = iteration_data.Size();
    auto src = gsl::make_span<const gsl::byte>(static_cast<const gsl::byte*>(iteration_data.DataRaw()), num_bytes);
    auto dst = output_span.subspan(i * bytes_per_iteration, bytes_per_iteration);
    gsl::copy(src, dst);
  }

  return Status::OK();
}

Status LoopImpl::Execute(FeedsFetchesManager* ffm, const FeedsFetchesManager* cached_ffm) {
  auto status = Status::OK();

  std::vector<MLValue> feeds;
  std::vector<MLValue> fetches;

  CreateInitialFeeds(feeds);

  auto& iter_num_value = *iter_num_mlvalue_.GetMutable<Tensor>()->MutableData<int64_t>();

  while (iter_num_value < max_trip_count_ && *condition_mlvalue_.GetMutable<Tensor>()->MutableData<bool>()) {
    if (iter_num_value != 0) {
      UpdateFeeds(fetches, feeds);
      fetches.clear();
    }

    // loop carried variables can change shape across iterations, and we don't know how many iterations
    // there will be to allocate loop outputs upfront. due to that we can't use a custom fetch allocator
    // for any outputs
    if (cached_ffm) {
      status = utils::ExecuteGraphWithCachedInfo(session_state_, *cached_ffm, feeds, fetches, {},
                                                 /*sequential_execution*/ true, context_.GetTerminateFlag(),
                                                 context_.Logger());
    } else {
      status = utils::ExecuteGraph(session_state_, *ffm, feeds, fetches, {},
                                   /*sequential_execution*/ true, context_.GetTerminateFlag(), context_.Logger(),
                                   /*cache_copy_info*/ true);

      // after the first execution, use the cached information
      cached_ffm = ffm;
    }

    ORT_RETURN_IF_ERROR(status);

    condition_mlvalue_ = fetches[0];

    ++iter_num_value;
  }

  // As the loop carried variables may change shape across iterations there's no way to avoid a copy
  // as we need the final shape.
  auto copy_tensor_from_mlvalue_to_output = [this](const MLValue& input, int output_idx) {
    auto& data = input.Get<Tensor>();
    Tensor* output = context_.Output(output_idx, data.Shape());
    auto src = gsl::make_span<const gsl::byte>(static_cast<const gsl::byte*>(data.DataRaw()), data.Size());
    auto dst = gsl::make_span<gsl::byte>(static_cast<gsl::byte*>(output->MutableDataRaw()), output->Size());
    gsl::copy(src, dst);
  };

  // copy to Loop output
  if (iter_num_value != 0) {
    for (int i = 0; i < num_loop_carried_vars_; ++i) {
      // need to allocate Loop output and copy MLValue from fetches
      copy_tensor_from_mlvalue_to_output(fetches[i + 1], i);  // skip cond
    }

    for (int i = num_loop_carried_vars_; i < num_outputs_; ++i) {
      // add last output
      auto& per_iteration_outputs = loop_output_tensors_[i - num_loop_carried_vars_];
      per_iteration_outputs.push_back(fetches[i + 1]);  // skip cond

      ORT_RETURN_IF_ERROR(ConcatenateLoopOutput(per_iteration_outputs, i));
    }
  } else {
    // no iterations.
    // copy input loop carried vars to output.
    for (int i = 0; i < num_loop_carried_vars_; ++i) {
      copy_tensor_from_mlvalue_to_output(feeds[i + 2], i);  // skip iter# and cond
    }

    // create empty outputs for loop outputs
    TensorShape empty;
    for (int i = num_loop_carried_vars_; i < num_outputs_; ++i) {
      ORT_IGNORE_RETURN_VALUE(context_.Output(i, empty));
    }
  }
  return status;
}
}  // namespace onnxruntime
