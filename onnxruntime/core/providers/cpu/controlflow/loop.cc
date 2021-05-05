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

#include "core/framework/allocator.h"
#include "core/framework/framework_common.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/framework/session_options.h"
#include "core/framework/TensorSeq.h"

#include "gsl/gsl"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
/*
ONNX_OPERATOR_SET_SCHEMA(
    Loop,
    13,
    OpSchema()
        .SetDoc(Loop_ver13_doc)
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
            OpSchema::Variadic,
            false,
            0)
        .Output(
            0,
            "v_final_and_scan_outputs",
            "Final N loop carried dependency values then K scan_outputs. "
            "Scan outputs must be Tensors.",
            "V",
            OpSchema::Variadic,
            false)
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
        .TypeConstraint(
            "V",
            []() {
              auto t = OpSchema::all_tensor_types();
              auto s = OpSchema::all_tensor_sequence_types();
              t.insert(t.end(), s.begin(), s.end());
              return t;
            }(),
            "All Tensor and Sequence types")
        .TypeConstraint(
            "I",
            {"tensor(int64)"},
            "tensor of int64, which should be a scalar.")
        .TypeConstraint(
            "B",
            {"tensor(bool)"},
            "tensor of bool, which should be a scalar.")
        .TypeAndShapeInferenceFunction(LoopInferenceFunction));
*/

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(Loop,
                                   1, 10,
                                   KernelDefBuilder()
                                       .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                                       .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                                       .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
                                   Loop);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(Loop,
                                   11, 12,
                                   KernelDefBuilder()
                                       .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                                       .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                                       .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
                                   Loop);

ONNX_CPU_OPERATOR_KERNEL(Loop,
                         13,
                         KernelDefBuilder()
                             .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                             .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                             .TypeConstraint("V", DataTypeImpl::AllTensorAndSequenceTensorTypes()),
                         Loop);

Loop::Info::Info(const onnxruntime::Node& node, const GraphViewer& subgraph_in)
    : subgraph(subgraph_in) {
  num_loop_carried_vars = static_cast<int>(node.InputDefs().size()) - 2;  // skip 'M' and 'cond'
  num_implicit_inputs = static_cast<int>(node.ImplicitInputDefs().size());
  num_subgraph_inputs = 2 + num_loop_carried_vars;  // iter_num, cond, loop carried vars
  num_outputs = static_cast<int>(node.OutputDefs().size());

  auto& subgraph_inputs = subgraph.GetInputs();
  auto& subgraph_outputs = subgraph.GetOutputs();

  // we know how many inputs we are going to call the subgraph with based on the Loop inputs,
  // and that value is in num_subgraph_inputs.
  // validate that the subgraph has that many inputs.
  ORT_ENFORCE(static_cast<size_t>(num_subgraph_inputs) == subgraph_inputs.size(),
              "Graph in 'body' attribute of Loop should have ", num_subgraph_inputs, " inputs. Found:",
              subgraph_inputs.size());

  // check num outputs are correct. the 'cond' output from the subgraph is not a Loop output, so diff is 1
  num_subgraph_outputs = static_cast<int>(subgraph_outputs.size());
  ORT_ENFORCE(num_subgraph_outputs - 1 == num_outputs,
              "'Loop' node has ", num_outputs, " outputs so the subgraph requires ", num_outputs + 1,
              " but has ", num_subgraph_outputs);

  subgraph_input_names.reserve(num_subgraph_inputs);
  for (int i = 0; i < num_subgraph_inputs; ++i) {
    subgraph_input_names.push_back(subgraph_inputs[i]->Name());
  }

  // save list of subgraph output names in their provided order to use when fetching the results
  // from each subgraph execution. the Loop outputs will match this order.
  subgraph_output_names.reserve(num_subgraph_outputs);
  for (int i = 0; i < num_subgraph_outputs; ++i) {
    auto& output = subgraph_outputs[i];
    subgraph_output_names.push_back(output->Name());
  }
}

class LoopImpl {
 public:
  LoopImpl(OpKernelContextInternal& context,
           const SessionState& session_state,
           const Loop::Info& info,
           const Loop::ConcatOutput& concat_output_func,
           void* stream);

  // Initialize by validating all the inputs, and allocating the output tensors
  Status Initialize();

  // Execute the batch, by iterating the sequence in each batch entry
  // and calling the subgraph with each item in the sequence.
  Status Execute(const FeedsFetchesManager& cached_ffm);

 private:
  void CreateInitialFeeds(std::vector<OrtValue>& feeds);
  void SaveOutputsAndUpdateFeeds(const std::vector<OrtValue>& last_outputs, std::vector<OrtValue>& next_inputs);

  // create the single Loop output from a collection of per-iteration outputs
  Status ConcatenateLoopOutput(std::vector<OrtValue>& per_iteration_output, int output_index);

  OpKernelContextInternal& context_;
  const SessionState& session_state_;
  const Loop::Info& info_;

  int64_t max_trip_count_;
  bool condition_;

  const std::vector<const OrtValue*>& implicit_inputs_;

  OrtValue iter_num_mlvalue_;
  OrtValue condition_mlvalue_;

  // collection of OrtValue outputs from each loop iteration for the loop outputs.
  // the order from the subgraph matches the order from the loop output
  std::vector<std::vector<OrtValue>> loop_output_tensors_;

  const Loop::ConcatOutput& concat_output_func_;
  void* stream_;
};

static Status ConcatenateCpuOutput(void* /*stream*/,
                                   std::vector<OrtValue>& per_iteration_output,
                                   void* output, size_t output_size_in_bytes) {
  const auto& first_output = per_iteration_output.front().Get<Tensor>();
  const auto& per_iteration_shape = first_output.Shape();
  size_t bytes_per_iteration = first_output.SizeInBytes();

  // we can't easily use a C++ template for the tensor element type,
  // so use a span for some protection but work in bytes
  gsl::span<gsl::byte> output_span = gsl::make_span<gsl::byte>(static_cast<gsl::byte*>(output),
                                                               output_size_in_bytes);

  for (size_t i = 0, num_iterations = per_iteration_output.size(); i < num_iterations; ++i) {
    auto& ort_value = per_iteration_output[i];
    auto& iteration_data = ort_value.Get<Tensor>();

    // sanity check
    if (bytes_per_iteration != iteration_data.SizeInBytes()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Inconsistent shape in loop output for output. ",
                             " Expected:", per_iteration_shape, " Got:", iteration_data.Shape());
    }

    auto src = gsl::make_span<const gsl::byte>(static_cast<const gsl::byte*>(iteration_data.DataRaw()),
                                               bytes_per_iteration);
    auto dst = output_span.subspan(i * bytes_per_iteration, bytes_per_iteration);
    gsl::copy(src, dst);
  }

  return Status::OK();
}

void Loop::Init(const OpKernelInfo& info) {
  // make sure the attribute was present even though we don't need it here.
  // The GraphProto is loaded as a Graph instance by main Graph::Resolve,
  // and a SessionState instance for executing the subgraph is created by InferenceSession.
  // This is available via Info().GetSubgraphSessionState("attribute_name") when Compute is called.
  ONNX_NAMESPACE::GraphProto proto;
  ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("body", &proto).IsOK());
  ORT_IGNORE_RETURN_VALUE(proto);

  concat_output_func_ = ConcatenateCpuOutput;
  stream_ = nullptr;
}

std::unique_ptr<OpKernel> Loop::Create(const OpKernelInfo& info, const ConcatOutput& concat_output_func, void* stream) {
  auto result = make_unique<Loop>(info);
  result->SetConcatOutputFunc(concat_output_func);
  result->SetComputeStream(stream);
  return result;
}

common::Status Loop::SetupSubgraphExecutionInfo(const SessionState& session_state,
                                                const std::string& attribute_name,
                                                const SessionState& subgraph_session_state) {
  ORT_ENFORCE(info_ == nullptr, "SetupSubgraphExecutionInfo should only be called once for each subgraph.");
  ORT_UNUSED_PARAMETER(attribute_name);

  const auto& node = Node();
  info_ = std::make_unique<Loop::Info>(node, subgraph_session_state.GetGraphViewer());

  // the Loop inputs are matched to subgraph feeds based on order.
  // we first need the names of the Loop inputs to determine what device they are available on
  std::vector<std::string> feed_names;
  feed_names.reserve(info_->num_subgraph_inputs + info_->num_implicit_inputs);

  // iter_num and cond subgraph inputs - created by the LoopImpl::Initialize so the name doesn't matter
  // as we skip them when we call FindDevicesForValues, and default them to always being on CPU.
  feed_names.push_back(info_->subgraph_input_names[0]);
  feed_names.push_back(info_->subgraph_input_names[1]);

  // add the names for the loop carried vars from the Loop input
  const auto& loop_inputs = node.InputDefs();
  for (int i = 0; i < info_->num_loop_carried_vars; ++i) {
    // + 2 to skip 'M' and 'cond' Loop inputs
    feed_names.push_back(loop_inputs[i + 2]->Name());
  }

  for (auto& entry : node.ImplicitInputDefs()) {
    feed_names.push_back(entry->Name());
  }

  // iter_num and cond are created on CPU via MakeScalarMLValue so skip those (they will correctly default to CPU).
  // use the SessionState from the control flow node to find the remaining input locations.
  size_t start_at = 2;
  std::vector<OrtDevice> feed_locations;
  ORT_RETURN_IF_ERROR(controlflow::detail::FindDevicesForValues(session_state, feed_names, feed_locations, start_at));

  // now update the feed names to use the subgraph input names for the loop carried vars so that we can determine
  // what device the subgraph needs them on
  for (int i = 0; i < info_->num_loop_carried_vars; ++i) {
    // +2 for both to skip the iter_num and cond values
    feed_names[i + 2] = info_->subgraph_input_names[i + 2];
  }

  std::unique_ptr<FeedsFetchesManager> ffm;
  ORT_RETURN_IF_ERROR(FeedsFetchesManager::Create(feed_names, info_->subgraph_output_names,
                                                  subgraph_session_state.GetOrtValueNameIdxMap(), ffm));
  ORT_RETURN_IF_ERROR(utils::InitializeFeedFetchCopyInfo(subgraph_session_state, *ffm));

  // setup the locations where we want the subgraph output to end up on
  std::vector<const OrtMemoryInfo*> fetch_locations;
  fetch_locations.reserve(info_->num_subgraph_outputs);

  // 'cond' is first output and we need it to be on CPU so we can read the latest value
  const auto& cpu_allocator_info = session_state.GetExecutionProviders()
                                       .Get(onnxruntime::kCpuExecutionProvider)
                                       ->GetAllocator(0, OrtMemTypeDefault)
                                       ->Info();
  fetch_locations.push_back(&cpu_allocator_info);

  // Loop state variables need to be where we can feed them in to the next iteration, so set the fetch location
  // to match the feed location.
  for (int i = 0; i < info_->num_loop_carried_vars; ++i) {
    // +2 for both to skip the iter_num and cond input values
    const auto& alloc_info = utils::FindMemoryInfoForValue(session_state, loop_inputs[i + 2]->Name());
    fetch_locations.push_back(&alloc_info);
  }

  // remaining outputs we want where the matching Loop output will be allocated
  const auto& loop_outputs = node.OutputDefs();
  for (size_t i = info_->num_loop_carried_vars, end = loop_outputs.size(); i < end; ++i) {
    const auto& alloc_info = utils::FindMemoryInfoForValue(session_state, loop_outputs[i]->Name());
    fetch_locations.push_back(&alloc_info);
  }

  utils::FinalizeFeedFetchCopyInfo(*ffm, feed_locations, fetch_locations);

  feeds_fetches_manager_ = std::move(ffm);

  return Status::OK();
}

Status Loop::Compute(OpKernelContext* ctx) const {
  auto* ctx_internal = static_cast<OpKernelContextInternal*>(ctx);
  auto* session_state = ctx_internal->SubgraphSessionState("body");
  ORT_ENFORCE(session_state, "Subgraph SessionState was not found for 'body' attribute.");
  ORT_ENFORCE(feeds_fetches_manager_, "CreateFeedsFetchesManager must be called prior to execution of graph.");

  LoopImpl loop_impl{*ctx_internal, *session_state, *info_, concat_output_func_, stream_};

  auto status = loop_impl.Initialize();
  ORT_RETURN_IF_ERROR(status);

  status = loop_impl.Execute(*feeds_fetches_manager_);

  return status;
}

LoopImpl::LoopImpl(OpKernelContextInternal& context,
                   const SessionState& session_state,
                   const Loop::Info& subgraph_info,
                   const Loop::ConcatOutput& concat_output_func,
                   void* stream)
    : context_(context),
      session_state_(session_state),
      info_(subgraph_info),
      implicit_inputs_(context_.GetImplicitInputs()),
      concat_output_func_(concat_output_func),
      stream_(stream) {
  auto* max_trip_count_tensor = context.Input<Tensor>(0);
  max_trip_count_ = max_trip_count_tensor ? *max_trip_count_tensor->Data<int64_t>() : INT64_MAX;

  auto cond_tensor = context.Input<Tensor>(1);
  condition_ = cond_tensor ? *cond_tensor->Data<bool>() : true;
}

Status LoopImpl::Initialize() {
  auto status = Status::OK();

  auto* max_trip_count_tensor = context_.Input<Tensor>(0);
  auto* cond_tensor = context_.Input<Tensor>(1);

  if (max_trip_count_tensor) {
    if (max_trip_count_tensor->Shape().Size() != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "'Loop' input 'M' should be a scalar tensor. Got shape of ",
                             max_trip_count_tensor->Shape());
    }
  }

  if (cond_tensor) {
    if (cond_tensor->Shape().Size() != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "'Loop' input 'cond' should be a scalar tensor. Got shape of ",
                             cond_tensor->Shape());
    }
  }

  auto& subgraph_inputs = info_.subgraph.GetInputs();

  // we need to know if the subgraph expects a rank 0 or rank 1 value for these, so a shape is required.
  ORT_RETURN_IF(subgraph_inputs[0]->Shape() == nullptr, "Loop subgraph input 0 has unknown shape: ",
                subgraph_inputs[0]->Name());
  ORT_RETURN_IF(subgraph_inputs[1]->Shape() == nullptr, "Loop subgraph input 1 has unknown shape: ",
                subgraph_inputs[1]->Name());

  auto iter_num_rank = subgraph_inputs[0]->Shape()->dim_size();
  auto condition_rank = subgraph_inputs[1]->Shape()->dim_size();

  // these need to be on CPU
  auto cpu_allocator = session_state_.GetExecutionProviders()
                           .Get(onnxruntime::kCpuExecutionProvider)
                           ->GetAllocator(0, OrtMemTypeDefault);
  iter_num_mlvalue_ = MakeScalarMLValue<int64_t>(cpu_allocator, 0, iter_num_rank);
  condition_mlvalue_ = MakeScalarMLValue<bool>(cpu_allocator, condition_, condition_rank);

  loop_output_tensors_.resize(info_.num_outputs - info_.num_loop_carried_vars);

  return status;
}

void LoopImpl::CreateInitialFeeds(std::vector<OrtValue>& feeds) {
  feeds.reserve(info_.num_subgraph_inputs + info_.num_implicit_inputs);

  // This ordering is the same as used in SetupSubgraphExecutionInfo
  feeds.push_back(iter_num_mlvalue_);
  feeds.push_back(condition_mlvalue_);

  // populate loop carried var inputs which conveniently start at slot 2 in both the Loop and subgraph inputs
  for (int i = 2; i < info_.num_subgraph_inputs; ++i) {
    feeds.push_back(*context_.GetInputMLValue(i));
  }

  // pass in implicit inputs as feeds. order matches
  for (const auto* entry : implicit_inputs_) {
    feeds.push_back(*entry);
  }
}

void LoopImpl::SaveOutputsAndUpdateFeeds(const std::vector<OrtValue>& last_outputs,
                                         std::vector<OrtValue>& next_inputs) {
  // last_output: cond, loop vars..., loop output...
  // next_input: iter_num, cond, loop_vars. iter_num is re-used

  // simple copy for cond and loop carried vars. start at 1 to skip iter_num in input
  for (int i = 1; i < info_.num_subgraph_inputs; ++i) {
    next_inputs[i] = last_outputs[i - 1];
  }

  // save loop outputs as we have to concatenate at the end
  for (int j = info_.num_loop_carried_vars; j < info_.num_outputs; ++j) {
    ORT_ENFORCE(last_outputs[j + 1].IsTensor(), "All scan outputs MUST be tensors");
    loop_output_tensors_[j - info_.num_loop_carried_vars].push_back(last_outputs[j + 1]);  // skip 'cond' in output
  }
}

Status LoopImpl::ConcatenateLoopOutput(std::vector<OrtValue>& per_iteration_output, int output_index) {
  const auto& first_output = per_iteration_output.front().Get<Tensor>();
  const auto& per_iteration_dims = first_output.Shape().GetDims();

  std::vector<int64_t> dims;
  dims.reserve(1 + per_iteration_output.size());

  // first dimension is number of iterations
  dims.push_back(gsl::narrow_cast<int64_t>(per_iteration_output.size()));
  std::copy(per_iteration_dims.cbegin(), per_iteration_dims.cend(), std::back_inserter(dims));

  TensorShape output_shape{dims};
  Tensor* output = context_.Output(output_index, output_shape);

  ORT_RETURN_IF_ERROR(concat_output_func_(stream_, per_iteration_output, output->MutableDataRaw(), output->SizeInBytes()));

  return Status::OK();
}

Status LoopImpl::Execute(const FeedsFetchesManager& ffm) {
  auto status = Status::OK();

  std::vector<OrtValue> feeds;
  std::vector<OrtValue> fetches;

  CreateInitialFeeds(feeds);

  auto& iter_num_value = *iter_num_mlvalue_.GetMutable<Tensor>()->MutableData<int64_t>();

  while (iter_num_value < max_trip_count_ && *condition_mlvalue_.GetMutable<Tensor>()->MutableData<bool>()) {
    if (iter_num_value != 0) {
      SaveOutputsAndUpdateFeeds(fetches, feeds);
      fetches.clear();
    }

    status = utils::ExecuteSubgraph(session_state_, ffm, feeds, fetches, {},
                                    ExecutionMode::ORT_SEQUENTIAL, context_.GetTerminateFlag(), context_.Logger());

    ORT_RETURN_IF_ERROR(status);

    condition_mlvalue_ = fetches[0];

    ++iter_num_value;
  }

  // As the loop carried variables may change shape across iterations there's no way to avoid a copy
  // as we need the final shape.
  auto copy_tensor_from_mlvalue_to_output = [this](const OrtValue& input, int output_idx) {
    auto type = input.Type();
    if (type == DataTypeImpl::GetType<Tensor>()) {
      auto& data = input.Get<Tensor>();
      Tensor* output = context_.Output(output_idx, data.Shape());
      session_state_.GetDataTransferMgr().CopyTensor(input.Get<Tensor>(), *output);
    } else if (type == DataTypeImpl::GetType<TensorSeq>()) {
      std::vector<Tensor> tensors;

      auto& data = input.Get<TensorSeq>();
      TensorSeq* output = context_.Output<TensorSeq>(output_idx);
      output->SetType(data.DataType());

      AllocatorPtr alloc;
      auto status = context_.GetTempSpaceAllocator(&alloc);
      if (!status.IsOK()) {
        ORT_THROW("Unable to get an allocator");
      }
      for (auto it = data.begin(), end = data.end(); it != end; ++it) {
        Tensor tmp(it->DataType(), onnxruntime::TensorShape(it->Shape()), alloc);
        session_state_.GetDataTransferMgr().CopyTensor(*it, tmp);
        tensors.push_back(std::move(tmp));
      }

      output->SetElements(std::move(tensors));
    }
  };

  // copy to Loop output
  if (iter_num_value != 0) {
    for (int i = 0; i < info_.num_loop_carried_vars; ++i) {
      // need to allocate Loop output and copy OrtValue from fetches
      copy_tensor_from_mlvalue_to_output(fetches[i + 1], i);  // skip cond
    }

    for (int i = info_.num_loop_carried_vars; i < info_.num_outputs; ++i) {
      // add last output
      auto& per_iteration_outputs = loop_output_tensors_[i - info_.num_loop_carried_vars];
      per_iteration_outputs.push_back(fetches[i + 1]);  // skip cond

      ORT_RETURN_IF_ERROR(ConcatenateLoopOutput(per_iteration_outputs, i));
    }
  } else {
    // no iterations.
    // copy input loop carried vars to output.
    for (int i = 0; i < info_.num_loop_carried_vars; ++i) {
      copy_tensor_from_mlvalue_to_output(feeds[i + 2], i);  // skip iter# and cond
    }

    // create empty outputs for loop outputs using the subgraph output shapes for the rank
    auto& graph_outputs = info_.subgraph.GetOutputs();

    for (int i = info_.num_loop_carried_vars; i < info_.num_outputs; ++i) {
      // get shape from subgraph output if possible to attempt to have the correct rank
      auto* graph_output = graph_outputs.at(i + 1);  // + 1 as first subgraph output is condition value
      auto* graph_output_shape = graph_output->Shape();

      std::vector<int64_t> output_dims;
      output_dims.reserve((graph_output_shape ? graph_output_shape->dim_size() : 0) + 1);
      output_dims.push_back(0);  // num iterations is first dim

      if (graph_output_shape) {
        const auto& tensor_shape = onnxruntime::utils::GetTensorShapeFromTensorShapeProto(*graph_output_shape);
        const auto& dims = tensor_shape.GetDims();

        // copy to output dims and use 0 for any symbolic dim
        std::for_each(dims.cbegin(), dims.cend(),
                      [&output_dims](const int64_t dim) { output_dims.push_back(dim < 0 ? 0 : dim); });
      } else {
        // TODO: We could try and call ExecuteGraph to get the output shape from fetches so the rank is correct,
        // however that could still fail as we would potentially be passing in invalid data.
        // Until we know this is required just output a warning and return the rank 1 empty output.
        LOGS(context_.Logger(), WARNING) << "Loop had zero iterations and the shape of subgraph output " << i + 1
                                         << " was not found. Defaulting to a rank 1 shape of {0}.";
      }

      ORT_IGNORE_RETURN_VALUE(context_.Output(i, TensorShape(output_dims)));
    }
  }
  return status;
}
}  // namespace onnxruntime
