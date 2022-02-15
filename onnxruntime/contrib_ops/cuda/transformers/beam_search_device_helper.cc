#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/math/topk_impl.h"
#include "core/providers/cuda/math/softmax.h"
#include "core/providers/cuda/shared_inc/accumulation_type.h"
#include "core/framework/ort_value.h"
#include "contrib_ops/cuda/bert/transformer_cuda_common.h"
#include "beam_search_impl.h"
#include <cuda_runtime.h>
#include "dump_cuda_tensor.h"

#ifdef DEBUG_BEAM_SEARCH
using namespace onnxruntime::contrib::cuda::transformers;
#endif

namespace onnxruntime {
namespace concurrency {
class ThreadPool;
}
}  // namespace onnxruntime

#include "beam_search_device_helper.h"

namespace onnxruntime {
namespace contrib {
namespace BeamSearchCudaDeviceHelper {

Status TopK(const Tensor* input, const int axis, const unsigned k, bool largest, bool sorted,
            AllocatorPtr allocator,
            void* stream,
            onnxruntime::concurrency::ThreadPool* /*threadpool*/,
            std::unique_ptr<Tensor>& output_values,
            std::unique_ptr<Tensor>& output_indices) {
  ORT_ENFORCE(nullptr != input);
  int32_t rank = static_cast<int32_t>(input->Shape().NumDimensions());

  ORT_ENFORCE(axis >= 0 && axis < rank);
  ORT_ENFORCE(k > 0 && k <= input->Shape().GetDims()[axis]);

  auto input_shape = input->Shape();
  auto output_shape = input_shape;
  output_shape[axis] = k;

  TArray<int64_t> elem_nums_cuda(input->Shape().GetDims());
  for (int32_t i = elem_nums_cuda.Size() - 2; i >= 0; --i) {
    elem_nums_cuda[i] *= elem_nums_cuda[i + 1];
  }

  int64_t dimension = input_shape[axis];
  int64_t N = elem_nums_cuda[0] / dimension;

  output_values = Tensor::Create(input->DataType(), output_shape, allocator);
  output_indices = Tensor::Create(DataTypeImpl::GetType<int64_t>(), output_shape, allocator);

  if (input->IsDataType<float>()) {
    return TopKImpl<float>(nullptr,  // We limit number of beams in BeamSearchParameters, so that K <= 256 and kernel is not needed
                           reinterpret_cast<cudaStream_t>(stream),
                           input->Data<float>(),
                           static_cast<float*>(output_values->MutableDataRaw()),
                           static_cast<int64_t*>(output_indices->MutableDataRaw()),
                           elem_nums_cuda,
                           static_cast<size_t>(elem_nums_cuda.Size()),
                           static_cast<int32_t>(axis),
                           static_cast<int64_t>(k),
                           static_cast<int64_t>(largest),
                           static_cast<int64_t>(sorted),
                           N,
                           dimension);
  } else if (input->IsDataType<MLFloat16>()) {
    return TopKImpl<MLFloat16>(nullptr,
                               reinterpret_cast<cudaStream_t>(stream),
                               input->Data<MLFloat16>(),
                               static_cast<MLFloat16*>(output_values->MutableDataRaw()),
                               static_cast<int64_t*>(output_indices->MutableDataRaw()),
                               elem_nums_cuda,
                               static_cast<size_t>(elem_nums_cuda.Size()),
                               static_cast<int32_t>(axis),
                               static_cast<int64_t>(k),
                               static_cast<int64_t>(largest),
                               static_cast<int64_t>(sorted),
                               N,
                               dimension);
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "BeamSearch op: An implementation for the input type ",
                         input->DataType(), " is not supported yet");
}

Status AddToFeeds(const IExecutionProvider* execution_provider,
                  OrtValue& input_ids,
                  OrtValue& position_ids,
                  OrtValue& attention_mask,
                  std::vector<OrtValue>& feeds,
                  IAllocatorUniquePtr<char>& buffer) {
  // Copy tensors to GPU, then add to feeds
  const CUDAExecutionProvider* provider = reinterpret_cast<const CUDAExecutionProvider*>(execution_provider);
  const TensorShape& shape = input_ids.Get<Tensor>().Shape();
  ORT_ENFORCE(shape.NumDimensions() == 2);
  const int64_t elements = shape[0] * shape[1];

  AllocatorPtr pinned_allocator = provider->GetAllocator(DEFAULT_CPU_ALLOCATOR_DEVICE_ID, OrtMemTypeCPU);
  cudaStream_t stream = static_cast<cudaStream_t>(provider->GetComputeStream());

  size_t bytes = (sizeof(int64_t) + sizeof(int64_t) + sizeof(float)) * elements;
  auto pinned_buffer = IAllocator::MakeUniquePtr<void>(pinned_allocator, bytes);
  char* pinned_data = static_cast<char*>(pinned_buffer.get());

  // Copy tensors to one pinned memory buffer (so that we only need copy to GPU once)
  memcpy(pinned_data, input_ids.Get<Tensor>().Data<int64_t>(), sizeof(int64_t) * elements);
  memcpy(pinned_data + sizeof(int64_t) * elements, position_ids.Get<Tensor>().Data<int64_t>(), sizeof(int64_t) * elements);
  memcpy(pinned_data + 2 * sizeof(int64_t) * elements, attention_mask.Get<Tensor>().Data<float>(), sizeof(float) * elements);

  if (!buffer) {
    buffer = provider->GetScratchBuffer<char>(bytes);
  }

  char* gpu_data = buffer.get();
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(gpu_data, pinned_data, bytes, cudaMemcpyHostToDevice, stream));

  // Create an event to make sure the async copy is finished before reading the data.
  onnxruntime::contrib::cuda::AutoDestoryCudaEvent new_event;
  cudaEvent_t& isCopyDone = new_event.Get();
  CUDA_RETURN_IF_ERROR(cudaEventCreate(&isCopyDone));
  CUDA_RETURN_IF_ERROR(cudaEventRecord(isCopyDone, stream));
  CUDA_RETURN_IF_ERROR(cudaEventSynchronize(isCopyDone));

  // TODO: allocate a buffer for subgraph inputs so that we can reuse the buffer in each subgraph call.
  OrtValue device_input_ids;
  OrtValue device_position_ids;
  OrtValue device_attention_mask;
  const OrtMemoryInfo& location = provider->GetAllocator(0, OrtMemTypeDefault)->Info();
  Tensor::InitOrtValue(DataTypeImpl::GetType<int64_t>(), shape, gpu_data, location, device_input_ids);
  Tensor::InitOrtValue(DataTypeImpl::GetType<int64_t>(), shape, gpu_data + sizeof(int64_t) * elements, location, device_position_ids);
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), shape, gpu_data + 2 * sizeof(int64_t) * elements, location, device_attention_mask);

  feeds.push_back(device_input_ids);
  feeds.push_back(device_position_ids);
  feeds.push_back(device_attention_mask);

  return Status::OK();
}

void InitBeamState(transformers::IBeamSearchState<float>* beam_state,
                   transformers::IBeamSearchCpuState<float>* cpu_state,
                   gsl::span<int64_t>& sequence_lengths,
                   int batch_size,
                   int num_beams,
                   gsl::span<const int64_t> input_ids_in_cpu,
                   int sequence_length,
                   int max_length,
                   void* stream) {
  // TODO: we can use another stream to avoid blocking subgraph execution.
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  cudaMemsetAsync(beam_state->next_token_logits.data(), 0, beam_state->next_token_logits.size_bytes(), cuda_stream);
  cudaMemsetAsync(beam_state->next_token_scores.data(), 0, beam_state->next_token_scores.size_bytes(), cuda_stream);
  cudaMemsetAsync(beam_state->next_tokens.data(), 0, beam_state->next_tokens.size_bytes(), cuda_stream);
  cudaMemsetAsync(beam_state->next_indices.data(), 0, beam_state->next_indices.size_bytes(), cuda_stream);

  // Initialize score of first beam of each group with 0 and the rest with -1e9.
  cuda::LaunchInitKernel(beam_state->beam_scores.data(), batch_size, num_beams, reinterpret_cast<cudaStream_t>(stream));

  // copy sequence lengths to GPU
  // since next_positions is only needed to update feeds after subgraph execution, so it is fine to use Async here.
  // cudaMemsetAsync(beam_state->next_positions.data(), 0, beam_state->next_positions.size_bytes(), cuda_stream);
  cudaMemcpyAsync(beam_state->next_positions.data(), sequence_lengths.data(), sequence_lengths.size_bytes(), cudaMemcpyHostToDevice, cuda_stream);

  memset(cpu_state->sequences_space.data(), 0, cpu_state->sequences_space.size_bytes());

  // Copy input_ids to sequences[0]
  gsl::span<int64_t> sequences_0 = cpu_state->sequences_space;
  int batch_beam_size = batch_size * num_beams;
  for (int i = 0; i < batch_beam_size; i++) {
    gsl::span<const int64_t> source = input_ids_in_cpu.subspan(i * sequence_length, sequence_length);
    gsl::span<int64_t> target = sequences_0.subspan(i * max_length, sequence_length);
    // cudaMemcpyAsync(target.data(), source.data(), source.size_bytes(), cudaMemcpyDeviceToDevice, cuda_stream);
    gsl::copy(source, target);
  }
}

Status ProcessLogits(const OrtValue& logits,                                            // logits output of subgraph
                     transformers::IBeamSearchState<float>* beam_state,                 // state
                     transformers::IBeamSearchCpuState<float>* cpu_state,               // state in CPU
                     transformers::ISequences* sequences,                               // sequences
                     AllocatorPtr& allocator,                                           // default allocator
                     onnxruntime::concurrency::ThreadPool* thread_pool,                 // thread pool (for CPU only)
                     transformers::ILogitsProcessorList<float>* /*logits_processors*/,  // logits processors
                     transformers::IBeamScorer<float>* beam_scorer,                     // beam scorer
                     const transformers::IBeamSearchParameters* parameters,             // parameters
                     int /*step*/,                                                      // iteration counter
                     void* stream,                                                      // cuda stream (for CUDA only)
                     const transformers::IConsoleDumper* dumper) {                      // tensor dumper
  int batch_size = parameters->batch_size;
  int num_beams = parameters->num_beams;
  int vocab_size = parameters->vocab_size;
  bool output_scores = parameters->output_scores;

  int batch_beam_size = batch_size * num_beams;
  const float* logits_data = logits.Get<Tensor>().Data<float>();

  // Logits has shape (batch_size * num_beams, input_length, vocab_size),
  // where input_length equals to parameters_->sequence_length for first subgraph call, and 1 for the remaining calls.
  const TensorShape& logits_shape = logits.Get<Tensor>().Shape();
  ORT_ENFORCE(logits_shape.NumDimensions() == 3);
  auto input_length = logits_shape[1];

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  // Get logits for the last token:
  //    next_token_logits = logits[:, -1, :], and the result shape is (batch_size * num_beams, vocab_size)
  // When input_length == 1, use logits directly in SoftmaxCPU below so it only need for input_length > 1.
  gsl::span<float>& next_token_logits = beam_state->next_token_logits;
  if (input_length > 1) {
    // TODO: use one kernel to replace a loop of memory copy.
    const float* current_logits = logits_data + (input_length - 1) * vocab_size;
    for (int i = 0; i < batch_beam_size; i++) {
      gsl::span<const float> source(current_logits, vocab_size);
      gsl::span<float> target = next_token_logits.subspan(i * vocab_size, vocab_size);
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target.data(), source.data(), sizeof(float) * vocab_size, cudaMemcpyDeviceToDevice, cuda_stream));
      current_logits += input_length * vocab_size;
    }
  }

#ifdef DEBUG_BEAM_SEARCH
  dumper->Print("logits", logits);
  dumper->Print("next_token_logits", next_token_logits.data(), batch_size, num_beams, vocab_size);
#endif

  // Get scores for candidates of next token: next_token_scores = log_softmax(next_token_logits, dim=-1)
  gsl::span<float>& next_token_scores = beam_state->next_token_scores;

  float* Y_data = next_token_scores.data();
  const float* X_data = input_length > 1 ? next_token_logits.data() : logits_data;
  dispatch_blockwise_softmax_forward<float, float, AccumulationType_t<float>, true>(
      cuda_stream, Y_data, X_data, vocab_size, vocab_size, batch_size * num_beams);

#ifdef DEBUG_BEAM_SEARCH
  dumper->Print("next_token_scores after softmax", next_token_scores.data(), batch_size, num_beams, vocab_size);
#endif

  // Apply all score processors that updates scores
  // logits_processors->Process(sequences, next_token_scores);
  // TODO: implement other processors.
  if (!parameters->vocab_mask.empty()) {
    cuda::LaunchVocabMaskKernel<float>(next_token_scores.data(), parameters->vocab_mask.data(),
                                       parameters->batch_size, parameters->num_beams, parameters->vocab_size,
                                       cuda_stream);
  }

#ifdef DEBUG_BEAM_SEARCH
  dumper->Print("next_token_scores after logits processor", next_token_scores.data(), batch_size, num_beams, vocab_size);
#endif

  // Add beam score to next token scores. Corresponding python code is like:
  //    next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
  cuda::LaunchAddProbsKernel(next_token_scores.data(), beam_state->beam_scores.data(), batch_size, num_beams, vocab_size, reinterpret_cast<cudaStream_t>(stream));

#ifdef DEBUG_BEAM_SEARCH
  dumper->Print("next_token_scores after adding beam_scores", next_token_scores.data(), batch_size, num_beams, vocab_size);
#endif

  if (output_scores) {
    // Append next token scores to the scores output.
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(beam_state->remaining_scores.data(), next_token_scores.data(), next_token_scores.size_bytes(), cudaMemcpyDeviceToDevice, cuda_stream));
    beam_state->remaining_scores = beam_state->remaining_scores.subspan(next_token_scores.size());
  }

  // Apply top-k selection like the following:
  //   next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
  //   next_token_scores, next_tokens = torch.topk(next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
  int64_t next_token_scores_dims[] = {batch_size, num_beams * vocab_size};
  TensorShape next_token_scores_shape(&next_token_scores_dims[0], 2);
  auto element_type = DataTypeImpl::GetType<float>();
  OrtValue next_token_scores_value;
  Tensor::InitOrtValue(element_type, next_token_scores_shape, next_token_scores.data(), allocator->Info(), next_token_scores_value);
  const Tensor& input = next_token_scores_value.Get<Tensor>();

  constexpr int axis = 1;
  const unsigned top_k = static_cast<unsigned>(2 * num_beams);
  constexpr bool largest = true;
  constexpr bool sorted = true;  // results returned in sorted order.

  std::unique_ptr<Tensor> topk_scores;
  std::unique_ptr<Tensor> topk_indices;
  ORT_RETURN_IF_ERROR(TopK(&input, axis, top_k, largest, sorted, allocator, stream, thread_pool, topk_scores, topk_indices));

#ifdef DEBUG_BEAM_SEARCH
  dumper->Print("topk_scores", *(topk_scores.get()));
  dumper->Print("topk_indices", *(topk_indices.get()));
#endif

  // Convert indices in range [0, num_beams * vocab_size) to token ID of range [0, vocab_size) like the following:
  //   next_indices = (next_tokens / vocab_size).long()
  //   next_tokens = next_tokens % vocab_size
  const int64_t* next_token_indices = topk_indices->Data<int64_t>();
  cuda::LaunchNextTokenKernel(next_token_indices, beam_state->next_indices.data(), beam_state->next_tokens.data(), batch_size, top_k, vocab_size, reinterpret_cast<cudaStream_t>(stream));

  const float* data = topk_scores->Data<float>();

#ifdef DEBUG_BEAM_SEARCH
  dumper->Print("next_scores before scorer", data, batch_size, top_k);
  dumper->Print("next_tokens before scorer", beam_state->next_tokens.data(), batch_size, top_k);
  dumper->Print("next_indices before scorer", beam_state->next_indices.data(), batch_size, top_k);
#endif

  CUDA_RETURN_IF_ERROR(cudaMemcpy(cpu_state->topk_scores.data(), data, topk_scores->Shape().Size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_RETURN_IF_ERROR(cudaMemcpy(cpu_state->topk_tokens.data(), beam_state->next_tokens.data(), beam_state->next_tokens.size_bytes(), cudaMemcpyDeviceToHost));
  CUDA_RETURN_IF_ERROR(cudaMemcpy(cpu_state->topk_indices.data(), beam_state->next_indices.data(), beam_state->next_indices.size_bytes(), cudaMemcpyDeviceToHost));

  gsl::span<const float> next_scores = gsl::make_span(cpu_state->topk_scores.data(), static_cast<typename gsl::span<float>::index_type>(topk_scores->Shape().Size()));
  gsl::span<const int64_t> next_tokens(cpu_state->topk_tokens.data(), beam_state->next_tokens.size());
  gsl::span<const int64_t> next_indices(cpu_state->topk_indices.data(), beam_state->next_indices.size());

  // TODO: convert beam scorer to CUDA
  beam_scorer->Process(
      sequences,
      next_scores,
      next_tokens,
      next_indices);
  return Status::OK();
}

Status DeviceCopy(gsl::span<float> target, gsl::span<const float> source, void* stream, int copyDirection) {
  assert(copyDirection >= 0 && copyDirection <= 3);
  if (stream == nullptr) {
    CUDA_RETURN_IF_ERROR(cudaMemcpy(target.data(), source.data(), source.size_bytes(), static_cast<cudaMemcpyKind>(copyDirection)));
  } else {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target.data(), source.data(), source.size_bytes(), static_cast<cudaMemcpyKind>(copyDirection), cuda_stream));
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
  }
  return Status::OK();
}

Status PickPastState(const std::vector<OrtValue>& last_outputs,
                     std::vector<OrtValue>& next_inputs,
                     gsl::span<const int64_t>& beam_indices,
                     AllocatorPtr allocator,
                     void* stream,
                     const transformers::IConsoleDumper* dumper) {
  for (size_t i = 1; i < last_outputs.size(); ++i) {
    const OrtValue& present = last_outputs[i];  // shape is like (2, batch_beam_size, 12, past_seq_len, 64)
    const TensorShape& past_shape = present.Get<Tensor>().Shape();

    // Create a tensor with same shape.
    // TODO: allocate one buffer for all layers, and use a CUDA kernel to copy key/value cache data.
    OrtValue past;
    auto past_type = DataTypeImpl::GetType<float>();
    Tensor::InitOrtValue(past_type, past_shape, allocator, past);

    auto block_size_per_beam = past_shape[2] * past_shape[3] * past_shape[4];
    auto past_key_size = past_shape[1] * past_shape[2] * past_shape[3] * past_shape[4];

    gsl::span<float> past_span = gsl::make_span<float>(past.GetMutable<Tensor>()->MutableData<float>(), past_shape.Size());
    gsl::span<const float> present_span = gsl::make_span<const float>(present.Get<Tensor>().Data<float>(), past_shape.Size());
    for (gsl::index j = 0; j < beam_indices.length(); j++) {
      int64_t beam_index = beam_indices[j];
      gsl::span<const float> present_key = present_span.subspan(beam_index * block_size_per_beam, block_size_per_beam);
      gsl::span<const float> present_value = present_span.subspan(past_key_size + beam_index * block_size_per_beam, block_size_per_beam);

      gsl::span<float> past_key = past_span.subspan(j * block_size_per_beam, block_size_per_beam);
      gsl::span<float> past_value = past_span.subspan(past_key_size + j * block_size_per_beam, block_size_per_beam);
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(past_key.data(), present_key.data(), present_key.size_bytes(), cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream)));
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(past_value.data(), present_value.data(), present_value.size_bytes(), cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream)));

#ifdef DEBUG_BEAM_SEARCH
      // if (i == 1)  // only dump past_0
      // {
      //   dumper->Print("past_key of beam", static_cast<int>(j), true);
      //   dumper->Print(nullptr, past_key.data(), 1, static_cast<int>(block_size_per_beam));
      //   dumper->Print("past_value of beam", static_cast<int>(j), true);
      //   dumper->Print(nullptr, past_value.data(), 1, static_cast<int>(block_size_per_beam));
      // }
#endif
    }

    next_inputs[i + 2] = past;
  }

  return Status::OK();
}

Status UpdateFeeds(
    AllocatorPtr allocator,
    void* stream,
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    OrtValue& position_ids,
    gsl::span<const int64_t> beam_next_tokens,
    gsl::span<const int64_t> beam_indices,
    int num_beams,
    const transformers::IConsoleDumper* dumper) {
  // Update input_ids with next tokens.
  int batch_beam_size = static_cast<int>(beam_next_tokens.length());
  int64_t dims[] = {batch_beam_size, 1};
  TensorShape input_ids_shape(&dims[0], 2);
  auto element_type = DataTypeImpl::GetType<int64_t>();
  OrtValue input_ids;
  Tensor::InitOrtValue(element_type, input_ids_shape, allocator, input_ids);
  int64_t* input_ids_data = input_ids.GetMutable<Tensor>()->MutableData<int64_t>();
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(input_ids_data, beam_next_tokens.data(), beam_next_tokens.size_bytes(), cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream)));
  next_inputs[0] = input_ids;

  // Update position IDs
  int64_t* position_data = position_ids.GetMutable<Tensor>()->MutableData<int64_t>();
  next_inputs[1] = position_ids;

  // Update attention mask
  const OrtValue& old_mask = next_inputs[2];
  const float* old_mask_data = old_mask.Get<Tensor>().Data<float>();
  int64_t mask_dims[] = {batch_beam_size, current_length};
  TensorShape mask_shape(&mask_dims[0], 2);
  OrtValue attention_mask;
  auto mask_type = DataTypeImpl::GetType<float>();
  Tensor::InitOrtValue(mask_type, mask_shape, allocator, attention_mask);
  float* mask_data = attention_mask.GetMutable<Tensor>()->MutableData<float>();

  // Launch kernel to update like the following:
  //  for (int i = 0; i < batch_beam_size; i++) {
  //    position_data[i]++;
  //  }
  //
  //  for (int i = 0; i < batch_beam_size; i++) {
  //    for (int j = 0; j < current_length - 1; j++) {
  //      mask_data[i * current_length + j] = old_mask_data[i * (current_length - 1) + j];
  //    }
  //    mask_data[i * current_length + current_length - 1] = 1.0f;
  //  }
  cuda::LaunchUpdateKernel<float>(old_mask_data, mask_data, position_data, batch_beam_size, current_length, reinterpret_cast<cudaStream_t>(stream));

  next_inputs[2] = attention_mask;

  // Make sure data is ready before next subgraph execution.
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));

#ifdef DEBUG_BEAM_SEARCH
  dumper->Print("input_ids", input_ids);
  dumper->Print("position_ids", position_ids);
  dumper->Print("attention_mask", attention_mask);
#endif

  // Update past state
  if (num_beams == 1) {
    // feed present_* output to past_* inputs one by one
    for (size_t i = 1; i < last_outputs.size(); ++i) {
      next_inputs[i + 2] = last_outputs[i];
    }
    return Status::OK();
  } else {
    return PickPastState(last_outputs, next_inputs, beam_indices, allocator, stream, dumper);
  }
}

}  // namespace BeamSearchCudaDeviceHelper
}  // namespace contrib
}  // namespace onnxruntime