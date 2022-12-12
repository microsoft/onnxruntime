// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <utility>
#include <memory>
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/math/topk_impl.h"
#include "core/providers/cuda/math/softmax.h"
#include "core/providers/cuda/shared_inc/accumulation_type.h"
#include "core/framework/ort_value.h"
#include "contrib_ops/cuda/bert/transformer_cuda_common.h"
#include <cuda_runtime.h>
#include "contrib_ops/cuda/transformers/beam_search_impl.h"
#include "contrib_ops/cuda/transformers/dump_cuda_tensor.h"
#include "contrib_ops/cpu/transformers/subgraph_t5_decoder.h"
#include "contrib_ops/cpu/transformers/subgraph_gpt.h"
#include "contrib_ops/cuda/transformers/beam_search_topk.h"

namespace onnxruntime {
namespace concurrency {
class ThreadPool;
}
}  // namespace onnxruntime

#include "generation_device_helper.h"

namespace onnxruntime {
namespace contrib {
namespace GenerationCudaDeviceHelper {

Status TopK(const Tensor* input, const int axis, const unsigned k, bool largest, bool sorted,
            AllocatorPtr allocator,
            void* stream,
            onnxruntime::concurrency::ThreadPool* /*threadpool*/,
            Tensor& output_values,
            Tensor& output_indices) {
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

  output_values = std::move(*Tensor::Create(input->DataType(), output_shape, allocator));
  output_indices = std::move(*Tensor::Create(DataTypeImpl::GetType<int64_t>(), output_shape, std::move(allocator)));

  if (input->IsDataType<float>()) {
    return TopKImpl<float>(nullptr,  // We limit number of beams in BeamSearchParameters, so K <= 256 and use NULL here
                           reinterpret_cast<cudaStream_t>(stream),
                           input->Data<float>(),
                           static_cast<float*>(output_values.MutableDataRaw()),
                           static_cast<int64_t*>(output_indices.MutableDataRaw()),
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
                               static_cast<MLFloat16*>(output_values.MutableDataRaw()),
                               static_cast<int64_t*>(output_indices.MutableDataRaw()),
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
                  std::initializer_list<OrtValue> inputs,
                  std::vector<OrtValue>& feeds,
                  IAllocatorUniquePtr<char>& buffer) {
  // Copy tensors to GPU, then add to feeds
  const CUDAExecutionProvider* provider = reinterpret_cast<const CUDAExecutionProvider*>(execution_provider);
  size_t total_bytes = 0;
  for (auto& input : inputs) {
    if (input.IsAllocated()) {
      total_bytes += input.Get<Tensor>().SizeInBytes();
    }
  }

  ORT_ENFORCE(total_bytes > 0);

  AllocatorPtr pinned_allocator = provider->GetAllocator(DEFAULT_CPU_ALLOCATOR_DEVICE_ID, OrtMemTypeCPU);
  cudaStream_t stream = static_cast<cudaStream_t>(provider->GetComputeStream());
  auto pinned_buffer = IAllocator::MakeUniquePtr<void>(pinned_allocator, total_bytes);
  char* pinned_data = static_cast<char*>(pinned_buffer.get());

  // Copy tensors to one pinned memory buffer (so that we only need copy to GPU once)
  char* destination = pinned_data;
  for (auto& input : inputs) {
    if (input.IsAllocated()) {
      const Tensor& tensor = input.Get<Tensor>();
      const size_t bytes = tensor.SizeInBytes();
      MLDataType dataType = tensor.DataType();
      if (dataType == DataTypeImpl::GetType<int32_t>()) {
        memcpy(destination, input.Get<Tensor>().Data<int32_t>(), bytes);
      } else if (dataType == DataTypeImpl::GetType<int64_t>()) {
        memcpy(destination, input.Get<Tensor>().Data<int64_t>(), bytes);
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                               "AddToFeeds: An implementation for the input type ",
                               dataType, " is not supported yet");
      }

      // Do not need alignment because GPT has int32 inputs (past is empty) and T5 encoder has int64 inputs.
      destination += bytes;
    }
  }

  if (!buffer) {
    buffer = provider->GetScratchBuffer<char>(total_bytes);
  }

  char* gpu_data = buffer.get();
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(gpu_data, pinned_data, total_bytes, cudaMemcpyHostToDevice, stream));

  // Create an event to make sure the async copy is finished before reading the data.
  onnxruntime::contrib::cuda::AutoDestoryCudaEvent new_event;
  cudaEvent_t& isCopyDone = new_event.Get();
  CUDA_RETURN_IF_ERROR(cudaEventCreate(&isCopyDone));
  CUDA_RETURN_IF_ERROR(cudaEventRecord(isCopyDone, stream));
  CUDA_RETURN_IF_ERROR(cudaEventSynchronize(isCopyDone));

  // TODO(tianleiwu): allocate a buffer for subgraph inputs so that we can reuse the buffer in each subgraph call.
  const OrtMemoryInfo& location = provider->GetAllocator(0, OrtMemTypeDefault)->Info();
  for (auto& input : inputs) {
    if (input.IsAllocated()) {
      const Tensor& tensor = input.Get<Tensor>();
      const TensorShape& shape = tensor.Shape();
      const size_t bytes = tensor.SizeInBytes();
      MLDataType dataType = tensor.DataType();

      OrtValue device_input;
      Tensor::InitOrtValue(dataType, shape, gpu_data, location, device_input);
      gpu_data += bytes;
      feeds.push_back(device_input);
    }
  }

  return Status::OK();
}

template <typename T>
void InitBeamState(transformers::IBeamSearchState<T>* beam_state,
                   gsl::span<int32_t>& sequence_lengths,
                   int batch_size,
                   int num_beams,
                   void* stream) {
  // TODO(tianleiwu): we can use another stream to avoid blocking subgraph execution.
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  cudaMemsetAsync(beam_state->next_token_logits.data(), 0, beam_state->next_token_logits.size_bytes(), cuda_stream);
  cudaMemsetAsync(beam_state->next_token_scores.data(), 0, beam_state->next_token_scores.size_bytes(), cuda_stream);
  cudaMemsetAsync(beam_state->next_tokens.data(), 0, beam_state->next_tokens.size_bytes(), cuda_stream);
  cudaMemsetAsync(beam_state->next_indices.data(), 0, beam_state->next_indices.size_bytes(), cuda_stream);
  cudaMemsetAsync(beam_state->next_scores.data(), 0, beam_state->next_scores.size_bytes(), cuda_stream);
  cudaMemsetAsync(beam_state->topk_buffer.data(), 0, beam_state->topk_buffer.size_bytes(), cuda_stream);

  // Initialize score of first beam of each group with 0 and the rest with -1e9.
  cuda::LaunchInitKernel(beam_state->beam_scores.data(), batch_size, num_beams, reinterpret_cast<cudaStream_t>(stream));

  // copy sequence lengths to GPU
  // since next_positions is only needed to update feeds after subgraph execution, so it is fine to use Async here.
  if (!beam_state->next_positions.empty()) {  // next_positions is empty for T5
    cudaMemcpyAsync(beam_state->next_positions.data(), sequence_lengths.data(), sequence_lengths.size_bytes(),
                    cudaMemcpyHostToDevice, cuda_stream);
  }
}

template <typename T>
void InitGreedyState(transformers::IGreedySearchState<T>* greedy_state,
                     gsl::span<int32_t>& sequence_lengths,
                     void* stream) {
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  cudaMemsetAsync(greedy_state->next_token_scores.data(), 0, greedy_state->next_token_scores.size_bytes(), cuda_stream);
  cudaMemsetAsync(greedy_state->next_positions.data(), 0, greedy_state->next_positions.size_bytes(), cuda_stream);

  cudaMemcpyAsync(greedy_state->next_positions.data(), sequence_lengths.data(), sequence_lengths.size_bytes(),
                  cudaMemcpyHostToDevice, cuda_stream);
}

template <typename T>
Status ProcessLogits(const OrtValue& logits,                                 // logits output of subgraph
                     transformers::IBeamSearchState<T>* beam_state,          // state
                     transformers::IBeamSearchCpuState* cpu_state,           // state in CPU
                     transformers::ISequences* sequences,                    // sequences
                     AllocatorPtr& allocator,                                // default allocator
                     onnxruntime::concurrency::ThreadPool* thread_pool,      // thread pool (for CPU only)
                     transformers::ILogitsProcessorList* logits_processors,  // logits processors
                     transformers::IBeamScorer* beam_scorer,                 // beam scorer
                     const transformers::IBeamSearchParameters* parameters,  // parameters
                     int step,                                               // iteration counter
                     void* stream,                                           // cuda stream (for CUDA only)
                     const transformers::IConsoleDumper* dumper) {           // tensor dumper

  ORT_UNUSED_PARAMETER(logits_processors);
  ORT_UNUSED_PARAMETER(thread_pool);

#ifndef DEBUG_GENERATION
  ORT_UNUSED_PARAMETER(dumper);
#endif

  int batch_size = parameters->batch_size;
  int num_beams = parameters->num_beams;
  int vocab_size = parameters->vocab_size;
  bool output_scores = parameters->output_scores;

  int batch_beam_size = batch_size * num_beams;

  typedef typename ToCudaType<T>::MappedType CudaT;
  const CudaT* logits_data = reinterpret_cast<const CudaT*>(logits.Get<Tensor>().Data<T>());

  // Logits has shape (batch_size * num_beams, input_length, padded_vocab_size),
  // where input_length equals to parameters_->sequence_length for first subgraph call, and 1 for the remaining calls.
  const TensorShape& logits_shape = logits.Get<Tensor>().Shape();
  ORT_ENFORCE(logits_shape.NumDimensions() == 3);
  auto input_length = logits_shape[1];
  auto logits_batch_size = logits_shape[0];

  // NOTE: `padded_vocab_size` MAY be different from `vocab_size`.
  // But the following implementation should work correctly if they are the same
  // or different.
  auto padded_vocab_size = static_cast<int>(logits_shape[2]);

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  // Get logits for the last token:
  //    next_token_logits = logits[:, -1, :], and the result shape is (batch_size * num_beams, vocab_size)
  // When input_length == 1, use logits directly in SoftmaxCPU below so it only need for input_length > 1.
  gsl::span<T>& next_token_logits = beam_state->next_token_logits;
  // TODO(tianleiwu): use one kernel to replace a loop of memory copy.
  if (input_length > 1 || logits_batch_size == batch_size) {
    // Move the pointer in increments of padded_vocab_size to account for any padding
    // if any in the logits weight of the MatMul.
    const CudaT* current_logits = logits_data + (input_length - 1) * padded_vocab_size;
    for (int i = 0; i < batch_beam_size; i++) {
      // We only copy what is relevant (i.e.) vocab_size as padded_vocab_size will contain
      // some logits corresponding to the "padded" vocab size which we will ignore
      // for token generation.
      gsl::span<const T> source(reinterpret_cast<const T*>(current_logits), vocab_size);
      gsl::span<T> target = next_token_logits.subspan(static_cast<size_t>(i) * vocab_size, vocab_size);
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target.data(), source.data(), sizeof(T) * vocab_size,
                                           cudaMemcpyDeviceToDevice, cuda_stream));
      if (logits_batch_size == batch_beam_size) {
        current_logits += input_length * padded_vocab_size;
      } else if (logits_batch_size == batch_size && i % num_beams == num_beams - 1) {
        current_logits += input_length * padded_vocab_size;
      }
    }
  }

#ifdef DEBUG_GENERATION
  dumper->Print("logits", logits);
  if (input_length > 1 || logits_batch_size == batch_size) {
    dumper->Print("next_token_logits", next_token_logits.data(), batch_size, num_beams, vocab_size);
  }
#endif

  // Get scores for candidates of next token: next_token_scores = log_softmax(next_token_logits, dim=-1)
  gsl::span<float>& next_token_scores = beam_state->next_token_scores;

  // The output will be float for consideration of precision and easy integration with remaining parts.
  float* Y_data = next_token_scores.data();
  bool is_reuse_logits_buffer = (input_length == 1 && logits_batch_size == batch_beam_size);

  const CudaT* X_data = is_reuse_logits_buffer ? logits_data : reinterpret_cast<const CudaT*>(next_token_logits.data());

  dispatch_blockwise_softmax_forward<CudaT, float, float, true>(
      cuda_stream, Y_data, X_data, vocab_size,
      is_reuse_logits_buffer ? padded_vocab_size : vocab_size,
      vocab_size,
      batch_size * num_beams);

#ifdef DEBUG_GENERATION
  dumper->Print("next_token_scores after softmax", next_token_scores.data(), batch_size, num_beams, vocab_size);
#endif

  // Sequences generated by beam scorer is currently stored in CPU.
  // Copy sequences to device only when repetition penalty or no repeat ngram is used in kernel
  BufferUniquePtr sequences_buffer;
  int current_sequence_length = sequences->GetSequenceLength();
  bool run_ngram = parameters->no_repeat_ngram_size > 0 && current_sequence_length >= parameters->no_repeat_ngram_size;
  if (parameters->repetition_penalty != 1.0f || run_ngram) {
    size_t bytes = SafeInt<size_t>(sizeof(int32_t)) * batch_beam_size * parameters->max_length;
    void* data = allocator->Alloc(bytes);
    BufferUniquePtr temp_buffer(data, BufferDeleter(allocator));
    sequences_buffer = std::move(temp_buffer);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(sequences_buffer.get(), sequences->GetSequence(0).data(), bytes,
                                         cudaMemcpyHostToDevice, cuda_stream));
  }

  cuda::LaunchLogitsProcessKernel<float>(
      next_token_scores.data(),
      parameters->vocab_mask.data(),
      step > 1 ? nullptr : parameters->prefix_vocab_mask.data(),  // prefix vocab mask is applied to first step only.
      parameters->batch_size,
      parameters->num_beams,
      parameters->vocab_size,
      (parameters->min_length > 0 && current_sequence_length < parameters->min_length) ? parameters->eos_token_id : -1,
      reinterpret_cast<int32_t*>(sequences_buffer.get()),
      parameters->max_length,
      current_sequence_length,
      parameters->repetition_penalty,
      parameters->no_repeat_ngram_size,
      cuda_stream);

#ifdef DEBUG_GENERATION
  dumper->Print("next_token_scores after logits process", next_token_scores.data(), batch_size, num_beams, vocab_size);
#endif
  // Add beam score to next token scores. Corresponding python code is like:
  //    next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
  cuda::LaunchAddProbsKernel(next_token_scores.data(), beam_state->beam_scores.data(),
                             batch_size, num_beams, vocab_size, cuda_stream);

#ifdef DEBUG_GENERATION
  dumper->Print("next_token_scores adding beam_scores", next_token_scores.data(), batch_size, num_beams, vocab_size);
#endif

  if (output_scores) {
    // Append next token scores to the scores output.
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(beam_state->remaining_scores.data(),
                                         next_token_scores.data(),
                                         next_token_scores.size_bytes(),
                                         cudaMemcpyDeviceToDevice,
                                         cuda_stream));
    beam_state->remaining_scores = beam_state->remaining_scores.subspan(next_token_scores.size());
  }

  if (num_beams <= 32) {
    constexpr size_t max_parts_of_vocab = 128;
    float* topk_tmp_buffer = beam_state->topk_buffer.data();
    float* topk_scores_1st_stage = topk_tmp_buffer;
    int32_t* topk_tokens_1st_stage = reinterpret_cast<int32_t*>(topk_scores_1st_stage + batch_beam_size * max_parts_of_vocab * 2 * num_beams);
    float* topk_scores_2nd_stage = reinterpret_cast<float*>(topk_tokens_1st_stage + batch_beam_size * max_parts_of_vocab * 2 * num_beams);
    int32_t* topk_tokens_2nd_stage = reinterpret_cast<int32_t*>(topk_scores_2nd_stage + batch_beam_size * 2 * num_beams);

    cuda::BeamSearchTopK(next_token_scores.data(),
                         batch_size,
                         num_beams,
                         vocab_size,
                         2 * num_beams,
                         topk_scores_1st_stage,
                         topk_tokens_1st_stage,
                         topk_scores_2nd_stage,
                         topk_tokens_2nd_stage,
                         beam_state->next_scores.data(),
                         beam_state->next_tokens.data(),
                         beam_state->next_indices.data(),
                         cuda_stream);

    // Select [batch_size, 2 * num_beams] from [batch_size * num_beams, 2 * num_beams]
#ifdef DEBUG_GENERATION
    dumper->Print("next_tokens before scorer", beam_state->next_tokens.data(), batch_size, 2 * num_beams);
    dumper->Print("next_indices before scorer", beam_state->next_indices.data(), batch_size, 2 * num_beams);
    dumper->Print("next_scores before scorer", beam_state->next_scores.data(), batch_size, 2 * num_beams);
#endif
  } else {
    // Apply top-k selection like the following:
    //   next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
    //   next_token_scores, next_tokens = torch.topk(next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
    // int64_t next_token_scores_dims[] = {batch_size, num_beams * vocab_size};
    int64_t next_token_scores_dims[] = {batch_size * num_beams, vocab_size};

    TensorShape next_token_scores_shape(&next_token_scores_dims[0], 2);
    auto element_type = DataTypeImpl::GetType<float>();
    OrtValue next_token_scores_value;
    Tensor::InitOrtValue(element_type, next_token_scores_shape, next_token_scores.data(), allocator->Info(),
                         next_token_scores_value);
    const Tensor& input = next_token_scores_value.Get<Tensor>();

    constexpr int axis = 1;
    const unsigned top_k = static_cast<unsigned>(2 * num_beams);
    constexpr bool largest = true;
    constexpr bool sorted = true;  // results returned in sorted order.

    std::unique_ptr<Tensor> topk_scores = Tensor::CreateDefault();
    std::unique_ptr<Tensor> topk_tokens = Tensor::CreateDefault();
    ORT_RETURN_IF_ERROR(TopK(&input, axis, top_k, largest, sorted, allocator, stream, thread_pool,
                             *topk_scores, *topk_tokens));

#ifdef DEBUG_GENERATION
    dumper->Print("topk_scores", *(topk_scores.get()));
    dumper->Print("topk_tokens", *(topk_tokens.get()));
#endif

    cuda::LaunchBatchTopKKernel(topk_scores->Data<float>(),
                                topk_tokens->Data<int64_t>(),
                                beam_state->next_indices.data(),
                                beam_state->next_tokens.data(),
                                beam_state->next_scores.data(),
                                batch_size,
                                num_beams,
                                2 * num_beams,
                                cuda_stream);
  }

  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(cpu_state->topk_scores.data(),
                                       beam_state->next_scores.data(),
                                       beam_state->next_scores.size_bytes(),
                                       cudaMemcpyDeviceToHost,
                                       cuda_stream));
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(cpu_state->topk_tokens.data(),
                                       beam_state->next_tokens.data(),
                                       beam_state->next_tokens.size_bytes(),
                                       cudaMemcpyDeviceToHost,
                                       cuda_stream));
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(cpu_state->topk_indices.data(),
                                       beam_state->next_indices.data(),
                                       beam_state->next_indices.size_bytes(),
                                       cudaMemcpyDeviceToHost,
                                       cuda_stream));
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(cuda_stream));

  gsl::span<const float> next_scores(cpu_state->topk_scores.data(), beam_state->next_scores.size());
  gsl::span<const int32_t> next_tokens(cpu_state->topk_tokens.data(), beam_state->next_tokens.size());
  gsl::span<const int32_t> next_indices(cpu_state->topk_indices.data(), beam_state->next_indices.size());

  // Limitation: beam scorer runs in CPU. It might be better to use CUDA kernel to replace it.
  beam_scorer->Process(
      sequences,
      next_scores,
      next_tokens,
      next_indices);
  return Status::OK();
}

template <typename T>
Status GreedySearchProcessLogits(
    const OrtValue& logits,                                 // logits output of subgraph
    transformers::IGreedySearchState<T>* greedy_state,      // state
    transformers::ISequences* sequences,                    // sequences
    AllocatorPtr& allocator,                                // default allocator
    onnxruntime::concurrency::ThreadPool* thread_pool,      // thread pool (for CPU only)
    transformers::ILogitsProcessorList* logits_processors,  // logits processors
    const transformers::IBeamSearchParameters* parameters,  // parameters
    int step,                                               // iteration counter
    void* stream,                                           // cuda stream (for CUDA only)
    const transformers::IConsoleDumper* dumper) {           // tensor dumper
  ORT_UNUSED_PARAMETER(logits_processors);

#ifndef DEBUG_GENERATION
  ORT_UNUSED_PARAMETER(dumper);
#endif

  int batch_size = parameters->batch_size;
  int vocab_size = parameters->vocab_size;
  bool output_scores = parameters->output_scores;

  int batch_beam_size = batch_size;

  typedef typename ToCudaType<T>::MappedType CudaT;
  const CudaT* logits_data = reinterpret_cast<const CudaT*>(logits.Get<Tensor>().Data<T>());

  // Logits has shape (batch_size, input_length, padded_vocab_size),
  // where input_length equals to parameters_->sequence_length for first subgraph call, and 1 for the remaining calls.
  const TensorShape& logits_shape = logits.Get<Tensor>().Shape();
  ORT_ENFORCE(logits_shape.NumDimensions() == 3);
  auto input_length = logits_shape[1];

  // NOTE: `padded_vocab_size` MAY be different from `vocab_size`.
  // But the following implementation should work correctly if they are the same
  // or different.
  auto padded_vocab_size = static_cast<int>(logits_shape[2]);

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  // Get logits for the last token:
  //    next_token_logits = logits[:, -1, :], and the result shape is (batch_size, vocab_size)
  // In greedy search, next_token_scores is next_token_logits.
  gsl::span<T>& next_token_scores = greedy_state->next_token_scores;

  // TODO(tianleiwu): use one kernel to replace a loop of memory copy.
  // Move the pointer in increments of padded_vocab_size to account for any padding
  // if any in the logits weight of the MatMul.
  const CudaT* current_logits = logits_data + (input_length - 1) * padded_vocab_size;
  for (int i = 0; i < batch_beam_size; i++) {
    // We only copy what is relevant (i.e.) vocab_size as padded_vocab_size will contain
    // some logits corresponding to the "padded" vocab size which we will ignore
    // for token generation.
    gsl::span<const T> source(reinterpret_cast<const T*>(current_logits), vocab_size);
    gsl::span<T> target = next_token_scores.subspan(i * vocab_size, vocab_size);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target.data(), source.data(), sizeof(T) * vocab_size,
                                         cudaMemcpyDeviceToDevice, cuda_stream));
    current_logits += input_length * padded_vocab_size;
  }

#ifdef DEBUG_GENERATION
  dumper->Print("logits", logits);
  dumper->Print("next_token_scores", next_token_scores.data(), batch_size, vocab_size);
#endif

  // Sequences generated by beam scorer is currently stored in CPU.
  // Copy sequences to device only when repetition penalty or no repeat ngram is used in kernel
  BufferUniquePtr sequences_buffer;
  int current_sequence_length = sequences->GetSequenceLength();
  if (parameters->repetition_penalty != 1.0f) {
    size_t bytes = SafeInt<size_t>(sizeof(int32_t)) * batch_beam_size * parameters->max_length;
    void* data = allocator->Alloc(bytes);
    BufferUniquePtr temp_buffer(data, BufferDeleter(allocator));
    sequences_buffer = std::move(temp_buffer);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(sequences_buffer.get(), sequences->GetSequence(0).data(), bytes,
                                         cudaMemcpyHostToDevice, cuda_stream));
  }

  cuda::LaunchLogitsProcessKernel<CudaT>(
      reinterpret_cast<CudaT*>(next_token_scores.data()),
      parameters->vocab_mask.data(),
      step > 1 ? nullptr : parameters->prefix_vocab_mask.data(),  // prefix vocab mask is applied to first step only.
      parameters->batch_size,
      parameters->num_beams,
      parameters->vocab_size,
      (parameters->min_length > 0 && current_sequence_length < parameters->min_length) ? parameters->eos_token_id : -1,
      reinterpret_cast<int32_t*>(sequences_buffer.get()),
      parameters->max_length,
      current_sequence_length,
      parameters->repetition_penalty,
      parameters->no_repeat_ngram_size,
      cuda_stream);

#ifdef DEBUG_GENERATION
  dumper->Print("next_token_scores after logits process", next_token_scores.data(), batch_size, vocab_size);
#endif

  // TODO(wy): support output_scores in greedy search
  ORT_UNUSED_PARAMETER(output_scores);

  // next_tokens = torch.argmax(scores, dim=-1)
  int64_t next_token_scores_dims[] = {static_cast<int64_t>(batch_size), vocab_size};
  TensorShape next_token_scores_shape(&next_token_scores_dims[0], 2);
  auto element_type = DataTypeImpl::GetType<T>();
  OrtValue next_token_scores_value;
  Tensor::InitOrtValue(element_type,
                       next_token_scores_shape,
                       next_token_scores.data(),
                       allocator->Info(),
                       next_token_scores_value);
  const Tensor& input = next_token_scores_value.Get<Tensor>();

  constexpr int axis = 1;
  constexpr unsigned top_k = static_cast<unsigned>(1);
  constexpr bool largest = true;
  constexpr bool sorted = false;

  auto topk_scores = Tensor::CreateDefault();
  auto topk_indices = Tensor::CreateDefault();
  ORT_RETURN_IF_ERROR(TopK(&input, axis, top_k, largest, sorted, allocator, stream, thread_pool,
                           *topk_scores, *topk_indices));

#ifdef DEBUG_GENERATION
  dumper->Print("topk_scores", *(topk_scores.get()));
  dumper->Print("topk_indices", *(topk_indices.get()));
#endif

  const int64_t* next_token_indices = topk_indices->Data<int64_t>();

  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(greedy_state->next_tokens_cpu.data(),
                                       next_token_indices,
                                       greedy_state->next_tokens_cpu.size_bytes(),
                                       cudaMemcpyDeviceToHost,
                                       cuda_stream));
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(cuda_stream));

#ifdef DEBUG_GENERATION
  dumper->Print("greedy_state->next_tokens", greedy_state->next_tokens.data(), batch_size, 1);
#endif
  return Status::OK();
}

template <typename T>
Status DeviceCopy(gsl::span<T> target, gsl::span<const T> source, void* stream, int copyDirection) {
  assert(copyDirection >= 0 && copyDirection <= 3);
  if (stream == nullptr) {
    CUDA_RETURN_IF_ERROR(cudaMemcpy(target.data(), source.data(), source.size_bytes(),
                                    static_cast<cudaMemcpyKind>(copyDirection)));
  } else {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target.data(), source.data(), source.size_bytes(),
                                         static_cast<cudaMemcpyKind>(copyDirection), cuda_stream));
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(cuda_stream));
  }
  return Status::OK();
}

template <typename T>
Status PickGptPastState(const std::vector<OrtValue>& last_outputs,
                        std::vector<OrtValue>& next_inputs,
                        gsl::span<const int32_t>& beam_indices,
                        AllocatorPtr allocator,
                        int gpt_subgraph_first_past_input_idx,
                        int gpt_subgraph_first_present_output_idx,
                        void* stream) {
  int num_present_tensors = static_cast<int>(last_outputs.size()) - gpt_subgraph_first_present_output_idx;
  for (int i = 0; i < num_present_tensors; ++i) {
    const OrtValue& present = last_outputs[gpt_subgraph_first_present_output_idx + i];

    // shape is like (2, batch_beam_size, 12, past_seq_len, 64)
    const TensorShape& past_shape = present.Get<Tensor>().Shape();
    auto block_size_per_beam = past_shape[2] * past_shape[3] * past_shape[4];
    auto past_key_size = past_shape[1] * past_shape[2] * past_shape[3] * past_shape[4];

    // Create a tensor with same shape.
    // TODO(tianleiwu): allocate one buffer for all layers, and use a CUDA kernel to copy key/value cache data.
    OrtValue past;
    auto past_type = DataTypeImpl::GetType<T>();
    Tensor::InitOrtValue(past_type, past_shape, allocator, past);

    gsl::span<T> past_span = gsl::make_span<T>(past.GetMutable<Tensor>()->MutableData<T>(), past_shape.Size());
    gsl::span<const T> present_span = gsl::make_span<const T>(present.Get<Tensor>().Data<T>(), past_shape.Size());
    for (size_t j = 0; j < beam_indices.size(); j++) {
      int32_t beam_index = beam_indices[j];
      gsl::span<const T> present_key = present_span.subspan(beam_index * block_size_per_beam, block_size_per_beam);
      gsl::span<const T> present_value = present_span.subspan(past_key_size + beam_index * block_size_per_beam,
                                                              block_size_per_beam);

      gsl::span<T> past_key = past_span.subspan(j * block_size_per_beam, block_size_per_beam);
      gsl::span<T> past_value = past_span.subspan(past_key_size + j * block_size_per_beam, block_size_per_beam);
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(past_key.data(), present_key.data(), present_key.size_bytes(),
                                           cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream)));
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(past_value.data(), present_value.data(), present_value.size_bytes(),
                                           cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream)));
    }

    next_inputs[gpt_subgraph_first_past_input_idx + i] = past;
  }

  return Status::OK();
}

// Copy present state to past state for T5 model
template <typename T>
Status PickT5PastState(const std::vector<OrtValue>& last_outputs,
                       std::vector<OrtValue>& next_inputs,
                       int num_present_tensors,
                       gsl::span<const int32_t>& beam_indices,
                       AllocatorPtr allocator,
                       int t5_decoder_first_past_input_idx,
                       int t5_decoder_first_present_output_idx,
                       void* stream) {
  for (int i = 0; i < num_present_tensors; ++i) {
    const OrtValue& present = last_outputs[t5_decoder_first_present_output_idx + i];

    // shape is like (batch_beam_size, 12, past_seq_len, 64)
    const TensorShape& past_shape = present.Get<Tensor>().Shape();
    auto block_size_per_beam = past_shape[1] * past_shape[2] * past_shape[3];

    // Create a tensor with same shape.
    // TODO(tianleiwu): allocate one buffer for all layers, and use a CUDA kernel to copy key/value cache data.
    OrtValue past;
    Tensor::InitOrtValue(DataTypeImpl::GetType<T>(), past_shape, allocator, past);

    gsl::span<T> past_span = gsl::make_span<T>(past.GetMutable<Tensor>()->MutableData<T>(), past_shape.Size());
    gsl::span<const T> present_span = gsl::make_span<const T>(present.Get<Tensor>().Data<T>(), past_shape.Size());
    for (size_t j = 0; j < beam_indices.size(); j++) {
      int32_t beam_index = beam_indices[j];
      gsl::span<const T> present_beam = present_span.subspan(beam_index * block_size_per_beam, block_size_per_beam);
      gsl::span<T> past_beam = past_span.subspan(j * block_size_per_beam, block_size_per_beam);
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(past_beam.data(), present_beam.data(), present_beam.size_bytes(),
                                           cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream)));
    }

    next_inputs[t5_decoder_first_past_input_idx + i] = past;
  }

  return Status::OK();
}

template <typename T>
Status UpdateGptFeeds(
    AllocatorPtr allocator,
    void* stream,
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    OrtValue& position_ids,
    bool increase_position,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices,
    int num_beams,
    int gpt_subgraph_first_past_input_idx,
    int gpt_subgraph_first_present_output_idx) {
  // Update input_ids with next tokens.
  int batch_beam_size = static_cast<int>(beam_next_tokens.size());
  int64_t dims[] = {batch_beam_size, 1};
  TensorShape input_ids_shape(&dims[0], 2);
  auto element_type = DataTypeImpl::GetType<int32_t>();
  OrtValue input_ids;
  Tensor::InitOrtValue(element_type, input_ids_shape, allocator, input_ids);
  int32_t* input_ids_data = input_ids.GetMutable<Tensor>()->MutableData<int32_t>();
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(input_ids_data, beam_next_tokens.data(), beam_next_tokens.size_bytes(),
                                       cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream)));
  next_inputs[0] = input_ids;

  // Update position IDs
  int32_t* position_data = increase_position ? position_ids.GetMutable<Tensor>()->MutableData<int32_t>() : nullptr;
  next_inputs[1] = position_ids;

  // Update attention mask
  const OrtValue& old_mask = next_inputs[2];
  const int32_t* old_mask_data = old_mask.Get<Tensor>().Data<int32_t>();
  int64_t mask_dims[] = {batch_beam_size, current_length};
  TensorShape mask_shape(&mask_dims[0], 2);
  OrtValue attention_mask;
  auto mask_type = DataTypeImpl::GetType<int32_t>();
  Tensor::InitOrtValue(mask_type, mask_shape, allocator, attention_mask);
  int32_t* mask_data = attention_mask.GetMutable<Tensor>()->MutableData<int32_t>();

  // Launch kernel to update position_ids and attention_mask for next iteration
  cuda::LaunchUpdateGptKernel(old_mask_data, mask_data, position_data, batch_beam_size, current_length,
                              reinterpret_cast<cudaStream_t>(stream));

  next_inputs[2] = attention_mask;

  // Update past state
  if (num_beams == 1) {
    const int k = gpt_subgraph_first_past_input_idx - gpt_subgraph_first_present_output_idx;
    // feed present_* output to past_* inputs one by one
    for (size_t i = gpt_subgraph_first_present_output_idx; i < last_outputs.size(); ++i) {
      next_inputs[i + k] = last_outputs[i];
    }
  } else {
    ORT_RETURN_IF_ERROR(PickGptPastState<T>(last_outputs, next_inputs, beam_indices, allocator,
                                            gpt_subgraph_first_past_input_idx,
                                            gpt_subgraph_first_present_output_idx, stream));
  }

  // Make sure data is ready before next subgraph execution.
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
  return Status::OK();
}

// Update decoder inputs given decoder outputs of last iteration.
template <typename T>
Status UpdateDecoderFeeds(
    AllocatorPtr allocator,
    void* stream,
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int num_present_tensors,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices,
    int num_beams,
    int t5_decoder_first_past_input_idx,
    int t5_decoder_first_present_output_idx,
    bool use_sequence_as_input_ids,
    int current_length,
    transformers::Sequences&,
    const transformers::IConsoleDumper* dumper) {
  // last_outputs: logits, present_key_self_0, present_value_self_0, ...
  // next_inputs: input_ids,
  //              encoder_attention_mask, encoder_hidden_states,
  //              past_key_self_0, past_value_self_0, ...
  //              past_key_cross_0, past_value_cross_0, ...
  // Only need copy beam next tokens to input_ids, and copy present_*_self_* to past_*_self_*,

  if (use_sequence_as_input_ids) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "BeamSearch CUDA Op does not support using sequence as input_ids in decoder input");
  }
  ORT_UNUSED_PARAMETER(current_length);

  // Update input_ids with next tokens.
  int batch_beam_size = static_cast<int>(beam_next_tokens.size());
  int64_t dims[] = {batch_beam_size, 1};
  TensorShape input_ids_shape(&dims[0], 2);
  auto element_type = DataTypeImpl::GetType<int32_t>();
  OrtValue input_ids;
  Tensor::InitOrtValue(element_type, input_ids_shape, allocator, input_ids);
  int32_t* input_ids_data = input_ids.GetMutable<Tensor>()->MutableData<int32_t>();
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(input_ids_data, beam_next_tokens.data(), beam_next_tokens.size_bytes(),
                                       cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream)));
  next_inputs[0] = input_ids;

#ifdef DEBUG_GENERATION
  dumper->Print("input_ids", input_ids);
#else
  ORT_UNUSED_PARAMETER(dumper);
#endif

  // Update past state
  ORT_ENFORCE(last_outputs.size() >= static_cast<size_t>(1 + num_present_tensors));
  // TODO(tianleiwu): remove num_beams==1 once GreedySearch operator is available.
  if (num_beams == 1) {
    // feed present_* output to past_* inputs one by one
    for (int i = 0; i < num_present_tensors; ++i) {
      next_inputs[t5_decoder_first_past_input_idx + i] =
          last_outputs[t5_decoder_first_present_output_idx + i];
      return Status::OK();
    }
  }

  return PickT5PastState<T>(last_outputs, next_inputs, num_present_tensors, beam_indices, allocator,
                            t5_decoder_first_past_input_idx, t5_decoder_first_present_output_idx, stream);
}

template <typename T>
Status ExpandBuffer(void* stream,
                    const OrtValue& input,
                    int num_beams,
                    AllocatorPtr allocator,
                    OrtValue& expanded,
                    bool only_copy_shape) {
  // Input shape (batch_size, xxx). The input is required with data type T.
  // Output shape (batch_size * num_beams, xxx)
  const TensorShape& input_shape = input.Get<Tensor>().Shape();
  const int64_t& batch_size = input_shape[0];
  const int64_t& chunk_size = static_cast<int64_t>(input_shape.Size() / batch_size);

  int64_t dims[4] = {0};
  input_shape.CopyDims(dims, input_shape.NumDimensions());
  dims[0] = batch_size * num_beams;
  TensorShape expanded_shape(&dims[0], input_shape.NumDimensions());

  MLDataType element_type = input.Get<Tensor>().DataType();
  ORT_ENFORCE(element_type == DataTypeImpl::GetType<T>());
  Tensor::InitOrtValue(element_type, expanded_shape, allocator, expanded);

  if (only_copy_shape) {
    return Status::OK();
  }

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  const T* input_data = input.Get<Tensor>().Data<T>();
  T* expanded_data = expanded.GetMutable<Tensor>()->MutableData<T>();
  T* target = expanded_data;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < num_beams; j++) {
      CUDA_RETURN_IF_ERROR(
          cudaMemcpyAsync(
              target,
              input_data + i * chunk_size,
              sizeof(T) * chunk_size,
              cudaMemcpyDeviceToDevice,
              cuda_stream));
      target += chunk_size;
    }
  }

  return Status::OK();
}

// Explicit template instantiations of functions
template void InitBeamState<float>(
    transformers::IBeamSearchState<float>* beam_state,
    gsl::span<int32_t>& sequence_lengths,
    int batch_size,
    int num_beams,
    void* stream);

template void InitGreedyState<float>(
    transformers::IGreedySearchState<float>* greedy_state,
    gsl::span<int32_t>& sequence_lengths,
    void* stream);

template Status ProcessLogits<float>(
    const OrtValue& logits,
    transformers::IBeamSearchState<float>* beam_state,
    transformers::IBeamSearchCpuState* cpu_state,
    transformers::ISequences* sequences,
    AllocatorPtr& allocator,
    onnxruntime::concurrency::ThreadPool* thread_pool,
    transformers::ILogitsProcessorList* logits_processors,
    transformers::IBeamScorer* beam_scorer,
    const transformers::IBeamSearchParameters* parameters,
    int step,
    void* stream,
    const transformers::IConsoleDumper* dumper);

template Status GreedySearchProcessLogits<float>(
    const OrtValue& logits,
    transformers::IGreedySearchState<float>* greedy_state,
    transformers::ISequences* sequences,
    AllocatorPtr& allocator,
    onnxruntime::concurrency::ThreadPool* thread_pool,
    transformers::ILogitsProcessorList* logits_processors,
    const transformers::IBeamSearchParameters* parameters,
    int step,
    void* stream,
    const transformers::IConsoleDumper* dumper);

template Status DeviceCopy<float>(
    gsl::span<float> target,
    gsl::span<const float> source,
    void* stream,
    int copyDirection);

template Status DeviceCopy<int32_t>(
    gsl::span<int32_t> target,
    gsl::span<const int32_t> source,
    void* stream,
    int copyDirection);

template Status UpdateGptFeeds<float>(
    AllocatorPtr allocator,
    void* stream,
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    OrtValue& position_ids,
    bool increase_position,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices,
    int num_beams,
    int gpt_subgraph_first_past_input_idx,
    int gpt_subgraph_first_present_output_idx);

// Float16
template void InitBeamState<MLFloat16>(
    transformers::IBeamSearchState<MLFloat16>* beam_state,
    gsl::span<int32_t>& sequence_lengths,
    int batch_size,
    int num_beams,
    void* stream);

template void InitGreedyState<MLFloat16>(
    transformers::IGreedySearchState<MLFloat16>* greedy_state,
    gsl::span<int32_t>& sequence_lengths,
    void* stream);

template Status ProcessLogits<MLFloat16>(
    const OrtValue& logits,
    transformers::IBeamSearchState<MLFloat16>* beam_state,
    transformers::IBeamSearchCpuState* cpu_state,
    transformers::ISequences* sequences,
    AllocatorPtr& allocator,
    onnxruntime::concurrency::ThreadPool* thread_pool,
    transformers::ILogitsProcessorList* logits_processors,
    transformers::IBeamScorer* beam_scorer,
    const transformers::IBeamSearchParameters* parameters,
    int step,
    void* stream,
    const transformers::IConsoleDumper* dumper);

template Status GreedySearchProcessLogits<MLFloat16>(
    const OrtValue& logits,
    transformers::IGreedySearchState<MLFloat16>* greedy_state,
    transformers::ISequences* sequences,
    AllocatorPtr& allocator,
    onnxruntime::concurrency::ThreadPool* thread_pool,
    transformers::ILogitsProcessorList* logits_processors,
    const transformers::IBeamSearchParameters* parameters,
    int step,
    void* stream,
    const transformers::IConsoleDumper* dumper);

template Status UpdateGptFeeds<MLFloat16>(
    AllocatorPtr allocator,
    void* stream,
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    OrtValue& position_ids,
    bool increase_position,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices,
    int num_beams,
    int gpt_subgraph_first_past_input_idx,
    int gpt_subgraph_first_present_output_idx);

template Status UpdateDecoderFeeds<float>(
    AllocatorPtr allocator,
    void* stream,
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int num_present_tensors,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices,
    int num_beams,
    int t5_decoder_first_past_input_idx,
    int t5_decoder_first_present_output_idx,
    bool use_sequence_as_input_ids,
    int current_length,
    transformers::Sequences& sequences,
    const transformers::IConsoleDumper* dumper);

template Status UpdateDecoderFeeds<MLFloat16>(
    AllocatorPtr allocator,
    void* stream,
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int num_present_tensors,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices,
    int num_beams,
    int t5_decoder_first_past_input_idx,
    int t5_decoder_first_present_output_idx,
    bool use_sequence_as_input_ids,
    int current_length,
    transformers::Sequences& sequences,
    const transformers::IConsoleDumper* dumper);

template Status ExpandBuffer<int32_t>(
    void* stream,
    const OrtValue& input,
    int num_beams,
    AllocatorPtr allocator,
    OrtValue& expanded,
    bool only_copy_shape);

template Status ExpandBuffer<float>(
    void* stream,
    const OrtValue& input,
    int num_beams,
    AllocatorPtr allocator,
    OrtValue& expanded,
    bool only_copy_shape);

template Status ExpandBuffer<MLFloat16>(
    void* stream,
    const OrtValue& input,
    int num_beams,
    AllocatorPtr allocator,
    OrtValue& expanded,
    bool only_copy_shape);
}  // namespace GenerationCudaDeviceHelper
}  // namespace contrib
}  // namespace onnxruntime
