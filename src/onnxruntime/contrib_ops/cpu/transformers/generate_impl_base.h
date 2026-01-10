// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "core/common/span_utils.h"
#include "contrib_ops/cpu/transformers/generation_shared.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

template <typename T>
gsl::span<T> AllocateBuffer(AllocatorPtr allocator,
                            BufferUniquePtr& buffer,
                            size_t elements,
                            bool fill = false,
                            T fill_value = T{}) {
  size_t bytes = SafeInt<size_t>(sizeof(T)) * elements;
  void* data = allocator->Alloc(bytes);
  BufferUniquePtr temp_buffer(data, BufferDeleter(allocator));
  buffer = std::move(temp_buffer);
  T* first = reinterpret_cast<T*>(buffer.get());
  auto span = gsl::make_span(first, elements);

  if (fill) {
    std::fill_n(first, elements, fill_value);
  }

  return span;
}

template <typename T>
gsl::span<T> AllocateBuffer(AllocatorPtr allocator,
                            IAllocatorUniquePtr<void>& buffer,
                            size_t elements,
                            Stream* stream,
                            bool fill = false,
                            T fill_value = T{}) {
  size_t bytes = SafeInt<size_t>(sizeof(T)) * elements;
  buffer = IAllocator::MakeUniquePtr<void>(allocator, bytes, false, stream);
  T* first = reinterpret_cast<T*>(buffer.get());
  auto span = gsl::make_span(first, elements);

  if (fill) {
    std::fill_n(first, elements, fill_value);
  }

  return span;
}

template <typename ElementType>
inline void AllocateTempBufferForGetGreedySearchTopOne(
    int32_t batch_size,
    AllocatorPtr allocator,
    IAllocatorUniquePtr<void>& buffer,
    gsl::span<ElementType>& stage_1_scores,  // shape (batch_size, parts_of_vocab)
    gsl::span<int32_t>& stage_1_tokens,      // shape (batch_size, parts_of_vocab)
    gsl::span<ElementType>& output_scores,   // shape (batch_size)
    gsl::span<int32_t>& output_tokens,       // shape (batch_size)
    Stream* stream) {
  constexpr size_t kMaxPartsPerVocab = 128;
  const size_t stage_1_element_size = kMaxPartsPerVocab * batch_size;
  const size_t output_element_size = batch_size;

  // Note: use float to allocate buffer for temporary value buffer to avoid unalignment
  size_t bytes = (stage_1_element_size + output_element_size) * (sizeof(float) + sizeof(int32_t));
  buffer = IAllocator::MakeUniquePtr<void>(allocator, bytes, false, stream);
  void* topk_data = buffer.get();

  ElementType* stage_1_scores_data = reinterpret_cast<ElementType*>(topk_data);
  stage_1_scores = gsl::make_span<ElementType>(stage_1_scores_data, stage_1_element_size);

  int32_t* stage_1_token_data = reinterpret_cast<int32_t*>(
      reinterpret_cast<float*>(stage_1_scores_data) + stage_1_element_size);
  stage_1_tokens = gsl::make_span<int32_t>(stage_1_token_data, stage_1_element_size);

  ElementType* output_score_data = reinterpret_cast<ElementType*>(stage_1_token_data + stage_1_element_size);
  output_scores = gsl::make_span<ElementType>(output_score_data, output_element_size);

  int32_t* output_token_data = reinterpret_cast<int32_t*>(
      reinterpret_cast<float*>(output_score_data) + output_element_size);
  output_tokens = gsl::make_span<int32_t>(output_token_data, output_element_size);
}

class GenerateBase {
 public:
  GenerateBase(OpKernelContextInternal& context,
               const SessionState& decoder_session_state,
               concurrency::ThreadPool* thread_pool,
               Stream* ort_stream,
               IConsoleDumper* cuda_dumper,
               const GenerationDeviceHelper::TopkFunc& topk_func,
               const GenerationDeviceHelper::DeviceCopyFunc<float>& device_copy_func)
      : context_(context),
        decoder_session_state_(decoder_session_state),
        thread_pool_(thread_pool),
        implicit_inputs_(context_.GetImplicitInputs()),
        ort_stream_(ort_stream),
        cuda_dumper_(cuda_dumper),
        cpu_allocator_(decoder_session_state.GetAllocator(
            decoder_session_state.GetExecutionProviders()
                .Get(onnxruntime::kCpuExecutionProvider)
                ->GetOrtDeviceByMemType(OrtMemTypeDefault))),
        temp_space_allocator_(nullptr),
        topk_func_(topk_func),
        device_copy_func_(device_copy_func) {
  }

  virtual ~GenerateBase() = default;

  // Initialize by validating all the inputs, and allocating the output tensors.
  virtual Status Initialize() = 0;

  // Validate inputs.
  virtual Status CheckInputs(const OpKernelContextInternal& context) = 0;

  Status CheckScalarInput(const std::string& name, int index, bool required) const {
    auto* scalar_tensor = context_.Input<Tensor>(index);
    if (scalar_tensor) {
      if (!scalar_tensor->Shape().IsScalar()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME,
                               FAIL,
                               "Node input ", name, " should be a scalar. Got shape of ",
                               scalar_tensor->Shape());
      }
    } else if (required) {
      return ORT_MAKE_STATUS(ONNXRUNTIME,
                             FAIL,
                             "Node input ", name, " is required");
    }
    return Status::OK();
  }

  template <typename ParametersT>
  Status CheckInputsImpl(const ParametersT& parameters,
                         const Tensor* input_ids,
                         const Tensor* vocab_mask,
                         const Tensor* prefix_vocab_mask,
                         const Tensor* attention_mask,
                         const Tensor* presence_mask,
                         const Tensor* decoder_input_ids) const {
    const auto& dims = input_ids->Shape().GetDims();
    if (parameters->model_type == IGenerationParameters::kModelTypeWhisper) {
      if (dims.size() != 3) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Input 'input_features' is expected to have 3 dimensions, got ", dims.size());
      }
      if (decoder_input_ids != nullptr) {
        const auto& decoder_dims = decoder_input_ids->Shape().GetDims();
        if (decoder_dims.size() != 2) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Input 'decoder_input_ids' is expected to have 2 dimensions, got ", decoder_dims.size());
        }
      }
    } else if (dims.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'input_ids' is expected to have 2 dimensions, got ", dims.size());
    }

    if (vocab_mask != nullptr) {  // vocab_mask is optional
      const auto& vocab_mask_dims = vocab_mask->Shape().GetDims();
      if (vocab_mask_dims.size() != 1) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Input 'vocab_mask' is expected to have 1 dimension, got ", vocab_mask_dims.size());
      }

      // There is dependency on vocab_size parameter, which shall be set before calling this function.
      if (static_cast<int>(vocab_mask_dims[0]) != parameters->vocab_size) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Input 'vocab_mask'  dimension 0 does not match with vocab_size's, got ",
                               vocab_mask_dims[0]);
      }

      // store vocab mask in parameters.
      parameters->vocab_mask = vocab_mask->DataAsSpan<int32_t>();
    }

    if (prefix_vocab_mask != nullptr) {  // prefix_vocab_mask is optional
      const auto& vocab_mask_dims = prefix_vocab_mask->Shape().GetDims();
      if (vocab_mask_dims.size() != 2) {
        return ORT_MAKE_STATUS(ONNXRUNTIME,
                               INVALID_ARGUMENT,
                               "Input 'prefix_vocab_mask' is expected to be 2 dimensions, got ",
                               vocab_mask_dims.size());
      }

      // prefix_vocab_mask first dimension should be same as the first dimension of input_ids
      if (static_cast<int>(vocab_mask_dims[0]) != static_cast<int>(dims[0])) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "input_ids and prefix_vocab_mask must have the same batch_size");
      }

      // There is dependency on vocab_size parameter, which shall be set before calling this function.
      if (static_cast<int>(vocab_mask_dims[1]) != parameters->vocab_size) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Input 'prefix_vocab_mask' shape[1] shall be vocab_size, got ", vocab_mask_dims[1]);
      }

      // store prefix vocab mask in parameters.
      parameters->prefix_vocab_mask = prefix_vocab_mask->DataAsSpan<int32_t>();
    }

    if (attention_mask != nullptr) {
      const auto& dims_attn = attention_mask->Shape().GetDims();
      if (parameters->model_type == IGenerationParameters::kModelTypeWhisper) {
        if (dims_attn.size() != 3) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Input 'attention_mask' is expected to have 3 dimensions, got ", dims_attn.size());
        }
      } else if (dims_attn.size() != 2) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Input 'attention_mask' is expected to have 2 dimensions, got ", dims_attn.size());
      }
      if (parameters->model_type != IGenerationParameters::kModelTypeWhisper && !SpanEq(dims_attn, dims)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Input 'attention_mask' is expected to have same shape as input_ids");
      }
    }
    if (presence_mask != nullptr) {
      const auto& dims_presence = presence_mask->Shape().GetDims();
      if (dims_presence.size() != 2) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Input 'presence_mask' is expected to have 2 dimensions, got ", dims_presence.size());
      }

      // presence_mask first dimension should be same as the first dimension of input_ids
      if (static_cast<int>(dims_presence[0]) != static_cast<int>(dims[0])) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "input_ids and presence_mask must have the same batch_size");
      }

      if (static_cast<int>(dims_presence[1]) != parameters->vocab_size) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Input 'presence_mask' shape[1] shall be vocab_size, got ", dims_presence[1]);
      }

      // store prefix vocab mask in parameters.
      parameters->presence_mask = presence_mask->DataAsSpan<int32_t>();
    }

    return Status::OK();
  }

 protected:
  bool IsCuda() const {
    return ort_stream_ != nullptr;
  }

  const IConsoleDumper* GetConsoleDumper() const {
    return IsCuda() ? cuda_dumper_ : &(cpu_dumper_);
  }

  OpKernelContextInternal& context_;

  const SessionState& decoder_session_state_;

  concurrency::ThreadPool* thread_pool_;

  const std::vector<const OrtValue*>& implicit_inputs_;

  Stream* ort_stream_;

  IConsoleDumper* cuda_dumper_;
  CpuTensorConsoleDumper cpu_dumper_;

  LogitsProcessorList logits_processors_;

  AllocatorPtr cpu_allocator_;
  AllocatorPtr temp_space_allocator_;

  // Device specific functions
  GenerationDeviceHelper::TopkFunc topk_func_;
  GenerationDeviceHelper::DeviceCopyFunc<float> device_copy_func_;
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
