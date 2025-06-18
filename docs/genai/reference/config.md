---
title: Config reference
description: Reference for the ONNX Runtime generate() API configuration file
has_children: false
parent: Reference
grand_parent: Generate API (Preview)
nav_order: 1
---

# Configuration reference

_Note: this API is in preview and is subject to change._

A configuration file called `genai_config.json` is generated automatically if the model is generated with the model builder. If you provide your own model, you can copy the example below and modify it for your scenario.

{: .no_toc }

* TOC placeholder
{:toc}

## Example file

Below is an example `genai_config.json` for a decoder-only style model:

```json
{
    "model": {
        "bos_token_id": 199999,
        "context_length": 131072,
        "decoder": {
            "session_options": {
                "log_id": "onnxruntime-genai",
                "provider_options": []
            },
            "filename": "model.onnx",
            "head_size": 128,
            "hidden_size": 3072,
            "inputs": {
                "input_ids": "input_ids",
                "attention_mask": "attention_mask",
                "past_key_names": "past_key_values.%d.key",
                "past_value_names": "past_key_values.%d.value"
            },
            "outputs": {
                "logits": "logits",
                "present_key_names": "present.%d.key",
                "present_value_names": "present.%d.value"
            },
            "num_attention_heads": 24,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8
        },
        "eos_token_id": [
            200020,
            199999
        ],
        "pad_token_id": 199999,
        "type": "phi3",
        "vocab_size": 200064
    },
    "search": {
        "diversity_penalty": 0.0,
        "do_sample": false,
        "early_stopping": true,
        "length_penalty": 1.0,
        "max_length": 131072,
        "min_length": 0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "num_return_sequences": 1,
        "past_present_share_buffer": true,
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_k": 1,
        "top_p": 1.0
    }
}
```

---

## Configuration structure

The configuration file is structured as a JSON object with two main sections: `model` and `search`.  


---

### Config

Top-level configuration object.

- **config_path**: *(string, internal)*  
  Path to the config directory (not in JSON).

- **model**: *(object)*  
  Model architecture and ONNX model configuration.

- **search**: *(object)*  
  Generation/search parameters.

---

### Config::Model

Describes the model architecture, files, and tokenization.

- **type**: *(string)*  
  The type of model.
  
  For decoder-only LLMs, this type can be "decoder", or one of the following specific types:
  - chatglm
  - gemma
  - gemma2
  - gemma3_text
  - granite
  - llama
  - mistral
  - nemotron
  - olmo
  - phi
  - phimoe
  - phi3
  - phi3small
  - qwen2
  - qwen3

  For decoder only LLMS that are split into a pipeline of models, use "decoder-pipeline".
  
  Other model types:
  - whisper
  - phi3v
  - phi4mm
  - gemma3
  - marian-ssru

- **pad_token_id**: *(int)*  
  The id of the padding token.

- **eos_token_id**: *(int or array of int)*  
  The id(s) of the end-of-sequence token(s).

- **bos_token_id**: *(int)*  
  The id of the beginning-of-sequence token.

- **sep_token_id**: *(int, optional)*  
  The id of the separation token.

- **decoder_start_token_id**: *(int, optional)*  
  The id of the decoder start token (for encoder-decoder models).

- **vocab_size**: *(int)*  
  The size of the vocabulary.

- **context_length**: *(int)*  
  The maximum length of sequence that the model can process.

- **encoder**: *(object, optional)*  
  For models like Whisper. See [Encoder](#modelencoder).

- **embedding**: *(object, optional)*  
  For models with embedding submodules. See [Embedding](#modelembedding).

- **vision**: *(object, optional)*  
  For models with vision submodules. See [Vision](#modelvision).

- **speech**: *(object, optional)*  
  For models with speech submodules. See [Speech](#modelspeech).

- **decoder**: *(object)*  
  Decoder ONNX model and configuration. See [Decoder](#modeldecoder).

---

#### Model::Encoder

- **filename**: *(string)*  
  Path to the encoder ONNX file.

- **hidden_size**: *(int)*  
  Hidden size of the encoder.

- **num_key_value_heads**: *(int)*  
  Number of key-value heads.

- **num_hidden_layers**: *(int)*  
  Number of hidden layers.

- **head_size**: *(int)*  
  Size of each attention head.

- **inputs**: *(object)*  
  - **input_features**: *(string)*  
    Name of the input features tensor.
  - **input_ids**: *(string)*  
    Name of the input ids tensor.
  - **attention_mask**: *(string)*  
    Name of the attention mask tensor.

- **outputs**: *(object)*  
  - **encoder_outputs**: *(string)*  
    Name of the encoder outputs tensor.

---

#### Model::Embedding

- **filename**: *(string)*  
  Path to the embedding ONNX file.

- **inputs**: *(object)*  
  - **input_ids**: *(string)*  
    Name of the input ids tensor.
  - **image_features**: *(string)*  
    Name of the image features tensor.
  - **audio_features**: *(string)*  
    Name of the audio features tensor.

- **outputs**: *(object)*  
  - **embeddings**: *(string)*  
    Name of the embeddings output tensor.

---

#### Model::Vision

- **filename**: *(string)*  
  Path to the vision ONNX file.

- **config_filename**: *(string, optional)*  
  Path to the vision processor config file.

- **adapter_filename**: *(string, optional)*  
  Path to the vision adapter file.

- **inputs**: *(object)*  
  - **pixel_values**: *(string)*  
    Name of the pixel values tensor.
  - **image_sizes**: *(string)*  
    Name of the image sizes tensor.
  - **attention_mask**: *(string)*  
    Name of the image attention mask tensor.

- **outputs**: *(object)*  
  - **image_features**: *(string)*  
    Name of the image features output tensor.

---

#### Model::Speech

- **filename**: *(string)*  
  Path to the speech ONNX file.

- **config_filename**: *(string, optional)*  
  Path to the speech processor config file.

- **adapter_filename**: *(string, optional)*  
  Path to the speech adapter file.

- **inputs**: *(object)*  
  - **audio_embeds**: *(string)*  
    Name of the audio embeddings tensor.
  - **attention_mask**: *(string)*  
    Name of the audio attention mask tensor.
  - **audio_sizes**: *(string)*  
    Name of the audio sizes tensor.
  - **audio_projection_mode**: *(string)*  
    Name of the audio projection mode tensor.

- **outputs**: *(object)*  
  - **audio_features**: *(string)*  
    Name of the audio features output tensor.

---

#### Model::Decoder

- **filename**: *(string)*  
  Path to the decoder ONNX file.

- **session_options**: *(object)*  
  See [SessionOptions](#sessionoptions).

- **hidden_size**: *(int)*  
  Size of the hidden layers.

- **num_attention_heads**: *(int)*  
  Number of attention heads.

- **num_key_value_heads**: *(int)*  
  Number of key-value heads.

- **num_hidden_layers**: *(int)*  
  Number of hidden layers.

- **head_size**: *(int)*  
  Size of each attention head.

- **sliding_window**: *(object, optional)*  
  Parameters for sliding window inference.
  - **window_size**: *(int)*  
    Size of the window.
  - **pad_value**: *(int)*  
    Padding value for inactive tokens.
  - **alignment**: *(string)*  
    "left" or "right".
  - **slide_key_value_cache**: *(bool)*  
    Whether to slide the key-value cache.

- **inputs**: *(object)*  
  - **input_ids**: *(string)*  
    Name of the input ids tensor.
  - **embeddings**: *(string)*  
    Name of the embeddings tensor.
  - **position_ids**: *(string)*  
    Name of the position ids tensor.
  - **attention_mask**: *(string)*  
    Name of the attention mask tensor.
  - **past_key_names**: *(string)*  
    Name pattern for past key tensors.
  - **past_value_names**: *(string)*  
    Name pattern for past value tensors.
  - **past_names**: *(string, optional)*  
    Name for combined key/value pairs.
  - **cross_past_key_names**: *(string, optional)*  
    Name for cross-attention past key tensors.
  - **cross_past_value_names**: *(string, optional)*  
    Name for cross-attention past value tensors.
  - **current_sequence_length**: *(string)*  
    Name of the current sequence length tensor.
  - **past_sequence_length**: *(string)*  
    Name of the past sequence length tensor.
  - **past_key_values_length**: *(string)*  
    Name of the past key values length tensor.
  - **total_sequence_length**: *(string)*  
    Name of the total sequence length tensor.
  - **encoder_hidden_states**: *(string)*  
    Name of the encoder hidden states tensor.
  - **rnn_prev_states**: *(string, optional)*  
    Name of the previous RNN states tensor.
  - **encoder_attention_mask**: *(string, optional)*  
    Name of the encoder attention mask tensor.

- **outputs**: *(object)*  
  - **logits**: *(string)*  
    Name of the logits output tensor.
  - **present_key_names**: *(string)*  
    Name pattern for present key tensors.
  - **present_value_names**: *(string)*  
    Name pattern for present value tensors.
  - **present_names**: *(string, optional)*  
    Name for combined present key/value pairs.
  - **cross_present_key_names**: *(string, optional)*  
    Name for cross-attention present key tensors.
  - **cross_present_value_names**: *(string, optional)*  
    Name for cross-attention present value tensors.
  - **rnn_states**: *(string, optional)*  
    Name of the RNN states output tensor.

- **pipeline**: *(array, optional)*  
  For pipeline models, a list of sub-models with their own filenames, session options, inputs, and outputs.

---

#### Model::Decoder::PipelineModel

- **model_id**: *(string)*  
  Identifier for the pipeline model.

- **filename**: *(string)*  
  Path to the ONNX file.

- **session_options**: *(object, optional)*  
  Session options for this pipeline model.

- **inputs**: *(array of string)*  
  List of input tensor names.

- **outputs**: *(array of string)*  
  List of output tensor names.

- **output_names_forwarder**: *(object)*  
  Mapping of output names to forward.

- **run_on_prompt**: *(bool)*  
  Whether to run this model on the prompt.

- **run_on_token_gen**: *(bool)*  
  Whether to run this model during token generation.

- **reset_session_idx**: *(int)*  
  Index of the session to reset for memory management.

---

### SessionOptions

Options passed to ONNX Runtime for model execution.

- **intra_op_num_threads**: *(int, optional)*  
  Number of threads to use for intra-op parallelism.

- **inter_op_num_threads**: *(int, optional)*  
  Number of threads to use for inter-op parallelism.

- **enable_cpu_mem_arena**: *(bool, optional)*  
  Enable/disable the CPU memory arena.

- **enable_mem_pattern**: *(bool, optional)*  
  Enable/disable memory pattern optimization.

- **disable_cpu_ep_fallback**: *(bool, optional)*  
  Disable fallback to CPU execution provider.

- **disable_quant_qdq**: *(bool, optional)*  
  Disable quantization QDQ.

- **enable_quant_qdq_cleanup**: *(bool, optional)*  
  Enable quantization QDQ cleanup.

- **ep_context_enable**: *(bool, optional)*  
  Enable execution provider context.

- **ep_context_embed_mode**: *(string, optional)*  
  Execution provider context embed mode.

- **ep_context_file_path**: *(string, optional)*  
  Path to execution provider context file.

- **log_id**: *(string, optional)*  
  Prefix for logging.

- **log_severity_level**: *(int, optional)*  
  Logging severity level.

- **enable_profiling**: *(string, optional)*  
  Enable profiling.

- **custom_ops_library**: *(string, optional)*  
  Path to custom ops library.

- **use_env_allocators**: *(bool)*  
  Use environment allocators.

- **config_entries**: *(array of [string, string] pairs)*  
  Additional config entries.

- **provider_options**: *(array of ProviderOptions)*  
  List of provider options.

- **providers**: *(array of string)*  
  List of providers to use at runtime.

- **graph_optimization_level**: *(string, optional)*  
  Graph optimization level.

---

#### ProviderOptions

- **name**: *(string)*  
  Name of the provider. One of:
  - cuda
  - DML
  - NvTensorRtRtx
  - OpenVINO
  - QNN
  - rocm
  - WebGPU
  - VitisAI

  If this option is not given, the provider defaults to CPU.


- **options**: *(array of [string, string] pairs)*  
  Provider-specific options.

---

### Search

Describes the generation/search parameters.

- **do_sample**: *(bool)*  
  Whether to use sampling (top-k/top-p) or deterministic search.

- **min_length**: *(int)*  
  Minimum length of the generated sequence.

- **max_length**: *(int)*  
  Maximum length of the generated sequence.

- **batch_size**: *(int)*  
  Number of sequences to generate in parallel.

- **num_beams**: *(int)*  
  Number of beams for beam search. 1 means no beam search.

- **num_return_sequences**: *(int)*  
  Number of sequences to return.

- **repetition_penalty**: *(float)*  
  Penalty for repeating tokens. 1.0 means no penalty.

- **top_k**: *(int)*  
  Top-K sampling parameter.

- **top_p**: *(float)*  
  Top-P (nucleus) sampling parameter.

- **temperature**: *(float)*  
  Sampling temperature.

- **early_stopping**: *(bool)*  
  Whether to stop beam search early.

- **no_repeat_ngram_size**: *(int)*  
  Size of n-grams that should not be repeated.

- **diversity_penalty**: *(float)*  
  Not supported.

- **length_penalty**: *(float)*  
  Controls the length of the output. >1.0 encourages longer sequences, <1.0 encourages shorter.

- **past_present_share_buffer**: *(bool)*  
  If true, the past and present buffer are shared for efficiency.

- **random_seed**: *(int)*  
  Seed for the random number generator. -1 means use a random device.

---

## Notes

- **session_options**:  
  Supports advanced ONNX Runtime options such as threading, memory arena, quantization, profiling, and custom ops. See the C++ `SessionOptions` struct for all possible fields.

- **inputs/outputs**:  
  The names and patterns here must match the actual ONNX model graph.

- **pipeline**:  
  For advanced models, a pipeline of sub-models can be specified, each with its own ONNX file, session options, and input/output mappings.

---

## Search combinations

1. **Beam search**
   - `num_beams > 1`
   - `do_sample = false`

2. **Greedy search**
   - `num_beams = 1`
   - `do_sample = false`

3. **Top P / Top K**
   - `do_sample = true`

---