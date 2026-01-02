---
title: C API
description: C API reference for ONNX Runtime generate() API
has_children: false
parent: API docs
grand_parent: Generate API (Preview)
nav_order: 3
---

# ONNX Runtime generate() C API

_Note: this API is in preview and is subject to change._

{: .no_toc }

* TOC placeholder
{:toc}

## Overview

This document describes the C API for ONNX Runtime GenAI. The API is not thread safe. Below are the main functions and types, with code snippets and descriptions for each.

---

## Global API

### OgaShutdown

Cleanly shutdown the genai library and its ONNX Runtime usage on process exit.

```c
OgaShutdown();
```

---

### OgaSetLogBool

Control the logging behavior of the library by setting boolean logging options.

```c
OgaResult* result = OgaSetLogBool("option_name", true);
```

---

### OgaSetLogString

Control the logging behavior of the library by setting string logging options. If the option name is `"filename"` and a valid file path is provided, logging will be directed to that file. An empty string will reset logging to the default destination (std::cerr).

```c
OgaResult* result = OgaSetLogString("filename", "/path/to/logfile.txt");
```

---

### OgaSetLogCallback

Register a callback function to receive log messages from the library.

```c
void log_callback(const char* string, size_t length) {
  // Handle log message
}
OgaResult* result = OgaSetLogCallback(log_callback);
```

---

### OgaResultGetError

Gets the error message from an `OgaResult`.

```c
const char* error = OgaResultGetError(result);
```

---

### OgaDestroyResult

Destroys an `OgaResult`.

```c
OgaDestroyResult(result);
```

---

### OgaDestroyString

Destroys a string returned by the API.

```c
OgaDestroyString(str);
```

---

## Model API

### OgaCreateModel

Creates a model from the given directory. The directory should contain a file called `genai_config.json`, which corresponds to the [configuration specification](../reference/config.md).

```c
OgaModel* model = NULL;
OgaResult* result = OgaCreateModel("path/to/model_dir", &model);
```

---

### OgaDestroyModel

Destroys the given model.

```c
OgaDestroyModel(model);
```

---

### OgaCreateModelWithRuntimeSettings

Creates a model with runtime settings.

```c
OgaRuntimeSettings* settings = NULL;
OgaCreateRuntimeSettings(&settings);
// ... configure settings ...
OgaModel* model = NULL;
OgaResult* result = OgaCreateModelWithRuntimeSettings("path/to/model_dir", settings, &model);
```

---

### OgaCreateModelFromConfig

Creates a model from a config object.

```c
OgaConfig* config = NULL;
OgaCreateConfig("path/to/model_dir", &config);
OgaModel* model = NULL;
OgaResult* result = OgaCreateModelFromConfig(config, &model);
```

---

### OgaModelGetType

Gets the type of the model. The returned string must be destroyed with `OgaDestroyString`.

```c
const char* type = NULL;
OgaResult* result = OgaModelGetType(model, &type);
OgaDestroyString(type);
```

---

### OgaModelGetDeviceType

Gets the device type used by the model. The returned string must be destroyed with `OgaDestroyString`.

```c
const char* device_type = NULL;
OgaResult* result = OgaModelGetDeviceType(model, &device_type);
OgaDestroyString(device_type);
```

---

## Config API

### OgaCreateConfig

Creates a configuration object from a config directory. The path is expected to be encoded in UTF-8.

```c
OgaConfig* config = NULL;
OgaResult* result = OgaCreateConfig("path/to/model_dir", &config);
```

---

### OgaConfigClearProviders

Clear the list of execution providers in the given config.

```c
OgaResult* result = OgaConfigClearProviders(config);
```

---

### OgaConfigAppendProvider

Appends an execution provider to the configuration. If the provider already exists, does nothing.

```c
OgaResult* result = OgaConfigAppendProvider(config, "CUDAExecutionProvider");
```

---

### OgaConfigSetProviderOption

Sets a provider option in the configuration.

```c
OgaResult* result = OgaConfigSetProviderOption(config, "CUDAExecutionProvider", "device_id", "0");
```

---

### OgaConfigAddModelData

Adds model data to load the model from memory. The model data must remain valid at least until the model is created. If using session options such as `session.use_ort_model_bytes_directly`, the model data must remain valid until the `OgaModel` is destroyed.

```c
OgaResult* result = OgaConfigAddModelData(config, "model.onnx", model_data, model_data_length);
```

---

### OgaConfigRemoveModelData

Removes model data previously added to the config.

```c
OgaResult* result = OgaConfigRemoveModelData(config, "model.onnx");
```

---

### OgaConfigSetDecoderProviderOptionsHardwareDeviceType

Filter execution provider devices by hardware device type property.

```c
OgaResult* result = OgaConfigSetDecoderProviderOptionsHardwareDeviceType(config, "provider_name", "GPU");
```

---

### OgaConfigSetDecoderProviderOptionsHardwareDeviceId

Filter execution provider devices by hardware device ID property.

```c
OgaResult* result = OgaConfigSetDecoderProviderOptionsHardwareDeviceId(config, "provider_name", 0);
```

---

### OgaConfigSetDecoderProviderOptionsHardwareVendorId

Filter execution provider devices by hardware vendor ID property.

```c
OgaResult* result = OgaConfigSetDecoderProviderOptionsHardwareVendorId(config, "provider_name", 0x1234);
```

---

### OgaConfigClearDecoderProviderOptionsHardwareDeviceType

Clear the hardware device type property.

```c
OgaResult* result = OgaConfigClearDecoderProviderOptionsHardwareDeviceType(config, "provider_name");
```

---

### OgaConfigClearDecoderProviderOptionsHardwareDeviceId

Clear the hardware device ID property.

```c
OgaResult* result = OgaConfigClearDecoderProviderOptionsHardwareDeviceId(config, "provider_name");
```

---

### OgaConfigClearDecoderProviderOptionsHardwareVendorId

Clear the hardware vendor ID property.

```c
OgaResult* result = OgaConfigClearDecoderProviderOptionsHardwareVendorId(config, "provider_name");
```

---

### OgaConfigOverlay

Overlay JSON on top of the configuration.

```c
OgaResult* result = OgaConfigOverlay(config, "{\"option\": \"value\"}");
```

---

### OgaDestroyConfig

Destroys the configuration object.

```c
OgaDestroyConfig(config);
```

---

## Runtime Settings API

### OgaCreateRuntimeSettings

Creates a runtime settings object.

```c
OgaRuntimeSettings* settings = NULL;
OgaCreateRuntimeSettings(&settings);
```

---

### OgaRuntimeSettingsSetHandle

Sets a named handle in the runtime settings.

```c
OgaRuntimeSettingsSetHandle(settings, "custom_handle", handle_ptr);
```

---

### OgaDestroyRuntimeSettings

Destroys the runtime settings object.

```c
OgaDestroyRuntimeSettings(settings);
```

---

## Tokenizer API

### OgaCreateTokenizer

Creates a tokenizer for the given model.

```c
OgaTokenizer* tokenizer = NULL;
OgaResult* result = OgaCreateTokenizer(model, &tokenizer);
```

---

### OgaDestroyTokenizer

Destroys the tokenizer.

```c
OgaDestroyTokenizer(tokenizer);
```

---

### OgaUpdateTokenizerOptions

Updates tokenizer options for the given tokenizer instance.

Supported options:
- `add_special_tokens`: Controls whether to add special tokens (e.g., BOS/EOS) during tokenization. Values: `"true"` / `"false"` or `"1"` / `"0"`. Default: `"false"`.
- `skip_special_tokens`: Controls whether to remove special tokens during detokenization. Values: `"true"` / `"false"` or `"1"` / `"0"`. Default: `"true"`.

```c
const char* keys[] = {"add_special_tokens", "skip_special_tokens"};
const char* values[] = {"true", "false"};
OgaResult* result = OgaUpdateTokenizerOptions(tokenizer, keys, values, 2);
```

---

### OgaTokenizerGetBosTokenId

Returns the BOS (Beginning of Sequence) token ID.

```c
int32_t token_id = 0;
OgaResult* result = OgaTokenizerGetBosTokenId(tokenizer, &token_id);
```

---

### OgaTokenizerGetEosTokenIds

Returns an array of EOS (End of Sequence) token IDs. The array is owned by the tokenizer and will be freed when the tokenizer is destroyed.

```c
const int32_t* eos_token_ids = NULL;
size_t token_count = 0;
OgaResult* result = OgaTokenizerGetEosTokenIds(tokenizer, &eos_token_ids, &token_count);
```

---

### OgaTokenizerGetPadTokenId

Returns the PAD (padding) token ID.

```c
int32_t token_id = 0;
OgaResult* result = OgaTokenizerGetPadTokenId(tokenizer, &token_id);
```

---

### OgaTokenizerEncode

Encodes a single string and adds the encoded sequence of tokens to the `OgaSequences`. The `OgaSequences` must be freed with `OgaDestroySequences` when it is no longer needed.

```c
OgaSequences* sequences = NULL;
OgaCreateSequences(&sequences);
OgaResult* result = OgaTokenizerEncode(tokenizer, "Hello world", sequences);
```

---

### OgaTokenizerEncodeBatch

Encodes a batch of strings and returns a single tensor output.

```c
const char* texts[] = {"Hello", "World"};
OgaTensor* tensor = NULL;
OgaResult* result = OgaTokenizerEncodeBatch(tokenizer, texts, 2, &tensor);
```

---

### OgaTokenizerDecodeBatch

Decodes a batch of token sequences and returns an array of strings.

```c
OgaStringArray* out_strings = NULL;
OgaResult* result = OgaTokenizerDecodeBatch(tokenizer, tensor, &out_strings);
OgaDestroyStringArray(out_strings);
```

---

### OgaTokenizerToTokenId

Converts a string to its corresponding token ID.

```c
int32_t token_id = 0;
OgaResult* result = OgaTokenizerToTokenId(tokenizer, "Hello", &token_id);
```

---

### OgaTokenizerDecode

Decodes a sequence of tokens into a string. The output string must be freed with `OgaDestroyString`.

```c
const char* out_string = NULL;
OgaResult* result = OgaTokenizerDecode(tokenizer, tokens, token_count, &out_string);
OgaDestroyString(out_string);
```

---

### OgaTokenizerApplyChatTemplate

Applies a chat template to input messages. The template can optionally include tools and generation prompt.

```c
const char* result_string = NULL;
OgaResult* result = OgaTokenizerApplyChatTemplate(tokenizer, NULL, messages_json, tools_json, true, &result_string);
OgaDestroyString(result_string);
```

---

### OgaCreateTokenizerStream

Creates a tokenizer stream for incremental decoding. This allows decoding tokens one at a time.

```c
OgaTokenizerStream* stream = NULL;
OgaResult* result = OgaCreateTokenizerStream(tokenizer, &stream);
```

---

### OgaCreateTokenizerStreamFromProcessor

Creates a tokenizer stream from a multi-modal processor for incremental decoding.

```c
OgaTokenizerStream* stream = NULL;
OgaResult* result = OgaCreateTokenizerStreamFromProcessor(processor, &stream);
```

---

### OgaDestroyTokenizerStream

Destroys the tokenizer stream.

```c
OgaDestroyTokenizerStream(stream);
```

---

### OgaTokenizerStreamDecode

Decodes a single token in the stream. If a word is generated, it will be returned in `out`. The chunk is valid until the next call or when the stream is destroyed.

```c
const char* chunk = NULL;
OgaResult* result = OgaTokenizerStreamDecode(stream, token, &chunk);
```

---

## Sequences API

### OgaCreateSequences

Creates an empty `OgaSequences` object.

```c
OgaSequences* sequences = NULL;
OgaResult* result = OgaCreateSequences(&sequences);
```

---

### OgaDestroySequences

Destroys the given `OgaSequences`.

```c
OgaDestroySequences(sequences);
```

---

### OgaSequencesCount

Returns the number of sequences.

```c
size_t count = OgaSequencesCount(sequences);
```

---

### OgaSequencesGetSequenceCount

Returns the number of tokens in the sequence at the given index.

```c
size_t token_count = OgaSequencesGetSequenceCount(sequences, 0);
```

---

### OgaSequencesGetSequenceData

Returns a pointer to the token data for the sequence at the given index. The pointer is valid until the `OgaSequences` is destroyed.

```c
const int32_t* data = OgaSequencesGetSequenceData(sequences, 0);
```

---

### OgaAppendTokenSequence

Appends multiple tokens to the sequences.

```c
const int32_t tokens[] = {1, 2, 3};
OgaResult* result = OgaAppendTokenSequence(tokens, 3, sequences);
```

---

### OgaAppendTokenToSequence

Appends a single token to the sequence at the given index. If the sequence does not exist and the index equals the current sequence count, a new sequence is created.

```c
OgaResult* result = OgaAppendTokenToSequence(token_id, sequences, sequence_index);
```

---

## Generator Params API

### OgaCreateGeneratorParams

Creates generator parameters for the given model.

```c
OgaGeneratorParams* params = NULL;
OgaResult* result = OgaCreateGeneratorParams(model, &params);
```

---

### OgaDestroyGeneratorParams

Destroys the given generator params.

```c
OgaDestroyGeneratorParams(params);
```

---

### OgaGeneratorParamsSetSearchNumber

Sets a numeric search option (e.g., temperature, top_k, max_length).

```c
OgaResult* result = OgaGeneratorParamsSetSearchNumber(params, "max_length", 128);
```

---

### OgaGeneratorParamsSetSearchBool

Sets a boolean search option (e.g., do_sample).

```c
OgaResult* result = OgaGeneratorParamsSetSearchBool(params, "do_sample", true);
```

---

### OgaGeneratorParamsTryGraphCaptureWithMaxBatchSize

Attempts to enable graph capture mode with a maximum batch size for improved performance.

```c
OgaResult* result = OgaGeneratorParamsTryGraphCaptureWithMaxBatchSize(params, 8);
```

---

### OgaGeneratorParamsSetGuidance

Sets guidance data for constrained generation (e.g., JSON schema, regex, Lark grammar).

Supported guidance types:
- `json_schema`: Constrains output to a specific JSON schema
- `regex`: Constrains output to match a regular expression
- `lark_grammar`: Constrains output to a Lark grammar

The `enable_ff_tokens` flag allows force-forwarding tokens that satisfy the grammar without calling the model, speeding up generation (only valid when batch_size=1 and beam_size=1).

```c
OgaResult* result = OgaGeneratorParamsSetGuidance(params, "json_schema", schema_string, true);
```

---

### OgaGeneratorParamsSetInputSequences

Sets the input ID sequences for the generator params.

```c
OgaResult* result = OgaGeneratorParamsSetInputSequences(params, sequences);
```

---

### OgaGeneratorParamsSetModelInput

Sets an additional model input for advanced use cases (e.g., LoRA models).

```c
OgaResult* result = OgaGeneratorParamsSetModelInput(params, "input_name", tensor);
```

---

### OgaGeneratorParamsSetInputs

Sets named tensors as inputs for the generator.

```c
OgaResult* result = OgaGeneratorParamsSetInputs(params, named_tensors);
```

---

## Generator API

### OgaCreateGenerator

Creates a generator from the given model and generator params.

```c
OgaGenerator* generator = NULL;
OgaResult* result = OgaCreateGenerator(model, params, &generator);
```

---

### OgaDestroyGenerator

Destroys the given generator.

```c
OgaDestroyGenerator(generator);
```

---

### OgaGenerator_IsDone

Checks if generation is complete (all sequences have reached termination conditions).

```c
bool done = OgaGenerator_IsDone(generator);
```

---

### OgaGenerator_IsSessionTerminated

Checks if the session is terminated.

```c
bool terminated = OgaGenerator_IsSessionTerminated(generator);
```

---

### OgaGenerator_AppendTokenSequences

Appends token sequences to the generator for seeding generation.

```c
OgaResult* result = OgaGenerator_AppendTokenSequences(generator, sequences);
```

---

### OgaGenerator_AppendTokens

Appends individual tokens to the generator.

```c
const int32_t input_ids[] = {1, 2, 3};
OgaResult* result = OgaGenerator_AppendTokens(generator, input_ids, 3);
```

---

### OgaGenerator_GenerateNextToken

Generates the next token. This performs one iteration of the generation loop.

```c
OgaResult* result = OgaGenerator_GenerateNextToken(generator);
```

---

### OgaGenerator_GetNextTokens

Returns the next tokens generated by the model. The count matches the batch size. The pointer is valid until the next `OgaGenerator` call.

```c
const int32_t* tokens = NULL;
size_t count = 0;
OgaResult* result = OgaGenerator_GetNextTokens(generator, &tokens, &count);
```

---

### OgaGenerator_RewindTo

Rewinds the sequence to a new length. This is useful when the user wants to rewind the generator to a specific length and continue generating from that point.

```c
OgaResult* result = OgaGenerator_RewindTo(generator, new_length);
```

---

### OgaGenerator_SetRuntimeOption

Sets a runtime option (e.g., `terminate_session`).

```c
OgaResult* result = OgaGenerator_SetRuntimeOption(generator, "terminate_session", "1");
```

---

### OgaGenerator_GetSequenceCount

Returns the number of tokens in the sequence at the given index.

```c
size_t count = OgaGenerator_GetSequenceCount(generator, 0);
```

---

### OgaGenerator_GetSequenceData

Returns a pointer to the sequence data at the given index. The sequence data is owned by the `OgaGenerator` and will be freed when it is destroyed. The caller must copy the data if it needs to be used after the generator is destroyed.

```c
const int32_t* data = OgaGenerator_GetSequenceData(generator, 0);
```

---

### OgaGenerator_SetModelInput

Sets an additional model input for advanced use cases.

```c
OgaResult* result = OgaGenerator_SetModelInput(generator, "input_name", tensor);
```

---

### OgaGenerator_SetInputs

Sets named tensors as additional inputs.

```c
OgaResult* result = OgaGenerator_SetInputs(generator, named_tensors);
```

---

### OgaGenerator_GetInput

Returns a copy of the model input identified by the given name as an `OgaTensor` on CPU. The buffer is owned by the returned tensor and will be released when the tensor is destroyed.

```c
OgaTensor* input_tensor = NULL;
OgaResult* result = OgaGenerator_GetInput(generator, "input_name", &input_tensor);
```

---

### OgaGenerator_GetOutput

Returns a copy of the model output identified by the given name as an `OgaTensor` on CPU. The buffer is owned by the returned tensor and will be released when the tensor is destroyed.

```c
OgaTensor* output_tensor = NULL;
OgaResult* result = OgaGenerator_GetOutput(generator, "output_name", &output_tensor);
```

---

### OgaGenerator_GetLogits

Returns a copy of the logits from the model as an `OgaTensor` on CPU. The logits contain only the last token logits even during prompt processing. The buffer is owned by the returned tensor and will be released when it is destroyed.

```c
OgaTensor* logits = NULL;
OgaResult* result = OgaGenerator_GetLogits(generator, &logits);
```

---

### OgaGenerator_SetLogits

Sets the logits for the generator. This is useful for guided generation. The tensor must have the same shape as the logits returned by `OgaGenerator_GetLogits`.

```c
OgaResult* result = OgaGenerator_SetLogits(generator, logits_tensor);
```

---

## Adapter API

### OgaCreateAdapters

Creates the object that manages the adapters. Used to load all the model adapters with reference counting support.

```c
OgaAdapters* adapters = NULL;
OgaResult* result = OgaCreateAdapters(model, &adapters);
```

---

### OgaDestroyAdapters

Destroys the adapters object.

```c
OgaDestroyAdapters(adapters);
```

---

### OgaLoadAdapter

Loads a model adapter from the given adapter file path and assigns it a unique name for later reference.

```c
OgaResult* result = OgaLoadAdapter(adapters, "adapter_file_path", "adapter_name");
```

---

### OgaUnloadAdapter

Unloads the adapter with the given name. Returns an error if the adapter is not found or is still in use.

```c
OgaResult* result = OgaUnloadAdapter(adapters, "adapter_name");
```

---

### OgaSetActiveAdapter

Sets the active adapter for the generator.

```c
OgaResult* result = OgaSetActiveAdapter(generator, adapters, "adapter_name");
```

---

## Tensor API

### OgaCreateTensorFromBuffer

Creates a tensor from an optional user-owned buffer. If a user-owned buffer is supplied, the tensor does not own the memory, so the data must remain valid for the lifetime of the tensor. If the data pointer is `NULL`, the tensor will allocate its own memory.

```c
int64_t shape[] = {1, 3, 224, 224};
OgaTensor* tensor = NULL;
OgaResult* result = OgaCreateTensorFromBuffer(data, shape, 4, OgaElementType_float32, &tensor);
```

---

### OgaTensorGetType

Returns the element type of the tensor.

```c
OgaElementType type;
OgaResult* result = OgaTensorGetType(tensor, &type);
```

---

### OgaTensorGetShapeRank

Returns the rank (number of dimensions) of the tensor.

```c
size_t rank;
OgaResult* result = OgaTensorGetShapeRank(tensor, &rank);
```

---

### OgaTensorGetShape

Copies the shape dimensions into the provided array. The array size must match the rank returned by `OgaTensorGetShapeRank`.

```c
int64_t shape[rank];
OgaResult* result = OgaTensorGetShape(tensor, shape, rank);
```

---

### OgaTensorGetData

Returns a pointer to the tensor data. The pointer should be cast to the actual data type of the tensor.

```c
void* data = NULL;
OgaResult* result = OgaTensorGetData(tensor, &data);
```

---

### OgaDestroyTensor

Destroys the tensor.

```c
OgaDestroyTensor(tensor);
```

---

## Images and Audios API

### OgaLoadImage

Loads a single image from a file path.

```c
OgaImages* images = NULL;
OgaResult* result = OgaLoadImage("image_path.jpg", &images);
```

---

### OgaLoadImages

Loads multiple images from file paths.

```c
OgaStringArray* image_paths = NULL;
OgaCreateStringArrayFromStrings(paths, count, &image_paths);
OgaImages* images = NULL;
OgaResult* result = OgaLoadImages(image_paths, &images);
OgaDestroyStringArray(image_paths);
```

---

### OgaLoadImagesFromBuffers

Loads multiple images from memory buffers.

```c
OgaImages* images = NULL;
OgaResult* result = OgaLoadImagesFromBuffers(image_data, image_sizes, count, &images);
```

---

### OgaDestroyImages

Destroys the images object.

```c
OgaDestroyImages(images);
```

---

### OgaLoadAudio

Loads a single audio from a file path.

```c
OgaAudios* audios = NULL;
OgaResult* result = OgaLoadAudio("audio_path.wav", &audios);
```

---

### OgaLoadAudios

Loads multiple audios from file paths.

```c
OgaStringArray* audio_paths = NULL;
OgaCreateStringArrayFromStrings(paths, count, &audio_paths);
OgaAudios* audios = NULL;
OgaResult* result = OgaLoadAudios(audio_paths, &audios);
OgaDestroyStringArray(audio_paths);
```

---

### OgaLoadAudiosFromBuffers

Loads multiple audios from memory buffers.

```c
OgaAudios* audios = NULL;
OgaResult* result = OgaLoadAudiosFromBuffers(audio_data, audio_sizes, count, &audios);
```

---

### OgaDestroyAudios

Destroys the audios object.

```c
OgaDestroyAudios(audios);
```

---

## Named Tensors API

### OgaCreateNamedTensors

Creates a named tensors object.

```c
OgaNamedTensors* named_tensors = NULL;
OgaResult* result = OgaCreateNamedTensors(&named_tensors);
```

---

### OgaNamedTensorsGet

Gets a tensor by name.

```c
OgaTensor* tensor = NULL;
OgaResult* result = OgaNamedTensorsGet(named_tensors, "input_name", &tensor);
```

---

### OgaNamedTensorsSet

Sets a tensor by name.

```c
OgaResult* result = OgaNamedTensorsSet(named_tensors, "input_name", tensor);
```

---

### OgaNamedTensorsDelete

Deletes a tensor by name.

```c
OgaResult* result = OgaNamedTensorsDelete(named_tensors, "input_name");
```

---

### OgaNamedTensorsCount

Returns the number of named tensors.

```c
size_t count = 0;
OgaResult* result = OgaNamedTensorsCount(named_tensors, &count);
```

---

### OgaNamedTensorsGetNames

Returns an `OgaStringArray` containing the names of all tensors. Must be freed with `OgaDestroyStringArray`.

```c
OgaStringArray* names = NULL;
OgaResult* result = OgaNamedTensorsGetNames(named_tensors, &names);
OgaDestroyStringArray(names);
```

---

### OgaDestroyNamedTensors

Destroys the named tensors object.

```c
OgaDestroyNamedTensors(named_tensors);
```

---

## Multi-Modal Processing API

### OgaCreateMultiModalProcessor

Creates a multi-modal processor for the given model.

```c
OgaMultiModalProcessor* processor = NULL;
OgaResult* result = OgaCreateMultiModalProcessor(model, &processor);
```

---

### OgaDestroyMultiModalProcessor

Destroys the multi-modal processor.

```c
OgaDestroyMultiModalProcessor(processor);
```

---

### OgaProcessorProcessImages

Processes images with an input prompt.

```c
OgaNamedTensors* input_tensors = NULL;
OgaResult* result = OgaProcessorProcessImages(processor, "prompt text", images, &input_tensors);
```

---

### OgaProcessorProcessImagesAndPrompts

Processes images with multiple input prompts.

```c
OgaNamedTensors* input_tensors = NULL;
OgaResult* result = OgaProcessorProcessImagesAndPrompts(processor, prompts, images, &input_tensors);
```

---

### OgaProcessorProcessAudios

Processes audios with an input prompt.

```c
OgaNamedTensors* input_tensors = NULL;
OgaResult* result = OgaProcessorProcessAudios(processor, "prompt text", audios, &input_tensors);
```

---

### OgaProcessorProcessAudiosAndPrompts

Processes audios with multiple input prompts.

```c
OgaNamedTensors* input_tensors = NULL;
OgaResult* result = OgaProcessorProcessAudiosAndPrompts(processor, prompts, audios, &input_tensors);
```

---

### OgaProcessorProcessImagesAndAudios

Processes images and/or audios with an input prompt.

```c
OgaNamedTensors* input_tensors = NULL;
OgaResult* result = OgaProcessorProcessImagesAndAudios(processor, "prompt", images, audios, &input_tensors);
```

---

### OgaProcessorProcessImagesAndAudiosAndPrompts

Processes images and/or audios with multiple input prompts.

```c
OgaNamedTensors* input_tensors = NULL;
OgaResult* result = OgaProcessorProcessImagesAndAudiosAndPrompts(processor, prompts, images, audios, &input_tensors);
```

---

### OgaProcessorDecode

Decodes a sequence of tokens using the multi-modal processor.

```c
const char* out_string = NULL;
OgaResult* result = OgaProcessorDecode(processor, tokens, token_count, &out_string);
OgaDestroyString(out_string);
```

---

## String Array API

### OgaCreateStringArray

Creates an empty string array.

```c
OgaStringArray* string_array = NULL;
OgaResult* result = OgaCreateStringArray(&string_array);
```

---

### OgaCreateStringArrayFromStrings

Creates a string array from an array of strings.

```c
const char* strs[] = {"string1", "string2", "string3"};
OgaStringArray* string_array = NULL;
OgaResult* result = OgaCreateStringArrayFromStrings(strs, 3, &string_array);
```

---

### OgaDestroyStringArray

Destroys the string array.

```c
OgaDestroyStringArray(string_array);
```

---

### OgaStringArrayAddString

Adds a string to the string array.

```c
OgaResult* result = OgaStringArrayAddString(string_array, "new_string");
```

---

### OgaStringArrayGetCount

Gets the number of strings in the string array.

```c
size_t count = 0;
OgaResult* result = OgaStringArrayGetCount(string_array, &count);
```

---

### OgaStringArrayGetString

Gets a string from the string array at the given index.

```c
const char* str = NULL;
OgaResult* result = OgaStringArrayGetString(string_array, 0, &str);
```

---

## Engine and Request API

### OgaCreateEngine

Creates an engine from the given model. The engine is responsible for managing and scheduling multiple requests, executing model inference, and coordinating batching, caching, and resource management for efficient processing.

```c
OgaEngine* engine = NULL;
OgaResult* result = OgaCreateEngine(model, &engine);
```

---

### OgaDestroyEngine

Destroys the given engine.

```c
OgaDestroyEngine(engine);
```

---

### OgaEngineStep

Runs one step of the engine if there are pending requests. Returns a request that has been processed and is ready to be queried for results. This function should be called repeatedly to process all requests.

```c
OgaRequest* ready_request = NULL;
OgaResult* result = OgaEngineStep(engine, &ready_request);
```

---

### OgaEngineHasPendingRequests

Checks if the engine has any pending requests to process.

```c
bool has_pending = false;
OgaResult* result = OgaEngineHasPendingRequests(engine, &has_pending);
```

---

### OgaEngineAddRequest

Adds a request to the engine for processing. The request will be processed in subsequent calls to `OgaEngineStep`.

```c
OgaResult* result = OgaEngineAddRequest(engine, request);
```

---

### OgaEngineRemoveRequest

Removes a request from the engine.

```c
OgaResult* result = OgaEngineRemoveRequest(engine, request);
```

---

### OgaCreateRequest

Creates a new request for the engine with the specified generator parameters.

```c
OgaRequest* request = NULL;
OgaResult* result = OgaCreateRequest(params, &request);
```

---

### OgaDestroyRequest

Destroys the given request and cleans up its resources.

```c
OgaDestroyRequest(request);
```

---

### OgaRequestAddTokens

Adds input sequences to the request, which are used to seed the generation process.

```c
OgaResult* result = OgaRequestAddTokens(request, tokens);
```

---

### OgaRequestSetOpaqueData

Sets custom user data on the request that is opaque to the engine and can be retrieved later.

```c
OgaResult* result = OgaRequestSetOpaqueData(request, user_data_pointer);
```

---

### OgaRequestGetOpaqueData

Retrieves the custom user data previously set on the request using `OgaRequestSetOpaqueData`.

```c
void* opaque_data = NULL;
OgaResult* result = OgaRequestGetOpaqueData(request, &opaque_data);
```

---

### OgaRequestHasUnseenTokens

Checks if the request has any unseen tokens that have not yet been queried.

```c
bool has_unseen = false;
OgaResult* result = OgaRequestHasUnseenTokens(request, &has_unseen);
```

---

### OgaRequestGetUnseenToken

Retrieves the next unseen token from the request. Unseen tokens are those generated by the model but not yet queried.

```c
int32_t token = 0;
OgaResult* result = OgaRequestGetUnseenToken(request, &token);
```

---

### OgaRequestIsDone

Checks if the request has finished processing. A request is done when one of the termination conditions has been reached.

```c
bool done = false;
OgaResult* result = OgaRequestIsDone(request, &done);
```

---

## GPU Device Management

### OgaSetCurrentGpuDeviceId

Sets the current GPU device ID.

```c
OgaResult* result = OgaSetCurrentGpuDeviceId(0);
```

---

### OgaGetCurrentGpuDeviceId

Gets the current GPU device ID.

```c
int device_id = 0;
OgaResult* result = OgaGetCurrentGpuDeviceId(&device_id);
```

---

## Execution Provider Registration

### OgaRegisterExecutionProviderLibrary

Registers an execution provider library with ONNX Runtime.

```c
OgaRegisterExecutionProviderLibrary("registration_name", "/path/to/provider_library.so");
```

---

### OgaUnregisterExecutionProviderLibrary

Unregisters an execution provider library from ONNX Runtime.

```c
OgaUnregisterExecutionProviderLibrary("registration_name");
```

---