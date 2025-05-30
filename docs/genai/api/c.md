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

This document describes the C API for ONNX Runtime GenAI.  
Below are the main functions and types, with code snippets and descriptions for each.

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

Gets the type of the model.

```c
const char* type = NULL;
OgaModelGetType(model, &type);
```

---

### OgaModelGetDeviceType

Gets the device type used by the model.

```c
const char* device_type = NULL;
OgaModelGetDeviceType(model, &device_type);
```

---

## Config API

### OgaCreateConfig

Creates a configuration object from a config path.

```c
OgaConfig* config = NULL;
OgaResult* result = OgaCreateConfig("path/to/model_dir", &config);
```

---

### OgaConfigClearProviders

Clears all providers from the configuration.

```c
OgaConfigClearProviders(config);
```

---

### OgaConfigAppendProvider

Appends a provider to the configuration.

```c
OgaConfigAppendProvider(config, "CUDAExecutionProvider");
```

---

### OgaConfigSetProviderOption

Sets a provider option in the configuration.

```c
OgaConfigSetProviderOption(config, "CUDAExecutionProvider", "device_id", "0");
```

---

### OgaConfigOverlay

Overlays a JSON string onto the configuration.

```c
OgaConfigOverlay(config, "{\"option\": \"value\"}");
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

### OgaTokenizerEncode

Encodes a single string and adds the encoded sequence of tokens to the OgaSequences.

```c
OgaSequences* sequences = NULL;
OgaCreateSequences(&sequences);
OgaTokenizerEncode(tokenizer, "Hello world", sequences);
```

---

### OgaTokenizerEncodeBatch

Encodes a batch of strings.

```c
const char* texts[] = {"Hello", "World"};
OgaTensor* tensor = NULL;
OgaTokenizerEncodeBatch(tokenizer, texts, 2, &tensor);
```

---

### OgaTokenizerToTokenId

Converts a string to its corresponding token ID.

```c
int32_t token_id = 0;
OgaTokenizerToTokenId(tokenizer, "Hello", &token_id);
```

---

### OgaTokenizerDecode

Decodes a sequence of tokens into a string.

```c
const char* out_string = NULL;
OgaTokenizerDecode(tokenizer, tokens, token_count, &out_string);
// Use out_string, then:
OgaDestroyString(out_string);
```

---

### OgaTokenizerApplyChatTemplate

Applies a chat template to messages and tools.

```c
const char* result = NULL;
OgaTokenizerApplyChatTemplate(tokenizer, "template", "messages", "tools", true, &result);
OgaDestroyString(result);
```

---

### OgaTokenizerDecodeBatch

Decodes a batch of token sequences.

```c
OgaStringArray* out_strings = NULL;
OgaTokenizerDecodeBatch(tokenizer, tensor, &out_strings);
// Use out_strings, then:
OgaDestroyStringArray(out_strings);
```

---

### OgaCreateTokenizerStream

Creates a tokenizer stream for incremental decoding.

```c
OgaTokenizerStream* stream = NULL;
OgaCreateTokenizerStream(tokenizer, &stream);
```

---

### OgaDestroyTokenizerStream

Destroys the tokenizer stream.

```c
OgaDestroyTokenizerStream(stream);
```

---

### OgaTokenizerStreamDecode

Decodes a single token in the stream.

```c
const char* chunk = NULL;
OgaTokenizerStreamDecode(stream, token, &chunk);
// chunk is valid until next call or stream is destroyed
```

---

## Sequences API

### OgaCreateSequences

Creates an empty OgaSequences object.

```c
OgaSequences* sequences = NULL;
OgaCreateSequences(&sequences);
```

---

### OgaDestroySequences

Destroys the given OgaSequences.

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

Returns a pointer to the token data for the sequence at the given index.

```c
const int32_t* data = OgaSequencesGetSequenceData(sequences, 0);
```

---

## Generator Params API

### OgaCreateGeneratorParams

Creates generator parameters for the given model.

```c
OgaGeneratorParams* params = NULL;
OgaCreateGeneratorParams(model, &params);
```

---

### OgaDestroyGeneratorParams

Destroys the given generator params.

```c
OgaDestroyGeneratorParams(params);
```

---

### OgaGeneratorParamsSetSearchNumber

Sets a numeric search option.

```c
OgaGeneratorParamsSetSearchNumber(params, "max_length", 128);
```

---

### OgaGeneratorParamsSetSearchBool

Sets a boolean search option.

```c
OgaGeneratorParamsSetSearchBool(params, "do_sample", true);
```

---

### OgaGeneratorParamsTryGraphCaptureWithMaxBatchSize

Attempts to enable graph capture mode with a maximum batch size.

```c
OgaGeneratorParamsTryGraphCaptureWithMaxBatchSize(params, 8);
```

---

### OgaGeneratorParamsSetInputIDs

Sets the input ids for the generator params.

```c
OgaGeneratorParamsSetInputIDs(params, input_ids, input_ids_count, sequence_length, batch_size);
```

---

### OgaGeneratorParamsSetInputSequences

Sets the input id sequences for the generator params.

```c
OgaGeneratorParamsSetInputSequences(params, sequences);
```

---

### OgaGeneratorParamsSetModelInput

Sets an additional model input.

```c
OgaGeneratorParamsSetModelInput(params, "input_name", tensor);
```

---

### OgaGeneratorParamsSetInputs

Sets named tensors as inputs.

```c
OgaGeneratorParamsSetInputs(params, named_tensors);
```

---

### OgaGeneratorParamsSetGuidance

Sets guidance data.

```c
OgaGeneratorParamsSetGuidance(params, "type", "data");
```

---

## Generator API

### OgaCreateGenerator

Creates a generator from the given model and generator params.

```c
OgaGenerator* generator = NULL;
OgaCreateGenerator(model, params, &generator);
```

---

### OgaDestroyGenerator

Destroys the given generator.

```c
OgaDestroyGenerator(generator);
```

---

### OgaGenerator_IsDone

Checks if generation is complete.

```c
bool done = OgaGenerator_IsDone(generator);
```

---

### OgaGenerator_AppendTokenSequences

Appends token sequences to the generator.

```c
OgaGenerator_AppendTokenSequences(generator, sequences);
```

---

### OgaGenerator_AppendTokens

Appends tokens to the generator.

```c
OgaGenerator_AppendTokens(generator, input_ids, input_ids_count);
```

---

### OgaGenerator_IsSessionTerminated

Checks if the session is terminated.

```c
bool terminated = OgaGenerator_IsSessionTerminated(generator);
```

---

### OgaGenerator_GenerateNextToken

Generates the next token.

```c
OgaGenerator_GenerateNextToken(generator);
```

---

### OgaGenerator_RewindTo

Rewinds the sequence to a new length.

```c
OgaGenerator_RewindTo(generator, new_length);
```

---

### OgaGenerator_SetRuntimeOption

Sets a runtime option.

```c
OgaGenerator_SetRuntimeOption(generator, "terminate_session", "1");
```

---

### OgaGenerator_GetSequenceCount

Returns the number of tokens in the sequence at the given index.

```c
size_t count = OgaGenerator_GetSequenceCount(generator, 0);
```

---

### OgaGenerator_GetSequenceData

Returns a pointer to the sequence data at the given index.

```c
const int32_t* data = OgaGenerator_GetSequenceData(generator, 0);
```

---

### OgaGenerator_GetOutput

Gets a named output tensor.

```c
OgaTensor* tensor = NULL;
OgaGenerator_GetOutput(generator, "output_name", &tensor);
```

---

### OgaGenerator_GetLogits

Gets the logits tensor.

```c
OgaTensor* logits = NULL;
OgaGenerator_GetLogits(generator, &logits);
```

---

### OgaGenerator_SetLogits

Sets the logits tensor.

```c
OgaGenerator_SetLogits(generator, tensor);
```

---

### OgaSetActiveAdapter

Sets the active adapter for the generator.

```c
OgaSetActiveAdapter(generator, adapters, "adapter_name");
```

---

## Adapter API

### OgaCreateAdapters

Creates the object that manages the adapters.

```c
OgaAdapters* adapters = NULL;
OgaCreateAdapters(model, &adapters);
```

---

### OgaLoadAdapter

Loads the model adapter from the given adapter file path and adapter name.

```c
OgaLoadAdapter(adapters, "adapter_file_path", "adapter_name");
```

---

### OgaUnloadAdapter

Unloads the adapter with the given identifier.

```c
OgaUnloadAdapter(adapters, "adapter_name");
```

---

## Tensor API

### OgaCreateTensorFromBuffer

Creates a tensor from a buffer.

```c
OgaTensor* tensor = NULL;
OgaCreateTensorFromBuffer(data, shape_dims, shape_dims_count, element_type, &tensor);
```

---

### OgaTensorGetType

Returns the element type of the tensor.

```c
OgaElementType type;
OgaTensorGetType(tensor, &type);
```

---

### OgaTensorGetShapeRank

Returns the rank (number of dimensions) of the tensor.

```c
size_t rank;
OgaTensorGetShapeRank(tensor, &rank);
```

---

### OgaTensorGetShape

Returns the shape of the tensor.

```c
int64_t shape[rank];
OgaTensorGetShape(tensor, shape, rank);
```

---

### OgaTensorGetData

Returns a pointer to the tensor data.

```c
void* data = NULL;
OgaTensorGetData(tensor, &data);
```

---

### OgaDestroyTensor

Destroys the tensor.

```c
OgaDestroyTensor(tensor);
```

---

## Images and Audios API

### OgaLoadImages

Loads images from file paths.

```c
OgaStringArray* image_paths = NULL;
OgaCreateStringArrayFromStrings(paths, count, &image_paths);
OgaImages* images = NULL;
OgaLoadImages(image_paths, &images);
OgaDestroyStringArray(image_paths);
```

---

### OgaLoadImagesFromBuffers

Loads images from memory buffers.

```c
OgaImages* images = NULL;
OgaLoadImagesFromBuffers(image_data, image_sizes, count, &images);
```

---

### OgaDestroyImages

Destroys the images object.

```c
OgaDestroyImages(images);
```

---

### OgaLoadAudios

Loads audios from file paths.

```c
OgaStringArray* audio_paths = NULL;
OgaCreateStringArrayFromStrings(paths, count, &audio_paths);
OgaAudios* audios = NULL;
OgaLoadAudios(audio_paths, &audios);
OgaDestroyStringArray(audio_paths);
```

---

### OgaLoadAudiosFromBuffers

Loads audios from memory buffers.

```c
OgaAudios* audios = NULL;
OgaLoadAudiosFromBuffers(audio_data, audio_sizes, count, &audios);
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
OgaCreateNamedTensors(&named_tensors);
```

---

### OgaNamedTensorsGet

Gets a tensor by name.

```c
OgaTensor* tensor = NULL;
OgaNamedTensorsGet(named_tensors, "input_name", &tensor);
```

---

### OgaNamedTensorsSet

Sets a tensor by name.

```c
OgaNamedTensorsSet(named_tensors, "input_name", tensor);
```

---

### OgaNamedTensorsDelete

Deletes a tensor by name.

```c
OgaNamedTensorsDelete(named_tensors, "input_name");
```

---

### OgaNamedTensorsCount

Returns the number of named tensors.

```c
size_t count = 0;
OgaNamedTensorsCount(named_tensors, &count);
```

---

### OgaNamedTensorsGetNames

Gets the names of all tensors.

```c
OgaStringArray* names = NULL;
OgaNamedTensorsGetNames(named_tensors, &names);
OgaDestroyStringArray(names);
```

---

### OgaDestroyNamedTensors

Destroys the named tensors object.

```c
OgaDestroyNamedTensors(named_tensors);
```

---

## Utility Functions

### OgaSetLogBool

Sets a boolean logging option.

```c
OgaSetLogBool("option_name", true);
```

---

### OgaSetLogString

Sets a string logging option.

```c
OgaSetLogString("option_name", "value");
```

---

### OgaSetCurrentGpuDeviceId

Sets the current GPU device ID.

```c
OgaSetCurrentGpuDeviceId(0);
```

---

### OgaGetCurrentGpuDeviceId

Gets the current GPU device ID.

```c
int device_id = 0;
OgaGetCurrentGpuDeviceId(&device_id);
```

---

### OgaResultGetError

Gets the error message from an OgaResult.

```c
const char* error = OgaResultGetError(result);
```

---

### OgaDestroyResult

Destroys an OgaResult.

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

### OgaDestroyBuffer

Destroys a buffer.

```c
OgaDestroyBuffer(buffer);
```

---

### OgaBufferGetType

Gets the type of the buffer.

```c
OgaDataType type = OgaBufferGetType(buffer);
```

---

### OgaBufferGetDimCount

Gets the number of dimensions of a buffer.

```c
size_t dim_count = OgaBufferGetDimCount(buffer);
```

---

### OgaBufferGetDims

Gets the dimensions of a buffer.

```c
size_t dims[dim_count];
OgaBufferGetDims(buffer, dims, dim_count);
```

---

### OgaBufferGetData

Gets the data from a buffer.

```c
const void* data = OgaBufferGetData(buffer);
```

---