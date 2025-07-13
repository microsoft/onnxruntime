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

## Model API

### Create model

Creates a model from the given directory. The directory should contain a file called `genai_config.json`, which corresponds to the [configuration specification](../reference/config.md).

#### Parameters
 * Input: config_path The path to the model configuration directory. The path is expected to be encoded in UTF-8.
 * Output:  out The created model.

#### Returns
 `OgaResult` containing the error message if the model creation failed.

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateModel(const char* config_path, OgaModel** out);
```

### Destroy model

Destroys the given model.


#### Parameters

* Input: model The model to be destroyed.
 
#### Returns
`void`

```c
OGA_EXPORT void OGA_API_CALL OgaDestroyModel(OgaModel* model);
```

### Generate

Generates an array of token arrays from the model execution based on the given generator params.

#### Parameters

* Input: model The model to use for generation.
* Input: generator_params The parameters to use for generation.
* Output:  out The generated sequences of tokens. The caller is responsible for freeing the sequences using OgaDestroySequences after it is done using the sequences.

#### Returns

OgaResult containing the error message if the generation failed.
 
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerate(const OgaModel* model, const OgaGeneratorParams* generator_params, OgaSequences** out);
```

## Tokenizer API

### Create Tokenizer

#### Parameters
* Input: model. The model for which the tokenizer should be created

#### Returns
`OgaResult` containing the error message if the tokenizer creation failed.

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateTokenizer(const OgaModel* model, OgaTokenizer** out);
```

### Destroy Tokenizer

```c
OGA_EXPORT void OGA_API_CALL OgaDestroyTokenizer(OgaTokenizer*);
```
### Encode

Encodes a single string and adds the encoded sequence of tokens to the OgaSequences. The OgaSequences must be freed with OgaDestroySequences when it is no longer needed.

#### Parameters

#### Returns

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerEncode(const OgaTokenizer*, const char* str, OgaSequences* sequences);
```

### ApplyChatTemplate

Processes the specified template with the provided input using the tokenizer, and outputs the resulting string. Optionally, it can include a generation prompt in the output. out_string must be freed with OgaDestroyString

#### Parameters

#### Returns

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerApplyChatTemplate(const OgaTokenizer* tokenizer, const char* template_str, const char* messages, bool add_generation_prompt, const char** out_string);
```

### Decode

Decode a single token sequence and returns a null terminated utf8 string. out_string must be freed with OgaDestroyString

#### Parameters

#### Returns

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerDecode(const OgaTokenizer*, const int32_t* tokens, size_t token_count, const char** out_string);
```

### Encode batch

#### Parameters
* 
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerEncodeBatch(const OgaTokenizer*, const char** strings, size_t count, TokenSequences** out);
```

### Decode batch

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerDecodeBatch(const OgaTokenizer*, const OgaSequences* tokens, const char*** out_strings);
```

### Destroy tokenizer strings

```c
OGA_EXPORT void OGA_API_CALL OgaTokenizerDestroyStrings(const char** strings, size_t count);
```

### Create tokenizer stream

OgaTokenizerStream is used to decoded token strings incrementally, one token at a time.

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateTokenizerStream(const OgaTokenizer*, OgaTokenizerStream** out);
```

### Destroy tokenizer stream

#### Parameters

```c
OGA_EXPORT void OGA_API_CALL OgaDestroyTokenizerStream(OgaTokenizerStream*);
```

### Decode stream

Decode a single token in the stream. If this results in a word being generated, it will be returned in 'out'. The caller is responsible for concatenating each chunk together to generate the complete result.
'out' is valid until the next call to OgaTokenizerStreamDecode or when the OgaTokenizerStream is destroyed
 
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerStreamDecode(OgaTokenizerStream*, int32_t token, const char** out);
```

## Generator Params API

### Create Generator Params

Creates a OgaGeneratorParams from the given model.

#### Parameters

* Input: model The model to use for generation.
* Output:  out The created generator params.

#### Returns

`OgaResult` containing the error message if the generator params creation failed.
 
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateGeneratorParams(const OgaModel* model, OgaGeneratorParams** out);
```

### Destroy Generator Params

Destroys the given generator params.

#### Parameters

* Input: generator_params The generator params to be destroyed.

#### Returns
`void`

```c
OGA_EXPORT void OGA_API_CALL OgaDestroyGeneratorParams(OgaGeneratorParams* generator_params);
```

### Set search option (number)

Set a search option where the option is a number

#### Parameters
* generator_params: The generator params object to set the parameter on
* name: the name of the parameter
* value: the value to set

#### Returns
`OgaResult` containing the error message if the generator params creation failed.

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetSearchNumber(OgaGeneratorParams* generator_params, const char* name, double value);
```

### Set search option (bool)

Set a search option where the option is a bool.

#### Parameters
* generator_params: The generator params object to set the parameter on
* name: the name of the parameter
* value: the value to set

#### Returns
`OgaResult` containing the error message if the generator params creation failed.

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetSearchBool(OgaGeneratorParams* generator_params, const char* name, bool value);
```

### Try graph capture with max batch size

Graph capture fixes the dynamic elements of the computation graph to constant values. It can provide more efficient execution in some environments. To execute in graph capture mode, the maximum batch size needs to be known ahead of time. This function can fail if there is not enough memory to allocate the specified maximum batch size.

#### Parameters

* generator_params: The generator params object to set the parameter on
* max_batch_size: The maximum batch size to allocate

#### Returns

`OgaResult` containing the error message if graph capture mode could not be configured with the specified batch size

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsTryGraphCaptureWithMaxBatchSize(OgaGeneratorParams* generator_params, int32_t max_batch_size);
```

### Set inputs

Sets the input ids for the generator params. The input ids are used to seed the generation.

#### Parameters

 * Input: generator_params The generator params to set the input ids on.
 * Input: input_ids The input ids array of size input_ids_count = batch_size * sequence_length.
 * Input: input_ids_count The total number of input ids.
 * Input: sequence_length The sequence length of the input ids.
 * Input: batch_size The batch size of the input ids.

#### Returns

`OgaResult` containing the error message if the setting of the input ids failed.
 
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetInputIDs(OgaGeneratorParams* generator_params, const int32_t* input_ids, size_t input_ids_count, size_t sequence_length, size_t batch_size);
```

### Set input sequence

Sets the input id sequences for the generator params. The input id sequences are used to seed the generation.

#### Parameters

 * Input: generator_params The generator params to set the input ids on.
 * Input: sequences The input id sequences.

#### Returns

OgaResult containing the error message if the setting of the input id sequences failed.
 
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetInputSequences(OgaGeneratorParams* generator_params, const OgaSequences* sequences);
```

### Set model input

Set an additional model input, aside from the input_ids.

### Parameters

* generator_params: The generator params to set the input on
* name: the name of the parameter to set
* tensor: the value of the parameter

### Returns

OgaResult containing the error message if the setting of the input failed.

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetWhisperInputFeatures(OgaGeneratorParams*, OgaTensor* tensor);
```


## Generator API

### Create Generator

Creates a generator from the given model and generator params.

#### Parameters

 * Input: model The model to use for generation.
 * Input: params The parameters to use for generation.
 * Output:  out The created generator.

#### Returns
`OgaResult` containing the error message if the generator creation failed.
 
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateGenerator(const OgaModel* model, const OgaGeneratorParams* params, OgaGenerator** out);
```

### Destroy generator

Destroys the given generator.

#### Parameters

* Input: generator The generator to be destroyed.

#### Returns
`void`

```c
OGA_EXPORT void OGA_API_CALL OgaDestroyGenerator(OgaGenerator* generator);
```

### Check if generation has completed

Returns true if the generator has finished generating all the sequences.

#### Parameters

* Input: generator The generator to check if it is done with generating all sequences.

#### Returns

True if the generator has finished generating all the sequences, false otherwise.
 
```c
OGA_EXPORT bool OGA_API_CALL OgaGenerator_IsDone(const OgaGenerator* generator);
```

### Run one iteration of the model

Computes the logits from the model based on the input ids and the past state. The computed logits are stored in the generator.

#### Parameters

* Input: generator The generator to compute the logits for.

#### Returns

OgaResult containing the error message if the computation of the logits failed.
 
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_ComputeLogits(OgaGenerator* generator);
```

### Generate next token

Generates the next token based on the computed logits using the configured generation parameters.

#### Parameters

 * Input: generator The generator to generate the next token for.

#### Returns

OgaResult containing the error message if the generation of the next token failed.

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GenerateNextToken(OgaGenerator* generator);
```


### Get number of tokens

Returns the number of tokens in the sequence at the given index.

#### Parameters

 * Input: generator The generator to get the count of the tokens for the sequence at the given index.
 * Input: index. The index at which to return the tokens

#### Returns

The number tokens in the sequence at the given index.
 
```c
OGA_EXPORT size_t OGA_API_CALL OgaGenerator_GetSequenceCount(const OgaGenerator* generator, size_t index);
```

### Get sequence

Returns a pointer to the sequence data at the given index. The number of tokens in the sequence is given by `OgaGenerator_GetSequenceCount`.

#### Parameters

* Input: generator The generator to get the sequence data for the sequence at the given index. The pointer to the sequence data at the given index. The sequence data is owned by the OgaGenerator and will be freed when the OgaGenerator is destroyed. The caller must copy the data if it needs to be used after the OgaGenerator is destroyed.
* Input: index. The index at which to get the sequence.

#### Returns

A pointer to the token sequence

```c
OGA_EXPORT const int32_t* OGA_API_CALL OgaGenerator_GetSequenceData(const OgaGenerator* generator, size_t index);
```

### Set Runtime Option

An API to set Runtime options, more parameters will be added to this generic API to support Runtime options. An example to use this API for terminating the current session would be to call the SetRuntimeOption with key as "terminate_session" and value as "1": OgaGenerator_SetRuntimeOption(generator, "terminate_session", "1")

More details on the current runtime options can be found [here](https://github.com/microsoft/onnxruntime-genai/blob/main/documents/Runtime_option.md).

#### Parameters

* Input: generator The generator on which the Runtime option needs to be set
* Input: key The key for setting the runtime option
* Input: value The value for the key provided

#### Returns
`void`

```c
OGA_EXPORT void OGA_API_CALL OgaGenerator_SetRuntimeOption(OgaGenerator* generator, const char* key, const char* value);
```

## Adapter API

This API is used to load and switch fine-tuned adapters, such as LoRA adapters.

### Create adapters

Creates the object that manages the adapters. This object is used to load all the model adapters. It is responsible for reference counting the loaded adapters.

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateAdapters(const OgaModel* model, OgaAdapters** out);
```

#### Parameters

* model: the `OgaModel`, which has previously been created

#### Results

* out: a reference to the list of `OgaAdapters` created

### Load adapter

Loads the model adapter from the given adapter file path and adapter name.

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaLoadAdapter(OgaAdapters* adapters, const char* adapter_file_path, const char* adapter_name);
```

#### Parameters

 * `adapters`: The OgaAdapters object into which to load the adapter.
 * `adapter_file_path`: The file path of the adapter to load.
 * `adapter_name`: A unique identifier for the adapter to be used for adapter querying

#### Return value

`OgaResult` containing an error message if the adapter failed to load.

### Unload adapter

Unloads the adapter with the given identifier from the set of previously loaded adapters. If the adapter is not found, or if it cannot be unloaded (when it is in use), an error is returned.

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaUnloadAdapter(OgaAdapters* adapters, const char* adapter_name);
```

#### Parameters

* `adapters`: The OgaAdapters object from which to unload the adapter.
*  `adapter_name`: The name of the adapter to unload.

#### Return value

`OgaResult` containing an error message if the adapter failed to unload. This can occur if the method is called with an adapter that is not already loaded or has been marked active by a `OgaGenerator` still in use.

### Set active adapter

Sets the adapter with the given adapter name as active for the given OgaGenerator object.

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaSetActiveAdapter(OgaGenerator* generator, OgaAdapters* adapters, const char* adapter_name);
```

#### Parameters

* `generator`: The OgaGenerator object to set the active adapter.
* `adapters`: The OgaAdapters object that manages the model adapters.
* `adapter_name`: The name of the adapter to set as active.

#### Return value

`OgaResult` containing an error message if the adapter failed to be set as active. This can occur if the method is called with an adapter that has not been previously loaded.

## Enums and structs

```c
typedef enum OgaDataType {
  OgaDataType_int32,
  OgaDataType_float32,
  OgaDataType_string,  // UTF8 string
} OgaDataType;
```

```c
typedef struct OgaResult OgaResult;
typedef struct OgaGeneratorParams OgaGeneratorParams;
typedef struct OgaGenerator OgaGenerator;
typedef struct OgaModel OgaModel;
typedef struct OgaBuffer OgaBuffer;
```


## Utility functions

### Set the GPU device ID

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaSetCurrentGpuDeviceId(int device_id);
```

### Get the GPU device ID

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaGetCurrentGpuDeviceId(int* device_id);
```

### Get error message

#### Parameters

* Input: result OgaResult that contains the error message.

#### Returns

Error message contained in the OgaResult. The const char* is owned by the OgaResult and can will be freed when the OgaResult is destroyed.
 
```c
OGA_EXPORT const char* OGA_API_CALL OgaResultGetError(OgaResult* result);
```

### Destroy result

#### Parameters

* Input: result OgaResult to be destroyed.

#### Returns
`void`

```c
OGA_EXPORT void OGA_API_CALL OgaDestroyResult(OgaResult*);
```

### Destroy string

#### Parameters
* Input: string to be destroyed

#### Returns

```c
OGA_EXPORT void OGA_API_CALL OgaDestroyString(const char*);
```

### Destroy buffer

#### Parameters
* Input: buffer to be destroyed

#### Returns
`void`

```c
OGA_EXPORT void OGA_API_CALL OgaDestroyBuffer(OgaBuffer*);
```

### Get buffer type

#### Parameters
* Input: the buffer

#### Returns

The type of the buffer

```c
OGA_EXPORT OgaDataType OGA_API_CALL OgaBufferGetType(const OgaBuffer*);
```

### Get the number of dimensions of a buffer

#### Parameters
* Input: the buffer

#### Returns
The number of dimensions in the buffer

```c
OGA_EXPORT size_t OGA_API_CALL OgaBufferGetDimCount(const OgaBuffer*);
```

### Get buffer dimensions

Get the dimensions of a buffer

#### Parameters
* Input: the buffer
* Output: a dimension array

#### Returns
`OgaResult`

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaBufferGetDims(const OgaBuffer*, size_t* dims, size_t dim_count);
```

### Get buffer data

Get the data from a buffer

#### Parameters

#### Returns
`void`

```c
OGA_EXPORT const void* OGA_API_CALL OgaBufferGetData(const OgaBuffer*);
```

### Create sequences

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateSequences(OgaSequences** out);
```

### Destroy sequences

#### Parameters

* Input: sequences OgaSequences to be destroyed.

#### Returns
`void`

#### Returns

```c
OGA_EXPORT void OGA_API_CALL OgaDestroySequences(OgaSequences* sequences);
```

### Get number of sequences

Returns the number of sequences in the OgaSequences

#### Parameters

* Input: sequences

#### Returns
The number of sequences in the OgaSequences
 
```c
OGA_EXPORT size_t OGA_API_CALL OgaSequencesCount(const OgaSequences* sequences);
```

### Get the number of tokens in a sequence

Returns the number of tokens in the sequence at the given index

#### Parameters

* Input: sequences

#### Returns

The number of tokens in the sequence at the given index
 
```c
OGA_EXPORT size_t OGA_API_CALL OgaSequencesGetSequenceCount(const OgaSequences* sequences, size_t sequence_index);
```

### Get sequence data

Returns a pointer to the sequence data at the given index. The number of tokens in the sequence is given by OgaSequencesGetSequenceCount

#### Parameters
* Input: sequences

#### Returns

The pointer to the sequence data at the given index. The pointer is valid until the OgaSequences is destroyed.
 
```c
OGA_EXPORT const int32_t* OGA_API_CALL OgaSequencesGetSequenceData(const OgaSequences* sequences, size_t sequence_index);
```
