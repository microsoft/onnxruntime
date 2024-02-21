---
title: C API
description: C API reference for ONNX Runtime GenAI
has_children: false
parent: API docs
grand_parent: Generative AI
nav_order: 2
---

# ONNX Runtime GenAI C API

_Note: this API is in preview and is subject to change._

{: .no_toc }

* TOC placeholder
{:toc}


## Overview

## Functions

### Create model

Creates a model from the given configuration directory and device type.

#### Parameters
 * Input: config_path The path to the model configuration directory. The path is expected to be encoded in UTF-8.
 * Input: device_type The device type to use for the model.
 * Output:  out The created model.

#### Returns
 OgaResult containing the error message if the model creation failed.
 

### Destroy model

#### Parameters

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateModel(const char* config_path, OgaDeviceType device_type, OgaModel** out);
```

#### Parameters


 Destroys the given model.
 * Input: model The model to be destroyed.
 
```c
OGA_EXPORT void OGA_API_CALL OgaDestroyModel(OgaModel* model);
```

### Create Tokenizer

#### Parameters

#### Returns

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateTokenizer(const OgaModel* model, OgaTokenizer** out);
```

### Destroy Tokenizer

#### Parameters

#### Returns

```c
OGA_EXPORT void OGA_API_CALL OgaDestroyTokenizer(OgaTokenizer*);
```

### Encode batch

#### Parameters

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerEncodeBatch(const OgaTokenizer*, const char** strings, size_t count, OgaSequences** out);
```

### Decode batch

#### Parameters

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerDecodeBatch(const OgaTokenizer*, const OgaSequences* tokens, const char*** out_strings);
```

### Destroy tokenizer strings

#### Parameters

```c
OGA_EXPORT void OGA_API_CALL OgaTokenizerDestroyStrings(const char** strings, size_t count);
```

### Create tokenizer stream


#### Parameters

```c
OgaTokenizerStream is to decoded token strings incrementally, one token at a time.
```

#### Parameters

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateTokenizerStream(const OgaTokenizer*, OgaTokenizerStream** out);
```

### Destroy tokenizer stream

#### Parameters

```c
OGA_EXPORT void OGA_API_CALL OgaDestroyTokenizerStream(OgaTokenizerStream*);
```

### Decode stream

Decode a single token in the stream. If this results in a word being generated, it will be 

#### Parameters

returned in 'out'.
 * The caller is responsible for concatenating each chunk together to generate the complete result.
 * 'out' is valid until the next call to OgaTokenizerStreamDecode or when the OgaTokenizerStream is destroyed
 
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerStreamDecode(OgaTokenizerStream*, int32_t token, const char** out);
```


### Create Generator

Creates a generator from the given model and generator params.

#### Parameters

 * Input: model The model to use for generation.
 * Input: params The parameters to use for generation.
 * Output:  out The created generator.

#### Returns
OgaResult containing the error message if the generator creation failed.
 
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

### Create generator params

Creates a OgaGeneratorParams from the given model.

#### Parameters

* Input: model The model to use for generation.
* Output:  out The created generator params.

#### Returns

OgaResult containing the error message if the generator params creation failed.
 
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateGeneratorParams(const OgaModel* model, OgaGeneratorParams** out);
```

### Destroy generator params

Destroys the given generator params.

#### Parameters

 * Input: generator_params The generator params to be destroyed.

#### Returns
`void`

```c
OGA_EXPORT void OGA_API_CALL OgaDestroyGeneratorParams(OgaGeneratorParams* generator_params);
```

### Set maximum length

Sets the maximum length that the generated sequence can have.

#### Parameters

* Input: params The generator params to set the maximum length on.
* Input: max_length The maximum length of the generated sequences.

#### Returns

`OgaResult` containing the error message if the setting of the maximum length failed.
 
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetMaxLength(OgaGeneratorParams* generator_params, size_t max_length);
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

 OgaResult containing the error message if the setting of the input ids failed.
 
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

### Encode

Encodes a single string and adds the encoded sequence of tokens to the OgaSequences. The OgaSequences must be freed with OgaDestroySequences when it is no longer needed.

#### Parameters

#### Returns

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerEncode(const OgaTokenizer*, const char* str, OgaSequences* sequences);
```

### Decode

Decode a single token sequence and returns a null terminated utf8 string. out_string must be freed with OgaDestroyString

#### Parameters

#### Returns

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerDecode(const OgaTokenizer*, const int32_t* tokens, size_t token_count, const char** out_string);
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

### Create generator params

Creates a OgaGeneratorParams from the given model.

#### Parameters

* Input: model The model to use for generation.
* Output:  out The created generator params.

#### Returns

OgaResult containing the error message if the generator params creation failed.
 
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateGeneratorParams(const OgaModel* model, OgaGeneratorParams** out);
```

### Destroy generator params

Destroys the given generator params.

#### Parameters

* Input: generator_params The generator params to be destroyed.

#### Returns
`void`

```c
OGA_EXPORT void OGA_API_CALL OgaDestroyGeneratorParams(OgaGeneratorParams* generator_params);
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

Generates the next token based on the computed logits using the greedy search.

#### Parameters

 * Input: generator The generator to generate the next token for.

#### Returns

OgaResult containing the error message if the generation of the next token failed.

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GenerateNextToken_Top(OgaGenerator* generator);
```

### Generate next token with Top K sampling

#### Parameters

#### Returns

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GenerateNextToken_TopK(OgaGenerator* generator, int k, float t);
```

### Generate next token with Top P sampling

#### Parameters

#### Returns

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GenerateNextToken_TopP(OgaGenerator* generator, float p, float t);
```

### Get number of tokens

 Returns the number of tokens in the sequence at the given index.

#### Parameters

 * Input: generator The generator to get the count of the tokens for the sequence at the given index.
 * Input: index. The index at which to return the tokens

#### Returns

The number tokens in the sequence at the given index.
 
```c
OGA_EXPORT size_t OGA_API_CALL OgaGenerator_GetSequenceLength(const OgaGenerator* generator, size_t index);
```

### Get sequence

Returns a pointer to the sequence data at the given index. The number of tokens in the sequence is given by OgaGenerator_GetSequenceLength.

#### Parameters

* Input: generator The generator to get the sequence data for the sequence at the given index. The pointer to the sequence data at the given index. The sequence data is owned by the OgaGenerator and will be freed when the OgaGenerator is destroyed. The caller must copy the data if it needs to be used after the OgaGenerator is destroyed.
* Input: index. The index at which to get the sequence.

#### Returns

A pointer to the token sequence

```c
OGA_EXPORT const int32_t* OGA_API_CALL OgaGenerator_GetSequence(const OgaGenerator* generator, size_t index);
```

## Enums and structs

```c
typedef enum OgaDeviceType {
  OgaDeviceTypeAuto,
  OgaDeviceTypeCPU,
  OgaDeviceTypeCUDA,
} OgaDeviceType;
```

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


#### Parameters

#### Returns

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

