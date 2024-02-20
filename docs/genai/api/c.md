---
title: C API
description: C API reference for ONNX Runtime GenAI
has_children: false
grand_parent: Generative AI
nav_order: 2
---

# ONNX Runtime GenAI C API



## Create model

/*
 * \brief Creates a model from the given configuration directory and device type.
 * \param[in] config_path The path to the model configuration directory. The path is expected to be encoded in UTF-8.
 * \param[in] device_type The device type to use for the model.
 * \param[out] out The created model.
 * \return OgaResult containing the error message if the model creation failed.
 */

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateModel(const char* config_path, OgaDeviceType device_type, OgaModel** out);
```

/*
 * \brief Destroys the given model.
 * \param[in] model The model to be destroyed.
 */
```c
OGA_EXPORT void OGA_API_CALL OgaDestroyModel(OgaModel* model);
```

## Create Tokenizer

```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateTokenizer(const OgaModel* model, OgaTokenizer** out);
OGA_EXPORT void OGA_API_CALL OgaDestroyTokenizer(OgaTokenizer*);
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerEncodeBatch(const OgaTokenizer*, const char** strings, size_t count, OgaSequences** out);
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerDecodeBatch(const OgaTokenizer*, const OgaSequences* tokens, const char*** out_strings);
OGA_EXPORT void OGA_API_CALL OgaTokenizerDestroyStrings(const char** strings, size_t count);
```


/* OgaTokenizerStream is to decoded token strings incrementally, one token at a time.
 */
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateTokenizerStream(const OgaTokenizer*, OgaTokenizerStream** out);
OGA_EXPORT void OGA_API_CALL OgaDestroyTokenizerStream(OgaTokenizerStream*);
```

/*
 * Decode a single token in the stream. If this results in a word being generated, it will be returned in 'out'.
 * The caller is responsible for concatenating each chunk together to generate the complete result.
 * 'out' is valid until the next call to OgaTokenizerStreamDecode or when the OgaTokenizerStream is destroyed
 */
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerStreamDecode(OgaTokenizerStream*, int32_t token, const char** out);
```


## Create Generator

/*
 * \brief Creates a generator from the given model and generator params.
 * \param[in] model The model to use for generation.
 * \param[in] params The parameters to use for generation.
 * \param[out] out The created generator.
 * \return OgaResult containing the error message if the generator creation failed.
 */
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateGenerator(const OgaModel* model, const OgaGeneratorParams* params, OgaGenerator** out);
```

/*
 * \brief Destroys the given generator.
 * \param[in] generator The generator to be destroyed.
 */
```c
OGA_EXPORT void OGA_API_CALL OgaDestroyGenerator(OgaGenerator* generator);
```

## Create and set generator input parameters

/*
 * \brief Creates a OgaGeneratorParams from the given model.
 * \param[in] model The model to use for generation.
 * \param[out] out The created generator params.
 * \return OgaResult containing the error message if the generator params creation failed.
 */
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateGeneratorParams(const OgaModel* model, OgaGeneratorParams** out);
```

/*
 * \brief Destroys the given generator params.
 * \param[in] generator_params The generator params to be destroyed.
 */
```c
OGA_EXPORT void OGA_API_CALL OgaDestroyGeneratorParams(OgaGeneratorParams* generator_params);
```

/*
 * \brief Sets the maximum length that the generated sequence can have.
 * \param[in] params The generator params to set the maximum length on.
 * \param[in] max_length The maximum length of the generated sequences.
 * \return OgaResult containing the error message if the setting of the maximum length failed.
 */
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetMaxLength(OgaGeneratorParams* generator_params, size_t max_length);
```

/*
 * \brief Sets the input ids for the generator params. The input ids are used to seed the generation.
 * \param[in] generator_params The generator params to set the input ids on.
 * \param[in] input_ids The input ids array of size input_ids_count = batch_size * sequence_length.
 * \param[in] input_ids_count The total number of input ids.
 * \param[in] sequence_length The sequence length of the input ids.
 * \param[in] batch_size The batch size of the input ids.
 * \return OgaResult containing the error message if the setting of the input ids failed.
 */
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetInputIDs(OgaGeneratorParams* generator_params, const int32_t* input_ids, size_t input_ids_count, size_t sequence_length, size_t batch_size);
```

/*
 * \brief Sets the input id sequences for the generator params. The input id sequences are used to seed the generation.
 * \param[in] generator_params The generator params to set the input ids on.
 * \param[in] sequences The input id sequences.
 * \return OgaResult containing the error message if the setting of the input id sequences failed.
 */
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetInputSequences(OgaGeneratorParams* generator_params, const OgaSequences* sequences);
```

## Tokenize and decode tokens

/* Encodes a single string and adds the encoded sequence of tokens to the OgaSequences. The OgaSequences must be freed with OgaDestroySequences
   when it is no longer needed.
 */
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerEncode(const OgaTokenizer*, const char* str, OgaSequences* sequences);
```

/* Decode a single token sequence and returns a null terminated utf8 string. out_string must be freed with OgaDestroyString
 */
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerDecode(const OgaTokenizer*, const int32_t* tokens, size_t token_count, const char** out_string);
```


## Generate output

### High level API

/*
 * \brief Generates an array of token arrays from the model execution based on the given generator params.
 * \param[in] model The model to use for generation.
 * \param[in] generator_params The parameters to use for generation.
 * \param[out] out The generated sequences of tokens. The caller is responsible for freeing the sequences using OgaDestroySequences
 *             after it is done using the sequences.
 * \return OgaResult containing the error message if the generation failed.
 */
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerate(const OgaModel* model, const OgaGeneratorParams* generator_params, OgaSequences** out);
```

/*
 * \brief Creates a OgaGeneratorParams from the given model.
 * \param[in] model The model to use for generation.
 * \param[out] out The created generator params.
 * \return OgaResult containing the error message if the generator params creation failed.
 */
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateGeneratorParams(const OgaModel* model, OgaGeneratorParams** out);
```

/*
 * \brief Destroys the given generator params.
 * \param[in] generator_params The generator params to be destroyed.
 */
```c
OGA_EXPORT void OGA_API_CALL OgaDestroyGeneratorParams(OgaGeneratorParams* generator_params);
```

### Low level API


/*
 * \brief Returns true if the generator has finished generating all the sequences.
 * \param[in] generator The generator to check if it is done with generating all sequences.
 * \return True if the generator has finished generating all the sequences, false otherwise.
 */
```c
OGA_EXPORT bool OGA_API_CALL OgaGenerator_IsDone(const OgaGenerator* generator);
```

/*
 * \brief Computes the logits from the model based on the input ids and the past state. The computed logits are stored in the generator.
 * \param[in] generator The generator to compute the logits for.
 * \return OgaResult containing the error message if the computation of the logits failed.
 */
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_ComputeLogits(OgaGenerator* generator);
```

/*
 * \brief Generates the next token based on the computed logits using the greedy search.
 * \param[in] generator The generator to generate the next token for.
 * \return OgaResult containing the error message if the generation of the next token failed.
 */
```c
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GenerateNextToken_Top(OgaGenerator* generator);

OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GenerateNextToken_TopK(OgaGenerator* generator, int k, float t);
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GenerateNextToken_TopP(OgaGenerator* generator, float p, float t);
```

/*
 * \brief Returns the number of tokens in the sequence at the given index.
 * \param[in] generator The generator to get the count of the tokens for the sequence at the given index.
 * \return The number tokens in the sequence at the given index.
 */
```c
OGA_EXPORT size_t OGA_API_CALL OgaGenerator_GetSequenceLength(const OgaGenerator* generator, size_t index);
```

/*
 * \brief Returns a pointer to the sequence data at the given index. The number of tokens in the sequence
 *        is given by OgaGenerator_GetSequenceLength
 * \param[in] generator The generator to get the sequence data for the sequence at the given index.
 * \return The pointer to the sequence data at the given index. The sequence data is owned by the OgaGenerator
 *         and will be freed when the OgaGenerator is destroyed. The caller must copy the data if it needs to
 *         be used after the OgaGenerator is destroyed.
 */
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

typedef enum OgaDataType {
  OgaDataType_int32,
  OgaDataType_float32,
  OgaDataType_string,  // UTF8 string
} OgaDataType;

typedef struct OgaResult OgaResult;
typedef struct OgaGeneratorParams OgaGeneratorParams;
typedef struct OgaGenerator OgaGenerator;
typedef struct OgaModel OgaModel;
typedef struct OgaBuffer OgaBuffer;
```

// OgaSequences is an array of token arrays where the number of token arrays can be obtained using
// OgaSequencesCount and the number of tokens in each token array can be obtained using 

```c
OgaSequencesGetSequenceCount.
typedef struct OgaSequences OgaSequences;
typedef struct OgaTokenizer OgaTokenizer;
typedef struct OgaTokenizerStream OgaTokenizerStream;
```

## Utility functions

/*
 * \param[in] result OgaResult that contains the error message.
 * \return Error message contained in the OgaResult. The const char* is owned by the OgaResult
 *         and can will be freed when the OgaResult is destroyed.
 */
```c
OGA_EXPORT const char* OGA_API_CALL OgaResultGetError(OgaResult* result);
```

/*
 * \param[in] result OgaResult to be destroyed.
 */
```c
OGA_EXPORT void OGA_API_CALL OgaDestroyResult(OgaResult*);
OGA_EXPORT void OGA_API_CALL OgaDestroyString(const char*);

OGA_EXPORT void OGA_API_CALL OgaDestroyBuffer(OgaBuffer*);
OGA_EXPORT OgaDataType OGA_API_CALL OgaBufferGetType(const OgaBuffer*);
OGA_EXPORT size_t OGA_API_CALL OgaBufferGetDimCount(const OgaBuffer*);
OGA_EXPORT OgaResult* OGA_API_CALL OgaBufferGetDims(const OgaBuffer*, size_t* dims, size_t dim_count);
OGA_EXPORT const void* OGA_API_CALL OgaBufferGetData(const OgaBuffer*);

OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateSequences(OgaSequences** out);
```


/*
 * \param[in] sequences OgaSequences to be destroyed.
 */
```c
OGA_EXPORT void OGA_API_CALL OgaDestroySequences(OgaSequences* sequences);
```

/*
 * \brief Returns the number of sequences in the OgaSequences
 * \param[in] sequences
 * \return The number of sequences in the OgaSequences
 */
```c
OGA_EXPORT size_t OGA_API_CALL OgaSequencesCount(const OgaSequences* sequences);
```

/*
 * \brief Returns the number of tokens in the sequence at the given index
 * \param[in] sequences
 * \return The number of tokens in the sequence at the given index
 */
```c
OGA_EXPORT size_t OGA_API_CALL OgaSequencesGetSequenceCount(const OgaSequences* sequences, size_t sequence_index);
```

/*
 * \brief Returns a pointer to the sequence data at the given index. The number of tokens in the sequence
 *        is given by OgaSequencesGetSequenceCount
 * \param[in] sequences
 * \return The pointer to the sequence data at the given index. The pointer is valid until the OgaSequences is destroyed.
 */
```c
OGA_EXPORT const int32_t* OGA_API_CALL OgaSequencesGetSequenceData(const OgaSequences* sequences, size_t sequence_index);

OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetWhisperInputFeatures(OgaGeneratorParams*, const int32_t* inputs, size_t count);
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetWhisperDecoderInputIDs(OgaGeneratorParams*, const int32_t* input_ids, size_t input_ids_count);
```
