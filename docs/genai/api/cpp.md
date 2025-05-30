---
title: C++ API
description: C++ API reference for ONNX Runtime GenAI C++ API
has_children: false
parent: API docs
grand_parent: Generate API (Preview)
nav_order: 4
---

# ONNX Runtime GenAI C++ API

_Note: this API is in preview and is subject to change._

{: .no_toc }

* TOC placeholder
{:toc}

## Overview

This document describes the C++ API for ONNX Runtime GenAI.  
Below are the main classes and methods, with code snippets and descriptions for each.

---

## OgaModel

### Create

Creates a model from a configuration directory, with optional runtime settings or config object.

```cpp
auto model = OgaModel::Create("path/to/model_dir");
auto model2 = OgaModel::Create("path/to/model_dir", *settings);
auto model3 = OgaModel::Create(*config);
```

---

### GetType

Gets the type of the model.

```cpp
auto type = model->GetType();
```

---

### GetDeviceType

Gets the device type used by the model.

```cpp
auto device_type = model->GetDeviceType();
```

---

## OgaConfig

### Create

Creates a configuration object from a config path.

```cpp
auto config = OgaConfig::Create("path/to/model_dir");
```

---

### ClearProviders

Clears all providers from the configuration.

```cpp
config->ClearProviders();
```

---

### AppendProvider

Appends a provider to the configuration.

```cpp
config->AppendProvider("CUDAExecutionProvider");
```

---

### SetProviderOption

Sets a provider option in the configuration.

```cpp
config->SetProviderOption("CUDAExecutionProvider", "device_id", "0");
```

---

### Overlay

Overlays a JSON string onto the configuration.

```cpp
config->Overlay("{\"option\": \"value\"}");
```

---

## OgaRuntimeSettings

### Create

Creates a runtime settings object.

```cpp
auto settings = OgaRuntimeSettings::Create();
```

---

### SetHandle

Sets a named handle in the runtime settings.

```cpp
settings->SetHandle("custom_handle", handle_ptr);
```

---

## OgaTokenizer

### Create

Creates a tokenizer for the given model.

```cpp
auto tokenizer = OgaTokenizer::Create(*model);
```

---

### Encode

Encodes a string and adds the encoded sequence of tokens to the provided OgaSequences.

```cpp
auto sequences = OgaSequences::Create();
tokenizer->Encode("Hello world", *sequences);
```

---

### EncodeBatch

Encodes a batch of strings.

```cpp
const char* texts[] = {"Hello", "World"};
auto tensor = tokenizer->EncodeBatch(texts, 2);
```

---

### ToTokenId

Converts a string to its corresponding token ID.

```cpp
int32_t token_id = tokenizer->ToTokenId("Hello");
```

---

### Decode

Decodes a sequence of tokens into a string.

```cpp
auto str = tokenizer->Decode(tokens, token_count);
```

---

### ApplyChatTemplate

Applies a chat template to messages and tools.

```cpp
auto templated = tokenizer->ApplyChatTemplate("template", "messages", "tools", true);
```

---

### DecodeBatch

Decodes a batch of token sequences.

```cpp
auto decoded = tokenizer->DecodeBatch(*tensor);
```

---

## OgaTokenizerStream

### Create

Creates a tokenizer stream for incremental decoding.

```cpp
auto stream = OgaTokenizerStream::Create(*tokenizer);
```

---

### Decode

Decodes a single token in the stream. If this results in a word being generated, it will be returned.

```cpp
const char* chunk = stream->Decode(token);
```

---

## OgaSequences

### Create

Creates an empty OgaSequences object.

```cpp
auto sequences = OgaSequences::Create();
```

---

### Count

Returns the number of sequences.

```cpp
size_t n = sequences->Count();
```

---

### SequenceCount

Returns the number of tokens in the sequence at the given index.

```cpp
size_t tokens = sequences->SequenceCount(0);
```

---

### SequenceData

Returns a pointer to the token data for the sequence at the given index.

```cpp
const int32_t* data = sequences->SequenceData(0);
```

---

### Append

Appends a sequence of tokens or a single token to the sequences.

```cpp
sequences->Append(tokens, token_count);
sequences->Append(token, sequence_index);
```

---

## OgaGeneratorParams

### Create

Creates generator parameters for the given model.

```cpp
auto params = OgaGeneratorParams::Create(*model);
```

---

### SetSearchOption

Sets a numeric search option.

```cpp
params->SetSearchOption("max_length", 128);
```

---

### SetSearchOptionBool

Sets a boolean search option.

```cpp
params->SetSearchOptionBool("do_sample", true);
```

---

### SetModelInput

Sets an additional model input.

```cpp
params->SetModelInput("input_name", *tensor);
```

---

### SetInputs

Sets named tensors as inputs.

```cpp
params->SetInputs(*named_tensors);
```

---

### SetGuidance

Sets guidance data.

```cpp
params->SetGuidance("type", "data");
```

---

## OgaGenerator

### Create

Creates a generator from the given model and parameters.

```cpp
auto generator = OgaGenerator::Create(*model, *params);
```

---

### IsDone

Checks if generation is complete.

```cpp
bool done = generator->IsDone();
```

---

### AppendTokenSequences

Appends token sequences to the generator.

```cpp
generator->AppendTokenSequences(*sequences);
```

---

### AppendTokens

Appends tokens to the generator.

```cpp
generator->AppendTokens(tokens, token_count);
```

---

### IsSessionTerminated

Checks if the session is terminated.

```cpp
bool terminated = generator->IsSessionTerminated();
```

---

### GenerateNextToken

Generates the next token.

```cpp
generator->GenerateNextToken();
```

---

### RewindTo

Rewinds the sequence to a new length.

```cpp
generator->RewindTo(new_length);
```

---

### SetRuntimeOption

Sets a runtime option.

```cpp
generator->SetRuntimeOption("terminate_session", "1");
```

---

### GetSequenceCount

Returns the number of tokens in the sequence at the given index.

```cpp
size_t count = generator->GetSequenceCount(0);
```

---

### GetSequenceData

Returns a pointer to the sequence data at the given index.

```cpp
const int32_t* data = generator->GetSequenceData(0);
```

---

### GetOutput

Gets a named output tensor.

```cpp
auto tensor = generator->GetOutput("output_name");
```

---

### GetLogits

Gets the logits tensor.

```cpp
auto logits = generator->GetLogits();
```

---

### SetLogits

Sets the logits tensor.

```cpp
generator->SetLogits(*tensor);
```

---

### SetActiveAdapter

Sets the active adapter for the generator.

```cpp
generator->SetActiveAdapter(*adapters, "adapter_name");
```

---

## OgaTensor

### Create

Creates a tensor from a buffer.

```cpp
auto tensor = OgaTensor::Create(data, shape, shape_dims_count, element_type);
```

---

### Type

Returns the element type of the tensor.

```cpp
auto type = tensor->Type();
```

---

### Shape

Returns the shape of the tensor.

```cpp
auto shape = tensor->Shape();
```

---

### Data

Returns a pointer to the tensor data.

```cpp
void* data = tensor->Data();
```

---

## OgaImages

### Load

Loads images from file paths or memory buffers.

```cpp
std::vector<const char*> image_paths = {"img1.png", "img2.png"};
auto images = OgaImages::Load(image_paths);

auto images2 = OgaImages::Load(image_data_ptrs, image_sizes, count);
```

---

## OgaAudios

### Load

Loads audios from file paths or memory buffers.

```cpp
std::vector<const char*> audio_paths = {"audio1.wav", "audio2.wav"};
auto audios = OgaAudios::Load(audio_paths);

auto audios2 = OgaAudios::Load(audio_data_ptrs, audio_sizes, count);
```

---

## OgaNamedTensors

### Create

Creates a named tensors object.

```cpp
auto named_tensors = OgaNamedTensors::Create();
```

---

### Get

Gets a tensor by name.

```cpp
auto tensor = named_tensors->Get("input_name");
```

---

### Set

Sets a tensor by name.

```cpp
named_tensors->Set("input_name", *tensor);
```

---

### Delete

Deletes a tensor by name.

```cpp
named_tensors->Delete("input_name");
```

---

### Count

Returns the number of named tensors.

```cpp
size_t count = named_tensors->Count();
```

---

### GetNames

Gets the names of all tensors.

```cpp
auto names = named_tensors->GetNames();
```

---

## OgaAdapters

### Create

Creates an adapters manager for the given model.

```cpp
auto adapters = OgaAdapters::Create(*model);
```

---

### LoadAdapter

Loads an adapter from file.

```cpp
adapters->LoadAdapter("adapter_file_path", "adapter_name");
```

---

### UnloadAdapter

Unloads an adapter by name.

```cpp
adapters->UnloadAdapter("adapter_name");
```

---

## OgaMultiModalProcessor

### Create

Creates a multi-modal processor for the given model.

```cpp
auto processor = OgaMultiModalProcessor::Create(*model);
```

---

### ProcessImages

Processes images and returns named tensors.

```cpp
auto named_tensors = processor->ProcessImages("prompt", images.get());
```

---

### ProcessAudios

Processes audios and returns named tensors.

```cpp
auto named_tensors = processor->ProcessAudios(audios.get());
```

---

### ProcessImagesAndAudios

Processes both images and audios.

```cpp
auto named_tensors = processor->ProcessImagesAndAudios("prompt", images.get(), audios.get());
```

---

### Decode

Decodes a sequence of tokens into a string.

```cpp
auto str = processor->Decode(tokens, token_count);
```

---

## OgaHandle

### Constructor / Destructor

Initializes and shuts down the global Oga runtime.

```cpp
OgaHandle handle;
```

---

## Oga Utility Functions

### SetLogBool

Sets a boolean logging option.

```cpp
Oga::SetLogBool("option_name", true);
```

---

### SetLogString

Sets a string logging option.

```cpp
Oga::SetLogString("option_name", "value");
```

---

### SetCurrentGpuDeviceId

Sets the current GPU device ID.

```cpp
Oga::SetCurrentGpuDeviceId(0);
```

---

### GetCurrentGpuDeviceId

Gets the current GPU device ID.

```cpp
int id = Oga::GetCurrentGpuDeviceId();
```

---