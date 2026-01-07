---
title: C# API
description: C# API reference for ONNX Runtime generate() API
has_children: false
parent: API docs
grand_parent: Generate API (Preview)
nav_order: 2
---

# ONNX Runtime generate() C# API

_Note: this API is in preview and is subject to change._

{: .no_toc }

* TOC placeholder
{:toc}

## Overview

This document describes the C# API for ONNX Runtime GenAI.

---

## Model class

### Constructor

Initialize a new model from the given model path.

```csharp
public Model(string modelPath)
```

Initialize a new model from an existing configuration.

```csharp
public Model(Config config)
```

---

## Config class

### Constructor

Initialize a new configuration object from a config path.

```csharp
public Config(string configPath)
```

---

### ClearProviders

Clear all providers from the configuration.

```csharp
public void ClearProviders()
```

---

### AppendProvider

Append a provider to the configuration.

```csharp
public void AppendProvider(string provider)
```

---

### SetProviderOption

Set a provider option in the configuration.

```csharp
public void SetProviderOption(string provider, string name, string value)
```

---

### AddModelData

Add in-memory model data to the configuration.

```csharp
public void AddModelData(string modelFilename, byte[] modelData)
```

---

### RemoveModelData

Remove model data that was previously added.

```csharp
public void RemoveModelData(string modelFilename)
```

---

### SetDecoderProviderOptionsHardwareDeviceType

Set the decoder hardware device type for a provider.

```csharp
public void SetDecoderProviderOptionsHardwareDeviceType(string provider, string hardware_device_type)
```

---

### SetDecoderProviderOptionsHardwareDeviceId

Set the decoder hardware device ID for a provider.

```csharp
public void SetDecoderProviderOptionsHardwareDeviceId(string provider, uint hardware_device_id)
```

---

### SetDecoderProviderOptionsHardwareVendorId

Set the decoder hardware vendor ID for a provider.

```csharp
public void SetDecoderProviderOptionsHardwareVendorId(string provider, uint hardware_vendor_id)
```

---

### ClearDecoderProviderOptionsHardwareDeviceType

Clear the decoder hardware device type setting for a provider.

```csharp
public void ClearDecoderProviderOptionsHardwareDeviceType(string provider)
```

---

### ClearDecoderProviderOptionsHardwareDeviceId

Clear the decoder hardware device ID setting for a provider.

```csharp
public void ClearDecoderProviderOptionsHardwareDeviceId(string provider)
```

---

### ClearDecoderProviderOptionsHardwareVendorId

Clear the decoder hardware vendor ID setting for a provider.

```csharp
public void ClearDecoderProviderOptionsHardwareVendorId(string provider)
```

---

## Tokenizer class

### Constructor

Initialize a tokenizer for the given model.

```csharp
public Tokenizer(Model model)
```

---

### Encode

Encode a string and return the encoded sequences.

```csharp
public Sequences Encode(string str)
```

---

### EncodeBatch

Encode a batch of strings and return the encoded sequences.

```csharp
public Sequences EncodeBatch(string[] strings)
```

---

### Decode

Decode a sequence of tokens into a string.

```csharp
public string Decode(ReadOnlySpan<int> sequence)
```

---

### DecodeBatch

Decode a batch of sequences into an array of strings.

```csharp
public string[] DecodeBatch(Sequences sequences)
```

---

### UpdateOptions

Update tokenizer options in bulk.

```csharp
public void UpdateOptions(Dictionary<string, string> options)
```

---

### ApplyChatTemplate

Apply a chat template to messages and tools.

```csharp
public string ApplyChatTemplate(string template, string messages, string tools, bool addGenerationPrompt)
```

---

### GetBosTokenId

Return the beginning-of-sequence token ID.

```csharp
public int GetBosTokenId()
```

---

### GetEosTokenIds

Return the end-of-sequence token IDs.

```csharp
public ReadOnlySpan<int> GetEosTokenIds()
```

---

### GetPadTokenId

Return the padding token ID.

```csharp
public int GetPadTokenId()
```

---

### CreateStream

Create a tokenizer stream for incremental decoding.

```csharp
public TokenizerStream CreateStream()
```

---

## TokenizerStream class

### Decode

Decode a single token in the stream and return the generated string chunk.

```csharp
public string Decode(int token)
```

---

## GeneratorParams class

### Constructor

Initialize generator parameters for the given model.

```csharp
public GeneratorParams(Model model)
```

---

### SetSearchOption (double)

Set a numeric search option.

```csharp
public void SetSearchOption(string searchOption, double value)
```

---

### SetSearchOption (bool)

Set a boolean search option.

```csharp
public void SetSearchOption(string searchOption, bool value)
```

---

### TryGraphCaptureWithMaxBatchSize

Attempt to enable graph capture mode with a maximum batch size (deprecated; logs a warning).

```csharp
public void TryGraphCaptureWithMaxBatchSize(int maxBatchSize)
```

---

### SetGuidance

Configure guided generation behavior.

```csharp
public void SetGuidance(string type, string data, bool enableFFTokens = false)
```

---

## Generator class

### Constructor

Initialize a generator from the given model and generator parameters.

```csharp
public Generator(Model model, GeneratorParams generatorParams)
```

---

### IsDone

Check if generation is complete.

```csharp
public bool IsDone()
```

---

### SetModelInput

Set a named model input tensor.

```csharp
public void SetModelInput(string name, Tensor value)
```

---

### SetInputs

Set multiple model inputs at once.

```csharp
public void SetInputs(NamedTensors namedTensors)
```

---

### AppendTokens

Append token IDs to the active sequence.

```csharp
public void AppendTokens(ReadOnlySpan<int> inputIDs)
```

---

### AppendTokenSequences

Append pre-built sequences.

```csharp
public void AppendTokenSequences(Sequences sequences)
```

---

### GenerateNextToken

Generate the next token.

```csharp
public void GenerateNextToken()
```

---

### RewindTo

Rewind the generator to a specified length.

```csharp
public void RewindTo(ulong newLength)
```

---

### GetNextTokens

Return the tokens generated in the most recent step.

```csharp
public ReadOnlySpan<int> GetNextTokens()
```

---

### GetSequence

Return the generated sequence at the given index.

```csharp
public ReadOnlySpan<int> GetSequence(ulong index)
```

---

### GetInput

Retrieve an input tensor by name.

```csharp
public Tensor GetInput(string inputName)
```

---

### GetOutput

Retrieve an output tensor by name.

```csharp
public Tensor GetOutput(string outputName)
```

---

### SetActiveAdapter

Set the active adapter on this generator instance.

```csharp
public void SetActiveAdapter(Adapters adapters, string adapterName)
```

---

## OnnxRuntimeGenAIException class

### Overview

Exception type thrown when GenAI operations fail.

```csharp
public class OnnxRuntimeGenAIException : Exception
```

---

## Sequences class

### NumSequences

Get the number of sequences.

```csharp
public ulong NumSequences { get; }
```

---

### Append

Append a token to the specified sequence.

```csharp
public void Append(int token, ulong sequenceIndex)
```

---

### Indexer

Get the sequence at the specified index.

```csharp
public ReadOnlySpan<int> this[ulong sequenceIndex]
```

---

## Tensor class

### Constructor

Initialize a tensor from a buffer.

```csharp
public Tensor(IntPtr data, long[] shape, ElementType elementType)
```

---

### Shape

Get the shape of the tensor.

```csharp
public long[] Shape()
```

---

### Type

Get the element type of the tensor.

```csharp
public ElementType Type()
```

---

### ElementsFromShape

Compute the total element count for a shape.

```csharp
public static long ElementsFromShape(long[] shape)
```

---

### NumElements

Return the number of elements in the tensor.

```csharp
public long NumElements()
```

---

### GetData

Return a read-only span over the tensor data.

```csharp
public ReadOnlySpan<T> GetData<T>()
```

---

## NamedTensors class

### Overview

Represent a disposable collection of named tensors returned by processors.

---

## Utils class

### SetLogBool

Set a boolean logging option.

```csharp
public static void SetLogBool(string name, bool value)
```

---

### SetLogString

Set a string logging option.

```csharp
public static void SetLogString(string name, string value)
```

---

### SetCurrentGpuDeviceId

Set the current GPU device ID.

```csharp
public static void SetCurrentGpuDeviceId(int deviceId)
```

---

### GetCurrentGpuDeviceId

Get the current GPU device ID.

```csharp
public static int GetCurrentGpuDeviceId()
```

---

## OgaHandle class

### Overview

Provide a disposable handle that triggers GenAI shutdown when disposed.

```csharp
public class OgaHandle : IDisposable
```

---

## Images class

### Load (paths)

Load images from file paths.

```csharp
public static Images Load(string[] imagePaths)
```

---

### Load (bytes)

Load images from in-memory data.

```csharp
public static Images Load(byte[] imageBytesData)
```

---

## Audios class

### Load (paths)

Load audio from file paths.

```csharp
public static Audios Load(string[] audioPaths)
```

---

### Load (bytes)

Load audio from in-memory data.

```csharp
public static Audios Load(byte[] audioBytesData)
```

---

## MultiModalProcessor class

### Constructor

Initialize a processor for multimodal inputs.

```csharp
public MultiModalProcessor(Model model)
```

---

### ProcessImages

Process text and images into named tensors.

```csharp
public NamedTensors ProcessImages(string prompt, Images images)
public NamedTensors ProcessImages(string[] prompts, Images images)
```

---

### ProcessAudios

Process text and audio into named tensors.

```csharp
public NamedTensors ProcessAudios(string prompt, Audios audios)
public NamedTensors ProcessAudios(string[] prompts, Audios audios)
```

---

### ProcessImagesAndAudios

Process text with both images and audio.

```csharp
public NamedTensors ProcessImagesAndAudios(string prompt, Images images, Audios audios)
public NamedTensors ProcessImagesAndAudios(string[] prompts, Images images, Audios audios)
```

---

### Decode

Decode token IDs to text.

```csharp
public string Decode(ReadOnlySpan<int> sequence)
```

---

### CreateStream

Create a tokenizer stream for multimodal decoding.

```csharp
public TokenizerStream CreateStream()
```

---

## Adapters class

### Constructor

Create an adapter container for a model.

```csharp
public Adapters(Model model)
```

---

### LoadAdapter

Load an adapter file into the container.

```csharp
public void LoadAdapter(string adapterPath, string adapterName)
```

---

### UnloadAdapter

Unload a previously loaded adapter.

```csharp
public void UnloadAdapter(string adapterName)
```

---

## OnnxRuntimeGenAIChatClientOptions class

### Properties

```csharp
public IList<string>? StopSequences { get; set; }
public Func<IEnumerable<ChatMessage>, ChatOptions?, string>? PromptFormatter { get; set; }
public bool EnableCaching { get; set; }
```

---

## OnnxRuntimeGenAIChatClient class

### Constructors

Create a chat client from a model path, model, or config.

```csharp
public OnnxRuntimeGenAIChatClient(string modelPath, OnnxRuntimeGenAIChatClientOptions? options = null)
public OnnxRuntimeGenAIChatClient(Model model, bool ownsModel = true, OnnxRuntimeGenAIChatClientOptions? options = null)
public OnnxRuntimeGenAIChatClient(Config config, bool ownsConfig = true, OnnxRuntimeGenAIChatClientOptions? options = null)
```

---

### GetResponseAsync

Generate a complete chat response.

```csharp
public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
```

---

### GetStreamingResponseAsync

Stream chat response updates.

```csharp
public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
```

---