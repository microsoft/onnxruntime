---
title: Java API
description: Java API reference for ONNX Runtime generate() API
has_children: false
parent: API docs
grand_parent: Generate API (Preview)
nav_order: 4
---

# ONNX Runtime generate() Java API

_Note: this API is in preview and is subject to change._

{: .no_toc }

* TOC placeholder
{:toc}

## Overview

This document describes the Java API for ONNX Runtime GenAI.  
Below are the main classes and methods, with code snippets and descriptions for each.

---

## Install and import

The Java API is delivered by the `ai.onnxruntime.genai` Java package. Package publication is pending. To build the package from source, see the [build from source guide](../howto/build-from-source.md).

```java
import ai.onnxruntime.genai.*;
```

---

## Model class

### Constructor (model path)

Initializes a new model from the given model path.

```java
public Model(String modelPath) throws GenAIException
```

---

### Constructor (config)

Initializes a new model using a pre-built configuration.

```java
public Model(Config config) throws GenAIException
```

---

### close

Releases native resources owned by the model.

```java
public void close()
```

---

## Config class

### Constructor

Initializes a new configuration object from a config path.

```java
public Config(String configPath) throws GenAIException
```

---

### clearProviders

Clears all providers from the configuration.

```java
public void clearProviders()
```

---

### appendProvider

Appends a provider to the configuration.

```java
public void appendProvider(String providerName)
```

---

### setProviderOption

Sets a provider option in the configuration.

```java
public void setProviderOption(String providerName, String optionKey, String optionValue)
```

---

### close

Releases native resources owned by the configuration.

```java
public void close()
```

---

## Tokenizer class

### Constructor

Initializes a tokenizer for the given model.

```java
public Tokenizer(Model model) throws GenAIException
```

---

### encode

Encodes a string into a sequence of token ids.

```java
public Sequences encode(String string) throws GenAIException
```

---

### encodeBatch

Encodes an array of strings into a sequence of token ids for each input.

```java
public Sequences encodeBatch(String[] strings) throws GenAIException
```

---

### decode

Decodes a sequence of token ids into text.

```java
public String decode(int[] sequence) throws GenAIException
```

---

### decodeBatch

Decodes a batch of sequences of token ids into text.

```java
public String[] decodeBatch(Sequences sequences) throws GenAIException
```

---

### getBosTokenId

Gets the beginning-of-sentence token id.

```java
public int getBosTokenId() throws GenAIException
```

---

### getPadTokenId

Gets the padding token id.

```java
public int getPadTokenId() throws GenAIException
```

---

### getEosTokenIds

Gets the end-of-sentence token ids.

```java
public int[] getEosTokenIds() throws GenAIException
```

---

### toTokenId

Converts a string to its token id.

```java
public int toTokenId(String str) throws GenAIException
```

---

### applyChatTemplate

Applies a chat template to format messages and tools.

```java
public String applyChatTemplate(
    String templateStr, String messages, String tools, boolean addGenerationPrompt)
    throws GenAIException
```

---

### updateOptions

Updates tokenizer options via key/value pairs.

```java
public void updateOptions(java.util.Map<String, String> options) throws GenAIException
```

---

### createStream

Creates a TokenizerStream object for streaming tokenization.

```java
public TokenizerStream createStream() throws GenAIException
```

---

### close

Releases native resources owned by the tokenizer.

```java
public void close()
```

---

## TokenizerStream class

### decode

Decodes a single token in the stream and returns the generated string chunk.

```java
public String decode(int token) throws GenAIException
```

---

### close

Releases native resources owned by the tokenizer stream.

```java
public void close()
```

---

## GeneratorParams class

### Constructor

Initializes generator parameters for the given model.

```java
public GeneratorParams(Model model) throws GenAIException
```

---

### setSearchOption (double)

Sets a numeric search option.

```java
public void setSearchOption(String optionName, double value) throws GenAIException
```

---

### setSearchOption (boolean)

Sets a boolean search option.

```java
public void setSearchOption(String optionName, boolean value) throws GenAIException
```

---

### close

Releases native resources owned by the generator parameters.

```java
public void close()
```

---

## Generator class

### Constructor

Constructs a Generator object with the given model and generator parameters.

```java
public Generator(Model model, GeneratorParams generatorParams) throws GenAIException
```

---

### iterator

Generates a token on each call to `next()` by calling `generateNextToken` internally.

```java
public java.util.Iterator<Integer> iterator()
```

---

### isDone

Checks if the generation process is done.

```java
public boolean isDone()
```

---

### setModelInput

Adds a tensor as a named model input.

```java
public void setModelInput(String name, Tensor tensor) throws GenAIException
```

---

### setInputs

Adds a batch of named tensors as model inputs.

```java
public void setInputs(NamedTensors namedTensors) throws GenAIException
```

---

### appendTokens

Appends token ids to the generator input.

```java
public void appendTokens(int[] inputIDs) throws GenAIException
```

---

### appendTokenSequences

Appends token sequences to the generator input.

```java
public void appendTokenSequences(Sequences sequences) throws GenAIException
```

---

### rewindTo

Rewinds the generator to a specific token length before continuing generation.

```java
public void rewindTo(long newLength) throws GenAIException
```

---

### generateNextToken

Generates the next token in the sequence using cached logits/state.

```java
public void generateNextToken() throws GenAIException
```

---

### getSequence

Retrieves a sequence of token ids for the specified sequence index.

```java
public int[] getSequence(long sequenceIndex) throws GenAIException
```

---

### getLastTokenInSequence

Retrieves the last token in the sequence for the specified sequence index.

```java
public int getLastTokenInSequence(long sequenceIndex) throws GenAIException
```

---

### getInput

Returns a copy of the named model input as a tensor.

```java
public Tensor getInput(String name) throws GenAIException
```

---

### getOutput

Returns a copy of the named model output as a tensor.

```java
public Tensor getOutput(String name) throws GenAIException
```

---

### setActiveAdapter

Activates a previously loaded adapter by name.

```java
public void setActiveAdapter(Adapters adapters, String adapterName) throws GenAIException
```

---

### close

Releases native resources owned by the generator.

```java
public void close()
```

---

## Sequences class

### numSequences

Gets the number of sequences in the collection.

```java
public long numSequences()
```

---

### getSequence

Gets the sequence at the specified index.

```java
public int[] getSequence(long sequenceIndex)
```

---

### close

Releases native resources owned by the sequences.

```java
public void close()
```

---

## Tensor class

### Constructor

Constructs a Tensor with the given data, shape, and element type.

```java
public Tensor(ByteBuffer data, long[] shape, ElementType elementType) throws GenAIException
```

---

### getType

Gets the tensor element type.

```java
public Tensor.ElementType getType()
```

---

### getShape

Gets the tensor shape.

```java
public long[] getShape()
```

---

### close

Releases native resources owned by the tensor.

```java
public void close()
```

---

## Images class

### Constructor

Loads images from the given path.

```java
public Images(String imagePath) throws GenAIException
```

---

### close

Releases native resources owned by the images.

```java
public void close()
```

---

## Audios class

### Constructor

Loads audio from the given path.

```java
public Audios(String audioPath) throws GenAIException
```

---

### close

Releases native resources owned by the audios.

```java
public void close()
```

---

## MultiModalProcessor class

### Constructor

Creates a processor for a given model.

```java
public MultiModalProcessor(Model model) throws GenAIException
```

---

### processImages (single prompt)

Processes a text prompt and images into named tensors.

```java
public NamedTensors processImages(String prompt, Images images) throws GenAIException
```

---

### processImages (batch prompts)

Processes batch prompts and images into named tensors.

```java
public NamedTensors processImages(String[] prompts, Images images) throws GenAIException
```

---

### processAudios (single prompt)

Processes a text prompt and audios into named tensors.

```java
public NamedTensors processAudios(String prompt, Audios audios) throws GenAIException
```

---

### processAudios (batch prompts)

Processes batch prompts and audios into named tensors.

```java
public NamedTensors processAudios(String[] prompts, Audios audios) throws GenAIException
```

---

### processImagesAndAudios (single prompt)

Processes a text prompt with images and audios into named tensors.

```java
public NamedTensors processImagesAndAudios(String prompt, Images images, Audios audios)
    throws GenAIException
```

---

### processImagesAndAudios (batch prompts)

Processes batch prompts with images and audios into named tensors.

```java
public NamedTensors processImagesAndAudios(String[] prompts, Images images, Audios audios)
    throws GenAIException
```

---

### decode

Decodes a token sequence produced by the processor back to text.

```java
public String decode(int[] sequence) throws GenAIException
```

---

### createStream

Creates a TokenizerStream tied to this processor for streaming tokenization.

```java
public TokenizerStream createStream() throws GenAIException
```

---

### close

Releases native resources owned by the processor.

```java
public void close()
```

---

## NamedTensors class

### Constructor

Wraps a native handle containing a named tensor collection.

```java
public NamedTensors(long handle)
```

---

### close

Releases native resources owned by the named tensors.

```java
public void close()
```

---

## Adapters class

### Constructor

Creates an adapter container bound to a model.

```java
public Adapters(Model model) throws GenAIException
```

---

### loadAdapter

Loads an adapter from disk and registers it under a name.

```java
public void loadAdapter(String adapterFilePath, String adapterName) throws GenAIException
```

---

### unloadAdapter

Unloads a previously loaded adapter.

```java
public void unloadAdapter(String adapterName) throws GenAIException
```

---

### close

Releases native resources owned by the adapters container.

```java
public void close()
```

---

## GenAIException class

### Constructors

Exceptions propagated from the native layer.

```java
GenAIException(String message)
GenAIException(String message, Exception innerException)
```

---