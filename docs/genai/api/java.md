---
title: Java API
description: Java API reference for ONNX Runtime generate() API
has_children: false
parent: API docs
grand_parent: Generate API (Preview)
nav_order: 1
---

# ONNX Runtime generate() Java API

_Note: this API is in preview and is subject to change._

{: .no_toc }

* TOC placeholder
{:toc}

## Install and import
//ADD LATER
The Java API is delivered by the onnxruntime-genai Java package.

```bash
//install onnxruntime-genai
```

```java
import onnxruntime_genai
```

## Exception Class

An exception which contains the error message and code produced by the native layer.

### Constructor

```java
public GenAIException(String message)
```

## SimpleGenAI Class

The `SimpleGenAI` class provides a simple usage example of the GenAI API. It works with a model that generates text based on a prompt, processing a single prompt at a time.
Usage:

Create an instance of the class with the path to the model. The path should also contain the GenAI configuration files.
Call createGeneratorParams with the prompt text.
Set any other search options via the GeneratorParams object as needed using `setSearchOption`.
Call generate with the GeneratorParams object and an optional listener.
The listener is used as a callback mechanism so that tokens can be used as they are generated. Create a class that implements the TokenUpdateListener interface and provide an instance of that class as the `listener` argument.

### Constructor

```java
public SimpleGenAI(String modelPath)
            throws GenAIException
```

#### Throws

`GenAIException`

### Generate

Generate text based on the prompt and settings in GeneratorParams. 

NOTE: This only handles a single sequence of input (i.e. a single prompt which equates to batch size of 1).

```java
public String generate(GeneratorParams generatorParams,
 Consumer<String> listener)
                throws GenAIException
```

#### Parameters

- `generatorParams`: the prompt and settings to run the model with.
- `listener`: optional callback for tokens to be provided as they are generated. 

NOTE: Token generation will be blocked until the listener's `accept` method returns.

#### Throws

`GenAIException`- on failure.

#### Returns

The generated text.

## GenAI Class

Description

### Initialize OS_ARCH_STR Method

Computes and initializes OS_ARCH_STR (such as linux-x64)

```java
private static String initOsArch()
```

### Check Android Method

Check if we're running on Android.

```java
static boolean isAndroid()
```

#### Returns

Returns True if the property java.vendor equals The Android Project, false otherwise.

### Cleanup Method

Marks the file for delete on exit.

```java
private static void cleanUp(File file)
```

#### Parameters

- `file`: the file to remove.

### Load Method

Load a shared library by name.

NOTE: If the library path is not specified via a system property then it attempts to extract the library from the classpath before loading it.

```java
private static void load(String library) throws IOException
```

#### Parameters

- `library`: the bare name of the library.

#### Throws

`IOException`- If the fild failed to read or write.

### Extract Method

Extracts the library from the classpath resources. Returns optional.empty if it failed to extract or couldn't be found.

```java
private static Optional<File> extractFromResources(String library)
```

#### Parameters

- `library`: the library name.

#### Returns

An optional containing the file if it is successfully extracted, or an empty optional if it failed to extract or couldn't be found.

### Map Library Method

Maps the library name into a platform dependent library filename. Converts macOS's "jnilib" to "dylib" but otherwise is the same as {@link System#mapLibraryName(String)}.

```java
private static String mapLibraryName(String library)
```

#### Parameters

- `library`: the library name.

#### Returns

The library filename.


## Model class

### Constructor

```java
Model(String modelPath)
```

### Create Tokenizer Method

Creates a Tokenizer instance for this model. The model contains the configuration information that determines the tokenizer to use.

```java
public Tokenizer createTokenizer()
                          throws GenAIException
```

#### Throws

`GenAIException`- if the call to the GenAI native API fails

#### Returns

`onnxruntime_genai.Model`

### Generate Method

```java
public Sequences generate(GeneratorParams generatorParams)
                   throws GenAIException
```

#### Parameters

- `generatorParams`: the generator parameters.

#### Throws

`GenAIException`- if the call to the GenAI native API fails.

#### Returns

The generated sequences.

### Generate Parameters Method

Creates a GeneratorParams instance for executing the model. 

NOTE: GeneratorParams internally uses the Model, so the Model instance must remain valid.

```java
public GeneratorParams createGeneratorParams()
                                      throws GenAIException
```

#### Throws

`GenAIException`- if the call to the GenAI native API fails.

#### Returns

The GeneratorParams instance.


## Tokenizer class

### Encode

Encodes a string into a sequence of token ids.

```java
public Sequences encode(String string)
                 throws GenAIException
```

#### Parameters

- `string`: text to encode as token ids.

#### Throws

`GenAIException`- if the call to the GenAI native API fails.

#### Returns

A Sequences object with a single sequence in it.


### Decode

Decodes a sequence of token ids into text.

```java
public String decode(int[] sequence)
              throws GenAIException
```

#### Parameters

- `sequence`: collection of token ids to decode to text

#### Throws

`GenAIException`- if the call to the GenAI native API fails.

#### Returns

The text representation of the sequence.


### Encode batch

Encodes an array of strings into a sequence of token ids for each input.

```java
public Sequences encodeBatch(String[] strings)
                      throws GenAIException
```

#### Parameters

- `strings`: collection of strings to encode as token ids.

#### Throws

`GenAIException`- if the call to the GenAI native API fails.

#### Returns

A Sequences object with one sequence per input string.

### Decode batch

Decodes a batch of sequences of token ids into text.

```java
public String[] decodeBatch(Sequences sequences)
                     throws GenAIException
```

#### Parameters

- `sequences`: a Sequences object with one or more sequences of token ids.

#### Throws

`GenAIException`- if the call to the GenAI native API fails.

#### Returns

An array of strings with the text representation of each sequence.


### Create tokenizer decoding stream

Creates a TokenizerStream object for streaming tokenization. This is used with Generator class to provide each token as it is generated.

```java
public TokenizerStream createStream()
                             throws GenAIException
```

#### Parameters

None

#### Throws

`GenAIException`- if the call to the GenAI native API fails.

#### Returns

The new TokenizerStream instance.


## TokenizerStream class

This class is used to convert individual tokens when using Generator.generateNextToken.

### Decode method

 
```java
public String decode(int token)
              throws GenAIException
```

#### Throws

`GenAIException`

## Tensor Class

Constructs a Tensor with the given data, shape, and element type.

```java
public Tensor(ByteBuffer data, long[] shape, ElementType elementType) throws GenAIException
```

#### Parameters

- `data`: the data for the Tensor. Must be a direct ByteBuffer.
- `shape`: the shape of the Tensor.
- `elementType`: the Type of elements in the Tensor.

#### Throws

`GenAIException`

## GeneratorParams class

The `GeneratorParams` class represents the parameters used for generating sequences with a model. Set the prompt using setInput, and any other search options using setSearchOption.

### Create a Generator Params object

```java
//find
```

### setSearchOption

```java
public void setSearchOption(String optionName,
 double value)
                     throws GenAIException
```

#### Throws

`GenAIException`

### setSearchOption

```java
public void setSearchOption(String optionName, boolean value)
                     throws GenAIException
```

#### Throws

`GenAIException`

### setInput

Sets the prompt/s for model execution. The `sequences` are created by using Tokenizer.Encode or EncodeBatch.

```java
public void setInput(Sequences sequences)
              throws GenAIException
```

#### Parameters
- `sequences`: sequences containing the encoded prompt.

#### Throws
`GenAIException`- if the call to the GenAI native API fails.

### setInput

Sets the prompt/s token ids for model execution. The `tokenIds` are the encoded parameters.

```java
public void setInput(int[] tokenIds, int sequenceLength, int batchSize)
              throws GenAIException
```

#### Parameters

- `tokenIds`: the token ids of the encoded prompt/s
- `sequenceLength`: the length of each sequence.
- `batchSize`: size of the batch. 

#### Throws

`GenAIException`- if the call to the GenAI native API fails. 

NOTE: all sequences in the batch must be the same length.

## Generator class

The Generator class generates output using a model and generator parameters.
The expected usage is to loop until isDone returns false. Within the loop, call computeLogits followed by generateNextToken.

The newly generated token can be retrieved with getLastTokenInSequence and decoded with TokenizerStream.Decode.

After the generation process is done, GetSequence can be used to retrieve the complete generated sequence if needed.

### Create a Generator

Constructs a Generator object with the given model and generator parameters.

```java
Generator(Model model, GeneratorParams generatorParams)
```

#### Parameters

- `model`: the model.
- `params`: the generator parameters.

#### Throws

`GenAIException`- if the call to the GenAI native API fails.

### Is generation done

Checks if the generation process is done.

```java
public boolean isDone()
```

#### Returns

Returns true if the generation process is done, false otherwise.

### Compute logits

Computes the logits for the next token in the sequence.

```java
public void computeLogits()
                   throws GenAIException
```

#### Throws

`GenAIException`- if the call to the GenAI native API fails.

### Get sequence

Retrieves a sequence of token ids for the specified sequence index.

```java
public int[] getSequence(long sequenceIndex)
                  throws GenAIException
```

#### Parameters
- `sequenceIndex`: the index of the sequence.

#### Throws

`GenAIException`- if the call to the GenAI native API fails.

#### Returns

An array of integers with the sequence of token ids.

### Generate next token

Generates the next token in the sequence.

```java
public void generateNextToken()
                       throws GenAIException
```

#### Throws

`GenAIException`- if the call to the GenAI native API fails.

### Get last token in sequence

Retrieves the last token in the sequence for the specified sequence index.

```java
public int getLastTokenInSequence(long sequenceIndex)
                           throws GenAIException
```

#### Parameters

- `sequenceIndex`: the index of the sequence.

#### Throws

`GenAIException`- if the call to the GenAI native API fails.

#### Returns

The last token in the sequence.

## Sequence Class

Represents a collection of encoded prompts/responses.

### numSequences Method

Gets the number of sequences in the collection. This is equivalent to the batch size.

```java
public long numSequences()
```

### Returns

The number of sequences.

