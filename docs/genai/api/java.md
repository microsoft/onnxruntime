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

## Install and import
The Java API is delivered by the ai.onnxruntime.genai Java package. Package publication is pending. To build the package from source, see the [build from source guide](../howto/build-from-source.md).

```java
import ai.onnxruntime.genai.*;
```

## SimpleGenAI Class

The `SimpleGenAI` class provides a simple usage example of the GenAI API. It works with a model that generates text based on a prompt, processing a single prompt at a time.
Usage:

Create an instance of the class with the path to the model. The path should also contain the GenAI configuration files.

```java
SimpleGenAI genAI = new SimpleGenAI(folderPath);
```

Call createGeneratorParams with the prompt text.
Set any other search options via the GeneratorParams object as needed using `setSearchOption`.

```java
GeneratorParams generatorParams = genAI.createGeneratorParams(promptText);
// .. set additional generator params before calling generate()
```

Call generate with the GeneratorParams object and an optional listener.

```java
String fullResponse = genAI.generate(generatorParams, listener);
```

The listener is used as a callback mechanism so that tokens can be used as they are generated. Create a class that implements the `Consumer<String>` interface and provide an instance of that class as the `listener` argument.

### Constructor

```java
public SimpleGenAI(String modelPath) throws GenAIException
```

#### Throws

`GenAIException`- on failure.

### Generate Method

Generate text based on the prompt and settings in GeneratorParams. 

NOTE: This only handles a single sequence of input (i.e. a single prompt which equates to batch size of 1).

```java
public String generate(GeneratorParams generatorParams, Consumer<String> listener) throws GenAIException
```

#### Parameters

- `generatorParams`: the prompt and settings to run the model with.
- `listener`: optional callback for tokens to be provided as they are generated. 

NOTE: Token generation will be blocked until the listener's `accept` method returns.

#### Throws

`GenAIException`- on failure.

#### Returns

The generated text.

#### Example

```java
SimpleGenAI generator = new SimpleGenAI(modelPath);
GeneratorParams params = generator.createGeneratorParams("What's 6 times 7?");
Consumer<String> listener = token -> logger.info("onTokenGenerate: " + token);
String result = generator.generate(params, listener);

logger.info("Result: " + result);
```

### createGenerateParams Method

Create the generator parameters and add the prompt text. The user can set other search options via the GeneratorParams object prior to running `generate`.

```java
public GeneratorParams createGeneratorParams(String prompt) throws GenAIException
```

#### Parameters

- `prompt`: the prompt text to encode.

#### Throws

`GenAIException`- on failure.

#### Returns

The generator parameters.

## Exception Class

An exception which contains the error message and code produced by the native layer.

### Constructor

```java
public GenAIException(String message)
```

#### Example

```java
catch (GenAIException e) {
  throw new GenAIException("Token generation loop failed.", e);
}
```

## Model class

### Constructor

```java
Model(String modelPath)
```

### Create Tokenizer Method

Creates a Tokenizer instance for this model. The model contains the configuration information that determines the tokenizer to use.

```java
public Tokenizer createTokenizer() throws GenAIException
```

#### Throws

`GenAIException`- if the call to the GenAI native API fails

#### Returns

The tokenizer instance.

### Generate Method

```java
public Sequences generate(GeneratorParams generatorParams) throws GenAIException
```

#### Parameters

- `generatorParams`: the generator parameters.

#### Throws

`GenAIException`- if the call to the GenAI native API fails.

#### Returns

The generated sequences.

#### Example

```java
Sequences output = model.generate(generatorParams);
```

### createGeneratorParams Method

Creates a GeneratorParams instance for executing the model. 

NOTE: GeneratorParams internally uses the Model, so the Model instance must remain valid.

```java
public GeneratorParams createGeneratorParams() throws GenAIException
```

#### Throws

`GenAIException`- if the call to the GenAI native API fails.

#### Returns

The GeneratorParams instance.

#### Example

```java
GeneratorParams params = generator.createGeneratorParams("What's 6 times 7?");
```

## Tokenizer class

### Encode Method

Encodes a string into a sequence of token ids.

```java
public Sequences encode(String string) throws GenAIException
```

#### Parameters

- `string`: text to encode as token ids.

#### Throws

`GenAIException`- if the call to the GenAI native API fails.

#### Returns

A Sequences object with a single sequence in it.

#### Example

```java
Sequences encodedPrompt = tokenizer.encode(prompt);
```

### Decode Method

Decodes a sequence of token ids into text.

```java
public String decode(int[] sequence) throws GenAIException
```

#### Parameters

- `sequence`: collection of token ids to decode to text

#### Throws

`GenAIException`- if the call to the GenAI native API fails.

#### Returns

The text representation of the sequence.

#### Example

```java
String result = tokenizer.decode(output_ids);
```

### encodeBatch Method

Encodes an array of strings into a sequence of token ids for each input.

```java
public Sequences encodeBatch(String[] strings) throws GenAIException
```

#### Parameters

- `strings`: collection of strings to encode as token ids.

#### Throws

`GenAIException`- if the call to the GenAI native API fails.

#### Returns

A Sequences object with one sequence per input string.

#### Example

```java
Sequences encoded = tokenizer.encodeBatch(inputs);
```

### decodeBatch Method

Decodes a batch of sequences of token ids into text.

```java
public String[] decodeBatch(Sequences sequences) throws GenAIException
```

#### Parameters

- `sequences`: a Sequences object with one or more sequences of token ids.

#### Throws

`GenAIException`- if the call to the GenAI native API fails.

#### Returns

An array of strings with the text representation of each sequence.

#### Example

```java
String[] decoded = tokenizer.decodeBatch(encoded);
```

### createStream Method

Creates a TokenizerStream object for streaming tokenization. This is used with Generator class to provide each token as it is generated.

```java
public TokenizerStream createStream() throws GenAIException
```

#### Throws

`GenAIException`- if the call to the GenAI native API fails.

#### Returns

The new TokenizerStream instance.

## TokenizerStream class

This class is used to convert individual tokens when using Generator.generateNextToken.

### Decode method

```java
public String decode(int token) throws GenAIException
```

#### Parameters

- `token`: int value for token

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

#### Example

Create a 2x2 Tensor with 32-bit float data.

```java
long[] shape = {2, 2};
ByteBuffer data = ByteBuffer.allocateDirect(4 * Float.BYTES);
FloatBuffer floatBuffer = data.asFloatBuffer();
floatBuffer.put(new float[] {1.0f, 2.0f, 3.0f, 4.0f});

Tensor tensor = new Tensor(data, shape, Tensor.ElementType.float32);
```

## GeneratorParams class

The `GeneratorParams` class represents the parameters used for generating sequences with a model. Set the prompt using setInput, and any other search options using setSearchOption.

### Create a Generator Params object

```java
GeneratorParams params = new GeneratorParams(model);
```

### setSearchOption Method

```java
public void setSearchOption(String optionName, double value) throws GenAIException
```

#### Throws

`GenAIException`

#### Example

Set search option to limit the model generation length.

```java
generatorParams.setSearchOption("max_length", 10);
```

### setSearchOption Method

```java
public void setSearchOption(String optionName, boolean value) throws GenAIException
```

#### Throws

`GenAIException`

#### Example

```java
generatorParams.setSearchOption("early_stopping", true);
```

### setInput Method

Sets the prompt/s for model execution. The `sequences` are created by using Tokenizer.Encode or EncodeBatch.

```java
public void setInput(Sequences sequences) throws GenAIException
```

#### Parameters
- `sequences`: sequences containing the encoded prompt.

#### Throws
`GenAIException`- if the call to the GenAI native API fails.

#### Example
```java
generatorParams.setInput(encodedPrompt);
```

### setInput Method

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

#### Example

```java
generatorParams.setInput(tokenIds, sequenceLength, batchSize);
```

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

### isDone Method

Checks if the generation process is done.

```java
public boolean isDone()
```

#### Returns

Returns true if the generation process is done, false otherwise.

### computeLogits Method

Computes the logits for the next token in the sequence.

```java
public void computeLogits() throws GenAIException
```

#### Throws

`GenAIException`- if the call to the GenAI native API fails.

### getSequence Method

Retrieves a sequence of token ids for the specified sequence index.

```java
public int[] getSequence(long sequenceIndex) throws GenAIException
```

#### Parameters
- `sequenceIndex`: the index of the sequence.

#### Throws

`GenAIException`- if the call to the GenAI native API fails.

#### Returns

An array of integers with the sequence of token ids.

#### Exmaple
```java
int[] outputIds = output.getSequence(i);
```

### generateNextToken Method

Generates the next token in the sequence.

```java
public void generateNextToken() throws GenAIException
```

#### Throws

`GenAIException`- if the call to the GenAI native API fails.

### getLastTokenInSequence Method

Retrieves the last token in the sequence for the specified sequence index.

```java
public int getLastTokenInSequence(long sequenceIndex) throws GenAIException
```

#### Parameters

- `sequenceIndex`: the index of the sequence.

#### Throws

`GenAIException`- if the call to the GenAI native API fails.

#### Returns

The last token in the sequence.

## Sequences Class

Represents a collection of encoded prompts/responses.

### numSequences Method

Gets the number of sequences in the collection. This is equivalent to the batch size.

```java
public long numSequences()
```

### Returns

The number of sequences.

#### Example
```java
int numSequences = (int) sequences.numSequences();
```

### getSequence Method

Gets the sequence at the specified index.

```java
public int[] getSequence(long sequenceIndex)
```

#### Parameters

- `sequenceIndex`: The index of the sequence.

#### Returns

The sequence as an array of integers.

