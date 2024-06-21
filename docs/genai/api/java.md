---
title: Java API
description: Java API reference for ONNX Runtime generate() API
has_children: false
parent: API docs
grand_parent: Generate API (Preview)
nav_order: 1
---

# Java API

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

`GenAIException` - If the call to the GenAI native API fails

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

`GenAIException` - If the call to the GenAI native API fails.

#### Returns

The generated sequences.

### Generate Parameters Method

Creates a GeneratorParams instance for executing the model. NOTE: GeneratorParams internally uses the Model, so the Model instance must remain valid.

```java
public GeneratorParams createGeneratorParams()
                                      throws GenAIException
```

#### Throws

`GenAIException` - If the call to the GenAI native API fails.

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

- `string`: Text to encode as token ids.

#### Throws

`GenAIException` - If the call to the GenAI native API fails.

#### Returns

A Sequences object with a single sequence in it.


### Decode

Decodes a sequence of token ids into text.

```java
public String decode(int[] sequence)
              throws GenAIException
```

#### Parameters

- `sequence`: Collection of token ids to decode to text

#### Throws

`GenAIException`- If the call to the GenAI native API fails.

#### Returns

The text representation of the sequence.


### Encode batch

Encodes an array of strings into a sequence of token ids for each input.

```java
public Sequences encodeBatch(String[] strings)
                      throws GenAIException
```

#### Parameters

- `strings`: Collection of strings to encode as token ids.

#### Throws

`GenAIException`- If the call to the GenAI native API fails.

#### Returns

A Sequences object with one sequence per input string.

### Decode batch

Decodes a batch of sequences of token ids into text.

```java
public String[] decodeBatch(Sequences sequences)
                     throws GenAIException
```

#### Parameters

- `sequences`: A Sequences object with one or more sequences of token ids.

#### Throws

`GenAIException`- If the call to the GenAI native API fails.

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

`GenAIException`- If the call to the GenAI native API fails.

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
- `sequences`: Sequences containing the encoded prompt.

#### Throws
`GenAIException`- If the call to the GenAI native API fails.


### setInput

Sets the prompt/s token ids for model execution. The `tokenIds` are the encoded parameters.

```java
public void setInput(int[] tokenIds, int sequenceLength, int batchSize)
              throws GenAIException
```

#### Parameters

- `tokenIds`: The token ids of the encoded prompt/s
- `sequenceLength`: The length of each sequence.
- `batchSize`: Size of the batch. 

#### Throws

`GenAIException`- If the call to the GenAI native API fails. NOTE: all sequences in the batch must be the same length.


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

- `model`: The model.
- `params`: The generator parameters.

#### Throws

`GenAIException`- If the call to the GenAI native API fails.


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

`GenAIException`- If the call to the GenAI native API fails.


### Get sequence

Retrieves a sequence of token ids for the specified sequence index.

```java
public int[] getSequence(long sequenceIndex)
                  throws GenAIException
```

#### Parameters
- `sequenceIndex`: The index of the sequence.

#### Throws

`GenAIException`- If the call to the GenAI native API fails.

#### Returns

An array of integers with the sequence of token ids.


### Generate next token

Generates the next token in the sequence.

```java
public void generateNextToken()
                       throws GenAIException
```

#### Throws

`GenAIException`- If the call to the GenAI native API fails.

### Get last token in sequence

Retrieves the last token in the sequence for the specified sequence index.

```java
public int getLastTokenInSequence(long sequenceIndex)
                           throws GenAIException
```

#### Parameters

- `sequenceIndex`: The index of the sequence.

#### Throws

`GenAIException`- If the call to the GenAI native API fails.

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

Generate text based on the prompt and settings in GeneratorParams. NOTE: This only handles a single sequence of input (i.e. a single prompt which equates to batch size of 1).

```java
public String generate(GeneratorParams generatorParams,
 Consumer<String> listener)
                throws GenAIException
```

#### Parameters

- `generatorParams`: The prompt and settings to run the model with.
- `listener`: Optional callback for tokens to be provided as they are generated. NOTE: Token generation will be blocked until the listener's `accept` method returns.

#### Throws

`GenAIException`- On failure.

#### Returns

The generated text.
