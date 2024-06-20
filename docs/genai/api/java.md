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
//add later
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

### createTokenizer

Creates a Tokenizer instance for this model. The model contains the configuration information that determines the tokenizer to use.

```java
public Tokenizer createTokenizer()
                          throws GenAIException
```

#### Returns

`onnxruntime_genai.Model`

### Generate method

```java
public Sequences generate(GeneratorParams generatorParams)
                   throws GenAIException
```

#### Parameters
- `generatorParams`: the generator parameters.

#### Returns

The generated sequences.


## Tokenizer class

### Create tokenizer object

```java
onnxruntime_genai.Model.Tokenizer(model: onnxruntime_genai.Model) -> onnxruntime_genai.Tokenizer
```

#### Parameters

- `model`: (Required) The model that was loaded by the `Model()`

#### Returns

- `Tokenizer`: The tokenizer object

### Encode

```java
onnxruntime_genai.Tokenizer.encode(text: str) -> numpy.ndarray[numpy.int32]
```

#### Parameters

- `text`: (Required)

#### Returns

`numpy.ndarray[numpy.int32]`: an array of tokens representing the prompt

### Decode

```java
onnxruntime_genai.Tokenizer.decode(tokens: numpy.ndarry[int]) -> str 
```

#### Parameters

- `numpy.ndarray[numpy.int32]`: (Required) a sequence of generated tokens


#### Returns

`str`: the decoded generated tokens


### Encode batch

```java
onnxruntime_genai.Tokenizer.encode_batch(texts: list[str]) -> numpy.ndarray[int, int]
```

#### Parameters

- `texts`: A list of inputs

#### Returns

`numpy.ndarray[int, int]`: The batch of tokenized strings

### Decode batch

```java
onnxruntime_genai.Tokenize.decode_batch(tokens: [[numpy.int32]]) -> list[str]
```

#### Parameters

- tokens

#### Returns

`texts`: a batch of decoded text


### Create tokenizer decoding stream


```java
onnxruntime_genai.Tokenizer.create_stream() -> TokenizerStream
```

#### Parameters

None

#### Returns

`onnxruntime_genai.TokenizerStream` The tokenizer stream object

## TokenizerStream class

This class accumulates the next displayable string (according to the tokenizer's vocabulary).

### Decode method

 
```java
onnxruntime_genai.TokenizerStream.decode(token: int32) -> str
```
  
#### Parameters

- `token`: (Required) A token to decode

#### Returns

`str`: If a displayable string has accumulated, this method returns it. If not, this method returns the empty string.

## GeneratorParams class

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

### setSearchOption

```java
public void setSearchOption(String optionName, boolean value)
                     throws GenAIException
```

### setInput

Sets the prompt/s for model execution. The `sequences` are created by using Tokenizer.Encode or EncodeBatch.

#### Parameters
- `sequences`: Sequences containing the encoded prompt.

#### Throws
GenAIException - If the call to the GenAI native API fails.


```java
public void setInput(Sequences sequences)
              throws GenAIException
```

### setInput

Sets the prompt/s token ids for model execution. The 'tokenIds' are the encoded parameters.

#### Parameters

- `tokenIds`: The token ids of the encoded prompt/s
- `sequenceLength`: The length of each sequence.
- `batchSize`: Size of the batch. 

#### Throws

GenAIException - If the call to the GenAI native API fails. (Note: all sequences in the batch must be the same length)

```java
public void setInput(int[] tokenIds, int sequenceLength, int batchSize)
              throws GenAIException
```


## Generator class

### Create a Generator

```java
Generator(Model model, GeneratorParams generatorParams)
```

#### Parameters

- `model`: (Required) The model to use for generation
- `params`: (Required) The set of parameters that control the generation

#### Returns

`onnxruntime_genai.Generator` The Generator object


### Is generation done

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

### Get output

Returns an output of the model.

```java
onnxruntime_genai.Generator.get_output(str: name) -> numpy.ndarray
```

#### Parameters
- `name`: the name of the model output

#### Returns
- `numpy.ndarray`: a multi dimensional array of the model outputs. The shape of the array is shape of the output.

#### Example

The following code returns the output logits of a model.

```java
logits = generator.get_output("logits")
```


### Generate next token

Using the current set of logits and the specified generator parameters, calculates the next batch of tokens, using Top P sampling.

```java
onnxruntime_genai.Generator.generate_next_token()
```

### Get next tokens

```java
onnxruntime_genai.Generator.get_next_tokens() -> numpy.ndarray[numpy.int32]
```

Returns

`numpy.ndarray[numpy.int32]`: The most recently generated tokens

### Get sequence

```java
onnxruntime_genai.Generator.get_sequence(index: int) -> numpy.ndarray[numpy.int32] 
```

- `index`: (Required) The index of the sequence in the batch to return