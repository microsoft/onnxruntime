---
title: Python API
description: Python API reference for ONNX Runtime GenAI
has_children: false
parent: API docs
grand_parent: Generative AI
nav_order: 2
---

# Python API
{: .no_toc }

* TOC placeholder
{:toc}

## Install and import

The Python API is delivered by the onnxruntime-genai Python package.

```bash
pip install onnxruntime-genai
```

```python
import onnxruntime_genai
```

## Model class

### Load the model

Loads the ONNX model(s) and configuration from a folder on disk.

```python
onnxruntime_genai.Model(model_folder: str, device: onnxruntime_genai.DeviceType) -> onnxruntime_genai.Model
```

#### Parameters

- `model_folder`: (required) Location of model and configuration on disk
- `device`: (optional) The device to run on. One of:
   - onnxruntime_genai.CPU
   - onnxruntime_genai.CUDA
   If not specified, defaults to CPU.

#### Return value

`onnxruntime_genai.Model`

### Create tokenizer object

```python
onnxruntime_genai.Model.create_tokenizer(model: onnxruntime_genai.Model) -> onnxruntime_genai.Tokenizer
```

#### Parameters

- `model`: (Required) The model that was loaded by the `Model()`

#### Return value

- `Tokenizer`


### Generate method

```python
onnxruntime_genai.Model.generate(params: GeneratorParams) -> XXXX 
```

#### Parameters
- `params`: (Required) Created by the `GenerateParams` method.

#### Return value

`numpy.int32 [ batch_size, max_length]`


## GeneratorParams class

### Create GeneratorParams object

```python
onnxruntime_genai.GeneratorParams(model: onnxruntime_genai.Model) -> onnxruntime_genai.GeneratorParams
```

#### Parameters

- `model`: (required) The model that was loaded by onnxruntime_genai.Model()

#### Return value

`onnxruntime_genai.GeneratorParams`

## Tokenizer class

Tokenizer objects are created from a Model.

### Encode

```python
onnxruntime_genai.Tokenizer.encode(prompt: str) -> numpy.int32
```

#### Parameters

- `prompt`: (Required)

#### Return value

`numpy.int32`: an array of tokens representing the prompt

### Decode

```python
onnxruntime_genai.Tokenizer.decode(numpy.int32) -> str 
```

#### Parameters

`numpy.int32`: (Required) a sequence of generated tokens


#### Return value

str: the decoded generated tokens


### Encode batch

```python
onnxruntime_genai.Tokenizer.encode_batch(texts: list[str]) -> 
```

#### Parameters

- `texts`: a list of inputs

#### Return value

`[[numpy.int32]]`: a 2 D array of tokens

### Decode batch

```python
onnxruntime_genai.Tokenize.decode_batch(tokens: [[numpy.int32]]) -> list[str]
```

#### Parameters

- tokens

#### Return value

`texts`: a batch of decoded text


### Create tokenizer decoding stream

Decodes one token at a time to allow for responsive user interfaces.

```python
onnxruntime_genai.Tokenizer.create_stream() -> TokenizerStream
```

#### Parameters

None

#### Return value

- TokenizerStream

## TokenizerStream class

This class keeps track of the generated token sequence, returning the next displayable string (according to the tokenizer's vocabulary) when decode is called. Explain empty string ...

### Decode method

```python
onnxruntime_genai.TokenizerStream.decode(token: int32) -> str
```
  
#### Parameters

- `token`: (Required) A token to decode

#### Returns

`str`: Next displayable text, if at the end of displayable block?, otherwise empty string

## Generator class

### Create a Generator

```python
onnxruntime_genai.Generator(model: Model, params: GeneratorParams) -> Generator
```

#### Parameters

- `model`: (Required) The model to use for generation
- `params`: (Required) The set of parameters that control the generation

#### Return value

- `onnxruntime_genai.Generator`


### Is generation done

Returns true when all sequences are at max length, or have reached the end of sequence.

```python
onnxruntime_genai.Generator.is_done() -> bool
```

### Compute logits

Runs the model through one iteration.

```python
onnxruntime_genai.Generator.compute_logits()
```

### Generate next token

Using the current set of logits and the specified generator parameters, calculates the next batch of tokens, using Top P sampling.

```python
onnxruntime_genai.Generator.generate_next_token()
```

### Generate next token with Top P sampling

Using the current set of logits and the specified generator parameters, calculates the next batch of tokens, using Top P sampling.

```python
onnxruntime_genai.Generator.generate_next_token_top_p()
```

### Generate next token with Top K sampling

Using the current set of logits and the specified generator parameters, calculates the next batch of tokens, using Top K sampling.

```python
onnxruntime_genai.Generator.generate_next_token_top_k()
```

### Generate next token with Top K and Top P sampling

Using the current set of logits and the specified generator parameters, calculates the next batch of tokens, using both Top K then Top P sampling.

```python
onnxruntime_genai.Generator.generate_next_token_top_k_top_p()
```

### Get next tokens

Returns the most recently generated tokens.

```python
onnxruntime_genai.Generator.get_next_tokens() -> [numpy.int32]
```

### Get sequence

```python
onnxruntime_genai.Generator.get_sequence(index: int) -> [numpy.int32] 
```

- `index`: (Required) The index of the sequence in the batch to return