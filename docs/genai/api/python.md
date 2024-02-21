---
title: Python API
description: Python API reference for ONNX Runtime GenAI
has_children: false
parent: API docs
grand_parent: Generative AI (Preview)
nav_order: 1
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

- `model_folder`: Location of model and configuration on disk
- `device`: The device to run on. One of:
   - onnxruntime_genai.CPU
   - onnxruntime_genai.CUDA
   If not specified, defaults to CPU.

#### Returns

`onnxruntime_genai.Model`

### Generate method

```python
onnxruntime_genai.Model.generate(params: GeneratorParams) -> numpy.ndarray[int, int]
```

#### Parameters
- `params`: (Required) Created by the `GenerateParams` method.

#### Returns

`numpy.ndarray[int, int]`: a two dimensional numpy array with dimensions equal to the size of the batch passed in and the maximum length of the sequence of tokens.


## GeneratorParams class

### Create GeneratorParams object

```python
onnxruntime_genai.GeneratorParams(model: onnxruntime_genai.Model) -> onnxruntime_genai.GeneratorParams
```

#### Parameters

- `model`: (required) The model that was loaded by onnxruntime_genai.Model()

#### Returns

`onnxruntime_genai.GeneratorParams`: The GeneratorParams object

## Tokenizer class

### Create tokenizer object

```python
onnxruntime_genai.Model.Tokenizer(model: onnxruntime_genai.Model) -> onnxruntime_genai.Tokenizer
```

#### Parameters

- `model`: (Required) The model that was loaded by the `Model()`

#### Returns

- `Tokenizer`: The tokenizer object

### Encode

```python
onnxruntime_genai.Tokenizer.encode(text: str) -> numpy.ndarray[numpy.int32]
```

#### Parameters

- `text`: (Required)

#### Returns

`numpy.ndarray[numpy.int32]`: an array of tokens representing the prompt

### Decode

```python
onnxruntime_genai.Tokenizer.decode(tokens: numpy.ndarry[int]) -> str 
```

#### Parameters

- `numpy.ndarray[numpy.int32]`: (Required) a sequence of generated tokens


#### Returns

`str`: the decoded generated tokens


### Encode batch

```python
onnxruntime_genai.Tokenizer.encode_batch(texts: list[str]) -> numpy.ndarray[int, int]
```

#### Parameters

- `texts`: A list of inputs

#### Returns

`numpy.ndarray[int, int]`: The batch of tokenized strings

### Decode batch

```python
onnxruntime_genai.Tokenize.decode_batch(tokens: [[numpy.int32]]) -> list[str]
```

#### Parameters

- tokens

#### Returns

`texts`: a batch of decoded text


### Create tokenizer decoding stream


```python
onnxruntime_genai.Tokenizer.create_stream() -> TokenizerStream
```

#### Parameters

None

#### Returns

`onnxruntime_genai.TokenizerStream` The tokenizer stream object

## TokenizerStream class

This class accumulates the next displayable string (according to the tokenizer's vocabulary).

### Decode method

 
```python
onnxruntime_genai.TokenizerStream.decode(token: int32) -> str
```
  
#### Parameters

- `token`: (Required) A token to decode

#### Returns

`str`: If a displayable string has accumulated, this method returns it. If not, this method returns the empty string.

## Generator Params class

### Create a Generator Params

```python
onnxruntime_genai.GeneratorParams(model: Model) -> GeneratorParams
```

### Input_ids member 

```python
onnxruntime_genai.GeneratorParams.input_ids = numpy.ndarray[numpy.int32, numpy.int32]
```

### Set search options method

```python
onnxruntime_genai.GeneratorParams.set_search_options(options: dict[str, Any])
```

### 

## Generator class

### Create a Generator

```python
onnxruntime_genai.Generator(model: Model, params: GeneratorParams) -> Generator
```

#### Parameters

- `model`: (Required) The model to use for generation
- `params`: (Required) The set of parameters that control the generation

#### Returns

`onnxruntime_genai.Generator` The Generator object


### Is generation done

```python
onnxruntime_genai.Generator.is_done() -> bool
```

#### Returns

Returns true when all sequences are at max length, or have reached the end of sequence.


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

```python
onnxruntime_genai.Generator.get_next_tokens() -> numpy.ndarray[numpy.int32]
```

Returns

`numpy.ndarray[numpy.int32]`: The most recently generated tokens

### Get sequence

```python
onnxruntime_genai.Generator.get_sequence(index: int) -> numpy.ndarray[numpy.int32] 
```

- `index`: (Required) The index of the sequence in the batch to return