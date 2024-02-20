---
title: Python API
description: Python API reference for ONNX Runtime GenAI
has_children: false
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
   - onnxruntime_genai.CPU
   If not specified, defaults to XXX

#### Return value

### Create a Generator

```python
onnxruntime_genai.Model.Generator(params: GeneratorParams) -> Generator
```

#### Parameters

- `params`: (Required) The set of parameters that control the generation

#### Return value

- `onnxruntime_genai.Generator`


### Generate

```python
onnxruntime_genai.Model.generate(params: GeneratorParams) -> XXXX 
```

#### Parameters
- `params`: (Required) Created by the `GenerateParams` method.

#### Return value

### Generate sequence

```python
onnxruntime_genai.Model.generate_sequence(input_ids: , params: )
```

#### Parameters

- `input_ids`: tokenized prompt
- `params`: dictionary of generation parameters


### Create GeneratorParameters class

```python
params=onnxruntime_genai.GeneratorParams(model: onnxruntime_genai.Model) -> onnxruntime_genai.GeneratorParams
```

#### Parameters

- `model`: (required) The model that was loaded by onnxruntime_genai.Model()

#### Return value


## Tokenizer class

### Create tokenizer

```python
create_tokenizer(model: onnxruntime_genai.Model) -> onnxruntime_genai.Tokenizer
```

#### Parameters

- `model`: (Required) The model that was loaded by the `Model()`

#### Return value

- `Tokenizer`


### Encode

```python
onnxruntime_genai.Tokenizer.encode(XXXX) -> XXXX
```

#### Parameters

#### Return value

### Decode

```python
onnxruntime_genai.StreamingTokenizer.decode(XXXX) -> XXXX
```

#### Parameters

#### Return value

### Encode batch

```python
onnxruntime_genai.Tokenizer.encode_batch(XXXX) -> XXXX
```

#### Parameters

#### Return value

### Decode batch

```python
onnxruntime_genai.decode_batch(XXXX) -> XXXX
```

#### Parameters

#### Return value



### Create streaming tokenizer

```python
create_stream(model: onnxruntime_genai.Model) -> TokenizerStream
```

#### Parameters

- `model`: (Required) The model that was loaded by the `Model()`

#### Return value

- TokenizerStream

### Decode token stream

```python
onnxruntime_genai.TokenizerStream.decode(token: ) -> token
```
  
## Generator class

### Is generation done

```python
onnxruntime_genai.Generator.is_done() -> bool
```

### Compute logits

```python
onnxruntime_genai.Generator.compute_logits() ->
```

### Generate next token

```python
onnxruntime_genai.Generator.generate_next_token() -> 
```

### Generate next token with Top P sampling

```python
onnxruntime_genai.Generator.generate_next_token_top_p() -> 
```

### Generate next token with Top K sampling

```python
onnxruntime_genai.Generator.generate_next_token_top_k() -> 
```

### Generate next token with Top K and Top P sampling

```python
onnxruntime_genai.Generator.generate_next_token_top_k_top_p() -> 
```

### Get next tokens

```python
onnxruntime_genai.Generator.generate_next_tokens() -> 
```

### Get sequence

```python
onnxruntime_genai.Generator.generate_next_token() -> 
```
