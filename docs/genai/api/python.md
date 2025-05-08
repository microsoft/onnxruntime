---
title: Python API
description: Python API reference for ONNX Runtime generate() API
has_children: false
parent: API docs
grand_parent: Generate API (Preview)
nav_order: 1
---

# Python API

_Note: this API is in preview and is subject to change._

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

### Load a model

Loads the ONNX model(s) and configuration from a folder on disk.

```python
onnxruntime_genai.Model(model_folder: str) -> onnxruntime_genai.Model
```

#### Parameters

- `model_folder`: Location of model and configuration on disk

#### Returns

`onnxruntime_genai.Model`

### Generate method

```python
onnxruntime_genai.Model.generate(params: GeneratorParams) -> numpy.ndarray[int, int]
```

#### Parameters
- `params`: (Required) Created by the `GeneratorParams` method.

#### Returns

`numpy.ndarray[int, int]`: a two dimensional numpy array with dimensions equal to the size of the batch passed in and the maximum length of the sequence of tokens.

### Device type

Return the device type that the model has been configured to run on.

```python
onnxruntime_genai.Model.device_type
```

#### Returns

`str`: a string describing the device that the loaded model will run on


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

### ApplyChatTemplate

```python
onnxruntime_genai.Tokenizer.apply_chat_template(template_str: str, messages: str, add_generation_prompt: bool) -> str 
```

#### Parameters

- `template_str`: (Optional) string representing the chat template, falls back to the default chat template from the tokenizer config.
- `messages`: (Required) string containing the input messages to be processed
- `add_generation_prompt`: (Required) boolean indicating whether to add a generation prompt to the output.


#### Returns

`str`: the messages with chat template tokens applied


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

## GeneratorParams class

### Create a Generator Params object

```python
onnxruntime_genai.GeneratorParams(model: Model) -> GeneratorParams
```

### Pad token id member

```python
onnxruntime_genai.GeneratorParams.pad_token_id
```

### EOS token id member

```python
onnxruntime_genai.GeneratorParams.eos_token_id
```

### vocab size member

```python
onnxruntime_genai.GeneratorParams.vocab_size
```

### input_ids member

```python
onnxruntime_genai.GeneratorParams.input_ids: numpy.ndarray[numpy.int32, numpy.int32]
```

### Set model input

```python
onnxruntime_genai.GeneratorParams.set_model_input(name: str, value: [])
```


### Set search options method

```python
onnxruntime_genai.GeneratorParams.set_search_options(options: dict[str, Any])
```

### Try graph capture with max batch size

```python
onnxruntime_genai.GeneratorParams.try_graph_capture_with_max_batch_size(max_batch_size: int)
```

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

### Get output

Returns an output of the model.

```python
onnxruntime_genai.Generator.get_output(str: name) -> numpy.ndarray
```

#### Parameters
- `name`: the name of the model output

#### Returns
- `numpy.ndarray`: a multi dimensional array of the model outputs. The shape of the array is shape of the output.

#### Example

The following code returns the output logits of a model.

```python
logits = generator.get_output("logits")
```


### Generate next token

Using the current set of logits and the specified generator parameters, calculates the next batch of tokens, using Top P sampling.

```python
onnxruntime_genai.Generator.generate_next_token()
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

## Adapter class

### Create

Create an Adapters object, using a model that has been loaded.

```python
model = ...
adapters = og.Adapters(model)
```

#### Parameters

* `model`: the model that the adapters will be used with

#### Return value

An `Adapter` object

### Load

Load an adapter from disk into an Adapter object in memory.

```python
onnxruntime_genai.Adapters(file: str, name: str) -> None
```

#### Parameters

* `file`: the location on disk from which to load the adapter
* `name`: the name of the adapter

#### Return value

None

### Set active adapter

Sets the actove adapter on a `Generator` object.

```python
onnxruntime_genai.Generator(adapters: Generators::Adapters, adapter: str) -> None
```

#### Parameters

* `adapters`: the adapters object, which has had the identified adapter loading into it
* `adapter`: the name of the adapter to set as active

#### Return value

None