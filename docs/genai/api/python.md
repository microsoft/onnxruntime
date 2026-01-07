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

```python
onnxruntime_genai.Model(config_path: str) -> Model
onnxruntime_genai.Model(config: onnxruntime_genai.Config) -> Model
```

#### Properties

- `type`: Returns the model type as a string.

  ```python
  model = onnxruntime_genai.Model("config.json")
  print(model.type)
  ```

- `device_type`: Returns the device type as a string.

  ```python
  print(model.device_type)
  ```

#### Methods

- `create_multimodal_processor() -> MultiModalProcessor`

  ```python
  processor = model.create_multimodal_processor()
  ```

---

## Config class

```python
onnxruntime_genai.Config(config_path: str) -> Config
```

#### Methods

- `append_provider(provider: str)`

  ```python
  config = onnxruntime_genai.Config("config.json")
  config.append_provider("CUDAExecutionProvider")
  ```

- `set_provider_option(option: str, value: str)`

  ```python
  config.set_provider_option("device_id", "0")
  ```

- `clear_providers()`

  ```python
  config.clear_providers()
  ```

- `add_model_data(model_filename: str, data: bytes | buffer)`

  ```python
  with open("decoder.onnx", "rb") as f:
      config.add_model_data("decoder.onnx", f.read())
  ```

- `remove_model_data(model_filename: str)`

  ```python
  config.remove_model_data("decoder.onnx")
  ```

- `overlay(config_path: str)`

  ```python
  config.overlay("config.overlay.json")
  ```

- `set_decoder_provider_options_hardware_device_type(device_type: int)`
- `set_decoder_provider_options_hardware_device_id(device_id: int)`
- `set_decoder_provider_options_hardware_vendor_id(vendor_id: int)`
- `clear_decoder_provider_options_hardware_device_type()`
- `clear_decoder_provider_options_hardware_device_id()`
- `clear_decoder_provider_options_hardware_vendor_id()`

---

## GeneratorParams class

```python
onnxruntime_genai.GeneratorParams(model: Model) -> GeneratorParams
```

#### Methods

- `try_graph_capture_with_max_batch_size(max_batch_size: int)`

  ```python
  params.try_graph_capture_with_max_batch_size(8)
  ```
  _Deprecated: emits a warning and will be removed in a future release._

- `set_search_options(**options)`

  ```python
  params.set_search_options(temperature=0.7, top_p=0.9)
  ```

- `set_guidance(type: str, data: str)`

  ```python
  params.set_guidance("prefix", "Once upon a time", enable_ff_tokens=False)
  ```

---

## Generator class

```python
onnxruntime_genai.Generator(model: Model, params: GeneratorParams) -> Generator
```

#### Methods

- `is_done() -> bool`

  ```python
  generator = onnxruntime_genai.Generator(model, params)
  done = generator.is_done()
  ```

- `get_input(name: str) -> numpy.ndarray`

  ```python
  input_ids = generator.get_input("input_ids")
  ```

- `get_output(name: str) -> numpy.ndarray`

  ```python
  output = generator.get_output("output_ids")
  ```

- `set_inputs(named_tensors: NamedTensors)`

  ```python
  named = onnxruntime_genai.NamedTensors()
  named["input_ids"] = np.array([1, 2, 3], dtype=np.int32)
  generator.set_inputs(named)
  ```

- `set_model_input(name: str, value: numpy.ndarray)`

  ```python
  generator.set_model_input("input_ids", np.array([1, 2, 3], dtype=np.int32))
  ```

- `append_tokens(tokens: numpy.ndarray[int32])`

  ```python
  generator.append_tokens(np.array([4, 5], dtype=np.int32))
  ```

- `append_tokens(tokens: onnxruntime_genai.Tensor)`

  ```python
  tensor = onnxruntime_genai.Tensor(np.array([4, 5], dtype=np.int32))
  generator.append_tokens(tensor)
  ```

- `get_logits() -> numpy.ndarray[float32]`

  ```python
  logits = generator.get_logits()
  ```

- `set_logits(new_logits: numpy.ndarray[float32])`

  ```python
  generator.set_logits(np.zeros_like(logits))
  ```

- `generate_next_token()`

  ```python
  generator.generate_next_token()
  ```

- `rewind_to(new_length: int)`

  ```python
  generator.rewind_to(2)
  ```

- `get_next_tokens() -> numpy.ndarray[int32]`

  ```python
  next_tokens = generator.get_next_tokens()
  ```

- `get_sequence(index: int) -> numpy.ndarray[int32]`

  ```python
  sequence = generator.get_sequence(0)
  ```

- `set_active_adapter(adapters: onnxruntime_genai.Adapters, adapter_name: str)`

  ```python
  adapters = onnxruntime_genai.Adapters(model)
  generator.set_active_adapter(adapters, "adapter_name")
  ```

---

## Tokenizer class

```python
onnxruntime_genai.Tokenizer(model: Model) -> Tokenizer
```

#### Methods

- `bos_token_id: int`

  ```python
  bos = tokenizer.bos_token_id
  ```

- `eos_token_ids: numpy.ndarray[int32]`

  ```python
  eos_ids = tokenizer.eos_token_ids
  ```

- `pad_token_id: int`

  ```python
  pad = tokenizer.pad_token_id
  ```

- `update_options(**options)`

  ```python
  tokenizer.update_options(space_replacement="▁")
  ```

- `encode(text: str) -> numpy.ndarray[int32]`

  ```python
  tokenizer = onnxruntime_genai.Tokenizer(model)
  tokens = tokenizer.encode("Hello world")
  ```

- `to_token_id(text: str) -> int`

  ```python
  token_id = tokenizer.to_token_id("Hello")
  ```

- `decode(tokens: numpy.ndarray[int32]) -> str`

  ```python
  text = tokenizer.decode(tokens)
  ```

- `apply_chat_template(messages: str, template_str: str | None = None, tools: str | None = None, add_generation_prompt: bool = True) -> str`

  ```python
  chat = tokenizer.apply_chat_template(messages="Hi!", template_str="{user}: {message}", add_generation_prompt=True)
  ```

- `encode_batch(texts: list[str]) -> onnxruntime_genai.Tensor`

  ```python
  batch_tensor = tokenizer.encode_batch(["Hello", "World"])
  ```

- `decode_batch(tokens: onnxruntime_genai.Tensor) -> list[str]`

  ```python
  texts = tokenizer.decode_batch(batch_tensor)
  ```

- `create_stream() -> TokenizerStream`

  ```python
  stream = tokenizer.create_stream()
  ```

---

## TokenizerStream class

```python
onnxruntime_genai.TokenizerStream(tokenizer: Tokenizer) -> TokenizerStream
```

#### Methods

- `decode(token: int32) -> str`

  ```python
  token_str = stream.decode(123)
  ```

---

## NamedTensors class

```python
onnxruntime_genai.NamedTensors() -> NamedTensors
```

#### Methods

- `__getitem__(name: str) -> onnxruntime_genai.Tensor`

  ```python
  tensor = named_tensors["input_ids"]
  ```

- `__setitem__(name: str, value: numpy.ndarray or onnxruntime_genai.Tensor)`

  ```python
  named_tensors["input_ids"] = np.array([1, 2, 3], dtype=np.int32)
  ```

- `__contains__(name: str) -> bool`

  ```python
  exists = "input_ids" in named_tensors
  ```

- `__delitem__(name: str)`

  ```python
  del named_tensors["input_ids"]
  ```

- `__len__() -> int`

  ```python
  length = len(named_tensors)
  ```

- `keys() -> list[str]`

  ```python
  keys = named_tensors.keys()
  ```

---

## Tensor class

```python
onnxruntime_genai.Tensor(array: numpy.ndarray) -> Tensor
```

#### Methods

- `shape() -> list[int]`

  ```python
  tensor = onnxruntime_genai.Tensor(np.array([1, 2, 3]))
  print(tensor.shape())
  ```

- `type() -> int`

  ```python
  print(tensor.type())
  ```

- `data() -> memoryview`

  ```python
  data = tensor.data()
  ```

- `as_numpy() -> numpy.ndarray`

  ```python
  arr = tensor.as_numpy()
  ```

---

## Adapters class

```python
onnxruntime_genai.Adapters(model: Model) -> Adapters
```

#### Methods

- `unload(adapter_name: str)`

  ```python
  adapters.unload("adapter_name")
  ```

- `load(file: str, name: str)`

  ```python
  adapters.load("adapter_file.bin", "adapter_name")
  ```

---

## Request class

```python
onnxruntime_genai.Request(params: GeneratorParams) -> Request
```

#### Methods

- `add_tokens(tokens: numpy.ndarray[int32])`

  ```python
  request.add_tokens(np.array([1, 2, 3], dtype=np.int32))
  ```

- `has_unseen_tokens() -> bool`

  ```python
  still_pending = request.has_unseen_tokens()
  ```

- `is_done() -> bool`

  ```python
  done = request.is_done()
  ```

- `get_unseen_token() -> int`

  ```python
  next_token = request.get_unseen_token()
  ```

- `set_opaque_data(data: object)` / `get_opaque_data() -> object | None`

  ```python
  request.set_opaque_data({"trace_id": "abc"})
  opaque = request.get_opaque_data()
  ```

---

## Engine class

```python
onnxruntime_genai.Engine(model: Model) -> Engine
```

#### Methods

- `add_request(request: Request)`

  ```python
  engine.add_request(request)
  ```

- `step()`

  ```python
  engine.step()
  ```

- `remove_request(request: Request)`

  ```python
  engine.remove_request(request)
  ```

- `has_pending_requests() -> bool`

  ```python
  pending = engine.has_pending_requests()
  ```

---

## MultiModalProcessor class

```python
onnxruntime_genai.MultiModalProcessor(model: Model) -> MultiModalProcessor
```

#### Methods

- `__call__(prompt: str | list[str] = None, images: Images = None, audios: Audios = None) -> onnxruntime_genai.Tensor`

  ```python
  result = processor(prompt="Describe this image", images=onnxruntime_genai.Images.open("image.png"))
  batched = processor(["Describe first", "Describe second"], images=onnxruntime_genai.Images.open("image1.png", "image2.png"))
  ```

- `create_stream() -> TokenizerStream`

  ```python
  stream = processor.create_stream()
  ```

- `decode(tokens: numpy.ndarray[int32]) -> str`

  ```python
  text = processor.decode(tokens)
  ```

---

## Images class

```python
onnxruntime_genai.Images.open(*image_paths: str) -> Images
onnxruntime_genai.Images.open_bytes(*image_datas: bytes) -> Images
```

```python
images = onnxruntime_genai.Images.open("image1.png", "image2.jpg")
with open("image1.png", "rb") as f:
    images_bytes = onnxruntime_genai.Images.open_bytes(f.read())
```

---

## Audios class

```python
onnxruntime_genai.Audios.open(*audio_paths: str) -> Audios
onnxruntime_genai.Audios.open_bytes(*audio_datas: bytes) -> Audios
```

```python
audios = onnxruntime_genai.Audios.open("audio1.wav")
with open("audio1.wav", "rb") as f:
    audios_bytes = onnxruntime_genai.Audios.open_bytes(f.read())
```

---

## Utility functions

- `onnxruntime_genai.set_log_options(**options)`

  ```python
  onnxruntime_genai.set_log_options(verbose=True)
  ```

- `onnxruntime_genai.set_log_callback(callback: Callable[[str], None] | None)`

  ```python
  onnxruntime_genai.set_log_callback(lambda message: print("LOG:", message))
  ```

- `onnxruntime_genai.is_cuda_available() -> bool`

  ```python
  print(onnxruntime_genai.is_cuda_available())
  ```

- `onnxruntime_genai.is_dml_available() -> bool`

  ```python
  print(onnxruntime_genai.is_dml_available())
  ```

- `onnxruntime_genai.is_rocm_available() -> bool`

  ```python
  print(onnxruntime_genai.is_rocm_available())
  ```

- `onnxruntime_genai.is_webgpu_available() -> bool`

  ```python
  print(onnxruntime_genai.is_webgpu_available())
  ```

- `onnxruntime_genai.is_qnn_available() -> bool`

  ```python
  print(onnxruntime_genai.is_qnn_available())
  ```

- `onnxruntime_genai.is_openvino_available() -> bool`

  ```python
  print(onnxruntime_genai.is_openvino_available())
  ```

- `onnxruntime_genai.set_current_gpu_device_id(device_id: int)`

  ```python
  onnxruntime_genai.set_current_gpu_device_id(0)
  ```

- `onnxruntime_genai.get_current_gpu_device_id() -> int`

  ```python
  print(onnxruntime_genai.get_current_gpu_device_id())
  ```

- `onnxruntime_genai.register_execution_provider_library(provider_name: str, path: str)`

  ```python
  onnxruntime_genai.register_execution_provider_library("MyEP", "/path/to/libMyEP.so")
  ```

- `onnxruntime_genai.unregister_execution_provider_library(provider_name: str)`

  ```python
  onnxruntime_genai.unregister_execution_provider_library("MyEP")
  ```