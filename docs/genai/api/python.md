---
title: Python API
description: Python API reference for ONNX Runtime GenAI
has_children: false
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

### GeneratorParameters class

```python
params=onnxruntime_genai.GeneratorParams(model: onnxruntime_genai.Model) -> onnxruntime_genai.GeneratorParams
```

#### Parameters

- `model`: (required) The model that was loaded by onnxruntime_genai.Model()

#### Return value


### Generate

```python
onnxruntime_genai.Model.generate(params: GeneratorParams) -> XXXX 
```

#### Parameters
- `params`: (Required) Created by the `GenerateParams` method.

#### Return value

### Generate sequence

```python
onnxruntime_genai.Model.generate_sequence(input_ids: , params: **kwargs)
```

#### Parameters

- `input_ids`: tokenized prompt
- `params`: dictionary of generation parameters

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
  

  pybind11::class_<PyGenerator>(m, "Generator")
      .def(pybind11::init<Model&, PyGeneratorParams&>())
      .def("is_done", &PyGenerator::IsDone)
      .def("compute_logits", &PyGenerator::ComputeLogits)
      .def("generate_next_token", &PyGenerator::GenerateNextToken)
      .def("generate_next_token_top", &PyGenerator::GenerateNextToken_Top)
      .def("generate_next_token_top_p", &PyGenerator::GenerateNextToken_TopP)
      .def("generate_next_token_top_k", &PyGenerator::GenerateNextToken_TopK)
      .def("generate_next_token_top_k_top_p", &PyGenerator::GenerateNextToken_TopK_TopP)
      .def("get_next_tokens", &PyGenerator::GetNextTokens)
      .def("get_sequence", &PyGenerator::GetSequence);

  m.def("is_cuda_available", []() {
#ifdef USE_CUDA
    return true;
#else
        return false;
#endif
  });
}

}  // namespace Generators