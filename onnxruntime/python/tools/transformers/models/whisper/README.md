# Whisper

## Exporting Whisper with Beam Search

There are several ways to export Whisper with beam search (using Whisper tiny as an example).

### Option 1: from source
```
$ git clone https://github.com/microsoft/onnxruntime
$ cd onnxruntime/onnxruntime/python/tools/transformers/models/whisper
$ python3 convert_to_onnx.py -m openai/whisper-tiny --output whispertiny --use_external_data_format
```

### Option 2: from wheel
```
$ python3 -m onnxruntime.transformers.models.whisper.convert_to_onnx -m openai/whisper-tiny --output whispertiny --use_external_data_format
```

### Option 3: end-to-end model from [Olive](https://github.com/microsoft/Olive/tree/main/examples/whisper)

Please follow the [README instructions](https://github.com/microsoft/Olive/tree/main/examples/whisper#prerequisites) in Olive.

### Option 4: from [Hugging Face Optimum](https://github.com/huggingface/optimum)

Run the following Python code to export:

```
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

model_name = "openai/whisper-large"
model = ORTModelForSpeechSeq2Seq.from_pretrained(
    model_name,
    export=True,
)
model.save_pretrained(model_name.split("/")[-1] + "-onnx")
```

## Exporting + Optimizing + Quantizing Whisper with Beam Search

Here are some additional examples for exporting Whisper with beam search.

Export with Forced Decoder Input Ids
```
# From source:
$ python3 convert_to_onnx.py -m openai/whisper-tiny --output whispertiny --use_external_data_format --use_forced_decoder_ids

# From wheel:
$ python3 -m onnxruntime.transformers.models.whisper.convert_to_onnx -m openai/whisper-tiny --output whispertiny --use_external_data_format --use_forced_decoder_ids
```

Export + Optimize for FP32
```
# From source:
$ python3 convert_to_onnx.py -m openai/whisper-tiny --output whispertiny --use_external_data_format --optimize_onnx --precision fp32

# From wheel:
$ python3 -m onnxruntime.transformers.models.whisper.convert_to_onnx -m openai/whisper-tiny --output whispertiny --use_external_data_format --optimize_onnx --precision fp32
```

Export + Optimize for FP16 and GPU
```
# From source:
$ python3 convert_to_onnx.py -m openai/whisper-tiny --output whispertiny --use_external_data_format --optimize_onnx --precision fp16 --use_gpu --provider cuda

# From wheel:
$ python3 -m onnxruntime.transformers.models.whisper.convert_to_onnx -m openai/whisper-tiny --output whispertiny --use_external_data_format --optimize_onnx --precision fp16 --use_gpu --provider cuda
```

Export + Quantize for INT8
```
# From source:
$ python3 convert_to_onnx.py -m openai/whisper-tiny --output whispertiny --use_external_data_format --precision int8 --quantize_embedding_layer

# From wheel:
$ python3 -m onnxruntime.transformers.models.whisper.convert_to_onnx -m openai/whisper-tiny --output whispertiny --use_external_data_format --precision int8 --quantize_embedding_layer
```

## Benchmark Whisper

Here are some examples of how you can benchmark Whisper across various end-to-end (E2E) implementations.

Note: In the below examples, `PyTorch` refers to running in PyTorch without `torch.compile` and `PyTorch 2.0` refers to running in PyTorch with `torch.compile`.

1. PyTorch (without `torch.compile`), FP32
```
python3 benchmark.py \
    --benchmark-type hf-pt \
    --audio-path 1272-141231-0002.mp3 \
    --model-name openai/whisper-large-v2 \
    --precision fp32 \
    --device cpu
```

2. PyTorch 2.0 (with `torch.compile`), FP16
```
python3 benchmark.py \
    --benchmark-type hf-pt2 \
    --audio-path 1272-141231-0002.mp3 \
    --model-name openai/whisper-large-v2 \
    --precision fp16 \
    --device cuda
```

3. Optimum + ONNX Runtime, FP32, export via Optimum
```
python3 benchmark.py \
    --benchmark-type hf-ort \
    --audio-path 1272-141231-0002.mp3 \
    --model-name openai/whisper-large-v2 \
    --hf-ort-model-path ./whisper-large-v2-onnx/ \
    --precision fp32 \
    --device cpu
```

4. ONNX Runtime, FP32, export via Olive or convert_to_onnx
```
python3 benchmark.py \
    --benchmark-type ort \
    --audio-path 1272-141231-0002.mp3 \
    --model-name openai/whisper-large-v2 \
    --ort-model-path ./wlarge-fp32/whisper-large_beamsearch.onnx \
    --precision fp32 \
    --device cpu
```

5. ONNX Runtime, FP16, export via Olive or convert_to_onnx
```
python3 benchmark.py \
    --benchmark-type ort \
    --audio-path 1272-141231-0002.mp3 \
    --model-name openai/whisper-large-v2 \
    --ort-model-path ./wlarge-fp32/whisper-large_all.onnx \
    --precision fp16 \
    --device cuda
```

6. ONNX Runtime, INT8, export via Olive or convert_to_onnx
```
python3 benchmark.py \
    --benchmark-type ort \
    --audio-path 1272-141231-0002.mp3 \
    --model-name openai/whisper-large-v2 \
    --ort-model-path ./wlarge-fp32/whisper-large_all.onnx \
    --precision fp32 \
    --device cpu
```
