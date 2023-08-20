# Whisper

## Exporting Whisper with Beam Search

There are several ways to export Whisper with beam search (using Whisper tiny as an example).

### Option 1: from convert_to_onnx

```
# From source
$ git clone https://github.com/microsoft/onnxruntime
$ cd onnxruntime/onnxruntime/python/tools/transformers/
$ python3 -m models.whisper.convert_to_onnx -m openai/whisper-tiny --output whispertiny --use_external_data_format

# From wheel
$ python3 -m onnxruntime.transformers.models.whisper.convert_to_onnx -m openai/whisper-tiny --output whispertiny --use_external_data_format
```

### Option 2: end-to-end model from [Olive](https://github.com/microsoft/Olive/tree/main/examples/whisper)

Please follow the [README instructions](https://github.com/microsoft/Olive/tree/main/examples/whisper#prerequisites) in Olive.

### Option 3: from [Hugging Face Optimum](https://github.com/huggingface/optimum)

Run the following Python code to export:

```
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

model_name = "openai/whisper-large-v2"
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
$ python3 -m models.whisper.convert_to_onnx -m openai/whisper-tiny --output whispertiny --use_external_data_format --use_forced_decoder_ids

# From wheel:
$ python3 -m onnxruntime.transformers.models.whisper.convert_to_onnx -m openai/whisper-tiny --output whispertiny --use_external_data_format --use_forced_decoder_ids
```

Export + Optimize for FP32
```
# From source:
$ python3 -m models.whisper.convert_to_onnx -m openai/whisper-tiny --output whispertiny --use_external_data_format --optimize_onnx --precision fp32

# From wheel:
$ python3 -m onnxruntime.transformers.models.whisper.convert_to_onnx -m openai/whisper-tiny --output whispertiny --use_external_data_format --optimize_onnx --precision fp32
```

Export + Optimize for FP16 and GPU
```
# From source:
$ python3 -m models.whisper.convert_to_onnx -m openai/whisper-tiny --output whispertiny --use_external_data_format --optimize_onnx --precision fp16 --use_gpu --provider cuda

# From wheel:
$ python3 -m onnxruntime.transformers.models.whisper.convert_to_onnx -m openai/whisper-tiny --output whispertiny --use_external_data_format --optimize_onnx --precision fp16 --use_gpu --provider cuda
```

Export + Quantize for INT8
```
# From source:
$ python3 -m models.whisper.convert_to_onnx -m openai/whisper-tiny --output whispertiny --use_external_data_format --precision int8 --quantize_embedding_layer

# From wheel:
$ python3 -m onnxruntime.transformers.models.whisper.convert_to_onnx -m openai/whisper-tiny --output whispertiny --use_external_data_format --precision int8 --quantize_embedding_layer
```

## Benchmark Whisper

Here are some examples of how you can benchmark Whisper across various end-to-end (E2E) implementations.

Note: In the below examples, `PyTorch` refers to running in PyTorch without `torch.compile` and `PyTorch 2.0` refers to running in PyTorch with `torch.compile`.

### Variants

1. PyTorch (without `torch.compile`), FP32
```
python3 -m models.whisper.benchmark \
    --benchmark-type hf-pt \
    --audio-path 1272-141231-0002.mp3 \
    --model-name openai/whisper-large-v2 \
    --precision fp32 \
    --device cpu
```

2. PyTorch 2.0 (with `torch.compile`), FP16
```
python3 -m models.whisper.benchmark \
    --benchmark-type hf-pt2 \
    --audio-path 1272-141231-0002.mp3 \
    --model-name openai/whisper-large-v2 \
    --precision fp16 \
    --device cuda
```

3. Optimum + ONNX Runtime, FP32, export via Optimum
```
python3 -m models.whisper.benchmark \
    --benchmark-type hf-ort \
    --audio-path 1272-141231-0002.mp3 \
    --model-name openai/whisper-large-v2 \
    --hf-ort-model-path ./whisper-large-v2-onnx/ \
    --precision fp32 \
    --device cpu
```

4. ONNX Runtime, FP32, export via Olive or convert_to_onnx
```
python3 -m models.whisper.benchmark \
    --benchmark-type ort \
    --audio-path 1272-141231-0002.mp3 \
    --model-name openai/whisper-large-v2 \
    --ort-model-path ./wlarge-fp32/whisper-large-v2_beamsearch.onnx \
    --precision fp32 \
    --device cpu
```

5. ONNX Runtime, FP16, export via Olive or convert_to_onnx
```
python3 -m models.whisper.benchmark \
    --benchmark-type ort \
    --audio-path 1272-141231-0002.mp3 \
    --model-name openai/whisper-large-v2 \
    --ort-model-path ./wlarge-fp32/whisper-large_all.onnx \
    --precision fp16 \
    --device cuda
```

6. ONNX Runtime, INT8, export via Olive or convert_to_onnx
```
python3 -m models.whisper.benchmark \
    --benchmark-type ort \
    --audio-path 1272-141231-0002.mp3 \
    --model-name openai/whisper-large-v2 \
    --ort-model-path ./wlarge-fp32/whisper-large-v2_all.onnx \
    --precision fp32 \
    --device cpu
```

You can profile a variant by adding the `--profile` flag.

### Benchmark All

You can use `benchmark_all.py` to benchmark across various platforms and automatically store the results in a CSV file. Here is an example.

```
python3 -m models.whisper.benchmark_all \
    --audio-path ./whisper-test-audios/ \
    --hf-ort-model-path ./whisper-large-v2-onnx/ \
    --ort-model-path ./wlarge-fp32/whisper-large-v2_all.onnx \
    --model-name openai/whisper-large-v2 \
    --precision fp32 \
    --device cpu
```

### Benchmarking on NVIDIA A100

Here is a benchmark for an MP3 file with 20.7s of audio.

#### FP16

| Engine        | Size     | Per-Token Latency | Real-Time Factor |
| ------------- | -------- | ----------------- | ---------------- |
| PyTorch       | Tiny     | 4.697 ms/token    | 0.004697         |
| PyTorch 2.0   | Tiny     | 3.406 ms/token    | 0.003406         |
| ONNX Runtime  | Tiny     | 0.746 ms/token    | 0.000746         |
| PyTorch       | Medium   | 17.837 ms/token   | 0.017387         |
| PyTorch 2.0   | Medium   | 18.124 ms/token   | 0.018124         |
| ONNX Runtime  | Medium   | 3.894 ms/token    | 0.003894         |
| PyTorch       | Large v2 | 23.470 ms/token   | 0.023470         |
| PyTorch 2.0   | Large v2 | 23.146 ms/token   | 0.023146         |
| ONNX Runtime  | Large v2 | 6.262 ms/token    | 0.006262         |

#### FP32

| Engine        | Size     | Per-Token Latency | Real-Time Factor |
| ------------- | -------- | ----------------- | ---------------- |
| PyTorch       | Tiny     | 6.220 ms/token    | 0.006220         |
| PyTorch 2.0   | Tiny     | 3.944 ms/token    | 0.003944         |
| ONNX Runtime  | Tiny     | 1.545 ms/token    | 0.001545         |
| PyTorch       | Medium   | 19.093 ms/token   | 0.019093         |
| PyTorch 2.0   | Medium   | 20.459 ms/token   | 0.020459         |
| ONNX Runtime  | Medium   | 9.440 ms/token    | 0.009440         |
| PyTorch       | Large v2 | 25.844 ms/token   | 0.025844         |
| PyTorch 2.0   | Large v2 | 26.397 ms/token   | 0.026397         |
| ONNX Runtime  | Large v2 | 7.492 ms/token    | 0.007492         |
