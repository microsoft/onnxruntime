# Whisper

## Exporting Whisper with Beam Search

There are two ways to export Whisper with beam search (using Whisper tiny as an example).

Option 1: from source
```
$ git clone https://github.com/microsoft/onnxruntime
$ cd onnxruntime/onnxruntime/python/tools/transformers/models/whisper
$ python3 convert_to_onnx.py -m openai/whisper-tiny --output whispertiny --use_external_data_format
```

Option 2: from wheel
```
$ python3 -m onnxruntime.transformers.models.whisper.convert_to_onnx -m openai/whisper-tiny --output whispertiny --use_external_data_format
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

### E2E variants

1) PyTorch (without `torch.compile`), FP32, Hugging Face `pipeline` API
```
python3 benchmark.py \
    --benchmark-type "HF + PT" \
    --hf-api pipeline \
    --audio-path 1272-141231-0002.mp3 \
    --precision fp32 \
    --model-size tiny \
    --batch-size 2 \
    --device cpu
```

2) PyTorch (without `torch.compile`), FP32, Hugging Face `generate()` and `decode()` API
```
python3 benchmark.py \
    --benchmark-type "HF + PT" \
    --hf-api gen-and-dec \
    --audio-path 1272-141231-0002.mp3 \
    --precision fp32 \
    --model-size tiny \
    --batch-size 2 \
    --device cpu
```

3) PyTorch 2.0 (with `torch.compile`), FP32, Hugging Face `pipeline` API
```
python3 benchmark.py \
    --benchmark-type "HF + PT2" \
    --hf-api pipeline \
    --audio-path whisper/tests/jfk.flac \
    --precision fp32 \
    --model-size tiny \
    --batch-size 2 \
    --device cpu
```

4) PyTorch 2.0 (with `torch.compile`), FP32, Hugging Face `generate()` and `decode()` API
```
python3 benchmark.py \
    --benchmark-type "HF + PT2" \
    --hf-api gen-and-dec \
    --audio-path whisper/tests/jfk.flac \
    --precision fp32 \
    --model-size tiny \
    --batch-size 2 \
    --device cpu
```

5) ONNX Runtime, FP32, Hugging Face `pipeline` API
```
python3 benchmark.py \
    --benchmark-type "HF + ORT" \
    --hf-api pipeline \
    --audio-path whisper/tests/jfk.flac \
    --precision fp32 \
    --model-size tiny \
    --batch-size 2 \
    --device cpu
```

6) ONNX Runtime, FP32, Hugging Face `generate()` and `decode()` API
```
python3 benchmark.py \
    --benchmark-type "HF + ORT" \
    --hf-api gen-and-dec \
    --audio-path whisper/tests/jfk.flac \
    --precision fp32 \
    --model-size tiny \
    --batch-size 2 \
    --device cpu
```

7) ONNX Runtime, FP32
```
# Creating the E2E ONNX model requires two steps:
# 1) Run `convert_to_onnx.py` (https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/models/whisper/convert_to_onnx.py)
# 2) Run `whisper_e2e.py` (https://github.com/microsoft/onnxruntime-extensions/blob/main/tutorials/whisper_e2e.py)
# Note: This benchmark example can work for the ONNX model produced at the end of either step.

python3 benchmark.py \
    --benchmark-type ORT \
    --ort-model-path wtiny_fp32/whisper-tiny_all.onnx \
    --audio-path 1272-141231-0002.mp3 \
    --precision fp32 \
    --model-size tiny \
    --batch-size 2 \
    --device cpu
```

### Individual component examples

You can also benchmark individual model components in Whisper.

1) "Whisper encoder" from Hugging Face Optimum export, FP32
```
python3 benchmark.py \
    --benchmark-type "ORT" \
    --ort-model-path hf-ort-optimum-whisper-tiny/fp32/encoder_model.onnx \
    --precision fp32 \
    --model-size tiny \
    --batch-size 2 \
    --device cpu
```

2) "Whisper decoder" from Hugging Face Optimum export, FP32
```
python3 benchmark.py \
    --benchmark-type "ORT" \
    --ort-model-path hf-ort-optimum-whisper-tiny/fp32/decoder_model.onnx \
    --precision fp32 \
    --model-size tiny \
    --batch-size 2 \
    --device cpu
```

3) "Whisper decoder-with-past" from Hugging Face Optimum export, FP32
```
python3 benchmark.py \
    --benchmark-type "ORT" \
    --ort-model-path hf-ort-optimum-whisper-tiny/fp32/decoder_with_past_model.onnx \
    --precision fp32 \
    --model-size tiny \
    --batch-size 2 \
    --device cpu
```

4) "Whisper encoder-decoder-init" from ONNX Runtime's custom export, FP32
```
python3 benchmark.py \
    --benchmark-type "ORT" \
    --ort-model-path wtiny_fp32/openai/whisper-tiny_encoder_decoder_init_fp32.onnx \
    --precision fp32 \
    --model-size tiny \
    --batch-size 2 \
    --device cpu
```

5) "Whisper decoder-with-past" from ONNX Runtime's custom export, FP32
```
python3 benchmark.py \
    --benchmark-type "ORT" \
    --ort-model-path wtiny_fp32/openai/whisper-tiny_decoder_fp32.onnx \
    --precision fp32 \
    --model-size tiny \
    --batch-size 2 \
    --device cpu
```

6) "Whisper with beam search op" from ONNX Runtime's export, FP32
```
python3 benchmark.py \
    --benchmark-type "ORT" \
    --ort-model-path wtiny_fp32/openai/whisper-tiny_beamsearch.onnx \
    --precision fp32 \
    --model-size tiny \
    --batch-size 2 \
    --device cpu
```
