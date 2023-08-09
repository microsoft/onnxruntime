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
