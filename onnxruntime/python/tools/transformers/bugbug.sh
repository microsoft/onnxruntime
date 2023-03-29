python -m onnxruntime.transformers.benchmark -m bert-large-uncased -b 1 4 8 16 32 64 128 -s 512 -t 1000 -o by_script -g -p fp16 -i 3 --use_mask_index
