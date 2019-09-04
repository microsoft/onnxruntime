This folder contains scripts for Nuphar:

* cntk_converter.py

Converts CNTK model to ONNX and generate test data, requires CNTK python wheels to run

* create_shared.cmd/sh

Generates JIT dll from where env NUPHAR_CACHE_PATH is set, to reduce JIT cost at runtime

* model_editor.py

Edits models like LSTM to Scan for Nuphar to run

* model_quantizer.py

Quantize MatMul in model dynamically wrt. input

* rnn_benchmark.py

Benchmark for LSTM/GRU/RNN with model_editor and model_quantizer to show Nuphar's speed up for those models

* symbolic_shape_infer.py

Run symbolic shape inference with sympy. Nuphar relies on shape inference to run efficiently.