# C/C++ Samples

The C/C++ samples are indented to be run built and run independently.
Please see the READMEs of the respective samples


`cmake -B build -S . -DONNX_RUNTIME_PATH=path/to/onnxruntime> -DTRTRTX_RUNTIME_PATH=<path/to/TRTRTX/libs> && cmake --build build --config Release`

ONNX_RUNTIME_PATH= should contain the ONNX Runtime headers and the DLLs and TRTRTX_RUNTIME_PATH should container TensorRT RTX libraries.
