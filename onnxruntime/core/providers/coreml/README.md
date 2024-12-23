# Background
1. ANE only supports float16
2. ANE is faster a lot than CPU and GPU
3. NN supports convert to float16 internally so float model can be applied to ANE
4. Some fp16 operators can't be fallback to CPU EP some times
5. ANE compiler for float16 graph is super slower, so we have to depend on model cache.

# How the current implementation works
1. rewrite all input/output datatype from float32 to float16, AddOperationInput/AddOperationOutput/AddIntermediateOperationOutput
2. rewrite all weights from float32 to float16 during create ML tensor
3. special handling for some operators like cast
4. special handling for some operators like bn, which has scalar float32 input
5. wrap subgraph with cast(float32->float16) for all float32 input/output
6. add provider option to enable auto cast

# different solutions
1. wrap subgraph with cast(float32->float16) for all float32 input/output
    1. pros: we don't need to change the model input/output datatype
    2. pros: we don't need to convert the input/output data in CPU, cast node can run in GPU
    3. pros: we don't need to record those float32 input/output for special cast
    1. cons: a new node is added to the graph, we need to maintain their naming mapping.
2. treat float32 as float16, just like what we do for int64
    1. cons: we need to do th cast in the entry of coreml predict
# benchmark
yolov11.onnx MacM1
run float16 model, 5ms
run float32 model, 10ms
run float32 model with auto cast, 3ms (reason is float16 model keeps resize node as float32)

## a weird issue
with solution 2:
    the time cost with float32 model and enable auto cast is 5ms
while with solution 1:
    the time cost with float32 model and enable auto cast is 3ms

This shouldn't be happened, because the only difference is the cast node is added to the graph or manually converting the input/output data in CPU.
