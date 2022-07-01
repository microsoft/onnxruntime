# Kernel Explorer 
Kernel Explorer is a tool to help develop, test, and profile GPU kernels.

## Example Usage
```
mkdir build
cd build
cmake -D onnxruntime_ROCM_HOME=/opt/rocm ..
cmake --build .
pytest ../kernels --verbose
python ../kernels/vector_add_test.py
```
