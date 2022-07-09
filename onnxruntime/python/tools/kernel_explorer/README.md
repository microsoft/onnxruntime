# Kernel Explorer 
Kernel Explorer hooks up GPU kernel code with a Python frontend to help develop, test, profile, and auto-tune GPU kernels.

## Example Usage
```
mkdir build
cd build
cmake -D onnxruntime_ROCM_HOME=/opt/rocm ..
cmake --build .
pytest ../kernels --verbose
python ../kernels/vector_add_test.py
```
