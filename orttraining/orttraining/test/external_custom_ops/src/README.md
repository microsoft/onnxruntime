This folder contains an example to demonstrate external custom operation schema specification use-case.
This use-case requires compiling onnxruntime with --enable_external_custom_op_schemas.
This flow is tested only on Ubuntu and it doesn't work on Windows. Steps to run the example.
1. Build onnxruntime with --build_wheel and --enable_external_custom_op_schemas
2. Install onnxruntime wheel in python environment
3. Install custom ops using "python3 -m pip install ." command in this folder
4. Test using "python3 test.py"