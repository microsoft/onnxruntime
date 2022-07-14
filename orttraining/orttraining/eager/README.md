# ONNX Runtime Eager Mode Support for PyTorch

## Build and test
1. sudo apt-get install python3-dev
2. Upgrade CMAKE to > version 3.18.1.
3. Create and activate a virtual environment:
		a. python3 -m venv ~/venvs/onnxruntime
		b. source ~/venvs/onnxruntime/bin/activate
4. pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
5. pip3 install parameterized
6. pip install packaging wheel
7. Sudo apt-get install mpich libmpich-dev libcomp-11-dev (or use '--use_mpi=False' in the build command)
8. Sudo apt-get update (optional if you think software is out of date and it might show out dated stuff to remove)
9. Clone microsoft/onnxruntime

To build, go to the root of the onnxruntime repository and run:
./build.sh --cmake_generator Ninja --config Debug --build_shared_lib --parallel --build_eager_mode --enable_training --enable_pybind  --build_wheel --skip_tests

To run the eager tests:
 PYTHONPATH=~/{onnxruntime repo}/build/Linux/Debug python3 ~/{onnxruntime repo}/orttraining/orttraining/eager/test/ort_ops.py

## Mapping aten cpp namespace functions to onnx ops

Useful links
- [Onnx Op Schema](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
- [Aten native ops](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml)

For mapping existing aten ops to onnx files start in `orttraining/orttraining/eager/opgen/opgen/atenops.py.` This file
drives the generator which the build runs and produces `orttraining/orttraining/eager/ort_aten.g.cpp`. Looking at the
generated code will be helpful to understand the aten op signature and how onnx ops are invoked.


In the simple case, add the aten op and the mapped onnx op to the `hand_implemented` list. As
an example `"aten::t": Transpose("self"),` maps the aten transpose function `t` to the onnx op `Transpose` and maps the
`self` input param of `t` to the first argument of `Transpose`.


Often mapping ten to onnx is not one to one, so composing is supported. Example
`"aten::zeros_like": ConstantOfShape(Shape("self")),`.


In some case more data shaping is required, so only the signature should be created such as `"aten::equal": SignatureOnly(),`.
The implementation is then added to `orttraining/orttraining/eager/ort_aten.cpp`

Please add tests for all ops. Tests are defined in `orttraining/orttraining/eager/test/ort_ops.py`

## Decisions worth noting
- Resizing the output tensor. Aten supports resizing any out tensor but prints a warning this is depracated and support
will end. With that in mind, we have decided to error in `resize_output` if the output tensor is not empty or already
the right shape.
