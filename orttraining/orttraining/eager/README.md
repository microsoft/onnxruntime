# ONNX Runtime Eager Mode Support for PyTorch

## Build Instrcutions

* Build Pytorch with commit: 0834a368181d18c8fd2429614fafca50fb412ce1
* Install the pytorch build to your current environment
* Build onnxruntime, make sure you enable training and python build together with eager mode:

```bash
./build.sh --enable_training --enalbe_pybind --build_eager_mode
```