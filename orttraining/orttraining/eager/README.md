# ONNX Runtime Eager Mode Support for PyTorch

* Until the ONNX Runtime Eager Mode backend is merged in PyTorch upstream ([PR #58248](https://github.com/pytorch/pytorch/pull/58248)), build PyTorch from [abock/pytorch](https://github.com/abock/pytorch/tree/dev/abock/ort-backend) (branch: `dev/abock/ort-backend`):

  ```bash
  $ git clone https://github.com/abock/pytorch -b dev/abock/ort-backend --recursive
  $ cd pytorch
  $ python setup.py develop --user
  ```

* If not build as noted above, ensure the eager-mode enabled PyTorch build is installed in the current environment.

* Build ONNX Runtime, making sure you enable training and python build together with eager mode:

  ```bash
  ./build.sh --cmake_generator=Ninja --enable_training --enable_pybind --build_eager_mode
  ```