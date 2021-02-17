## Installing ORT Test Dependencies Locally

To install all development test dependencies for ONNX Runtime training, run the below two commands:


Due to a [bug on DeepSpeed](https://github.com/microsoft/DeepSpeed/issues/663), we need a two step process for installing dependencies ORTModule dependencies.

1. Install first set of dependencies:

   ```sh
   pip install -r tools/ci_build/github/linux/docker/scripts/training/requirements.txt

2. Install second set of dependencies for ortmodule:
   ```sh
   pip install -r tools/ci_build/github/linux/docker/scripts/training/ortmodule/requirements.txt
