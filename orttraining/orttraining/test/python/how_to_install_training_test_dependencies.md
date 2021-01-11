## Installing ORT Test Dependencies Locally

To install all development test dependencies for ONNX Runtime training, run the below two commands:

1. Install first set of dependencies:

   ```sh
   pip install -r tools/ci_build/github/linux/docker/scripts/training/requirements.txt
   ```
2. Install second set of dependencies:

   ```sh
   pip install -r tools/ci_build/github/linux/docker/scripts/training/secondary/requirements.txt
   ```
