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

>Note: pip runs through the requirements.txt twice. First time downloading the packages and running each package's setup.py and the second time actually installing it. ```deepspeed``` has ```import torch``` in its setup.py. So having deepspeed as a part of requirements.txt even though torch package has already been collected results in an error. Work around is to install deepspeed separately after the requirements.txt is installed through secondary/requirements.txt. Which is why we have a two step process for installing dependencies.