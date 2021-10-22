## Getting Started

This is a simple guide on how the distributed CI pipeline works and how it can be leveraged.

### The Pipeline

The distributed CI pipeline is intended for running tests that require a distributed environment (for example, tests that need to be run with ```mpirun```).
The pipeline ```yml``` file is defined in [```tools/ci_build/github/azure-pipelines/orttraining-linux-gpu-distributed-test-ci-pipeline.yml```](https://github.com/microsoft/onnxruntime/blob/master/tools/ci_build/github/azure-pipelines/orttraining-linux-gpu-distributed-test-ci-pipeline.yml).
The pipeline runs on every pull request commit under the [```orttraining-distributed```](https://dev.azure.com/onnxruntime/onnxruntime/_build?definitionId=140&_a=summary) check.
The flow of events in the pipeline are:

1. Clone the git repository and checkout the branch that needs to run for the CI (the pull request).
2. Build the docker container installing all dependencies that are needed for the distributed tests (for example, ```open-mpi```)
3. Run all tests defined in the file [```orttraining/orttraining/test/python/orttraining_distributed_tests.py```](https://github.com/microsoft/onnxruntime/blob/master/orttraining/orttraining/test/python/orttraining_distributed_tests.py) through the script [```orttraining/orttraining/test/python/launch_test.py```](https://github.com/microsoft/onnxruntime/blob/master/orttraining/orttraining/test/python/launch_test.py)
4. Report the status of the tests.

## Running Locally

To run the entire set of distributed tests locally, run the following command from the build directory:
```sh
python orttraining_distributed_tests.py
```

> **Note**: these set of tests can only be run on a machine with multiple gpus and the test will terminate if the number of gpus is less than 2.

## Adding Tests to the Pipeline

Follow the below steps to add new distributed tests that will run in this pipeline.

1. Create a new python file that can be called as a script. Let's call this ```dummy_distributed_test.py``` as an example.
2. Make sure this ```dummy_distributed_test.py``` can be called and executed using either ```python dummy_distributed_test.py``` or using ```mpirun -n <num_gpus> -x NCCL_DEBUG=INFO python dummy_distributed_test.py```. A real example of such a test file is [```orttraining/orttraining/test/python/orttraining_test_checkpoint.py```](https://github.com/microsoft/onnxruntime/blob/master/orttraining/orttraining/test/python/orttraining_test_checkpoint.py).
3. Create a new function in ```orttraining/orttraining/test/python/orttraining_distributed_tests.py```
   ```python
   def run_dummy_distributed_tests(cwd, log):
       log.debug('Running: Dummy distributed tests')

       command = [sys.executable, 'dummy_distributed_test.py']

       run_subprocess(command, cwd=cwd, log=log).check_returncode()
   ```
   Refer to ```run_checkpoint_tests()``` for an example.
4. Add a call to the ```run_dummy_distributed_tests()``` in the ```main()``` function in ```orttraining/orttraining/test/python/orttraining_distributed_tests.py```
   ```python
   run_dummy_distributed_tests(cwd, log)
   ```
   Refer to ```run_checkpoint_tests()``` for an example.
5. Call the distributed test suite on a local machine and ensure there are no failures.
   ```sh
   python orttraining_distributed_tests.py
   ```

> **Note**: If the test requires multiple ```run_subprocess()``` calls, restructure the test file(s) such that they have a single entry point. Refer to ```orttraining/orttraining/test/python/orttraining_test_checkpoint.py``` for an example.

Once the above has been tried and tested, submit a pull request and the tests should be executed in the [distributed CI pipeline](https://dev.azure.com/onnxruntime/onnxruntime/_build?definitionId=140&_a=summary). Make sure to search for ```'Running: Dummy distributed tests'``` in the pipeline logs to ensure that the newly added tests were successfully run in the pipeline.
