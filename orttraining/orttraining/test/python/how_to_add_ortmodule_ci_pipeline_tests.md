## Getting Started

This is a simple guide on how the ortmodule CI pipeline works and how it can be leveraged.

### The Pipeline

The ortmodule CI pipeline is intended for running tests related to the ```ORTModule``` class.
The pipeline ```yml``` file is defined in [```tools/ci_build/github/azure-pipelines/orttraining-linux-gpu-test-ci-pipeline.yml```](https://github.com/microsoft/onnxruntime/blob/thiagofc/ortmodule-api/tools/ci_build/github/azure-pipelines/orttraining-linux-gpu-test-ci-pipeline.yml).
The pipeline runs on every pull request commit to the branch ```thiagofc/ortmodule```.

## Running Locally

To run the entire set of ortmodule tests locally, run the following command from the build directory:
```sh
python orttraining_ortmodule_tests.py
```

## Adding Tests to the Pipeline

Follow the below steps to add new ortmodule tests that will run in this pipeline.

1. Create a new python file that can be called as a script. Let's call this ```dummy_ortmodule_test.py``` as an example.
2. Make sure this ```dummy_ortmodule_test.py``` can be called and executed using ```python dummy_ortmodule_test.py```.
3. Create a new function in ```orttraining/orttraining/test/python/orttraining_ortmodule_tests.py```
   ```python
   def run_dummy_ortmodule_tests(cwd, log):
       log.debug('Running: Dummy ortmodule tests')

       command = [sys.executable, 'dummy_ortmodule_test.py']

       run_subprocess(command, cwd=cwd, log=log).check_returncode()
   ```
4. Add a call to the ```run_dummy_ortmodule_tests()``` in the ```main()``` function in ```orttraining/orttraining/test/python/orttraining_ortmodule_tests.py```
   ```python
   run_dummy_ortmodule_tests(cwd, log)
   ```
5. Call the ortmodule test suite on a local machine and ensure there are no failures.
   ```sh
   python orttraining_ortmodule_tests.py
   ```

> **Note**: If the test requires multiple ```run_subprocess()``` calls, restructure the test file(s) such that they have a single entry point.

Once the above has been tried and tested, submit a pull request and the tests should be executed in the ortmodule ci pipeline. Make sure to search for ```'Running: Dummy ortmodule tests'``` in the pipeline logs to ensure that the newly added tests were successfully run in the pipeline.
