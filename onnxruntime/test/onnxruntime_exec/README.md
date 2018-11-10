# Compile and Build Onnxruntime Executor
```
>> bld.bat
```

Assuming the command was executed from (PATH_2_MONTREAL)\tests\onnxruntime_exec folder. This command will retrieve the latest version of Onnxruntime and build it in debug mode.
Once build is done, open the onnxruntime_exec.sln and build it in debug mode for debugging.

# Loading Model
```
>> loturt_exec.exe -m modelfile
```

The above command will load the model from modelfile and return the status of model loading [success/fail].

# Predicting with Model
```
>> loturt_exec.exe -m modelfile [-t testfile]
```

[-t testfile] is optional. When specified, the loturt_exec.exe will compute prediction/score/probability/etc. on every row of the file using the model and output it to stdout in CSV form.
The format of input testfile is CSV without any header. As of now 11/14, only ints/floats are supported type in CSV.


# Model Debugging
* Install python runtime for project Montreal.

    ```
    >> Powershell ./build.ps1
    ```

    * The first run will take the longest time. This is because we are setting up a python environment for the first time. Subsequent times will be quicker.
    * The python environment is located at runtime\Python
    * The build script will create and install the winmltools python package. To update this package, you will need to re-run the build script or copy the changed files into python\lib\site-packages\winmltools.

* Install CoreMLTools for Python 3:
	```
	mkdir coremltools
	git clone --recursive https://github.com/apple/coremltools.git
	runtime\python\python.exe -m pip install -e coremltools/
    ```

* Run the test in local mode (don't run the following script in 'mode' other than local model for your testing.). You can use any python enviroment as long as winmltools are there.

    ```
    >> $(PATH_2_MONTREAL)\runtime\Python\python.exe fn_model_conversion.py -m local -j TAEF_JSONS -s MODEL_SAVE_PATH
    ```

* Investigate the stdout of the above script to see where the model has problem. The model file is save in MODEL_SAVE_PATH. If its MODEL_LOADING_FAILURE or PREDICTION_FAILURE, run it through loturt_exec.exe. Unless loturt_exec.exe returns some prediction the model has problem.


* Please note that **loturt_exec.exe is currently not working on Image models.**

* Get text representation of coreml model on console
	```
    tests\scrtips\model_viewer_coreml.py model.mlmodel
    ```

* Get text representation of winml(onnx) model on console.
	```
    tests\scrtips\model_viewer.py model.onnx
    ```
