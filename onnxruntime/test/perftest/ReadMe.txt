onnxruntime_perf_test [options...] <model_path> <result_file>

Options:
        -m [test_mode]: Specifies the test mode. Value coulde be 'duration' or 'times'. Provide 'duration' to run the test for a fix duration, and 'times' to repeated for a certain times. Default:'duration'.
        -e [cpu|cuda]: Specifies the provider 'cpu' or 'cuda'. Default:'cpu'.\n"
        -r [repeated_times]: Specifies the repeated times if running in 'times' test mode.Default:1000.
        -t [seconds_to_run]: Specifies the seconds to run for 'duration' mode. Default:600.
        -p [profile_file]: Specifies the profile name to enable profiling and dump the profile data to the file.
        -v: Show verbose information
        -h: help

Model path and input data dependency:
    Performance test uses the same input structure as onnx_test_runner. It requrires the directory trees as below: 
    
    --ModelName
        --test_data_set_0
            --input0.pb
        --test_data_set_2
	        --input0.pb
        --model.onnx
    The path of model.onnx needs to be provided as <model_path> argument.

How to download sample test data from VSTS drop:
   1) Download drop app from https://aiinfra.artifacts.visualstudio.com/_apis/drop/client/exe
      Unzip the downloaded file and add lib/net45 dir to your PATH
   2) Download the test data by using this command:
      drop get -a -s https://aiinfra.artifacts.visualstudio.com/DefaultCollection -n Lotus/testdata/onnx/model/16 -d C:\testdata
	  You may change C:\testdata to any directory in your disk.
   Full document: https://www.1eswiki.com/wiki/VSTS_Drop


How to run performance tests for batch of models:
   1) Download the driver by using this command:
      drop get -a -s https://aiinfra.artifacts.visualstudio.com/DefaultCollection -n Lotus/test/perfdriver/$(perfTestDriverVersion) -d C:\perfdriver
      You may change C:\perfdriver to any directory in your disk.
      Currently, the $(perfTestDriverVersion) is 6
   2) Run the PerfTestDriver.py under python environment with proper arguments.