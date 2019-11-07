# ONNXRuntime Performance Test

This tool provides the performance results using the ONNX Runtime with the specific execution provider to run the inference for a given model using the sample input test data. This tool can provide a reliable measurement for the inference latency usign ONNX Runtime on the device. The options to use with the tool are listed below:

`onnxruntime_perf_test [options...] model_path result_file`

Options:

	-A: Disable memory arena.
	
	-M: Disable memory pattern.
	
	-P: Use parallel executor instead of sequential executor.
	
	-c: [parallel runs]: Specifies the (max) number of runs to invoke simultaneously. Default:1.
	
	-e: [cpu|cuda|mkldnn|tensorrt|ngraph|openvino|nuphar|acl]: Specifies the execution provider 'cpu','cuda','mkldnn','tensorrt', 'ngraph', 'openvino', 'nuphar' or 'acl'. Default is 'cpu'.
        
	-m: [test_mode]: Specifies the test mode. Value coulde be 'duration' or 'times'. Provide 'duration' to run the test for a fix duration, and 'times' to repeated for a certain times. Default:'duration'.
        
	-o: [optimization level]: Default is 1. Valid values are 0 (disable), 1 (basic), 2 (extended), 99 (all). Please see __onnxruntime_c_api.h__ (enum GraphOptimizationLevel) for the full list of all optimization levels.
	
	-u: [path to save optimized model]: Default is empty so no optimized model would be saved.
	
	-p: [profile_file]: Specifies the profile name to enable profiling and dump the profile data to the file.
	
	-r: [repeated_times]: Specifies the repeated times if running in 'times' test mode.Default:1000.
        
	-s: Show statistics result, like P75, P90.

	-t: [seconds_to_run]: Specifies the seconds to run for 'duration' mode. Default:600.
        
	-v: Show verbose information.
        
	-x: [intra_op_num_threads]: Sets the number of threads used to parallelize the execution within nodes. A value of 0 means the test will auto-select a default. Must >=0.
	
	-y: [inter_op_num_threads]: Sets the number of threads used to parallelize the execution of the graph (across nodes), A value of 0 means the test will auto-select a default. Must >=0.
	
	-h: help.

Model path and input data dependency:
    Performance test uses the same input structure as *onnx_test_runner* tool. It requrires the directory trees as below:

    --ModelName
        --test_data_set_0
            --input0.pb
        --test_data_set_2
            --input0.pb
        --model.onnx
    
The path of model.onnx needs to be provided as `<model_path>` argument.

__Sample output__ from the tool will look something like this:

	Total time cost:58.8053
	Total iterations:1000
	Average time cost:58.8053 ms
	Total run time:58.8102 s
	Min Latency is 0.0559777sec
	Max Latency is 0.0623472sec
	P50 Latency is 0.0587108sec
	P90 Latency is 0.0599845sec
	P95 Latency is 0.0605676sec
	P99 Latency is 0.0619517sec
	P999 Latency is 0.0623472se
