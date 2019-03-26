onnxruntime_test_runner [options...] <data_root>
Options:
        -j [models]: Specifies the number of models to run simultaneously.
        -c [runs]: Specifies the number of Session::Run() to invoke simultaneously for each model.
        -n [test_case_name]: Specifies a single test case to run.
        -p [PLANNER_TYPE]: PLANNER_TYPE could be 'seq' or 'simple'. Default: 'simple'.
        -e [EXECUTION_PROVIDER]: EXECUTION_PROVIDER could be 'cpu' or 'cuda'. Default: 'cpu'.
        -h: help

The debug version of this program depends on dbghelp.dll. Please make sure it's in your PATH.

How to run node tests:
1. Install onnx from onnxruntime\cmake\external\onnx
   Steps:
   "C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
   C:
   cd C:\src\onnxruntime\onnxruntime\cmake\external\onnx
   set PATH="C:\Program Files\CMake\bin";C:\python35;C:\python35\scripts;C:\Program Files\CMake\bin;%PATH%
   set PATH=%PATH%;C:\os\onnx\protoc
   set INCLUDE=%INCLUDE%;C:\os\protobuf-2.6.1\src
   set LIB=%LIB%;C:\os\protobuf-2.6.1\vsprojects\x64\Release
   python setup.py install

2. Execute test data generator:
       backend-test-tools generate-data -o <some_empty_folder>
   e.g.
       backend-test-tools generate-data -o C:\testdata
    backend-test-tools is a tool under C:\Python35\Scripts (If your python was installed to C:\Python35)

3. compile onnxruntime_test_runner and run
      onnxruntime_test_runner <test_data_dir>
	e.g.
	  onnxruntime_test_runner C:\testdata\node


How to run model tests:
1. Download test data from VSTS drop
   1) Download drop app from https://aiinfra.artifacts.visualstudio.com/_apis/drop/client/exe
      Unzip the downloaded file and add lib/net45 dir to your PATH
   2) Download the test data by using this command:
      drop get -a -s https://aiinfra.artifacts.visualstudio.com/DefaultCollection -n Lotus/testdata/onnx/model/16 -d C:\testdata
	  You may change C:\testdata to any directory in your disk.
   Full document: https://www.1eswiki.com/wiki/VSTS_Drop

2. compile onnxruntime_test_runner and run
   onnxruntime_test_runner <test_data_dir>
   e.g.
	 onnxruntime_test_runner C:\testdata
