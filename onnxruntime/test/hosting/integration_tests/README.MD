# ONNX Runtime Hosting Application Integration Tests

## Preparation

Tests validation depends on protobuf generated *_pb2.py. So we need to have a sucessful hosting application build to have it generated in the build folder under `hosting_test` subfolder. The following instruction assume you are in the folder. Otherwise, tests will fail due to `ModuleNotFoundError`.

## Functional Tests

Functional test will be run when build with `--build_hosting --enable_hosting_tests`. To run it separately, here is the command line:

```Bash
/usr/bin/python3 ./test_main.py <hosting_app_path> <mnist_model_path> <test_data_path>
```

## Model Zoo Tests

To run this set of tests, a prepared test data set need to be downloaded from [Azure Blob Storage](https://onnxhostingdev.blob.core.windows.net/testing/hosting_test_data_20190422.zip) and unzip to a folder, e.g. /home/foo/bar/model_zoo_test. It contains:

* ONNX models from [ONNX Model Zoo](https://github.com/onnx/models) with opset 7/8/9.
* HTTP request json and protobuf files
* Expected response json and protobuf files

If you only need the request and response data. Here is the [link](https://onnxhostingdev.blob.core.windows.net/testing/hosting_test_data_req_resp_only.zip) to download.

To run the full model zoo tests, here is the command line:

```Bash
/usr/bin/python3 ./model_zoo_tests.py <hosting_app_path> <model_path> <test_data_path>
```

For example:

```Bash
/usr/bin/python3 ./model_zoo_tests.py /some/where/hosting_app /home/foo/bar/model_zoo_test /home/foo/bar/model_zoo_test
```

If those models are in different folder but in the same structure as the test data, you could also do 

```Bash
/usr/bin/python3 ./model_zoo_tests.py /some/where/hosting_app /home/my/models/ /home/foo/bar/model_zoo_test/
```

All tests are running in sequential order.