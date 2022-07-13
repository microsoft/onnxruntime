<h1><span style="color:red">Note: ONNX Runtime Server has been deprecated.</span></h1>

# How to Use build ONNX Runtime Server for Prediction
ONNX Runtime Server provides an easy way to start an inferencing server for prediction with both HTTP and GRPC endpoints.

The CLI command to build the server is

Default CPU:
```
python3 /onnxruntime/tools/ci_build/build.py --build_dir /onnxruntime/build --config Release --build_server --parallel --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER
```

# How to Use ONNX Runtime Server for Prediction

 The CLI command to start the server is shown below:

```
$ ./onnxruntime_server
Version: <Build number>
Commit ID: <The latest commit ID>

the option '--model_path' is required but missing
Allowed options:
  -h [ --help ]                Shows a help message and exits
  --log_level arg (=info)      Logging level. Allowed options (case sensitive):
                               verbose, info, warning, error, fatal
  --model_path arg             Path to ONNX model
  --address arg (=0.0.0.0)     The base HTTP address
  --http_port arg (=8001)      HTTP port to listen to requests
  --num_http_threads arg (=<# of your cpu cores>) Number of http threads
  --grpc_port arg (=50051)     GRPC port to listen to requests
```

**Note**: The only mandatory argument for the program here is `model_path`

## Start the Server

To host an ONNX model as an inferencing server, simply run:

```
./onnxruntime_server --model_path /<your>/<model>/<path>
```

## HTTP Endpoint

The prediction URL for HTTP endpoint is in this format:

```
http://<your_ip_address>:<port>/v1/models/<your-model-name>/versions/<your-version>:predict
```

**Note**: Since we currently only support one model, the model name and version can be any string length > 0. In the future, model_names and versions will be verified.

### Request and Response Payload

The request and response need to be a protobuf message. The Protobuf definition can be found [here](../server/protobuf/predict.proto).

A protobuf message could have two formats: binary and JSON. Usually the binary payload has better latency, in the meanwhile the JSON format is easy for human readability. 

The HTTP request header field `Content-Type` tells the server how to handle the request and thus it is mandatory for all requests. Requests missing `Content-Type` will be rejected as `400 Bad Request`.

* For `"Content-Type: application/json"`, the payload will be deserialized as JSON string in UTF-8 format
* For `"Content-Type: application/vnd.google.protobuf"`, `"Content-Type: application/x-protobuf"` or `"Content-Type: application/octet-stream"`, the payload will be consumed as protobuf message directly.

Clients can control the response type by setting the request with an `Accept` header field and the server will serialize in your desired format. The choices currently available are the same as the `Content-Type` header field. If this field is not set in the request, the server will use the same type as your request.

### Inferencing

To send a request to the server, you can use any tool which supports making HTTP requests. Here is an example using `curl`:

```
curl  -X POST -d "@predict_request_0.json" -H "Content-Type: application/json" http://127.0.0.1:8001/v1/models/mymodel/versions/3:predict
```

or

```
curl -X POST --data-binary "@predict_request_0.pb" -H "Content-Type: application/octet-stream" -H "Foo: 1234"  http://127.0.0.1:8001/v1/models/mymodel/versions/3:predict
```

### Interactive tutorial notebook

A simple Jupyter notebook demonstrating the usage of ONNX Runtime server to host an ONNX model and perform inferencing can be found [here](https://github.com/onnx/tutorials/blob/master/tutorials/OnnxRuntimeServerSSDModel.ipynb).

## GRPC Endpoint

If you prefer using the GRPC endpoint, the protobuf could be found [here](../server/protobuf/prediction_service.proto). You could generate your client and make a GRPC call to it. To learn more about how to generate the client code and call to the server, please refer to [the tutorials of GRPC](https://grpc.io/docs/tutorials/).

## Advanced Topics

### Number of Worker Threads

You can change this to optimize server utilization. The default is the number of CPU cores on the host machine.

### Request ID and Client Request ID

For easy tracking of requests, we provide the following header fields:

* `x-ms-request-id`: will be in the response header, no matter the request result. It will be a GUID/uuid with dash, e.g. `72b68108-18a4-493c-ac75-d0abd82f0a11`. If the request headers contain this field, the value will be ignored.
* `x-ms-client-request-id`: a field for clients to tracking their requests. The content will persist in the response headers.

### rsyslog Support

If you prefer using an ONNX Runtime Server with [rsyslog](https://www.rsyslog.com/) support([build instruction](https://www.onnxruntime.ai/docs/how-to/build.html#build-onnx-runtime-server-on-linux)), you should be able to see the log in `/var/log/syslog` after the ONNX Runtime Server runs. For detail about how to use rsyslog, please reference [here](https://www.rsyslog.com/category/guides-for-rsyslog/).

## Report Issues

If you see any issues or want to ask questions about the server, please feel free to do so in this repo with the version and commit id from the command line. 

