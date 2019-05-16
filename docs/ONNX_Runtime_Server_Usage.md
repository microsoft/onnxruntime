<h1><span style="color:red">Note: ONNX Runtime Server is still in beta state. It's currently not ready for production environments.</span></h1>

# How to Use ONNX Runtime Server REST API for Prediction

ONNX Runtime Server provides a REST API for prediction. The goal of the project is to make it easy to "host" any ONNX model as a RESTful service. The CLI command to start the service is shown below:

```
$ ./onnxruntime_server
the option '--model_path' is required but missing
Allowed options:
  -h [ --help ]                Shows a help message and exits
  --log_level arg (=info)      Logging level. Allowed options (case sensitive):
                               verbose, info, warning, error, fatal
  --model_path arg             Path to ONNX model
  --address arg (=0.0.0.0)     The base HTTP address
  --http_port arg (=8001)      HTTP port to listen to requests
  --num_http_threads arg (=<# of your cpu cores>) Number of http threads


```

Note: The only mandatory argument for the program here is `model_path`

## Start the Server

To host an ONNX model as a REST API server, run:

```
./onnxruntime_server --model_path /<your>/<model>/<path>
```

The prediction URL is in this format:

```
http://<your_ip_address>:<port>/v1/models/<your-model-name>/versions/<your-version>:predict
```

**Note**: Since we currently only support one model, the model name and version can be any string length > 0. In the future, model_names and versions will be verified.

## Request and Response Payload

An HTTP request can be a Protobuf message in two formats: binary or JSON. The HTTP request header field `Content-Type` tells the server how to handle the request and thus it is mandatory for all requests. Requests missing `Content-Type` will be rejected as `400 Bad Request`.

* For `"Content-Type: application/json"`, the payload will be deserialized as JSON string in UTF-8 format
* For `"Content-Type: application/vnd.google.protobuf"`, `"Content-Type: application/x-protobuf"` or `"Content-Type: application/octet-stream"`, the payload will be consumed as protobuf message directly.

The Protobuf definition can be found [here](https://github.com/Microsoft/onnxruntime/blob/master/onnxruntime/server/protobuf/predict.proto).

## Inferencing

To send a request to the server, you can use any tool which supports making HTTP requests. Here is an example using `curl`:

```
curl  -X POST -d "@predict_request_0.json" -H "Content-Type: application/json" http://127.0.0.1:8001/v1/models/mymodel/versions/3:predict
```

or

```
curl -X POST --data-binary "@predict_request_0.pb" -H "Content-Type: application/octet-stream" -H "Foo: 1234"  http://127.0.0.1:8001/v1/models/mymodel/versions/3:predict
```

Clients can control the response type by setting the request with an `Accept` header field and the server will serialize in your desired format. The choices currently available are the same as the `Content-Type` header field.

## Advanced Topics

### Number of HTTP Threads

You can change this to optimize server utilization. The default is the number of CPU cores on the host machine.

### Request ID and Client Request ID

For easy tracking of requests, we provide the following header fields:

* `x-ms-request-id`: will be in the response header, no matter the request result. It will be a GUID/uuid with dash, e.g. `72b68108-18a4-493c-ac75-d0abd82f0a11`. If the request headers contain this field, the value will be ignored.
* `x-ms-client-request-id`: a field for clients to tracking their requests. The content will persist in the response headers.

Here is an example of a client sending a request:

#### Client Side

```
$ curl -v -X POST --data-binary "@predict_request_0.pb" -H "Content-Type: application/octet-stream" -H "Foo: 1234" -H "x-ms-client-request-id: my-request-001" -H "Accept: application/json"  http://127.0.0.1:8001/v1/models/mymodel/versions/3:predict
Note: Unnecessary use of -X or --request, POST is already inferred.
*   Trying 127.0.0.1...
* Connected to 127.0.0.1 (127.0.0.1) port 8001 (#0)
> POST /v1/models/mymodel/versions/3:predict HTTP/1.1
> Host: 127.0.0.1:8001
> User-Agent: curl/7.47.0
> Content-Type: application/octet-stream
> x-ms-client-request-id: my-request-001
> Accept: application/json
> Content-Length: 3179
> Expect: 100-continue
> 
* Done waiting for 100-continue
* We are completely uploaded and fine
< HTTP/1.1 200 OK
< Content-Type: application/json
< x-ms-request-id: 72b68108-18a4-493c-ac75-d0abd82f0a11
< x-ms-client-request-id: my-request-001
< Content-Length: 159
< 
* Connection #0 to host 127.0.0.1 left intact
{"outputs":{"Sample_Output_Name":{"dims":["1","10"],"dataType":1,"rawData":"6OpzRFquGsSFdM1FyAEnRFtRZcRa9NDEUBj0xI4ydsJIS0LE//CzxA==","dataLocation":"DEFAULT"}}}%
```

#### Server Side

And here is what the output on the server side looks like with logging level of verbose:

```
2019-04-04 23:48:26.395200744 [V:onnxruntime:72b68108-18a4-493c-ac75-d0abd82f0a11, predict_request_handler.cc:40 Predict] Name: mymodel Version: 3 Action: predict
2019-04-04 23:48:26.395289437 [V:onnxruntime:72b68108-18a4-493c-ac75-d0abd82f0a11, predict_request_handler.cc:46 Predict] x-ms-client-request-id: [my-request-001]
2019-04-04 23:48:26.395540707 [I:onnxruntime:InferenceSession, inference_session.cc:736 Run] Running with tag: 72b68108-18a4-493c-ac75-d0abd82f0a11
2019-04-04 23:48:26.395596858 [V:VLOG1:72b68108-18a4-493c-ac75-d0abd82f0a11, inference_session.cc:976 CreateLoggerForRun] Created logger for run with id of 72b68108-18a4-493c-ac75-d0abd82f0a11
2019-04-04 23:48:26.395731391 [I:onnxruntime:72b68108-18a4-493c-ac75-d0abd82f0a11, sequential_executor.cc:42 Execute] Begin execution
2019-04-04 23:48:26.395763319 [V:VLOG1:72b68108-18a4-493c-ac75-d0abd82f0a11, sequential_executor.cc:45 Execute] Size of execution plan vector: 12
2019-04-04 23:48:26.396228981 [V:VLOG1:72b68108-18a4-493c-ac75-d0abd82f0a11, sequential_executor.cc:156 Execute] Releasing node ML values after computing kernel: Convolution28
2019-04-04 23:48:26.396580161 [V:VLOG1:72b68108-18a4-493c-ac75-d0abd82f0a11, sequential_executor.cc:156 Execute] Releasing node ML values after computing kernel: Plus30
2019-04-04 23:48:26.396623732 [V:VLOG1:72b68108-18a4-493c-ac75-d0abd82f0a11, sequential_executor.cc:197 ReleaseNodeMLValues] Releasing mlvalue with index: 10
2019-04-04 23:48:26.396878822 [V:VLOG1:72b68108-18a4-493c-ac75-d0abd82f0a11, sequential_executor.cc:156 Execute] Releasing node ML values after computing kernel: ReLU32
2019-04-04 23:48:26.397091882 [V:VLOG1:72b68108-18a4-493c-ac75-d0abd82f0a11, sequential_executor.cc:156 Execute] Releasing node ML values after computing kernel: Pooling66
2019-04-04 23:48:26.397126243 [V:VLOG1:72b68108-18a4-493c-ac75-d0abd82f0a11, sequential_executor.cc:197 ReleaseNodeMLValues] Releasing mlvalue with index: 11
2019-04-04 23:48:26.397772701 [V:VLOG1:72b68108-18a4-493c-ac75-d0abd82f0a11, sequential_executor.cc:156 Execute] Releasing node ML values after computing kernel: Convolution110
2019-04-04 23:48:26.397818174 [V:VLOG1:72b68108-18a4-493c-ac75-d0abd82f0a11, sequential_executor.cc:197 ReleaseNodeMLValues] Releasing mlvalue with index: 13
2019-04-04 23:48:26.398060592 [V:VLOG1:72b68108-18a4-493c-ac75-d0abd82f0a11, sequential_executor.cc:156 Execute] Releasing node ML values after computing kernel: Plus112
2019-04-04 23:48:26.398095300 [V:VLOG1:72b68108-18a4-493c-ac75-d0abd82f0a11, sequential_executor.cc:197 ReleaseNodeMLValues] Releasing mlvalue with index: 14
2019-04-04 23:48:26.398257563 [V:VLOG1:72b68108-18a4-493c-ac75-d0abd82f0a11, sequential_executor.cc:156 Execute] Releasing node ML values after computing kernel: ReLU114
2019-04-04 23:48:26.398426740 [V:VLOG1:72b68108-18a4-493c-ac75-d0abd82f0a11, sequential_executor.cc:156 Execute] Releasing node ML values after computing kernel: Pooling160
2019-04-04 23:48:26.398466031 [V:VLOG1:72b68108-18a4-493c-ac75-d0abd82f0a11, sequential_executor.cc:197 ReleaseNodeMLValues] Releasing mlvalue with index: 15
2019-04-04 23:48:26.398542823 [V:VLOG1:72b68108-18a4-493c-ac75-d0abd82f0a11, sequential_executor.cc:156 Execute] Releasing node ML values after computing kernel: Times212_reshape0
2019-04-04 23:48:26.398599687 [V:VLOG1:72b68108-18a4-493c-ac75-d0abd82f0a11, sequential_executor.cc:156 Execute] Releasing node ML values after computing kernel: Times212_reshape1
2019-04-04 23:48:26.398692631 [V:VLOG1:72b68108-18a4-493c-ac75-d0abd82f0a11, sequential_executor.cc:156 Execute] Releasing node ML values after computing kernel: Times212
2019-04-04 23:48:26.398731471 [V:VLOG1:72b68108-18a4-493c-ac75-d0abd82f0a11, sequential_executor.cc:197 ReleaseNodeMLValues] Releasing mlvalue with index: 17
2019-04-04 23:48:26.398832735 [V:VLOG1:72b68108-18a4-493c-ac75-d0abd82f0a11, sequential_executor.cc:156 Execute] Releasing node ML values after computing kernel: Plus214
2019-04-04 23:48:26.398873229 [V:VLOG1:72b68108-18a4-493c-ac75-d0abd82f0a11, sequential_executor.cc:197 ReleaseNodeMLValues] Releasing mlvalue with index: 19
2019-04-04 23:48:26.398922929 [V:VLOG1:72b68108-18a4-493c-ac75-d0abd82f0a11, sequential_executor.cc:160 Execute] Fetching output.
2019-04-04 23:48:26.398956560 [V:VLOG1:72b68108-18a4-493c-ac75-d0abd82f0a11, sequential_executor.cc:163 Execute] Done with execution.
```