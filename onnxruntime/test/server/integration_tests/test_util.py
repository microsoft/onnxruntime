# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import base64
import struct
import math
import subprocess
import time
import requests
import json
import datetime
import socket
import errno
import sys
import urllib.request

import predict_pb2
import onnx_ml_pb2
import numpy

def test_log(str):
    print('[Test Log][{0}] {1}'.format(datetime.datetime.now(), str))


def is_process_killed(pid):
    if sys.platform.startswith("win"):
        process_name = 'onnxruntime_host.exe'
        call = 'TASKLIST', '/FI', 'imagename eq {0}'.format(process_name)
        output = subprocess.check_output(call).decode('utf-8')
        print(output)
        last_line = output.strip().split('\r\n')[-1]
        return not last_line.lower().startswith(process_name)
    else:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        else:
            return True

def prepare_mnist_model(target_path):
    # TODO: This need to be replaced by test data on build machine after merged to upstream master. 
    if not os.path.isfile(target_path):
        test_log('Downloading model from blob storage: https://ortsrvdev.blob.core.windows.net/test-data/mnist.onnx to {0}'.format(target_path))
        urllib.request.urlretrieve('https://ortsrvdev.blob.core.windows.net/test-data/mnist.onnx', target_path)
    else:
        test_log('Found mnist model at {0}'.format(target_path))


def decode_base64_string(s, count_and_type):
    b = base64.b64decode(s)
    r = struct.unpack(count_and_type, b)

    return r


def compare_floats(a, b, rel_tol=0.0001, abs_tol=0.0001):
    if not math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
        test_log('Not match with relative tolerance {0} and absolute tolerance {1}: {2} and {3}'.format(rel_tol, abs_tol, a, b))
        return False

    return True


def wait_service_up(server, port, timeout=1):
    s = socket.socket()
    if timeout:
        end = time.time() + timeout

    while True:
        try:
            if timeout:
                next_timeout = end - time.time()
                if next_timeout < 0:
                    return False
                else:
            	    s.settimeout(next_timeout)
            
            s.connect((server, port))
        except socket.timeout as err:
            if timeout:
                return False
        except Exception as err:
            pass
        else:
            s.close()
            return True


def launch_server_app(cmd, server_ip, server_port, wait_server_ready_in_seconds):
    test_log('Launching server app: [{0}]'.format(' '.join(cmd)))
    server_app_proc = subprocess.Popen(cmd)
    test_log('Server app PID: {0}'.format(server_app_proc.pid))
    test_log('Wait up to {0} second(s) for server initialization'.format(wait_server_ready_in_seconds))
    wait_service_up(server_ip, server_port, wait_server_ready_in_seconds)

    return server_app_proc


def shutdown_server_app(server_app_proc, wait_for_server_off_in_seconds):
    if server_app_proc is not None:
        test_log('Shutdown server app')
        server_app_proc.kill()

        while not is_process_killed(server_app_proc.pid):
            server_app_proc.wait(timeout=wait_for_server_off_in_seconds)
            test_log('PID {0} has been killed: {1}'.format(server_app_proc.pid, is_process_killed(server_app_proc.pid)))

        # Additional sleep to make sure the resource has been freed.
        time.sleep(1)

    return True


def make_http_request(url, request_headers, payload):
    test_log('POST Request Started')
    resp = requests.post(url, headers=request_headers, data=payload)
    test_log('POST Request Done')
    return resp


def json_response_validation(cls, resp, expected_resp_json_file):
    cls.assertEqual(resp.status_code, 200)
    cls.assertTrue(resp.headers.get('x-ms-request-id'))
    cls.assertEqual(resp.headers.get('Content-Type'), 'application/json')

    with open(expected_resp_json_file) as f:
        expected_result = json.loads(f.read())

    actual_response = json.loads(resp.content.decode('utf-8'))
    cls.assertTrue(actual_response['outputs'])

    for output in expected_result['outputs'].keys():
        cls.assertTrue(actual_response['outputs'][output])
        cls.assertTrue(actual_response['outputs'][output]['dataType'])
        cls.assertEqual(actual_response['outputs'][output]['dataType'], expected_result['outputs'][output]['dataType'])
        cls.assertTrue(actual_response['outputs'][output]['dims'])
        cls.assertEqual(actual_response['outputs'][output]['dims'], expected_result['outputs'][output]['dims'])
        cls.assertTrue(actual_response['outputs'][output]['rawData'])

        count = 1
        for x in actual_response['outputs'][output]['dims']:
            count = count * int(x)

        actual_array = decode_base64_string(actual_response['outputs'][output]['rawData'], '{0}f'.format(count))
        expected_array = decode_base64_string(expected_result['outputs'][output]['rawData'], '{0}f'.format(count))
        cls.assertEqual(len(actual_array), len(expected_array))
        cls.assertEqual(len(actual_array), count)
        for i in range(0, count):
            cls.assertTrue(compare_floats(actual_array[i], expected_array[i], rel_tol=0.001))


def pb_response_validation(cls, resp, expected_resp_pb_file):
    cls.assertEqual(resp.status_code, 200)
    cls.assertTrue(resp.headers.get('x-ms-request-id'))
    cls.assertEqual(resp.headers.get('Content-Type'), 'application/octet-stream')

    actual_result = predict_pb2.PredictResponse()
    actual_result.ParseFromString(resp.content)

    expected_result = predict_pb2.PredictResponse()
    with open(expected_resp_pb_file, 'rb') as f:
        expected_result.ParseFromString(f.read())

    for k in expected_result.outputs.keys():
        cls.assertEqual(actual_result.outputs[k].data_type, expected_result.outputs[k].data_type)

        count = 1
        for i in range(0, len(expected_result.outputs[k].dims)):
            cls.assertEqual(actual_result.outputs[k].dims[i], expected_result.outputs[k].dims[i])
            count = count * int(actual_result.outputs[k].dims[i])

        actual_array = numpy.frombuffer(actual_result.outputs[k].raw_data, dtype=numpy.float32)
        expected_array = numpy.frombuffer(expected_result.outputs[k].raw_data, dtype=numpy.float32)
        cls.assertEqual(len(actual_array), len(expected_array))
        cls.assertEqual(len(actual_array), count)
        for i in range(0, count):
            cls.assertTrue(compare_floats(actual_array[i], expected_array[i], rel_tol=0.001))
