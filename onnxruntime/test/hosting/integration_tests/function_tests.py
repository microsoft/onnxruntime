# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import subprocess
import time
import os
import requests
import json
import numpy

import test_util
import onnx_ml_pb2
import predict_pb2

class HttpJsonPayloadTests(unittest.TestCase):
    server_ip = '127.0.0.1'
    server_port = 54321
    url_pattern = 'http://{0}:{1}/v1/models/{2}/versions/{3}:predict'
    hosting_app_path = ''
    test_data_path = ''
    model_path = ''
    log_level = 'verbose'
    hosting_app_proc = None
    wait_server_ready_in_seconds = 1

    @classmethod
    def setUpClass(cls):
        cmd = [cls.hosting_app_path, '--http_port', str(cls.server_port), '--model_path', os.path.join(cls.model_path, 'mnist.onnx'), '--logging_level', cls.log_level]
        print('Launching hosting app: [{0}]'.format(' '.join(cmd)))
        cls.hosting_app_proc = subprocess.Popen(cmd)
        print('Hosting app PID: {0}'.format(cls.hosting_app_proc.pid))
        print('Sleep {0} second(s) to wait for server initialization'.format(cls.wait_server_ready_in_seconds))
        time.sleep(cls.wait_server_ready_in_seconds)


    @classmethod
    def tearDownClass(cls):
        print('Shutdown hosting app')
        cls.hosting_app_proc.kill()
        print('PID {0} has been killed: {1}'.format(cls.hosting_app_proc.pid, test_util.is_process_killed(cls.hosting_app_proc.pid)))


    def test_mnist_happy_path(self):
        input_data_file = os.path.join(self.test_data_path, 'mnist_test_data_set_0_input.json')
        output_data_file = os.path.join(self.test_data_path, 'mnist_test_data_set_0_output.json')

        with open(input_data_file, 'r') as f:
            request_payload = f.read()

        with open(output_data_file, 'r') as f:
            expected_response_json = f.read()
            expected_response = json.loads(expected_response_json)

        request_headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'x-ms-client-request-id': 'This~is~my~id'
        }

        url = self.url_pattern.format(self.server_ip, self.server_port, 'default_model', 12345)
        print(url)
        r = requests.post(url, headers=request_headers, data=request_payload)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers.get('Content-Type'), 'application/json')
        self.assertTrue(r.headers.get('x-ms-request-id'))
        self.assertEqual(r.headers.get('x-ms-client-request-id'), 'This~is~my~id')

        actual_response = json.loads(r.content.decode('utf-8'))

        # Note:
        # The 'dims' field is defined as "repeated int64" in protobuf.
        # When it is serialized to JSON, all int64/fixed64/uint64 numbers are converted to string
        # Reference: https://developers.google.com/protocol-buffers/docs/proto3#json

        self.assertTrue(actual_response['outputs'])
        self.assertTrue(actual_response['outputs']['Plus214_Output_0'])
        self.assertTrue(actual_response['outputs']['Plus214_Output_0']['dims'])
        self.assertEqual(actual_response['outputs']['Plus214_Output_0']['dims'], ['1', '10'])
        self.assertTrue(actual_response['outputs']['Plus214_Output_0']['dataType'])
        self.assertEqual(actual_response['outputs']['Plus214_Output_0']['dataType'], 1)
        self.assertTrue(actual_response['outputs']['Plus214_Output_0']['rawData'])
        actual_data = test_util.decode_base64_string(actual_response['outputs']['Plus214_Output_0']['rawData'], '10f')
        expected_data = test_util.decode_base64_string(expected_response['outputs']['Plus214_Output_0']['rawData'], '10f')

        for i in range(0, 10):
            self.assertTrue(test_util.compare_floats(actual_data[i], expected_data[i]))


    def test_mnist_invalid_url(self):
        url = self.url_pattern.format(self.server_ip, self.server_port, 'default_model', -1)
        print(url)

        request_headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        r = requests.post(url, headers=request_headers, data={'foo': 'bar'})
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.headers.get('Content-Type'), 'application/json')
        self.assertTrue(r.headers.get('x-ms-request-id'))


    def test_mnist_invalid_content_type(self):
        input_data_file = os.path.join(self.test_data_path, 'mnist_test_data_set_0_input.json')
        url = self.url_pattern.format(self.server_ip, self.server_port, 'default_model', 12345)
        print(url)

        request_headers = {
            'Content-Type': 'application/abc',
            'Accept': 'application/json',
            'x-ms-client-request-id': 'This~is~my~id'
        }

        with open(input_data_file, 'r') as f:
            request_payload = f.read()

        r = requests.post(url, headers=request_headers, data=request_payload)
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.headers.get('Content-Type'), 'application/json')
        self.assertTrue(r.headers.get('x-ms-request-id'))
        self.assertEqual(r.headers.get('x-ms-client-request-id'), 'This~is~my~id')
        self.assertEqual(r.content.decode('utf-8'), '{"error_code": 400, "error_message": "Missing or unknown \'Content-Type\' header field in the request"}\n')


    def test_mnist_missing_content_type(self):
        input_data_file = os.path.join(self.test_data_path, 'mnist_test_data_set_0_input.json')
        url = self.url_pattern.format(self.server_ip, self.server_port, 'default_model', 12345)
        print(url)

        request_headers = {
            'Accept': 'application/json'
        }

        with open(input_data_file, 'r') as f:
            request_payload = f.read()

        r = requests.post(url, headers=request_headers, data=request_payload)
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.headers.get('Content-Type'), 'application/json')
        self.assertTrue(r.headers.get('x-ms-request-id'))
        self.assertEqual(r.content.decode('utf-8'), '{"error_code": 400, "error_message": "Missing or unknown \'Content-Type\' header field in the request"}\n')


class HttpProtobufPayloadTests(unittest.TestCase):
    server_ip = '127.0.0.1'
    server_port = 54321
    url_pattern = 'http://{0}:{1}/v1/models/{2}/versions/{3}:predict'
    hosting_app_path = ''
    test_data_path = ''
    model_path = ''
    log_level = 'verbose'
    hosting_app_proc = None
    wait_server_ready_in_seconds = 1

    @classmethod
    def setUpClass(cls):
        cmd = [cls.hosting_app_path, '--http_port', str(cls.server_port), '--model_path', os.path.join(cls.model_path, 'mnist.onnx'), '--logging_level', cls.log_level]
        print('Launching hosting app: [{0}]'.format(' '.join(cmd)))
        cls.hosting_app_proc = subprocess.Popen(cmd)
        print('Hosting app PID: {0}'.format(cls.hosting_app_proc.pid))
        print('Sleep {0} second(s) to wait for server initialization'.format(cls.wait_server_ready_in_seconds))
        time.sleep(cls.wait_server_ready_in_seconds)


    @classmethod
    def tearDownClass(cls):
        print('Shutdown hosting app')
        cls.hosting_app_proc.kill()
        print('PID {0} has been killed: {1}'.format(cls.hosting_app_proc.pid, test_util.is_process_killed(cls.hosting_app_proc.pid)))


    def test_mnist_happy_path(self):
        input_data_file = os.path.join(self.test_data_path, 'mnist_test_data_set_0_input.pb')
        output_data_file = os.path.join(self.test_data_path, 'mnist_test_data_set_0_output.pb')

        with open(input_data_file, 'rb') as f:
            request_payload = f.read()

        content_type_headers = ['application/x-protobuf', 'application/octet-stream', 'application/vnd.google.protobuf']

        for h in content_type_headers:
            request_headers = {
                'Content-Type': h,
                'Accept': 'application/x-protobuf'
            }

            url = self.url_pattern.format(self.server_ip, self.server_port, 'default_model', 12345)
            print(url)
            r = requests.post(url, headers=request_headers, data=request_payload)
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.headers.get('Content-Type'), 'application/x-protobuf')
            self.assertTrue(r.headers.get('x-ms-request-id'))

            actual_result = predict_pb2.PredictResponse()
            actual_result.ParseFromString(r.content)

            expected_result = predict_pb2.PredictResponse()
            with open(output_data_file, 'rb') as f:
                expected_result.ParseFromString(f.read())

            for k in expected_result.outputs.keys():
                self.assertEqual(actual_result.outputs[k].data_type, expected_result.outputs[k].data_type)

            count = 1
            for i in range(0, len(expected_result.outputs['Plus214_Output_0'].dims)):
                self.assertEqual(actual_result.outputs['Plus214_Output_0'].dims[i], expected_result.outputs['Plus214_Output_0'].dims[i])
                count = count * int(actual_result.outputs['Plus214_Output_0'].dims[i])

            actual_array = numpy.frombuffer(actual_result.outputs['Plus214_Output_0'].raw_data, dtype=numpy.float32)
            expected_array = numpy.frombuffer(expected_result.outputs['Plus214_Output_0'].raw_data, dtype=numpy.float32)
            self.assertEqual(len(actual_array), len(expected_array))
            self.assertEqual(len(actual_array), count)
            for i in range(0, count):
                self.assertTrue(test_util.compare_floats(actual_array[i], expected_array[i], rel_tol=0.001))


    def test_respect_accept_header(self):
        input_data_file = os.path.join(self.test_data_path, 'mnist_test_data_set_0_input.pb')

        with open(input_data_file, 'rb') as f:
            request_payload = f.read()

        accept_headers = ['application/x-protobuf', 'application/octet-stream', 'application/vnd.google.protobuf']

        for h in accept_headers:
            request_headers = {
                'Content-Type': 'application/x-protobuf',
                'Accept': h
            }

            url = self.url_pattern.format(self.server_ip, self.server_port, 'default_model', 12345)
            print(url)
            r = requests.post(url, headers=request_headers, data=request_payload)
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.headers.get('Content-Type'), h)


    def test_missing_accept_header(self):
        input_data_file = os.path.join(self.test_data_path, 'mnist_test_data_set_0_input.pb')

        with open(input_data_file, 'rb') as f:
            request_payload = f.read()

        request_headers = {
            'Content-Type': 'application/x-protobuf',
        }

        url = self.url_pattern.format(self.server_ip, self.server_port, 'default_model', 12345)
        print(url)
        r = requests.post(url, headers=request_headers, data=request_payload)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers.get('Content-Type'), 'application/octet-stream')


    def test_any_accept_header(self):
        input_data_file = os.path.join(self.test_data_path, 'mnist_test_data_set_0_input.pb')

        with open(input_data_file, 'rb') as f:
            request_payload = f.read()

        request_headers = {
            'Content-Type': 'application/x-protobuf',
            'Accept': '*/*'
        }

        url = self.url_pattern.format(self.server_ip, self.server_port, 'default_model', 12345)
        print(url)
        r = requests.post(url, headers=request_headers, data=request_payload)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers.get('Content-Type'), 'application/octet-stream')


if __name__ == '__main__':
    unittest.main()
