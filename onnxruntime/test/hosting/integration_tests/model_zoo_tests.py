# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import os
import test_util


class ModelZooTests(unittest.TestCase):
    server_ip = '127.0.0.1'
    server_port = 54321
    url_pattern = 'http://{0}:{1}/v1/models/{2}/versions/{3}:predict'
    hosting_app_path = '/home/klein/code/onnxruntime/build/Linux/Debug/onnxruntime_hosting'
    log_level = 'verbose'
    server_ready_in_seconds = 2
    need_data_preparation = False
    need_data_cleanup = False
    model_zoo_test_data_path = '/home/klein/code/temp/64/test_data'
    supported_opsets = ['opset_7', 'opset_8', 'opset_9']
    skipped_models = ['tiny_yolov2']

    def test_models_from_model_zoo(self):
        json_request_headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        pb_request_headers = {
            'Content-Type': 'application/octet-stream',
            'Accept': 'application/octet-stream'
        }

        for opset in self.supported_opsets:
            test_data_folder = os.path.join(self.model_zoo_test_data_path, opset)
            model_folders = []
            for name in os.listdir(test_data_folder):
                if name in self.skipped_models:
                    continue

                item = os.path.join(test_data_folder, name)
                if os.path.isdir(item):
                    model_folders.append(item)

            for model in model_folders:
                test_util.test_log('Current model: {0}'.format(model))
                hosting_app_proc = None
                try:
                    cmd = [self.hosting_app_path, '--http_port', str(self.server_port), '--model_path', os.path.join(model, 'model.onnx'), '--logging_level', self.log_level]
                    test_util.test_log(cmd)
                    hosting_app_proc = test_util.launch_hosting_app(cmd, self.server_ready_in_seconds)

                    test_util.test_log('Run tests...')
                    tests = [os.path.join(model, name) for name in os.listdir(model) if os.path.isdir(os.path.join(model, name))]
                    for test in tests:
                        test_util.test_log('Current: {0}'.format(test))

                        test_util.test_log('JSON payload testing ....')
                        url = self.url_pattern.format(self.server_ip, self.server_port, 'default_model', 12345)
                        with open(os.path.join(test, 'request.json')) as f:
                            request_payload = f.read()
                        resp = test_util.make_http_request(url, json_request_headers, request_payload)
                        test_util.json_response_validation(self, resp, os.path.join(test, 'response.json'))

                        test_util.test_log('Protobuf payload testing ....')
                        url = self.url_pattern.format(self.server_ip, self.server_port, 'default_model', 54321)
                        with open(os.path.join(test, 'request.pb'), 'rb') as f:
                            request_payload = f.read()
                        resp = test_util.make_http_request(url, pb_request_headers, request_payload)
                        test_util.pb_response_validation(self, resp, os.path.join(test, 'response.pb'))
                finally:
                    test_util.shutdown_hosting_app(hosting_app_proc)


if __name__ == '__main__':
    unittest.main()
