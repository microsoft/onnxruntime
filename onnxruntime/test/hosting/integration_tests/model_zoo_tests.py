import unittest

import os
import requests
import json

import test_util

class ModelZooTests(unittest.TestCase):
    server_ip = '127.0.0.1'
    server_port = 54321
    url_pattern = 'http://{0}:{1}/v1/models/{2}/versions/{3}:predict'
    hosting_app_path = '/home/klein/code/onnxruntime/build/Linux/Debug/onnxruntime_hosting'
    log_level = 'verbose'
    wait_server_ready_in_seconds = 2
    need_data_preparation = False
    need_data_cleanup = False
    model_zoo_path = ''
    model_zoo_test_data_path = '/home/klein/code/temp/65/test_data'
    supported_opsets = ['opset_8', 'opset_9']

    @classmethod
    def setUpClass(cls):
        if cls.need_data_preparation:
            print('Preparing test data from ONNX Model Zoo')
            test_util.prepare_test_data(cls.model_zoo_path, cls.model_zoo_test_data_path, cls.supported_opsets)
        else:
            print('Skip test data preparation')

        # cmd = [cls.hosting_app_path, '--http_port', str(cls.server_port), '--model_path', os.path.join(cls.model_path, 'mnist.onnx'), '--logging_level', cls.log_level]
        # cls.hosting_app_proc = test_util.launch_hosting_app(cmd, cls.wait_server_ready_in_seconds)


    @classmethod
    def tearDownClass(cls):
        # test_util.shutdown_hosting_app(cls.hosting_app_proc)

        if cls.need_data_cleanup:
            print('Clean up test data from ONNX Model Zoo')
            test_util.clean_up_test_data(cls.test_data_path)
            print('Clean up done!')
        else:
            print('Skip test data clean up')


    def test_models_from_model_zoo_with_json(self):
        request_headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        for opset in self.supported_opsets:
            test_data_folder = os.path.join(self.model_zoo_test_data_path, opset)
            model_folders = []
            for name in os.listdir(test_data_folder):
                item = os.path.join(test_data_folder, name)
                if os.path.isdir(item):
                    model_folders.append(item)

            for model in model_folders:
                print('Current model: {0}'.format(model))
                hosting_app_proc = None
                try:
                    cmd = [self.hosting_app_path, '--http_port', str(self.server_port), '--model_path', os.path.join(model, 'model.onnx'), '--logging_level', self.log_level]
                    print(cmd)
                    hosting_app_proc = test_util.launch_hosting_app(cmd, self.wait_server_ready_in_seconds)

                    print('Run tests...')
                    tests = [os.path.join(model, name) for name in os.listdir(model) if os.path.isdir(os.path.join(model, name))]
                    for test in tests:
                        print('Current: {0}'.format(test))
                        url = self.url_pattern.format(self.server_ip, self.server_port, 'default_model', 12345)
                        with open(os.path.join(test, 'request.json')) as f:
                            request_payload = f.read()
                        resp = test_util.make_http_request(url, request_headers, request_payload)
                        test_util.json_response_validation(self, resp, os.path.join(test, 'response.json'))
                finally:
                    test_util.shutdown_hosting_app(hosting_app_proc)


if __name__=='__main__':
    unittest.main()