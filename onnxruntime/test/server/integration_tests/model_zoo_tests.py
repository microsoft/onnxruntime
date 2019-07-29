# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import random
import os
import sys


class ModelZooTests(unittest.TestCase):
    server_ip = '127.0.0.1'
    server_port = 54321
    grpc_port = 56789
    url_pattern = 'http://{0}:{1}/v1/models/{2}/versions/{3}:predict'
    server_app_path = ''  # Required
    log_level = 'verbose'
    server_ready_in_seconds = 10
    server_off_in_seconds = 100
    need_data_preparation = False
    need_data_cleanup = False
    model_zoo_model_path = ''  # Required
    model_zoo_test_data_path = ''  # Required
    supported_opsets = ['opset7', 'opset8', 'opset9', 'opset_7', 'opset_8', 'opset_9']
    skipped_models = [
        ('opset7', 'tf_inception_v2'),  # Known issue
    ]

    def __test_model(self, model_path, data_paths):
        json_request_headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        pb_request_headers = {
            'Content-Type': 'application/octet-stream',
            'Accept': 'application/octet-stream'
        }

        server_app_proc = None
        try:
            onnx_file_path = ''
            for fname in os.listdir(model_path):
                if not fname.startswith(".") and fname.endswith(".onnx") and os.path.isfile(os.path.join(model_path, fname)):
                    onnx_file_path = os.path.join(model_path, fname)
                    break

            if onnx_file_path == '':
                raise FileNotFoundError('Could not find any *.onnx file in {0}'.format(model_path))
            
            cmd = [self.server_app_path, '--http_port', str(self.server_port), '--model_path', onnx_file_path, '--log_level', self.log_level, '--grpc_port', str(self.grpc_port)]
            test_util.test_log(cmd)
            server_app_proc = test_util.launch_server_app(cmd, self.server_ip, self.server_port,
                                                            self.server_ready_in_seconds)

            test_util.test_log('[{0}] Run tests...'.format(model_path))
            for test in data_paths:
                test_util.test_log('[{0}] Current: {0}'.format(model_path, test))

                test_util.test_log('[{0}] JSON payload testing ....'.format(model_path))
                url = self.url_pattern.format(self.server_ip, self.server_port, 'default_model', 12345)
                with open(os.path.join(test, 'request.json')) as f:
                    request_payload = f.read()
                resp = test_util.make_http_request(url, json_request_headers, request_payload)
                test_util.json_response_validation(self, resp, os.path.join(test, 'response.json'))

                test_util.test_log('[{0}] Protobuf payload testing ....'.format(model_path))
                url = self.url_pattern.format(self.server_ip, self.server_port, 'default_model', 54321)
                with open(os.path.join(test, 'request.pb'), 'rb') as f:
                    request_payload = f.read()
                resp = test_util.make_http_request(url, pb_request_headers, request_payload)
                test_util.pb_response_validation(self, resp, os.path.join(test, 'response.pb'))

                test_util.test_log('[{0}] GRPC testing ....'.format(model_path))
                uri = ("{}:{}".format(self.server_ip, self.grpc_port))
                with open(os.path.join(test, 'request.pb'), 'rb') as f:
                    request_payload = f.read()
                with grpc.insecure_channel(uri) as channel:
                    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
                    resp = stub.Predict(request_payload)
                test_util.pb_response_validation(self, resp, os.path.join(test, 'response.pb'))
        finally:
            test_util.shutdown_server_app(server_app_proc, self.server_off_in_seconds)


    def test_models_from_model_zoo(self):
        model_data_map = {}
        for opset in self.supported_opsets:
            test_data_folder = os.path.join(self.model_zoo_test_data_path, opset)
            model_file_folder = os.path.join(self.model_zoo_model_path, opset)

            if os.path.isdir(test_data_folder):
                for name in os.listdir(test_data_folder):
                    if (opset, name) in self.skipped_models:
                        test_util.test_log("  Skip {0}:{1}".format(opset, name))
                        continue

                    if os.path.isdir(os.path.join(test_data_folder, name)):
                        current_dir = os.path.join(test_data_folder, name)
                        model_data_map[os.path.join(model_file_folder, name)] = [os.path.join(current_dir, name) for name in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, name))]

        test_util.test_log('Planned models and test data:')
        for model_data, data_paths in model_data_map.items():
            test_util.test_log(model_data)
            for data in data_paths:
                test_util.test_log('\t\t{0}'.format(data))
        test_util.test_log('-----------------------')

        self.server_port = random.randint(30000, 40000)
        self.grpc_port = self.server_port + 1
        for model_path, data_paths in model_data_map.items():
            self.__test_model(model_path, data_paths)


if __name__ == '__main__':
    sys.path.append(sys.argv[4])
    sys.path.append(sys.argv[5])

    import test_util

    loader = unittest.TestLoader()

    test_classes = [ModelZooTests]

    test_suites = []
    for tests in test_classes:
        tests.server_app_path = sys.argv[1]
        tests.model_zoo_model_path = sys.argv[2]
        tests.model_zoo_test_data_path = sys.argv[3]

        test_suites.append(loader.loadTestsFromTestCase(tests))

    suites = unittest.TestSuite(test_suites)
    runner = unittest.TextTestRunner(verbosity=2)

    results = runner.run(suites)
