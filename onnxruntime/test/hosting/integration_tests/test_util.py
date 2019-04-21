import os
import base64
import struct
import math
import subprocess
import time
import requests
import json

def is_process_killed(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def decode_base64_string(s, count_and_type):
    b = base64.b64decode(s)
    r = struct.unpack(count_and_type, b)

    return r


def compare_floats(a, b, rel_tol=0.0001):
    if not math.isclose(a, b, rel_tol=rel_tol):
        print('Not match with relative tolerance {0}: {1} and {2}'.format(rel_tol, a, b))
        return False

    return True


def prepare_test_data(model_zoo_path, output_path, supported_opsets):
    return


def clean_up_test_data(test_data_path):
    return


def launch_hosting_app(cmd, wait_server_ready_in_seconds):
    print('Launching hosting app: [{0}]'.format(' '.join(cmd)))
    hosting_app_proc = subprocess.Popen(cmd)
    print('Hosting app PID: {0}'.format(hosting_app_proc.pid))
    print('Sleep {0} second(s) to wait for server initialization'.format(wait_server_ready_in_seconds))
    time.sleep(wait_server_ready_in_seconds)

    return hosting_app_proc


def shutdown_hosting_app(hosting_app_proc):
    if hosting_app_proc is not None:
        print('Shutdown hosting app')
        hosting_app_proc.kill()
        print('PID {0} has been killed: {1}'.format(hosting_app_proc.pid, is_process_killed(hosting_app_proc.pid)))

    return


def make_http_request(url, request_headers, payload):
    return requests.post(url, headers=request_headers, data=payload)


def json_response_validation(cls, resp, expected_resp_json_file):
    cls.assertEqual(resp.status_code, 200)
    cls.assertTrue(resp.headers.get('x-ms-request-id'))
    cls.assertEqual(resp.headers.get('Content-Type'), 'application/json')

    with open(expected_resp_json_file) as f:
        expected_result = json.loads(f.read())

    actual_response = json.loads(resp.content.decode('utf-8'))
    cls.assertTrue(actual_response['outputs'])
