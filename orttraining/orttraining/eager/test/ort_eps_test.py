# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import torch
import onnxruntime_pybind11_state as torch_ort
import os
import sys


def is_windows():
    return sys.platform.startswith("win")


from io import StringIO
import sys
import threading
import time


class OutputGrabber(object):
    """
    Class used to grab standard output or another stream.
    """

    escape_char = "\b"

    def __init__(self, stream=None, threaded=False):
        self.origstream = stream
        self.threaded = threaded
        if self.origstream is None:
            self.origstream = sys.stdout
        self.origstreamfd = self.origstream.fileno()
        self.capturedtext = ""
        # Create a pipe so the stream can be captured:
        self.pipe_out, self.pipe_in = os.pipe()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self):
        """
        Start capturing the stream data.
        """
        self.capturedtext = ""
        # Save a copy of the stream:
        self.streamfd = os.dup(self.origstreamfd)
        # Replace the original stream with our write pipe:
        os.dup2(self.pipe_in, self.origstreamfd)
        if self.threaded:
            # Start thread that will read the stream:
            self.workerThread = threading.Thread(target=self.readOutput)
            self.workerThread.start()
            # Make sure that the thread is running and os.read() has executed:
            time.sleep(0.01)

    def stop(self):
        """
        Stop capturing the stream data and save the text in `capturedtext`.
        """
        # Print the escape character to make the readOutput method stop:
        self.origstream.write(self.escape_char)
        # Flush the stream to make sure all our data goes in before
        # the escape character:
        self.origstream.flush()
        if self.threaded:
            # wait until the thread finishes so we are sure that
            # we have until the last character:
            self.workerThread.join()
        else:
            self.readOutput()
        # Close the pipe:
        os.close(self.pipe_in)
        os.close(self.pipe_out)
        # Restore the original stream:
        os.dup2(self.streamfd, self.origstreamfd)
        # Close the duplicate stream:
        os.close(self.streamfd)

    def readOutput(self):
        """
        Read the stream data (one byte at a time)
        and save the text in `capturedtext`.
        """
        while True:
            char = os.read(self.pipe_out, 1).decode(self.origstream.encoding)
            if not char or self.escape_char in char:
                break
            self.capturedtext += char


class OrtEPTests(unittest.TestCase):
    def get_test_execution_provider_path(self):
        if is_windows():
            return os.path.join(".", "test_execution_provider.dll")
        else:
            return os.path.join(".", "libtest_execution_provider.so")

    def test_import_custom_eps(self):
        torch_ort.set_device(0, "CPUExecutionProvider", {})

        torch_ort._register_provider_lib("TestExecutionProvider", self.get_test_execution_provider_path(), {})
        # capture std out
        with OutputGrabber() as out:
            torch_ort.set_device(1, "TestExecutionProvider", {"device_id": "0", "some_config": "val"})
            ort_device = torch_ort.device(1)
        assert "My EP provider created, with device id: 0, some_option: val" in out.capturedtext
        with OutputGrabber() as out:
            torch_ort.set_device(2, "TestExecutionProvider", {"device_id": "1", "some_config": "val"})
            ort_device = torch_ort.device(1)
        assert "My EP provider created, with device id: 1, some_option: val" in out.capturedtext
        # test the reusing EP instance
        with OutputGrabber() as out:
            torch_ort.set_device(3, "TestExecutionProvider", {"device_id": "0", "some_config": "val"})
            ort_device = torch_ort.device(1)
        assert "My EP provider created, with device id: 0, some_option: val" not in out.capturedtext
        # test clear training ep instance pool
        torch_ort.clear_training_ep_instances()
        with OutputGrabber() as out:
            torch_ort.set_device(3, "TestExecutionProvider", {"device_id": "0", "some_config": "val"})
            ort_device = torch_ort.device(1)
        assert "My EP provider created, with device id: 0, some_option: val" in out.capturedtext

    @unittest.skip("Test fails with newest pytorch version.")
    def test_print(self):
        x = torch.ones(1, 2)
        ort_x = x.to("ort")
        with OutputGrabber() as out:
            print(ort_x)
        assert "tensor([[1., 1.]], device='ort:0')" in out.capturedtext


if __name__ == "__main__":
    unittest.main()
