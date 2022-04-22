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
            char = os.read(self.pipe_out,1).decode(self.origstream.encoding)
            if not char or self.escape_char in char:
                break
            self.capturedtext += char

import torch
from onnxruntime.capi import _pybind_state as torch_ort_eager
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from onnxruntime.training import optim, orttrainer, orttrainer_options
import unittest

def my_loss(x, target):
    return F.nll_loss(F.log_softmax(x, dim=1), target)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, target):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return my_loss(out, target)

class OrtEPTests(unittest.TestCase):
    def test_external_graph_transformer_triggering(self):
        input_size = 784
        hidden_size = 500
        num_classes = 10
        batch_size = 128
        model = NeuralNet(input_size, hidden_size, num_classes)
        
        model_desc = {'inputs': [('x', [batch_size, input_size]),
                                    ('target', [batch_size,])],
                                    'outputs': [('loss', [], True)]}
        optim_config = optim.SGDConfig()
        opts =  orttrainer.ORTTrainerOptions({'device':{'id':'cpu'}})
        model = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
        # because orttrainer is lazy initialized, feed in a random data to trigger the graph transformer
        data = torch.rand(batch_size, input_size)
        target = torch.randint(0, 10, (batch_size,))
        
        with OutputGrabber() as out:
            loss = model.train_step(data, target)
        assert '******************Trigger Customized Graph Transformer:  MyGraphTransformer!' in out.capturedtext

if __name__ == '__main__':
    unittest.main()