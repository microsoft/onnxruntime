# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import unittest
import pytest
import sys
import copy
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from helper import get_name
import onnxruntime
from onnxruntime.capi.ort_trainer import ORTTrainer, IODescription, ModelDescription, LossScaler, generate_sample, save_checkpoint, load_checkpoint

SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))

def ort_trainer_learning_rate_description():
    return IODescription('Learning_Rate', [1, ], torch.float32)


def remove_extra_info(model_desc):
    simple_model_desc = copy.deepcopy(model_desc)
    for input_desc in simple_model_desc.inputs_:
        input_desc.dtype_ = None
        input_desc.num_classes_ = None
    for output_desc in simple_model_desc.outputs_:
        output_desc.dtype_ = None
        output_desc.num_classes_ = None
    return simple_model_desc

def bert_model_description():
    vocab_size = 30528
    input_ids_desc = IODescription('input_ids', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes=vocab_size)
    segment_ids_desc = IODescription('segment_ids', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes=2)
    input_mask_desc = IODescription('input_mask', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes=2)
    masked_lm_labels_desc = IODescription('masked_lm_labels', ['batch', 'max_seq_len_in_batch'], torch.int64,
                                          num_classes=vocab_size)
    next_sentence_labels_desc = IODescription('next_sentence_labels', ['batch', ], torch.int64, num_classes=2)
    loss_desc = IODescription('loss', [], torch.float32)

    return ModelDescription([input_ids_desc, segment_ids_desc, input_mask_desc, masked_lm_labels_desc,
                             next_sentence_labels_desc], [loss_desc])

def map_optimizer_attributes(name):
    no_decay_keys = ["bias", "gamma", "beta", "LayerNorm"]
    no_decay = any(no_decay_key in name for no_decay_key in no_decay_keys)
    if no_decay:
        return {"alpha": 0.9, "beta": 0.999, "lambda": 0.0, "epsilon": 1e-6}
    else:
        return {"alpha": 0.9, "beta": 0.999, "lambda": 0.01, "epsilon": 1e-6}

def generate_sample_batch(desc, batch_size, device):
    desc_ = copy.deepcopy(desc)
    desc_.shape_[0] = batch_size
    sample = generate_sample(desc_, device)
    return sample

def create_ort_trainer(gradient_accumulation_steps,
                        use_mixed_precision,
                        allreduce_post_accumulation,
                        use_simple_model_desc=True,
                        loss_scaler=None,
                        deepspeed_zero_stage=0):
    model_desc = bert_model_description()
    simple_model_desc = remove_extra_info(model_desc) if use_simple_model_desc else model_desc
    learning_rate_description = ort_trainer_learning_rate_description()
    device = torch.device("cuda", 0)

    onnx_model = onnx.load(get_name("bert_toy_postprocessed.onnx"))

    model = ORTTrainer(onnx_model, None, simple_model_desc, "LambOptimizer",
                       map_optimizer_attributes,
                       learning_rate_description,
                       device,
                       gradient_accumulation_steps=gradient_accumulation_steps,
                       world_rank=0, world_size=1,
                       loss_scaler=loss_scaler,
                       use_mixed_precision=use_mixed_precision,
                       allreduce_post_accumulation=allreduce_post_accumulation,
                       deepspeed_zero_stage = deepspeed_zero_stage)

    return model, model_desc, device

def runBertTrainingTest(gradient_accumulation_steps,
                        use_mixed_precision,
                        allreduce_post_accumulation,
                        use_simple_model_desc=True,
                        use_internel_loss_scale=False):
    torch.manual_seed(1)
    onnxruntime.set_seed(1)

    loss_scaler = LossScaler("ort_test_input_loss_scalar", True) if use_internel_loss_scale else None

    model, model_desc, device = create_ort_trainer(gradient_accumulation_steps,
                        use_mixed_precision,
                        allreduce_post_accumulation,
                        use_simple_model_desc,
                        loss_scaler)

    if loss_scaler is None:
        loss_scaler = LossScaler(model.loss_scale_input_name, True)

    input_ids_batches = []
    segment_ids_batches = []
    input_mask_batches = []
    masked_lm_labels_batches = []
    next_sentence_labels_batches = []
    batch_size = 16
    num_batches = 8
    for batch in range(num_batches):
        input_ids_batches = [*input_ids_batches, generate_sample_batch(model_desc.inputs_[0], batch_size, device)]
        segment_ids_batches = [*segment_ids_batches, generate_sample_batch(model_desc.inputs_[1], batch_size, device)]
        input_mask_batches = [*input_mask_batches, generate_sample_batch(model_desc.inputs_[2], batch_size, device)]
        masked_lm_labels_batches = [*masked_lm_labels_batches, generate_sample_batch(model_desc.inputs_[3], batch_size, device)]
        next_sentence_labels_batches = [*next_sentence_labels_batches, generate_sample_batch(model_desc.inputs_[4], batch_size, device)]

    lr_batch_list = [0.0000000e+00, 4.6012269e-07, 9.2024538e-07, 1.3803681e-06, 1.8404908e-06,
                     2.3006135e-06, 2.7607362e-06, 3.2208588e-06, 3.6809815e-06]

    actual_losses = []
    actual_all_finites = []

    for batch_count in range(num_batches):
        input_ids = generate_sample_batch(model_desc.inputs_[0], batch_size, device)
        segment_ids = generate_sample_batch(model_desc.inputs_[1], batch_size, device)
        input_mask = generate_sample_batch(model_desc.inputs_[2], batch_size, device)
        masked_lm_labels = generate_sample_batch(model_desc.inputs_[3], batch_size, device)
        next_sentence_labels = generate_sample_batch(model_desc.inputs_[4], batch_size, device)
        lr = lr_batch_list[batch_count]

        learning_rate = torch.tensor([lr]).to(device)
        training_args = [input_ids,
                         segment_ids,
                         input_mask,
                         masked_lm_labels,
                         next_sentence_labels,
                         learning_rate]
        if use_mixed_precision:
            if not use_internel_loss_scale:
                loss_scale = torch.tensor([loss_scaler.loss_scale_]).to(device)
                training_args.append(loss_scale)
            actual_loss = model.train_step(*training_args)
            if isinstance(actual_loss, (list, tuple)):
                assert len(actual_loss) == 2
                actual_loss, actual_all_finite = actual_loss
                if not use_internel_loss_scale:
                    loss_scaler.update_loss_scale(actual_all_finite.item())
                    actual_all_finites = [*actual_all_finites, actual_all_finite.cpu().numpy().item(0)]

            actual_losses = [*actual_losses, actual_loss.cpu().numpy().item(0)]
        else:
            loss = model(*training_args)
            actual_losses = [*actual_losses, loss.cpu().numpy().item(0)]

        if batch_count == num_batches - 1:
            # test eval_step api with fetches at the end of the training.
            # if eval_step is called during the training, it will affect the actual training loss (training session is stateful).
            eval_loss = model.eval_step(input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels, fetches=['loss'])
            eval_loss = eval_loss.cpu().numpy().item(0)

    # If using internal loss scale, all_finites are handled internally too.
    if use_mixed_precision and not use_internel_loss_scale:
        return actual_losses, actual_all_finites, eval_loss
    else:
        return actual_losses, eval_loss

class MNISTWrapper():
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(MNISTWrapper.NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)
            self.register_buffer("bias_buffer", torch.tensor(1e-6))

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            out = torch.add(out, self.bias_buffer.to(out.dtype))
            return out

    class NeuralNetWithLoss(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(MNISTWrapper.NeuralNetWithLoss, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)

        def forward(self, x, target):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return F.nll_loss(F.log_softmax(out, dim=1), target), out

    def my_loss(x, target):
        return F.nll_loss(F.log_softmax(x, dim=1), target)

    def train_with_trainer(self, learningRate, trainer, device, train_loader, epoch):
        actual_losses = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.reshape(data.shape[0], -1)

            loss, _ = trainer.train_step(data, target, torch.tensor([learningRate]))

            args_log_interval = 100
            if batch_idx % args_log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                actual_losses = [*actual_losses, loss.cpu().numpy().item()]

        return actual_losses

    # TODO: comple this once ORT training can do evaluation.
    def test_with_trainer(self, trainer, device, test_loader):
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data.reshape(data.shape[0], -1)
                output = F.log_softmax(trainer.eval_step((data), fetches=['probability']), dim=1)
                test_loss += F.nll_loss(output, target, reduction='sum').item()     # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)                           # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        return test_loss, correct / len(test_loader.dataset)

    def mnist_model_description():
        input_desc = IODescription('input1', ['batch', 784], torch.float32)
        label_desc = IODescription('label', ['batch', ], torch.int64, num_classes=10)
        loss_desc = IODescription('loss', [], torch.float32)
        probability_desc = IODescription('probability', ['batch', 10], torch.float32)
        return ModelDescription([input_desc, label_desc], [loss_desc, probability_desc])

    def get_loaders(self):
        # TODO: Remove this temporary fix for urllib.error.HTTPError: HTTP Error 403: Forbidden
        # once a more permanent solution can be found.
        # Fix as per https://github.com/pytorch/vision/issues/1938#issuecomment-789986996
        from six.moves import urllib
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        # TODO: Remove this temporary fix when the issue https://github.com/pytorch/vision/issues/3549 is resolved
        # Resource http://yann.lecun.com/exdb/mnist/ is not available
        datasets.MNIST.resources = [
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
        ]

        args_batch_size = 64
        args_test_batch_size = 1000

        kwargs = {'num_workers': 0, 'pin_memory': True}
        # set shuffle to False to get deterministic data set among different torch version
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(SCRIPT_DIR, 'data'), train=True, download=True,
                           transform=transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=args_batch_size, shuffle=False, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(SCRIPT_DIR, 'data'), train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=args_test_batch_size, shuffle=False, **kwargs)

        return train_loader, test_loader

    def get_model(self):
        input_size = 784
        hidden_size = 500
        num_classes = 10

        # warning: changes the pytorch random generator state
        model = MNISTWrapper.NeuralNet(input_size, hidden_size, num_classes)
        model_desc = MNISTWrapper.mnist_model_description()
        return model, model_desc

    def get_model_with_internal_loss(self):
        input_size = 784
        hidden_size = 500
        num_classes = 10

        # warning: changes the pytorch random generator state
        model = MNISTWrapper.NeuralNetWithLoss(input_size, hidden_size, num_classes)
        model_desc = MNISTWrapper.mnist_model_description()
        return model, model_desc

    def get_trainer(self, model, model_desc, device, onnx_opset_ver=12, frozen_weights=[],
                    internal_loss_fn=False, get_lr_this_step=None, optimizer="SGDOptimizer"):
        loss_fn = MNISTWrapper.my_loss if not internal_loss_fn else None
        return ORTTrainer(model, loss_fn, model_desc, optimizer, None, IODescription('Learning_Rate', [1, ],
                                torch.float32), device, _opset_version=onnx_opset_ver, frozen_weights=frozen_weights,
                                get_lr_this_step=get_lr_this_step)

class TestOrtTrainer(unittest.TestCase):

    def run_mnist_training_and_testing(onnx_opset_ver):
        torch.manual_seed(1)
        device = torch.device("cuda")

        mnist = MNISTWrapper()
        train_loader, test_loader = mnist.get_loaders()
        model, model_desc = mnist.get_model()
        trainer = mnist.get_trainer(model, model_desc, device, onnx_opset_ver=onnx_opset_ver)

        learningRate = 0.01
        args_epochs = 2
        expected_losses = [2.312044143676758, 0.8018650412559509, 0.5819257497787476, 0.47025489807128906,
                        0.35800155997276306, 0.41124576330184937, 0.2731882333755493, 0.4201386570930481,
                        0.39458805322647095, 0.38380366563796997, 0.2722422480583191, 0.24230478703975677,
                        0.23505745828151703, 0.33442264795303345, 0.21140924096107483, 0.31545233726501465,
                        0.18556523323059082, 0.3453553020954132, 0.29598352313041687, 0.3595045208930969]

        expected_test_losses = [0.3145490005493164, 0.256188737487793]
        expected_test_accuracies = [0.9075, 0.9265]

        actual_losses = []
        actual_test_losses, actual_accuracies = [], []
        for epoch in range(1, args_epochs + 1):
            actual_losses = [*actual_losses, *mnist.train_with_trainer(learningRate, trainer, device, train_loader, epoch)]

            test_loss, accuracy = mnist.test_with_trainer(trainer, device, test_loader)
            actual_test_losses = [*actual_test_losses, test_loss]
            actual_accuracies = [*actual_accuracies, accuracy]

            # if you update outcomes, also do so for resume from checkpoint test
            # args_checkpoint_epoch = 1
            # if epoch == args_checkpoint_epoch:
                # state = {'rng_state': torch.get_rng_state(), 'model': trainer.state_dict()}
                # torch.save(state, get_name("ckpt_mnist.pt"))


        print("actual_losses=", actual_losses)
        print("actual_test_losses=", actual_test_losses)
        print("actual_accuracies=", actual_accuracies)

        # to update expected outcomes, enable pdb and run the test with -s and copy paste outputs
        # import pdb; pdb.set_trace()
        rtol = 1e-03
        assert_allclose(expected_losses, actual_losses, rtol=rtol, err_msg="loss mismatch")
        assert_allclose(expected_test_losses, actual_test_losses, rtol=rtol, err_msg="test loss mismatch")
        assert_allclose(expected_test_accuracies, actual_accuracies, rtol=rtol, err_msg="test accuracy mismatch")

    def testMNISTTrainingAndTestingOpset12(self):
        TestOrtTrainer.run_mnist_training_and_testing(onnx_opset_ver = 12)

    def testMNISTResumeTrainingAndTesting(self):
        torch.manual_seed(1)
        device = torch.device("cuda")

        mnist = MNISTWrapper()
        train_loader, test_loader = mnist.get_loaders()
        model, model_desc = mnist.get_model()

        learningRate = 0.01
        args_epochs = 2
        args_checkpoint_epoch = 1
        # should match those in test without checkpointing
        expected_losses = [0.26509523391723633, 0.24135658144950867, 0.2397943139076233, 0.3351520597934723,
                        0.20998981595039368, 0.31488314270973206, 0.18481917679309845, 0.34727591276168823,
                        0.2971782684326172, 0.3609251379966736]

        expected_test_losses = [0.25632242965698243]
        expected_test_accuracies = [0.9264]

        actual_losses = []
        actual_test_losses, actual_accuracies = [], []

        # restore from checkpoint
        resume_trainer = mnist.get_trainer(model, model_desc, device)
        checkpoint = torch.load(get_name("ckpt_mnist.pt"), map_location="cpu")
        torch.set_rng_state(checkpoint['rng_state'])
        resume_trainer.load_state_dict(checkpoint['model'], strict=True)

        # continue ..
        for epoch in range(args_checkpoint_epoch + 1, args_epochs + 1):
            actual_losses = [*actual_losses, *mnist.train_with_trainer(learningRate, resume_trainer, device, train_loader, epoch)]

            test_loss, accuracy = mnist.test_with_trainer(resume_trainer, device, test_loader)
            actual_test_losses = [*actual_test_losses, test_loss]
            actual_accuracies = [*actual_accuracies, accuracy]

        print("actual_losses=", actual_losses)
        print("actual_test_losses=", actual_test_losses)
        print("actual_accuracies=", actual_accuracies)

        # to update expected outcomes, enable pdb and run the test with -s and copy paste outputs
        # import pdb; pdb.set_trace()
        rtol = 1e-03
        assert_allclose(expected_losses, actual_losses, rtol=rtol, err_msg="loss mismatch")
        assert_allclose(expected_test_losses, actual_test_losses, rtol=rtol, err_msg="test loss mismatch")
        assert_allclose(expected_test_accuracies, actual_accuracies, rtol=rtol, err_msg="test accuracy mismatch")

    def testMNISTStateDict(self):
        torch.manual_seed(1)
        device = torch.device("cuda")

        mnist = MNISTWrapper()
        train_loader, test_loader = mnist.get_loaders()
        model, model_desc = mnist.get_model()

        trainer = mnist.get_trainer(model, model_desc, device)
        state_dict = trainer.state_dict()
        assert state_dict == {}

        learningRate = 0.02
        epoch = 0

        data, target = next(iter(train_loader))
        data, target = data.to(device), target.to(device)
        data = data.reshape(data.shape[0], -1)

        loss, _ = trainer.train_step(data, target, torch.tensor([learningRate]))

        state_dict = trainer.state_dict()
        assert state_dict.keys() == {'fc1.bias', 'fc1.weight', 'fc2.bias', 'fc2.weight', 'bias_buffer'}

    def testMNISTSaveAsONNX(self):
        torch.manual_seed(1)
        device = torch.device("cuda")
        onnx_file_name = 'mnist.onnx'
        if os.path.exists(onnx_file_name):
            os.remove(onnx_file_name)

        mnist = MNISTWrapper()
        train_loader, test_loader = mnist.get_loaders()
        model, model_desc = mnist.get_model()

        trainer = mnist.get_trainer(model, model_desc, device)
        trainer.save_as_onnx(onnx_file_name)
        assert not os.path.exists(onnx_file_name)

        learningRate = 0.02
        epoch = 0

        data, target = next(iter(train_loader))
        data, target = data.to(device), target.to(device)
        data = data.reshape(data.shape[0], -1)

        loss, _ = trainer.train_step(data, target, torch.tensor([learningRate]))

        trainer.save_as_onnx(onnx_file_name)
        assert os.path.exists(onnx_file_name)

    def testMNISTDevice(self):
        torch.manual_seed(1)
        device = torch.device("cuda")

        mnist = MNISTWrapper()
        train_loader, test_loader = mnist.get_loaders()
        model, model_desc = mnist.get_model()

        for model_device in [torch.device('cpu'), torch.device('cuda')]:
            model.to(model_device)
            trainer = mnist.get_trainer(model, model_desc, device)
            learningRate = 0.02
            epoch = 0

            data, target = next(iter(train_loader))
            data, target = data.to(device), target.to(device)
            data = data.reshape(data.shape[0], -1)

            loss, _ = trainer.train_step(data, target, torch.tensor([learningRate]))

    def testMNISTInitializerNames(self):
        torch.manual_seed(1)
        device = torch.device("cuda")

        mnist = MNISTWrapper()
        train_loader, test_loader = mnist.get_loaders()
        model, model_desc = mnist.get_model()

        trainer = mnist.get_trainer(model, model_desc, device)
        learningRate = 0.02
        epoch = 0

        data, target = next(iter(train_loader))
        data, target = data.to(device), target.to(device)
        data = data.reshape(data.shape[0], -1)

        loss, _ = trainer.train_step(data, target, torch.tensor([learningRate]))

        assert (set([n.name for n in trainer.onnx_model_.graph.initializer])-set(['bias_buffer'])) \
            == set([n for n, t in model.named_parameters()])

    def testMNISTInitializerNamesWithInternalLoss(self):
        torch.manual_seed(1)
        device = torch.device("cuda")

        mnist = MNISTWrapper()
        train_loader, test_loader = mnist.get_loaders()
        model, model_desc = mnist.get_model_with_internal_loss()


        def get_lr_this_step(global_step):
            learningRate = 0.02
            return torch.tensor([learningRate])

        trainer = mnist.get_trainer(model, model_desc, device, internal_loss_fn=True,
                                    get_lr_this_step=get_lr_this_step)
        epoch = 0

        data, target = next(iter(train_loader))
        data, target = data.to(device), target.to(device)
        data = data.reshape(data.shape[0], -1)

        loss, _ = trainer.train_step(data, target)

        assert set([n.name for n in trainer.onnx_model_.graph.initializer]) \
            == set([n for n, t in model.named_parameters()])

    def testMNISTFrozenWeight(self):
        torch.manual_seed(1)
        device = torch.device("cuda")

        mnist = MNISTWrapper()
        train_loader, test_loader = mnist.get_loaders()
        model, model_desc = mnist.get_model()

        trainer = mnist.get_trainer(model, model_desc, device, frozen_weights=['fc1.weight'])

        learningRate = 0.02
        epoch = 0

        data, target = next(iter(train_loader))
        data, target = data.to(device), target.to(device)
        data = data.reshape(data.shape[0], -1)

        loss, _ = trainer.train_step(data, target, torch.tensor([learningRate]))

        fc1_trainstep_1 = trainer.state_dict()['fc1.weight']
        fc2_trainstep_1 = trainer.state_dict()['fc2.weight']

        loss, _ = trainer.train_step(data, target, torch.tensor([learningRate]))

        fc1_trainstep_2 = trainer.state_dict()['fc1.weight']
        fc2_trainstep_2 = trainer.state_dict()['fc2.weight']
        assert np.array_equal(fc1_trainstep_1, fc1_trainstep_2) and \
            not np.array_equal(fc2_trainstep_1, fc2_trainstep_2)

    def testMNISTTorchBuffer(self):
        torch.manual_seed(1)
        device = torch.device("cuda")

        mnist = MNISTWrapper()
        train_loader, test_loader = mnist.get_loaders()
        model, model_desc = mnist.get_model()

        trainer = mnist.get_trainer(model, model_desc, device)

        learningRate = 0.02
        epoch = 0

        data, target = next(iter(train_loader))
        data, target = data.to(device), target.to(device)
        data = data.reshape(data.shape[0], -1)

        loss, _ = trainer.train_step(data, target, torch.tensor([learningRate]))

        fc1_trainstep_1 = trainer.state_dict()['fc1.weight']
        bias_buffer_trainstep_1 = trainer.state_dict()['bias_buffer']

        loss, _ = trainer.train_step(data, target, torch.tensor([learningRate]))

        fc1_trainstep_2 = trainer.state_dict()['fc1.weight']
        bias_buffer_trainstep_2 = trainer.state_dict()['bias_buffer']
        assert not np.array_equal(fc1_trainstep_1, fc1_trainstep_2) and \
            np.array_equal(bias_buffer_trainstep_1, bias_buffer_trainstep_2)

    def testMNISTFrozenWeightCheckpoint(self):
        torch.manual_seed(1)
        device = torch.device("cuda")

        mnist = MNISTWrapper()
        train_loader, test_loader = mnist.get_loaders()
        model, model_desc = mnist.get_model()

        trainer = mnist.get_trainer(model, model_desc, device, frozen_weights=['fc1.weight'])

        learningRate = 0.02
        epoch = 0

        # do one train step
        data, target = next(iter(train_loader))
        data, target = data.to(device), target.to(device)
        data = data.reshape(data.shape[0], -1)

        loss, _ = trainer.train_step(data, target, torch.tensor([learningRate]))

        # do one eval step
        data, target = next(iter(train_loader))
        data, target = data.to(device), target.to(device)
        data = data.reshape(data.shape[0], -1)

        loss, _ = trainer.eval_step(data, target)

        # save checkpoint, load model and compare
        state_dict = trainer.state_dict()

        new_model, _ = mnist.get_model()
        trainer = mnist.get_trainer(new_model, model_desc, device, frozen_weights=['fc1.weight'])
        trainer.load_state_dict(state_dict)

        ckpt_loss, _ = trainer.eval_step(data, target)
        assert loss == ckpt_loss

        loaded_state_dict = trainer.state_dict()
        assert state_dict.keys() == loaded_state_dict.keys()

    def testMNISTTrainingCheckpoint(self):
        torch.manual_seed(1)
        device = torch.device("cuda")

        mnist = MNISTWrapper()
        train_loader, test_loader = mnist.get_loaders()
        model, model_desc = mnist.get_model()

        trainer = mnist.get_trainer(model, model_desc, device,
            optimizer='LambOptimizer', frozen_weights=['fc1.weight'])

        learningRate = 0.02
        epoch = 0

        # do 5 train step
        for i in range(5):
            data, target = next(iter(train_loader))
            data, target = data.to(device), target.to(device)
            data = data.reshape(data.shape[0], -1)

            loss, _ = trainer.train_step(data, target, torch.tensor([learningRate]))

        # do one eval step
        data, target = next(iter(train_loader))
        data, target = data.to(device), target.to(device)
        data = data.reshape(data.shape[0], -1)

        loss, _ = trainer.eval_step(data, target)

        # save checkpoint, load model and compare
        state_dict = trainer.state_dict()

        new_model, _ = mnist.get_model()
        trainer = mnist.get_trainer(new_model, model_desc, device,
            optimizer='LambOptimizer', frozen_weights=['fc1.weight'])
        trainer.load_state_dict(state_dict)

        ckpt_loss, _ = trainer.eval_step(data, target)
        assert loss == ckpt_loss

        loaded_state_dict = trainer.state_dict()
        assert state_dict.keys() == loaded_state_dict.keys()
        for key in state_dict:
            assert np.array_equal(state_dict[key], loaded_state_dict[key])

    def testBertTrainingBasic(self):
        expected_losses = [11.027887, 11.108191, 11.055356, 11.040912, 10.960277, 11.02691, 11.082471, 10.920979]
        expected_eval_loss = [10.958977]
        actual_losses, actual_eval_loss = runBertTrainingTest(
            gradient_accumulation_steps=1, use_mixed_precision=False, allreduce_post_accumulation=False)

        # to update expected outcomes, enable pdb and run the test with -s and copy paste outputs
        # print('losses expected: ', expected_losses)
        # print('losses actual:   ', actual_losses)
        # print('eval_loss expected: ', expected_eval_loss)
        # print('eval_loss actual:   ', actual_eval_loss)
        # import pdb; pdb.set_trace()

        rtol = 1e-03
        assert_allclose(expected_losses, actual_losses, rtol=rtol, err_msg="loss mismatch")
        assert_allclose(expected_eval_loss, actual_eval_loss, rtol=rtol, err_msg="evaluation loss mismatch")

    def testBertTrainingGradientAccumulation(self):
        expected_losses = [11.027887, 11.108191, 11.055354, 11.040904, 10.960266, 11.026897, 11.082475, 10.920998]
        expected_eval_loss = [10.958998]

        actual_losses, actual_eval_loss = runBertTrainingTest(
            gradient_accumulation_steps=4, use_mixed_precision=False, allreduce_post_accumulation=False)

        # to update expected outcomes, enable pdb and run the test with -s and copy paste outputs
        # print('losses expected: ', expected_losses)
        # print('losses actual:   ', actual_losses)
        # print('eval_loss expected: ', expected_eval_loss)
        # print('eval_loss actual:   ', actual_eval_loss)
        # import pdb; pdb.set_trace()

        rtol = 1e-03
        assert_allclose(expected_losses, actual_losses, rtol=rtol, err_msg="loss mismatch")
        assert_allclose(expected_eval_loss, actual_eval_loss, rtol=rtol, err_msg="evaluation loss mismatch")

    def testBertCheckpointingBasic(self):
        model,_,_ = create_ort_trainer(gradient_accumulation_steps=1,
                        use_mixed_precision=False,
                        allreduce_post_accumulation=True,
                        use_simple_model_desc=True,
                        loss_scaler=None)
        sd = model.state_dict()

        # modify one of the default values
        sd['bert.encoder.layer.0.attention.output.LayerNorm.weight'] +=1
        model.load_state_dict(sd)

        ckpt_dir = 'testdata'
        save_checkpoint(model, ckpt_dir, 'bert_toy_save_test')
        del model

        # create new model
        model2,_,_ = create_ort_trainer(gradient_accumulation_steps=1,
                        use_mixed_precision=False,
                        allreduce_post_accumulation=True,
                        use_simple_model_desc=True,
                        loss_scaler=None)

        # load changed checkpoint
        load_checkpoint(model2, ckpt_dir, 'bert_toy_save_test')
        loaded_sd = model2.state_dict()

        for k,v in loaded_sd.items():
            assert torch.all(torch.eq(v, sd[k]))

    def testWrapModelLossFnStateDict(self):
        torch.manual_seed(1)
        device = torch.device("cuda")
        class LinearModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 4)
            def forward(self, y=None, x=None):
                if y is not None:
                    return self.linear(x) + y
                else:
                    return self.linear(x) + torch.ones(2, 4)

        pt_model = LinearModel()
        data = torch.randn(2, 2)
        label = torch.tensor([0, 1], dtype=torch.int64)
        input_desc = IODescription('x', [2, 2], torch.float32)
        label_desc = IODescription('label', [2, ], torch.int64, num_classes=4)
        output_desc = IODescription('output', [2, 4], torch.float32)
        loss_desc = IODescription('loss', [], torch.float32)
        model_desc = ModelDescription([input_desc, label_desc], [loss_desc, output_desc])
        def loss_fn(x, label):
            return F.nll_loss(F.log_softmax(x, dim=1), label)

        def get_lr_this_step(global_step):
            learningRate = 0.02
            return torch.tensor([learningRate])

        ort_trainer = ORTTrainer(
            pt_model, loss_fn, model_desc, "SGDOptimizer", None,
            IODescription('Learning_Rate', [1, ], torch.float32), device,
            get_lr_this_step=get_lr_this_step)
        ort_trainer.train_step(x=data, label=label)
        state_dict = ort_trainer.state_dict()
        assert state_dict.keys() == {'linear.bias', 'linear.weight'}

if __name__ == '__main__':
    unittest.main(module=__name__, buffer=True)
