# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import sys
import copy
from numpy.testing import assert_allclose, assert_array_equal

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from onnxruntime.capi.ort_trainer import ORTTrainer, IODescription, ModelDescription, LossScaler, generate_sample

def ort_trainer_learning_rate_description():
    return IODescription('Learning_Rate', [1, ], torch.float32)


def bert_model_description():
    vocab_size = 30528
    input_ids_desc = IODescription('input_ids', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes=vocab_size)
    segment_ids_desc = IODescription('segment_ids', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes=2)
    input_mask_desc = IODescription('input_mask', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes=2)
    masked_lm_labels_desc = IODescription('masked_lm_labels', ['batch', 'max_seq_len_in_batch'], torch.int64,
                                          num_classes=vocab_size)
    next_sentence_labels_desc = IODescription('next_sentence_labels', ['batch', ], torch.int64, num_classes=2)
    loss_desc = IODescription('loss', [], torch.float32)
    # probability_desc = IODescription('probability', ['batch', 10], torch.float32)

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

def runBertTrainingTest(gradient_accumulation_steps, use_mixed_precision, allreduce_post_accumulation):
    model_desc = bert_model_description()
    learning_rate_description = ort_trainer_learning_rate_description()
    device = torch.device("cuda", 0)

    onnx_model = onnx.load("/bert_ort/liqun/onnxruntime/onnxruntime/test/testdata/bert_toy_postprocessed.onnx")

    model = ORTTrainer(onnx_model, None, model_desc, "LambOptimizer",
                       map_optimizer_attributes,
                       learning_rate_description,
                       device, postprocess_model=None,
                       gradient_accumulation_steps=gradient_accumulation_steps,
                       world_rank=-1, world_size=1,
                       use_mixed_precision=use_mixed_precision,
                       allreduce_post_accumulation=allreduce_post_accumulation)

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
        if use_mixed_precision:
            loss_scale = torch.tensor(loss_scaler.loss_scale_).to(device)
            actual_loss = model.train_step((input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels, learning_rate, loss_scale))
            if isinstance(actual_loss, (list, tuple)):
                assert len(actual_loss) == 2
                actual_loss, actual_all_finite = actual_loss
                loss_scaler.update_loss_scale(actual_all_finite.item())
                actual_all_finites = [*actual_all_finites, actual_all_finite.cpu().numpy().item(0)]

            actual_losses = [*actual_losses, actual_loss.cpu().numpy().item(0)]
        else:
            loss = model((input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels, learning_rate))
            actual_losses = [*actual_losses, loss.cpu().numpy().item(0)]

        if batch_count == num_batches - 1:
            # test eval_step api with fetches at the end of the training.
            # if eval_step is called during the training, it will affect the actual training loss (training session is stateful),
            eval_loss = model.eval_step((input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels), fetches=['loss'])
            eval_loss = eval_loss.cpu().numpy().item(0)

    if use_mixed_precision:
        return actual_losses, actual_all_finites, eval_loss
    else:
        return actual_losses, eval_loss

class TestOrtTrainer(unittest.TestCase):

    def testMNISTTrainingAndTesting(self):
        class NeuralNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super(NeuralNet, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size) 
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, num_classes)  

            def forward(self, x):
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                return out

        def my_loss(x, target):
            return F.nll_loss(F.log_softmax(x, dim=1), target)

        def train_with_trainer(learningRate, trainer, device, train_loader, epoch):
            actual_losses = []
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                data = data.reshape(data.shape[0], -1)

                loss = trainer.train_step((data, target, torch.tensor([learningRate])))

                args_log_interval = 100
                if batch_idx % args_log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
                    actual_losses = [*actual_losses, loss.cpu().numpy().item()]

            return actual_losses

        # TODO: comple this once ORT training can do evaluation.
        def test_with_trainer(trainer, device, test_loader):
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

        torch.manual_seed(1)

        args_batch_size = 64
        args_test_batch_size = 1000

        kwargs = {'num_workers': 0, 'pin_memory': True}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([transforms.ToTensor(), 
                                                         transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=args_batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=args_test_batch_size, shuffle=True, **kwargs)

        device = torch.device("cuda")

        input_size = 784
        hidden_size = 500
        num_classes = 10
        model = NeuralNet(input_size, hidden_size, num_classes)

        model_desc = mnist_model_description()

        trainer = ORTTrainer(model, my_loss, model_desc, "SGDOptimizer", None, IODescription('Learning_Rate', [1, ], 
                             torch.float32), device)

        learningRate = 0.01
        args_epochs = 2
        expected_losses = [2.345372200012207, 0.8558371067047119, 0.6692017316818237, 0.5240862369537354,
                           0.4327302575111389, 0.2800341844558716, 0.2419648915529251, 0.263438880443573,
                           0.3994610905647278, 0.3097628951072693, 0.4905158281326294, 0.374204158782959,
                           0.19508624076843262, 0.2650184631347656, 0.4114145040512085, 0.24791213870048523,
                           0.16095051169395447, 0.18530189990997314, 0.1688750684261322, 0.23382069170475006]
        expected_test_losses = [0.30860821228027346, 0.2554518310546875]
        expected_test_accuracies = [0.9144, 0.9288]

        actual_losses = []
        actual_test_losses, actual_accuracies = [], []
        for epoch in range(1, args_epochs + 1):
            actual_losses = [*actual_losses, *train_with_trainer(learningRate, trainer, device, train_loader, epoch)]

            test_loss, accuracy = test_with_trainer(trainer, device, test_loader)
            actual_test_losses = [*actual_test_losses, test_loss]
            actual_accuracies = [*actual_accuracies, accuracy]

        print("actual_losses=", actual_losses)
        print("actual_test_losses=", actual_test_losses)
        print("actual_accuracies=", actual_accuracies)

        # to update expected outcomes, enable pdb and run the test with -s and copy paste outputs
        # import pdb; pdb.set_trace()

        assert_allclose(expected_losses, actual_losses, err_msg="loss mismatch")
        assert_allclose(expected_test_losses, actual_test_losses, err_msg="test loss mismatch")
        assert_allclose(expected_test_accuracies, actual_accuracies, err_msg="test accuracy mismatch")

    def testBertTrainingBasic(self):
        torch.manual_seed(1)
        expected_losses = [
            11.049949645996094, 11.13171100616455, 11.025514602661133, 11.0499267578125,
            10.893335342407227, 10.978937149047852, 11.082908630371094, 10.98244571685791]
        expected_eval_loss = [11.070235252380371]
        actual_losses, actual_eval_loss = runBertTrainingTest(
            gradient_accumulation_steps=1, use_mixed_precision=False, allreduce_post_accumulation=False)
        print('actual_losses ', actual_losses)
        print('eval_loss', actual_eval_loss)

        # to update expected outcomes, enable pdb and run the test with -s and copy paste outputs
        # import pdb; pdb.set_trace()

        assert_allclose(expected_losses, actual_losses, err_msg="loss mismatch")
        assert_allclose(expected_eval_loss, actual_eval_loss, err_msg="evaluation loss mismatch")

    def testBertTrainingMixedPrecision(self):
        torch.manual_seed(1)
        expected_losses = [11.0546875, 11.125, 11.0234375, 11.0546875, 10.890625, 10.9765625, 11.078125, 10.984375]
        expected_all_finites = [False, True, True, True, True, True, True, True]
        expected_eval_loss = [11.0703125]
        actual_losses, actual_all_finites, actual_eval_loss = runBertTrainingTest(
            gradient_accumulation_steps=1, use_mixed_precision=True, allreduce_post_accumulation=False)
        print('actual_losses ', actual_losses)
        print('actual_all_finite ', actual_all_finites)
        print('eval_loss', actual_eval_loss)

        # to update expected outcomes, enable pdb and run the test with -s and copy paste outputs
        # import pdb; pdb.set_trace()

        assert_allclose(expected_losses, actual_losses, err_msg="loss mismatch")
        assert_array_equal(expected_all_finites, actual_all_finites, "all_finite mismatch")
        assert_allclose(expected_eval_loss, actual_eval_loss, err_msg="evaluation loss mismatch")

    def testBertTrainingGradientAccumulationMixedPrecision(self):
        torch.manual_seed(1)
        expected_losses = [11.0546875, 11.125, 11.0234375, 11.0546875, 10.890625, 10.9765625, 11.078125, 10.984375]
        expected_all_finites = [False, True]
        expected_eval_loss = [11.0703125]
        actual_losses, actual_all_finites, actual_eval_loss = runBertTrainingTest(
            gradient_accumulation_steps=4, use_mixed_precision=True, allreduce_post_accumulation=False)
        print('actual_losses ', actual_losses)
        print('actual_all_finite ', actual_all_finites)
        print('eval_loss', actual_eval_loss)

        # to update expected outcomes, enable pdb and run the test with -s and copy paste outputs
        # import pdb; pdb.set_trace()

        assert_allclose(expected_losses, actual_losses, err_msg="loss mismatch")
        assert_array_equal(expected_all_finites, actual_all_finites, "all_finite mismatch")
        assert_allclose(expected_eval_loss, actual_eval_loss, err_msg="evaluation loss mismatch")


if __name__ == '__main__':
    unittest.main(module=__name__, buffer=True)
