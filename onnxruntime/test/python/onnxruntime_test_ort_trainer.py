# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import pytest
import sys
import copy
from numpy.testing import assert_allclose, assert_array_equal

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

from helper import get_name
import onnxruntime
from onnxruntime.capi.ort_trainer import ORTTrainer, IODescription, ModelDescription, LossScaler, generate_sample

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

def runBertTrainingTest(gradient_accumulation_steps,
                        use_mixed_precision,
                        allreduce_post_accumulation,
                        use_simple_model_desc=True,
                        use_internel_loss_scale=False):
    model_desc = bert_model_description()
    simple_model_desc = remove_extra_info(model_desc) if use_simple_model_desc else model_desc
    learning_rate_description = ort_trainer_learning_rate_description()
    device = torch.device("cuda", 0)

    torch.manual_seed(1)
    onnxruntime.seed(1)

    onnx_model = onnx.load(get_name("bert_toy_postprocessed.onnx"))

    loss_scaler = LossScaler("ort_test_input_loss_scalar", True) if use_internel_loss_scale else None

    model = ORTTrainer(onnx_model, None, simple_model_desc, "LambOptimizer",
                       map_optimizer_attributes,
                       learning_rate_description,
                       device, postprocess_model=None,
                       gradient_accumulation_steps=gradient_accumulation_steps,
                       world_rank=0, world_size=1,
                       loss_scaler=loss_scaler,
                       use_mixed_precision=use_mixed_precision,
                       allreduce_post_accumulation=allreduce_post_accumulation)

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
            # if eval_step is called during the training, it will affect the actual training loss (training session is stateful),
            eval_loss = model.eval_step(input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels, fetches=['loss'])
            eval_loss = eval_loss.cpu().numpy().item(0)

    # If using internal loss scale, all_finites are handled internally too.
    if use_mixed_precision and not use_internel_loss_scale:
        return actual_losses, actual_all_finites, eval_loss
    else:
        return actual_losses, eval_loss

class TestOrtTrainer(unittest.TestCase):
    def testBertTrainingBasic(self):
        expected_losses = [
            11.02906322479248, 11.094074249267578, 11.00899887084961, 11.06129264831543,
            11.029067039489746, 11.040265083312988, 11.046793937683105, 10.993699073791504]
        expected_eval_loss = [10.9691801071167]
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
        expected_losses = [
            11.02906322479248, 11.094074249267578, 11.008995056152344, 11.061283111572266,
            11.029059410095215, 11.04024887084961, 11.04680347442627, 10.993708610534668]
        expected_eval_loss = [10.969207763671875]
        
        actual_losses, actual_eval_loss = runBertTrainingTest(
            gradient_accumulation_steps=4, use_mixed_precision=False, allreduce_post_accumulation=False)

        # to update expected outcomes, enable pdb and run the test with -s and copy paste outputs
        # print('losses expected: ', expected_losses)
        # print('losses actual:   ', actual_losses)
        # print('eval_loss expected: ', expected_eval_loss)
        # print('eval_loss actual:   ', actual_eval_loss)
        # import pdb; pdb.set_trace()

        rtol = 1e-03
        assert_allclose(expected_losses, actual_losses, err_msg="loss mismatch")
        assert_allclose(expected_eval_loss, actual_eval_loss, err_msg="evaluation loss mismatch")

if __name__ == '__main__':
    unittest.main(module=__name__, buffer=True)
