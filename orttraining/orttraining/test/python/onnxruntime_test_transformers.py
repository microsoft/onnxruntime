from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import shutil
import pytest
import sys
import os
from enum import Enum

# set transformer_repo to the huggingface repo location
transformer_repo = ''
nvidia_deep_learning_examples_repo = ''

sys.path.append(transformer_repo)

from transformers import is_torch_available

# from configuration_common_test import ConfigTester

# from modeling_common_test import (CommonTestCases, ids_tensor, floats_tensor)

import random
global_rng = random.Random()

def ids_tensor(shape, vocab_size, rng=None, name=None):
    """Creates a random int32 tensor of the shape within the vocab size."""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()


def floats_tensor(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor of the shape within the vocab size."""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return torch.tensor(data=values, dtype=torch.float).view(shape).contiguous()


if is_torch_available():
    from transformers import (BertConfig, BertModel, BertForMaskedLM,
                              BertForNextSentencePrediction, BertForPreTraining,
                              BertForQuestionAnswering, BertForSequenceClassification,
                              BertForTokenClassification, BertForMultipleChoice)
    from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP
else:
    pytestmark = pytest.mark.skip("Require Torch")

import onnxruntime as ort
from onnxruntime.capi.ort_trainer import ORTTrainer, IODescription, ModelDescription, LossScaler, generate_sample
import torch

from helpers.bert_postprocess import postprocess_model

# sys.path.append(os.path.join(nvidia_deep_learning_examples_repo, 'PyTorch/LanguageModeling/BERT'))
# from run_pretraining import postprocess_model


def map_optimizer_attributes(name):
    no_decay_keys = ["bias", "gamma", "beta", "LayerNorm"]
    no_decay = any(no_decay_key in name for no_decay_key in no_decay_keys)
    if no_decay:
        return {"alpha": 0.9, "beta": 0.999, "lambda": 0.0, "epsilon": 1e-6}
    else:
        return {"alpha": 0.9, "beta": 0.999, "lambda": 0.01, "epsilon": 1e-6}

from torch.utils.data import Dataset, DataLoader
class OrtTestDataset(Dataset):
    def __init__(self, input_desc, seq_len, device):
        import copy
        self.input_desc_ = copy.deepcopy(input_desc)
        for input_desc in self.input_desc_:
            shape_ = []
            for i, axis in enumerate(input_desc.shape_):
                if axis == 'max_seq_len_in_batch':
                    shape_ = shape_ + [seq_len, ]
                elif axis != 'batch':
                    shape_ = input_desc.shape_[i]
            input_desc.shape_ = shape_
        self.device_ = device

    def __len__(self):
        return 100

    def __getitem__(self, item):
        input_batch = []
        for input_desc in self.input_desc_:
            input_sample = generate_sample(input_desc, self.device_)
            input_batch.append(input_sample)
        return input_batch

def create_ort_test_dataloader(input_desc, batch_size, seq_len, device):
    dataset = OrtTestDataset(input_desc, seq_len, device)
    return DataLoader(dataset, batch_size=batch_size)

class BatchArgsOption(Enum):
    List = 1
    Dict = 2
    ListAndDict = 3

def split_batch(batch, input_desc, args_count):
    total_argument_count = len(input_desc)
                # batch=[input_ids[batch, seglen], attention_mask[batch, seglen], token_type_ids[batch,seglen], token_type_ids[batch, seglen]]
    args = []   # (input_ids[batch, seglen], attention_mask[batch, seglen])
    kwargs = {} # {'token_type_ids': token_type_ids[batch,seglen], 'position_ids': token_type_ids[batch, seglen]}
    for i in range(args_count):
        args = args + [batch[i]]

    for i in range(args_count, total_argument_count):
        kwargs[input_desc[i].name_] = batch[i]

    return args, kwargs

def run_test(model, model_desc, device, args, gradient_accumulation_steps, fp16,
    allreduce_post_accumulation, get_lr_this_step, use_internal_get_lr_this_step, loss_scaler, use_internal_loss_scaler, 
    batch_args_option):
    dataloader = create_ort_test_dataloader(model_desc.inputs_, args.batch_size, args.seq_len, device)

    model = ORTTrainer(model, None, model_desc, "LambOptimizer",
        map_optimizer_attributes=map_optimizer_attributes,
        learning_rate_description=IODescription('Learning_Rate', [1,], torch.float32),
        device=device, postprocess_model=postprocess_model,
        gradient_accumulation_steps=gradient_accumulation_steps,                
        # BertLAMB default initial settings: b1=0.9, b2=0.999, e=1e-6
        world_rank=args.local_rank, world_size=args.world_size,
        use_mixed_precision=fp16,
        allreduce_post_accumulation=allreduce_post_accumulation,
        get_lr_this_step=get_lr_this_step if use_internal_get_lr_this_step else None,
        loss_scaler=loss_scaler if use_internal_loss_scaler else None)

    # trainig loop
    eval_batch = None
    model.train()
    for step, batch in enumerate(dataloader):
        if eval_batch is None:
            eval_batch = batch

        if not use_internal_get_lr_this_step:
            lr = get_lr_this_step(step)
            learning_rate = torch.tensor([lr])

        if not use_internal_loss_scaler and fp16:
            loss_scale = torch.tensor(loss_scaler.loss_scale_)

        if batch_args_option == BatchArgsOption.List:
            if not use_internal_get_lr_this_step:
                batch = batch + [learning_rate, ]
            if not use_internal_loss_scaler and fp16:
                batch = batch + [loss_scale, ]
            outputs = model(*batch)
        elif batch_args_option == BatchArgsOption.Dict:
            args, kwargs = split_batch(batch, model_desc.inputs_, 0)
            if not use_internal_get_lr_this_step:
                kwargs['Learning_Rate'] = learning_rate
            if not use_internal_loss_scaler and fp16:
                kwargs[model.loss_scale_input_name] = loss_scale
            outputs = model(*args, **kwargs)
        else:
            args_count = int(len(model_desc.inputs_) / 2)   # approx helf args, half kwargs
            args, kwargs = split_batch(batch, model_desc.inputs_, args_count)
            if not use_internal_get_lr_this_step:
                kwargs['Learning_Rate'] = learning_rate
            if not use_internal_loss_scaler and fp16:
                kwargs[model.loss_scale_input_name] = loss_scale
            outputs = model(*args, **kwargs)

    # eval
    model.eval()
    if batch_args_option == BatchArgsOption.List:
        outputs = model(*batch)
    elif batch_args_option == BatchArgsOption.Dict:
        args, kwargs = split_batch(batch, model_desc.inputs_, 0)
        outputs = model(*args, **kwargs)
    else:
        args_count = int(len(model_desc.inputs_) / 2)   # approx helf args, half kwargs
        args, kwargs = split_batch(batch, model_desc.inputs_, args_count)
        outputs = model(*args, **kwargs)

    return (output.cpu().numpy() for output in outputs)


class BertModelTest(unittest.TestCase):

    all_model_classes = (BertModel, BertForMaskedLM, BertForNextSentencePrediction,
                         BertForPreTraining, BertForQuestionAnswering, BertForSequenceClassification,
                         BertForTokenClassification) if is_torch_available() else ()

    class BertModelTester(object):

        def __init__(self,
                     parent,
                     batch_size=13,
                     seq_length=7,
                     is_training=True,
                     use_input_mask=True,
                     use_token_type_ids=True,
                     use_labels=True,
                     vocab_size=99,
                     hidden_size=32,
                     num_hidden_layers=5,
                     num_attention_heads=4,
                     intermediate_size=37,
                     hidden_act="gelu",
                     hidden_dropout_prob=0.1,
                     attention_probs_dropout_prob=0.1,
                     max_position_embeddings=512,
                     type_vocab_size=16,
                     type_sequence_label_size=2,
                     initializer_range=0.02,
                     num_labels=3,
                     num_choices=4,
                     scope=None,
                     device='cpu',
                     ):
            self.parent = parent
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.is_training = is_training
            self.use_input_mask = use_input_mask
            self.use_token_type_ids = use_token_type_ids
            self.use_labels = use_labels
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.intermediate_size = intermediate_size
            self.hidden_act = hidden_act
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.type_sequence_label_size = type_sequence_label_size
            self.initializer_range = initializer_range
            self.num_labels = num_labels
            self.num_choices = num_choices
            self.scope = scope
            self.device = device

            # 1. superset of bert input/output descs
            # see BertPreTrainedModel doc
            self.input_ids_desc = IODescription('input_ids', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes=self.vocab_size)
            self.attention_mask_desc = IODescription('attention_mask', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes=2)
            self.token_type_ids_desc = IODescription('token_type_ids', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes=2)
            self.position_ids_desc = IODescription('position_ids', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes=self.max_position_embeddings)
            self.head_mask_desc = IODescription('head_mask', [self.num_hidden_layers, self.num_attention_heads], torch.int64, num_classes=2)
            self.inputs_embeds_desc = IODescription('inputs_embeds', ['batch', 'max_seq_len_in_batch', self.hidden_size], torch.float32)

            self.encoder_hidden_states_desc = IODescription('encoder_hidden_states', ['batch', 'max_seq_len_in_batch', self.hidden_size], torch.float32)
            self.encoder_attention_mask_desc = IODescription('encoder_attention_mask', ['batch', 'max_seq_len_in_batch'], torch.float32)

            # see BertForPreTraining doc
            self.masked_lm_labels_desc = IODescription('masked_lm_labels', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes=self.vocab_size)
            self.next_sentence_label_desc = IODescription('next_sentence_label', ['batch',], torch.int64, num_classes=2)

            # outputs
            self.loss_desc = IODescription('loss', [1,], torch.float32)
            self.prediction_scores_desc = IODescription('prediction_scores', ['batch', 'max_seq_len_in_batch', self.vocab_size], torch.float32)

            self.seq_relationship_scores_desc = IODescription('seq_relationship_scores', ['batch', 2], torch.float32)   # IODescription('seq_relationship_scores', ['batch', 'max_seq_len_in_batch', 2], torch.float32)
            self.hidden_states_desc = IODescription('hidden_states', [self.num_hidden_layers, 'batch', 'max_seq_len_in_batch', self.hidden_size], torch.float32)
            self.attentions_desc = IODescription('attentions', [self.num_hidden_layers, 'batch', self.num_attention_heads, 'max_seq_len_in_batch', 'max_seq_len_in_batch'], torch.float32)
            self.last_hidden_state_desc = IODescription('last_hidden_state', ['batch', 'max_seq_len_in_batch', self.hidden_size], torch.float32)
            self.pooler_output_desc = IODescription('pooler_output', ['batch', self.hidden_size], torch.float32)

            # BertForPreTraining forward:
            # def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
            # 	position_ids??=None, head_mask??=None, inputs_embeds??=None,
            #     masked_lm_labels=None, next_sentence_label=None):
            #
            # create_and_check_bert_for_pretraining calls BertForPreTraining:
            # model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids,
            #     masked_lm_labels=token_labels, next_sentence_label=sequence_labels)

        def BertForPreTraining_descs(self):
            return ModelDescription(
                [self.input_ids_desc, self.attention_mask_desc, self.token_type_ids_desc, self.masked_lm_labels_desc, self.next_sentence_label_desc],
                # returns loss_desc if both masked_lm_labels_desc, next_sentence_label are provided
                # hidden_states_desc, attentions_desc shall be included according to config.output_attentions, config.output_hidden_states
                [self.loss_desc, self.prediction_scores_desc, self.seq_relationship_scores_desc, 
                #hidden_states_desc, attentions_desc
                ])

        def prepare_config_and_inputs(self):
            input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).to(self.device)

            input_mask = None
            if self.use_input_mask:
                input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2).to(self.device)

            token_type_ids = None
            if self.use_token_type_ids:
                token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size).to(self.device)

            sequence_labels = None
            token_labels = None
            choice_labels = None
            if self.use_labels:
                sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size).to(self.device)
                token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels).to(self.device)
                choice_labels = ids_tensor([self.batch_size], self.num_choices).to(self.device)

            config = BertConfig(
                vocab_size_or_config_json_file=self.vocab_size,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                hidden_act=self.hidden_act,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                max_position_embeddings=self.max_position_embeddings,
                type_vocab_size=self.type_vocab_size,
                is_decoder=False,
                initializer_range=self.initializer_range)

            return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

        def prepare_config_and_inputs_for_decoder(self):
            config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels = self.prepare_config_and_inputs()

            config.is_decoder = True
            encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
            encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

            return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels, encoder_hidden_states, encoder_attention_mask

        def check_loss_output(self, result):
            self.parent.assertListEqual(
                list(result["loss"].size()),
                [])

        def create_and_check_bert_model(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
            model = BertModel(config=config)
            model.to(input_ids.device)
            model.eval()

            sequence_output, pooled_output = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)

            # failed because there is not loss output
            model_desc = ModelDescription([self.input_ids_desc, self.attention_mask_desc, self.token_type_ids_desc],
                [self.last_hidden_state_desc, self.pooler_output_desc])
            args_gradient_accumulation_steps = 8
            args_local_rank = 0
            args_world_size = 1
            args_fp16 = True
            args_allreduce_post_accumulation = True

            model = ORTTrainer(model, None, model_desc, "LambOptimizer",
                               map_optimizer_attributes=map_optimizer_attributes,
                               learning_rate_description=IODescription('Learning_Rate', [1, ], torch.float32),
                               device=self.device, postprocess_model=postprocess_model,
                               gradient_accumulation_steps=args_gradient_accumulation_steps,
                               world_rank=args_local_rank, world_size=args_world_size,
                               use_mixed_precision=True if args_fp16 else False,
                               allreduce_post_accumulation=True if args_allreduce_post_accumulation else False)

            sequence_output, pooled_output = model(input_ids, token_type_ids=token_type_ids)
            sequence_output, pooled_output = model(input_ids)

            result = {
                "sequence_output": sequence_output,
                "pooled_output": pooled_output,
            }
            self.parent.assertListEqual(
                list(result["sequence_output"].size()),
                [self.batch_size, self.seq_length, self.hidden_size])
            self.parent.assertListEqual(list(result["pooled_output"].size()), [self.batch_size, self.hidden_size])

        def create_and_check_bert_model_as_decoder(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels, encoder_hidden_states, encoder_attention_mask):
            model = BertModel(config)
            model.eval()
            sequence_output, pooled_output = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask)
            sequence_output, pooled_output = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, encoder_hidden_states=encoder_hidden_states)
            sequence_output, pooled_output = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)

            result = {
                "sequence_output": sequence_output,
                "pooled_output": pooled_output,
            }
            self.parent.assertListEqual(
                list(result["sequence_output"].size()),
                [self.batch_size, self.seq_length, self.hidden_size])
            self.parent.assertListEqual(list(result["pooled_output"].size()), [self.batch_size, self.hidden_size])

        def create_and_check_bert_for_masked_lm(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
            model = BertForMaskedLM(config=config)
            model.eval()
            loss, prediction_scores = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, masked_lm_labels=token_labels)

            #####
            model_desc = ModelDescription([self.input_ids_desc, self.attention_mask_desc, self.token_type_ids_desc, self.masked_lm_labels_desc],
                [self.loss_desc, self.prediction_scores_desc])
            args_gradient_accumulation_steps = 8
            args_local_rank = 0
            args_world_size = 1
            args_fp16 = True
            args_allreduce_post_accumulation = True

            model = ORTTrainer(model, None, model_desc, "LambOptimizer",
                               map_optimizer_attributes=map_optimizer_attributes,
                               learning_rate_description=IODescription('Learning_Rate', [1, ], torch.float32),
                               device=self.device, postprocess_model=postprocess_model,
                               gradient_accumulation_steps=args_gradient_accumulation_steps,
                               world_rank=args_local_rank, world_size=args_world_size,
                               use_mixed_precision=True if args_fp16 else False,
                               allreduce_post_accumulation=True if args_allreduce_post_accumulation else False)
            model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, masked_lm_labels=token_labels)

        def create_and_check_bert_model_for_masked_lm_as_decoder(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels, encoder_hidden_states, encoder_attention_mask):
            model = BertForMaskedLM(config=config)
            model.eval()
            loss, prediction_scores = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, masked_lm_labels=token_labels, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask)
            loss, prediction_scores = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, masked_lm_labels=token_labels, encoder_hidden_states=encoder_hidden_states)
            result = {
                "loss": loss,
                "prediction_scores": prediction_scores,
            }
            self.parent.assertListEqual(
                list(result["prediction_scores"].size()),
                [self.batch_size, self.seq_length, self.vocab_size])
            self.check_loss_output(result)

        def create_and_check_bert_for_next_sequence_prediction(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
            model = BertForNextSentencePrediction(config=config)
            model.eval()
            loss, seq_relationship_score = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, next_sentence_label=sequence_labels)
            result = {
                "loss": loss,
                "seq_relationship_score": seq_relationship_score,
            }
            self.parent.assertListEqual(
                list(result["seq_relationship_score"].size()),
                [self.batch_size, 2])
            self.check_loss_output(result)

        def create_and_check_bert_for_pretraining(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
            model = BertForPreTraining(config=config)
            model.eval()
            loss, prediction_scores, seq_relationship_score = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids,
                                                                    masked_lm_labels=token_labels, next_sentence_label=sequence_labels)
            model_desc = ModelDescription([self.input_ids_desc, self.attention_mask_desc, self.token_type_ids_desc,
                                           self.masked_lm_labels_desc, self.next_sentence_label_desc],
                                          [self.loss_desc, self.prediction_scores_desc, self.seq_relationship_scores_desc])

            import argparse
            args_ = argparse.Namespace(fp16=True, amp_opt_level='O1')

            from collections import namedtuple
            MyArgs = namedtuple("MyArgs", 
                "local_rank world_size max_steps learning_rate warmup_proportion batch_size seq_len")
            args = MyArgs(local_rank=0, world_size=1, max_steps=100, learning_rate=0.00001, warmup_proportion=0.01, batch_size=13, seq_len=7)

            from helpers.utils import get_lr
            def get_lr_this_step(global_step):
                return get_lr(args, global_step)
            loss_scaler = LossScaler('loss_scale_input_name', True, up_scale_window=2000)

            option_gradient_accumulation_steps = [8]
            option_fp16 = [True, False]
            option_allreduce_post_accumulation = True
            option_use_internal_get_lr_this_step = False
            option_use_internal_loss_scaler = False
            # TODO: with with fetches

            for gradient_accumulation_steps in option_gradient_accumulation_steps:
                for fp16 in option_fp16:
                    for option_split_batch in BatchArgsOption:                
                        loss_ort, prediction_scores_ort, seq_relationship_score_ort =\
                            run_test(model, model_desc, self.device, args, gradient_accumulation_steps, fp16,
                                     option_allreduce_post_accumulation,
                                     get_lr_this_step, option_use_internal_get_lr_this_step,
                                     loss_scaler, option_use_internal_loss_scaler,
                                     option_split_batch)

                        print(loss_ort)
                        print(prediction_scores_ort)
                        print(seq_relationship_score_ort)

        def create_and_check_bert_for_question_answering(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
            model = BertForQuestionAnswering(config=config)
            model.eval()
            loss, start_logits, end_logits = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids,
                                                   start_positions=sequence_labels, end_positions=sequence_labels)
            result = {
                "loss": loss,
                "start_logits": start_logits,
                "end_logits": end_logits,
            }
            self.parent.assertListEqual(
                list(result["start_logits"].size()),
                [self.batch_size, self.seq_length])
            self.parent.assertListEqual(
                list(result["end_logits"].size()),
                [self.batch_size, self.seq_length])
            self.check_loss_output(result)

        def create_and_check_bert_for_sequence_classification(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
            config.num_labels = self.num_labels
            model = BertForSequenceClassification(config)
            model.eval()
            loss, logits = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=sequence_labels)
            result = {
                "loss": loss,
                "logits": logits,
            }
            self.parent.assertListEqual(
                list(result["logits"].size()),
                [self.batch_size, self.num_labels])
            self.check_loss_output(result)

        def create_and_check_bert_for_token_classification(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
            config.num_labels = self.num_labels
            model = BertForTokenClassification(config=config)
            model.eval()
            loss, logits = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
            result = {
                "loss": loss,
                "logits": logits,
            }
            self.parent.assertListEqual(
                list(result["logits"].size()),
                [self.batch_size, self.seq_length, self.num_labels])
            self.check_loss_output(result)

        def create_and_check_bert_for_multiple_choice(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
            config.num_choices = self.num_choices
            model = BertForMultipleChoice(config=config)
            model.eval()
            multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
            multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
            multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
            loss, logits = model(multiple_choice_inputs_ids,
                                 attention_mask=multiple_choice_input_mask,
                                 token_type_ids=multiple_choice_token_type_ids,
                                 labels=choice_labels)
            result = {
                "loss": loss,
                "logits": logits,
            }
            self.parent.assertListEqual(
                list(result["logits"].size()),
                [self.batch_size, self.num_choices])
            self.check_loss_output(result)

        def prepare_config_and_inputs_for_common(self):
            config_and_inputs = self.prepare_config_and_inputs()
            (config, input_ids, token_type_ids, input_mask,
             sequence_labels, token_labels, choice_labels) = config_and_inputs
            inputs_dict = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': input_mask}
            return config, inputs_dict

    def setUp(self):
        self.model_tester = BertModelTest.BertModelTester(self)
        # self.config_tester = ConfigTester(self, config_class=BertConfig, hidden_size=37)

    # def test_config(self):
    #     self.config_tester.run_common_tests()

    # def test_bert_model(self, use_cuda=False):
    #     # ^^ This could be a real fixture
    #     if use_cuda:
    #         self.model_tester.device = "cuda"
    #     config_and_inputs = self.model_tester.prepare_config_and_inputs()
    #     self.model_tester.create_and_check_bert_model(*config_and_inputs)

    # def test_bert_model_as_decoder(self):
    #     config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
    #     self.model_tester.create_and_check_bert_model_as_decoder(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_bert_for_masked_lm(*config_and_inputs)

    # def test_for_masked_lm_decoder(self):
    #     config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
    #     self.model_tester.create_and_check_bert_model_for_masked_lm_as_decoder(*config_and_inputs)

    # def test_for_multiple_choice(self):
    #     config_and_inputs = self.model_tester.prepare_config_and_inputs()
    #     self.model_tester.create_and_check_bert_for_multiple_choice(*config_and_inputs)

    # def test_for_next_sequence_prediction(self):
    #     config_and_inputs = self.model_tester.prepare_config_and_inputs()
    #     self.model_tester.create_and_check_bert_for_next_sequence_prediction(*config_and_inputs)

    def test_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_bert_for_pretraining(*config_and_inputs)

    # def test_for_question_answering(self):
    #     config_and_inputs = self.model_tester.prepare_config_and_inputs()
    #     self.model_tester.create_and_check_bert_for_question_answering(*config_and_inputs)

    # def test_for_sequence_classification(self):
    #     config_and_inputs = self.model_tester.prepare_config_and_inputs()
    #     self.model_tester.create_and_check_bert_for_sequence_classification(*config_and_inputs)

    # def test_for_token_classification(self):
    #     config_and_inputs = self.model_tester.prepare_config_and_inputs()
    #     self.model_tester.create_and_check_bert_for_token_classification(*config_and_inputs)

    # @pytest.mark.slow
    # def test_model_from_pretrained(self):
    #     cache_dir = "/tmp/transformers_test/"
    #     for model_name in list(BERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
    #         model = BertModel.from_pretrained(model_name, cache_dir=cache_dir)
    #         shutil.rmtree(cache_dir)
    #         self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
