# adapted from run_glue.py of huggingface transformers

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import unittest
import numpy as np
from numpy.testing import assert_allclose

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    GlueDataset,
    GlueDataTrainingArguments,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

import onnxruntime
from onnxruntime.capi.ort_trainer import ORTTrainer, LossScaler, ModelDescription, IODescription
from onnxruntime.capi._pybind_state import get_mpi_context_local_rank, get_mpi_context_local_size, get_mpi_context_world_rank, get_mpi_context_world_size

from orttraining_transformer_trainer import ORTTransformerTrainer

import torch

logger = logging.getLogger(__name__)

def verify_old_and_new_api_are_equal(results_per_api):
    new_api_results = results_per_api[True]
    old_api_results = results_per_api[False]
    for key in new_api_results.keys():
        assert_allclose(new_api_results[key], old_api_results[key])

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

class ORTGlueTest(unittest.TestCase):

    def setUp(self):
        # configurations not to be changed accoss tests
        self.max_seq_length = 128
        self.train_batch_size = 8
        self.learning_rate = 2e-5
        self.num_train_epochs = 3.0
        self.local_rank = -1
        self.world_size = 1
        self.overwrite_output_dir = True
        self.gradient_accumulation_steps = 1
        self.data_dir = "/bert_data/hf_data/glue_data/"
        self.output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "glue_test_output/")
        self.cache_dir = '/tmp/glue/'
        self.logging_steps = 10
        self.rtol = 1e-02

    def test_roberta_with_mrpc(self):
        expected_acc = 0.8676470588235294
        expected_f1 = 0.9035714285714286
        expected_acc_and_f1 = 0.885609243697479
        expected_loss = 0.3022572344862947

        results_per_api = dict()
        for use_new_api in [True, False]:
            results = self.run_glue(model_name="roberta-base", task_name="MRPC", fp16=False, use_new_api=use_new_api)
            assert_allclose(results['acc'], expected_acc, rtol=self.rtol)
            assert_allclose(results['f1'], expected_f1, rtol=self.rtol)
            assert_allclose(results['acc_and_f1'], expected_acc_and_f1, rtol=self.rtol)
            assert_allclose(results['loss'], expected_loss, rtol=self.rtol)
            results_per_api[use_new_api] = results

        verify_old_and_new_api_are_equal(results_per_api)

    def test_roberta_fp16_with_mrpc(self):
        expected_acc = 0.8897058823529411
        expected_f1 = 0.9197860962566845
        expected_acc_and_f1 = 0.9047459893048129
        expected_loss = 0.3035417107098243

        results_per_api = dict()
        for use_new_api in [True, False]:
            results = self.run_glue(model_name="roberta-base", task_name="MRPC", fp16=True, use_new_api=use_new_api)
            assert_allclose(results['acc'], expected_acc, rtol=self.rtol)
            assert_allclose(results['f1'], expected_f1, rtol=self.rtol)
            assert_allclose(results['acc_and_f1'], expected_acc_and_f1, rtol=self.rtol)
            assert_allclose(results['loss'], expected_loss, rtol=self.rtol)
            results_per_api[use_new_api] = results

        verify_old_and_new_api_are_equal(results_per_api)

    def test_bert_with_mrpc(self):
        if self.local_rank == -1:
            expected_acc = 0.8553921568627451
            expected_f1 = 0.8970331588132635
            expected_acc_and_f1 = 0.8762126578380043
            expected_loss = 0.42737212419217707
        elif self.local_rank == 0:
            expected_acc = 0.8308823529411765
            expected_f1 = 0.881646655231561
            expected_acc_and_f1 = 0.8562645040863688
            expected_loss = 0.42491564023144107

        if self.local_rank == -1:
            # not parallel case, we can run both new and old api tests
            results_per_api = dict()
            for use_new_api in [True, False]:
                results = self.run_glue(model_name="bert-base-cased", task_name="MRPC", fp16=False, use_new_api=use_new_api)
                results_per_api[use_new_api] = results

            verify_old_and_new_api_are_equal(results_per_api)
        else:
            # with parallel training, TrainingArguments can only be created once (due to its cached _setup_devices)
            # thus we can only choose one test case to run.
            results = self.run_glue(model_name="bert-base-cased", task_name="MRPC", fp16=False, use_new_api=True)

        if self.local_rank in [-1, 0]:
            assert_allclose(results['acc'], expected_acc, rtol=self.rtol)
            assert_allclose(results['f1'], expected_f1, rtol=self.rtol)
            assert_allclose(results['acc_and_f1'], expected_acc_and_f1, rtol=self.rtol)
            assert_allclose(results['loss'], expected_loss, rtol=self.rtol)

    def test_bert_fp16_with_mrpc(self):
        expected_acc = 0.8529411764705882
        expected_f1 = 0.8972602739726027
        expected_acc_and_f1 = 0.8751007252215954
        expected_loss = 0.412924896998732

        results_per_api = dict()
        for use_new_api in [True, False]:
            results = self.run_glue(model_name="bert-base-cased", task_name="MRPC", fp16=True, use_new_api=use_new_api)
            assert_allclose(results['acc'], expected_acc, rtol=self.rtol)
            assert_allclose(results['f1'], expected_f1, rtol=self.rtol)
            assert_allclose(results['acc_and_f1'], expected_acc_and_f1, rtol=self.rtol)
            assert_allclose(results['loss'], expected_loss, rtol=self.rtol)
            results_per_api[use_new_api] = results

        verify_old_and_new_api_are_equal(results_per_api)

    def model_to_desc(self, model_name, model):
        if model_name.startswith('bert') or model_name.startswith('xlnet'):
            new_model_desc = {
                'inputs': [
                    ('input_ids', ['batch', 'max_seq_len_in_batch'],),
                    ('attention_mask', ['batch', 'max_seq_len_in_batch'],),
                    ('token_type_ids', ['batch', 'max_seq_len_in_batch'],),
                    ('labels', ['batch', ],)],
                'outputs': [('loss', [], True),
                            ('logits', ['batch', 2])]}
            model_desc = ModelDescription([
                IODescription('input_ids', ['batch', 'max_seq_len_in_batch']),
                IODescription('attention_mask', ['batch', 'max_seq_len_in_batch']),
                IODescription('token_type_ids', ['batch', 'max_seq_len_in_batch']),
                IODescription('labels', ['batch',])], [
                IODescription('loss', []),
                IODescription('logits', ['batch', 2])])
        elif model_name.startswith('roberta'):
            new_model_desc = {
                'inputs': [
                    ('input_ids', ['batch', 'max_seq_len_in_batch'],),
                    ('attention_mask', ['batch', 'max_seq_len_in_batch'],),
                    ('labels', ['batch', ],)],
                'outputs': [('loss', [], True),
                            ('logits', ['batch', 2])]}
            model_desc = ModelDescription([
                IODescription('input_ids', ['batch', 'max_seq_len_in_batch']),
                IODescription('attention_mask', ['batch', 'max_seq_len_in_batch']),
                IODescription('labels', ['batch',])], [
                IODescription('loss', []),
                IODescription('logits', ['batch', 2])])
        else:
            raise RuntimeError("unsupported base model name {}.".format(model_name))

        return model_desc, new_model_desc

    def run_glue(self, model_name, task_name, fp16, use_new_api):
        model_args = ModelArguments(model_name_or_path=model_name, cache_dir=self.cache_dir)
        data_args = GlueDataTrainingArguments(
            task_name=task_name, data_dir=os.path.join(self.data_dir, task_name),
            max_seq_length=self.max_seq_length)

        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_dir, task_name), do_train=True, do_eval=True,
            per_gpu_train_batch_size=self.train_batch_size,
            learning_rate=self.learning_rate, num_train_epochs=self.num_train_epochs,
            local_rank=self.local_rank,
            overwrite_output_dir=self.overwrite_output_dir, gradient_accumulation_steps=self.gradient_accumulation_steps,
            fp16=fp16, logging_steps=self.logging_steps)

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        )
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            bool(training_args.local_rank != -1),
            training_args.fp16,
        )
        logger.info("Training/evaluation parameters %s", training_args)

        set_seed(training_args.seed)
        onnxruntime.set_seed(training_args.seed)

        try:
            num_labels = glue_tasks_num_labels[data_args.task_name]
            output_mode = glue_output_modes[data_args.task_name]
        except KeyError:
            raise ValueError("Task not found: %s" % (data_args.task_name))

        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

        train_dataset = (
            GlueDataset(data_args, tokenizer=tokenizer)
            if training_args.do_train
            else None
        )

        eval_dataset = (
            GlueDataset(data_args, tokenizer=tokenizer, mode="dev")
            if training_args.do_eval
            else None
        )

        def compute_metrics(p: EvalPrediction) -> Dict:
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(data_args.task_name, preds, p.label_ids)

        model_desc, new_model_desc = self.model_to_desc(model_name, model)
        # Initialize the ORTTrainer within ORTTransformerTrainer
        trainer = ORTTransformerTrainer(
            model=model,
            model_desc=model_desc,
            new_model_desc=new_model_desc,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            use_new_api=use_new_api,
            world_size=self.world_size,
        )

        # Training
        if training_args.do_train:
            trainer.train()
            trainer.save_model()

        # Evaluation
        results = {}
        if training_args.do_eval and training_args.local_rank in [-1, 0]:
            logger.info("*** Evaluate ***")

            result = trainer.evaluate()

            logger.info("***** Eval results {} *****".format(data_args.task_name))
            for key, value in result.items():
               logger.info("  %s = %s", key, value)

            results.update(result)

        return results

if __name__ == "__main__":
    local_rank = get_mpi_context_local_rank()
    world_size = get_mpi_context_world_size()
    if world_size > 1:
        # mpi launch
        logger.warning("mpirun launch, local_rank / world_size: %s : % s", local_rank, world_size)

        # TrainingArguments._setup_devices will call torch.distributed.init_process_group(backend="nccl")
        # pytorch expects following environment settings (which would be set if launched with torch.distributed.launch).

        os.environ['RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'

        from onnxruntime.capi._pybind_state import set_cuda_device_id
        set_cuda_device_id(local_rank)

        test = ORTGlueTest()
        test.setUp()
        test.local_rank = local_rank
        test.world_size = world_size
        test.test_bert_with_mrpc()
    else:
        unittest.main()
