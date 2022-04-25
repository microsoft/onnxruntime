# adapted from Trainer.py of huggingface transformers

import json
import logging
import os
import random

from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from transformers.data.data_collator import DataCollator, DefaultDataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import TrainingArguments

import onnxruntime
from orttraining_test_bert_postprocess import postprocess_model
from onnxruntime.capi.ort_trainer import ORTTrainer, LossScaler, ModelDescription, IODescription

from onnxruntime.training import (
    _utils,
    amp,
    optim,
    orttrainer,
    TrainStepInfo,
    model_desc_validation as md_val,
    orttrainer_options as orttrainer_options,
)
from onnxruntime.training.optim import LinearWarmupLRScheduler, _LRScheduler

try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


def is_tensorboard_available():
    return _has_tensorboard


logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    onnxruntime.set_seed(seed)


class EvalPrediction(NamedTuple):
    predictions: np.ndarray
    label_ids: np.ndarray


class PredictionOutput(NamedTuple):
    predictions: np.ndarray
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]


class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float


def get_linear_schedule_with_warmup(num_warmup_steps, num_training_steps, base_lr):
    def lr_lambda_linear(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    def lambda_lr_get_lr(current_global_step):
        # LambdaLR increment self.last_epoch at evert sept()
        return base_lr * lr_lambda_linear(current_global_step)

    return lambda_lr_get_lr


class ORTTransformerTrainer:
    """ """

    model: PreTrainedModel
    args: TrainingArguments
    train_dataset: Dataset
    eval_dataset: Dataset
    compute_metrics: Callable[[EvalPrediction], Dict]

    def __init__(
        self,
        model: PreTrainedModel,
        model_desc: dict,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        compute_metrics: Callable[[EvalPrediction], Dict],
        world_size: Optional[int] = 1,
    ):
        """ """

        self.model = model
        self.model_desc = model_desc
        self.args = args
        self.world_size = world_size
        self.data_collator = DefaultDataCollator()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        set_seed(self.args.seed)
        # Create output directory if needed
        if self.args.local_rank in [-1, 0]:
            os.makedirs(self.args.output_dir, exist_ok=True)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = (
            SequentialSampler(self.train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(self.train_dataset)
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator.collate_batch,
        )

    def get_eval_dataloader(self) -> DataLoader:
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator.collate_batch,
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator.collate_batch,
        )

    def train(self):
        """
        Main training entry point.
        """
        train_dataloader = self.get_train_dataloader()

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        lr_scheduler = orttrainer.optim.LinearWarmupLRScheduler(t_total, self.args.warmup_steps / float(t_total))

        loss_scaler = amp.DynamicLossScaler() if self.args.fp16 else None
        device = self.args.device.type

        device = f"{device}:{self.args.device.index}" if self.args.device.index else f"{device}:0"
        options = orttrainer.ORTTrainerOptions(
            {
                "batch": {"gradient_accumulation_steps": self.args.gradient_accumulation_steps},
                "device": {"id": device},
                "mixed_precision": {"enabled": self.args.fp16, "loss_scaler": loss_scaler},
                "debug": {
                    "deterministic_compute": True,
                },
                "utils": {"grad_norm_clip": False},
                "distributed": {
                    # we are running single node multi gpu test. thus world_rank = local_rank
                    # and world_size = self.args.n_gpu
                    "world_rank": max(0, self.args.local_rank),
                    "world_size": int(self.world_size),
                    "local_rank": max(0, self.args.local_rank),
                    "allreduce_post_accumulation": True,
                },
                "lr_scheduler": lr_scheduler,
            }
        )

        param_optimizer = list(self.model.named_parameters())
        params = [
            {
                "params": [n for n, p in param_optimizer if "bias" in n or "LayerNorm.weight" in n],
                "weight_decay_mode": 1,
            },
            {
                "params": [n for n, p in param_optimizer if not ("bias" in n or "LayerNorm.weight" in n)],
                "weight_decay_mode": 1,
            },
        ]

        optim_config = optim.AdamConfig(params=params, lr=2e-5, do_bias_correction=True)
        self.model = orttrainer.ORTTrainer(self.model, self.model_desc, optim_config, options=options)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataloader.dataset))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.args.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        tr_loss = 0.0
        logging_loss = 0.0
        train_iterator = trange(
            epochs_trained,
            int(num_train_epochs),
            desc="Epoch",
            disable=self.args.local_rank not in [-1, 0],
        )

        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=self.args.local_rank not in [-1, 0])
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                tr_loss += self._training_step(self.model, inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps and (step + 1) == len(epoch_iterator)
                ):
                    global_step += 1

                    if self.args.local_rank in [-1, 0]:
                        if (self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0) or (
                            global_step == 1 and self.args.logging_first_step
                        ):
                            logs = {}
                            if self.args.evaluate_during_training:
                                results = self.evaluate()
                                for key, value in results.items():
                                    eval_key = "eval_{}".format(key)
                                    logs[eval_key] = value

                            loss_scalar = (tr_loss - logging_loss) / self.args.logging_steps

                            logs["loss"] = loss_scalar
                            logging_loss = tr_loss

                            epoch_iterator.write(json.dumps({**logs, **{"step": global_step}}))

                if self.args.max_steps > 0 and global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and global_step > self.args.max_steps:
                train_iterator.close()
                break

        logger.info("\n\nTraining completed. \n\n")
        return TrainOutput(global_step, tr_loss / global_step)

    def _training_step(self, model, inputs: Dict[str, torch.Tensor]) -> float:
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        outputs = model.train_step(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        return loss.item()

    def save_model(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_as_onnx(os.path.join(output_dir, "transformer.onnx"))

    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation and return metrics.

        Returns:
            A dict containing:
                - the eval loss
                - the potential metrics computed from the predictions
        """
        eval_dataloader = self.get_eval_dataloader()

        output = self._prediction_loop(eval_dataloader, description="Evaluation")
        return output.metrics

    def predict(self, test_dataset: Dataset) -> PredictionOutput:
        """
        Run prediction and return predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in evaluate().
        """
        test_dataloader = self.get_test_dataloader(test_dataset)
        return self._prediction_loop(test_dataloader, description="Prediction")

    def _prediction_loop(self, dataloader: DataLoader, description: str) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """

        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", len(dataloader.dataset))
        logger.info("  Batch size = %d", dataloader.batch_size)
        eval_losses: List[float] = []
        preds: np.ndarray = None
        label_ids: np.ndarray = None

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "masked_lm_labels"])

            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                outputs = self.model.eval_step(**inputs)

                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            if inputs.get("labels") is not None:
                if label_ids is None:
                    label_ids = inputs["labels"].detach().cpu().numpy()
                else:
                    label_ids = np.append(label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["loss"] = np.mean(eval_losses)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)
