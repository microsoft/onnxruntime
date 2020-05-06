import json
import logging
import os
import random
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
# from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm, trange

from transformers.data.data_collator import DataCollator, DefaultDataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import TrainingArguments

import onnxruntime
from orttraining_test_bert_postprocess import postprocess_model
from orttraining_test_utils import get_lr
from onnxruntime.capi.ort_trainer import ORTTrainer, LossScaler, ModelDescription, IODescription

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


class ORTTransformerTrainer:
    """
    """

    model: PreTrainedModel
    args: TrainingArguments
    data_collator: DataCollator
    train_dataset: Optional[Dataset]
    eval_dataset: Optional[Dataset]
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    prediction_loss_only: bool
    tb_writer: Optional["SummaryWriter"] = None

    def __init__(
        self,
        model: PreTrainedModel,
        model_desc: ModelDescription,
        args: TrainingArguments,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        prediction_loss_only=False,
    ):
        """
        """

        from collections import namedtuple
        MyArgs = namedtuple("MyArgs", 
            "local_rank world_size max_steps learning_rate warmup_proportion batch_size seq_len")
        lr_args = MyArgs(local_rank=0, world_size=1, max_steps=100, learning_rate=0.00001, warmup_proportion=0.01, batch_size=13, seq_len=7)

        def get_lr_this_step(global_step):
            return get_lr(lr_args, global_step)
        loss_scaler = LossScaler('loss_scale_input_name', True, up_scale_window=2000)

        def map_optimizer_attributes(name):
            no_decay_keys = ["bias", "gamma", "beta", "LayerNorm"]
            no_decay = any(no_decay_key in name for no_decay_key in no_decay_keys)
            if no_decay:
                return {"alpha": 0.9, "beta": 0.999, "lambda": 0.0, "epsilon": 1e-6}
            else:
                return {"alpha": 0.9, "beta": 0.999, "lambda": 0.01, "epsilon": 1e-6}

        model = ORTTrainer(model, None,
            model_desc, 
            "AdamOptimizer",
            # TODO: how to support AdamW
            map_optimizer_attributes=map_optimizer_attributes,
            learning_rate_description=IODescription('Learning_Rate', [1,], torch.float32),
            device=args.device,
            postprocess_model=postprocess_model,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            # BertLAMB default initial settings: b1=0.9, b2=0.999, e=1e-6
            world_rank=0, world_size=1,
            use_mixed_precision=args.fp16,
            allreduce_post_accumulation=True,
            get_lr_this_step=get_lr_this_step,
            loss_scaler=loss_scaler,
            _opset_version=12)


        self.model = model
        self.args = args
        if data_collator is not None:
            self.data_collator = data_collator
        else:
            self.data_collator = DefaultDataCollator()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.prediction_loss_only = prediction_loss_only
        if is_tensorboard_available() and self.args.local_rank in [-1, 0]:
            self.tb_writer = SummaryWriter(log_dir=self.args.logging_dir)
        if not is_tensorboard_available():
            logger.warning(
                "You are instantiating a Trainer but Tensorboard is not installed. You should consider installing it."
            )
        set_seed(self.args.seed)
        # Create output directory if needed
        if self.args.local_rank in [-1, 0]:
            os.makedirs(self.args.output_dir, exist_ok=True)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = (
            RandomSampler(self.train_dataset) if self.args.local_rank == -1 else DistributedSampler(self.train_dataset)
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator.collate_batch,
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        return DataLoader(
            eval_dataset if eval_dataset is not None else self.eval_dataset,
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

        model = self.model

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())

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
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=self.args.local_rank not in [-1, 0],
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
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    # TODO:
                    # if self.args.fp16:
                    #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    # else:
                    #     torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

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
                            learning_rate_scalar = scheduler.get_last_lr()[0]
                            logs["learning_rate"] = learning_rate_scalar
                            logs["loss"] = loss_scalar
                            logging_loss = tr_loss

                            if self.tb_writer:
                                for k, v in logs.items():
                                    self.tb_writer.add_scalar(k, v, global_step)
                            epoch_iterator.write(json.dumps({**logs, **{"step": global_step}}))

                if self.args.max_steps > 0 and global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and global_step > self.args.max_steps:
                train_iterator.close()
                break

        if self.tb_writer:
            self.tb_writer.close()

        logger.info("\n\nTraining completed. \n\n")
        return TrainOutput(global_step, tr_loss / global_step)

    def _training_step(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor]
    ) -> float:
        model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        return loss.item()

    def save_model(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_as_onnx(os.path.join(output_dir, "transformer.onnx"))

    def evaluate(
        self, eval_dataset: Optional[Dataset] = None, prediction_loss_only: Optional[bool] = None
    ) -> Dict[str, float]:
        """
        Run evaluation and return metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent.

        Args:
            eval_dataset: (Optional) Pass a dataset if you wish to override
            the one on the instance.
        Returns:
            A dict containing:
                - the eval loss
                - the potential metrics computed from the predictions
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

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

    def _prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", len(dataloader.dataset))
        logger.info("  Batch size = %d", dataloader.batch_size)
        eval_losses: List[float] = []
        preds: np.ndarray = None
        label_ids: np.ndarray = None
        self.model.eval()

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "masked_lm_labels"])

            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            if not prediction_loss_only:
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
