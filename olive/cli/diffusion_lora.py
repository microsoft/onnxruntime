# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from argparse import ArgumentParser
from copy import deepcopy
from typing import ClassVar

from olive.cli.base import (
    BaseOliveCLICommand,
    add_logging_options,
    add_save_config_file_options,
    add_shared_cache_options,
    add_telemetry_options,
    update_shared_cache_options,
)
from olive.common.utils import set_nested_dict_value
from olive.constants import DiffusersModelVariant
from olive.passes.diffusers.lora import LRSchedulerType, MixedPrecision
from olive.telemetry import action


class DiffusionLoraCommand(BaseOliveCLICommand):
    allow_unknown_args: ClassVar[bool] = True

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "diffusion-lora",
            help="Train LoRA adapters for diffusion models (SD 1.5, SDXL, Flux).",
        )

        # Model options
        sub_parser.add_argument(
            "-m",
            "--model_name_or_path",
            type=str,
            required=True,
            help="HuggingFace model name or local path (e.g., 'runwayml/stable-diffusion-v1-5').",
        )
        sub_parser.add_argument(
            "-o",
            "--output_path",
            type=str,
            default="diffusion-lora-adapter",
            help="Output path for the LoRA adapter. Default: diffusion-lora-adapter.",
        )

        # Model type
        sub_parser.add_argument(
            "--model_variant",
            type=str,
            default=DiffusersModelVariant.AUTO,
            choices=[t.value for t in DiffusersModelVariant],
            help="Type of diffusion model. Default: auto-detect.",
        )

        # LoRA options
        lora_group = sub_parser.add_argument_group("LoRA options")
        lora_group.add_argument(
            "-r",
            "--lora_r",
            type=int,
            default=16,
            help="LoRA rank. SD: 4-16, Flux: 16-64. Default: 16.",
        )
        lora_group.add_argument(
            "--alpha",
            type=float,
            default=None,
            help="LoRA alpha for scaling. Default: same as r.",
        )
        lora_group.add_argument(
            "--lora_dropout",
            type=float,
            default=0.0,
            help="LoRA dropout probability. Default: 0.0.",
        )
        lora_group.add_argument(
            "--target_modules",
            type=str,
            default=None,
            help="Target modules for LoRA (comma-separated). Default: auto-detect based on model type.",
        )

        # DreamBooth options
        db_group = sub_parser.add_argument_group("DreamBooth options")
        db_group.add_argument(
            "--dreambooth",
            action="store_true",
            help="Enable DreamBooth training for learning specific subjects.",
        )
        db_group.add_argument(
            "--instance_prompt",
            type=str,
            default=None,
            help="Fixed prompt for all images in DreamBooth mode. Required when --dreambooth is set. "
            "Example: 'a photo of sks dog'.",
        )
        db_group.add_argument(
            "--with_prior_preservation",
            action="store_true",
            help="Enable prior preservation to prevent language drift. Requires --class_prompt.",
        )
        db_group.add_argument(
            "--class_prompt",
            type=str,
            default=None,
            help="Prompt for class images in prior preservation. Required when --with_prior_preservation is set. "
            "Example: 'a photo of a dog'.",
        )
        db_group.add_argument(
            "--class_data_dir",
            type=str,
            default=None,
            help="Directory containing class images. If not provided or has fewer than --num_class_images, "
            "images will be auto-generated.",
        )
        db_group.add_argument(
            "--num_class_images",
            type=int,
            default=200,
            help="Number of class images for prior preservation. Default: 200.",
        )
        db_group.add_argument(
            "--prior_loss_weight",
            type=float,
            default=1.0,
            help="Weight of prior preservation loss. Default: 1.0.",
        )

        # Data options
        data_group = sub_parser.add_argument_group("Data options")
        data_group.add_argument(
            "-d",
            "--data_dir",
            type=str,
            help="Path to local image folder with training images.",
        )
        data_group.add_argument(
            "--data_name",
            type=str,
            help="HuggingFace dataset name (e.g., 'linoyts/Tuxemon').",
        )
        data_group.add_argument(
            "--data_split",
            type=str,
            default="train",
            help="Dataset split to use. Default: train.",
        )
        data_group.add_argument(
            "--image_column",
            type=str,
            default="image",
            help="Column name for images in HuggingFace dataset. Default: image.",
        )
        data_group.add_argument(
            "--caption_column",
            type=str,
            default=None,
            help="Column name for captions in HuggingFace dataset.",
        )
        data_group.add_argument(
            "--base_resolution",
            type=int,
            default=None,
            help="Base resolution for training. Auto-detected if model_variant is specified (SD1.5: 512, SDXL/Flux: 1024).",
        )

        # Training options
        train_group = sub_parser.add_argument_group("Training options")
        train_group.add_argument(
            "--max_train_steps",
            type=int,
            default=1000,
            help="Maximum training steps. Default: 1000.",
        )
        train_group.add_argument(
            "--learning_rate",
            type=float,
            default=1e-4,
            help="Learning rate. Default: 1e-4.",
        )
        train_group.add_argument(
            "--train_batch_size",
            type=int,
            default=1,
            help="Training batch size. Default: 1.",
        )
        train_group.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=4,
            help="Gradient accumulation steps. Default: 4.",
        )
        train_group.add_argument(
            "--mixed_precision",
            type=str,
            default=MixedPrecision.BF16,
            choices=[m.value for m in MixedPrecision],
            help="Mixed precision training. Default: bf16.",
        )
        train_group.add_argument(
            "--lr_scheduler",
            type=str,
            default=LRSchedulerType.CONSTANT,
            choices=[s.value for s in LRSchedulerType],
            help="Learning rate scheduler. Default: constant.",
        )
        train_group.add_argument(
            "--lr_warmup_steps",
            type=int,
            default=0,
            help="Learning rate warmup steps. Default: 0.",
        )
        train_group.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Random seed for reproducibility.",
        )

        # Flux-specific options
        flux_group = sub_parser.add_argument_group("Flux options")
        flux_group.add_argument(
            "--guidance_scale",
            type=float,
            default=3.5,
            help="Guidance scale for Flux training. Default: 3.5.",
        )

        # Output options
        sub_parser.add_argument(
            "--merge_lora",
            action="store_true",
            help="Merge LoRA into base model instead of saving adapter only.",
        )

        add_shared_cache_options(sub_parser)
        add_logging_options(sub_parser)
        add_save_config_file_options(sub_parser)
        add_telemetry_options(sub_parser)
        sub_parser.set_defaults(func=DiffusionLoraCommand)

    @action
    def run(self):
        return self._run_workflow()

    def _get_run_config(self, tempdir: str) -> dict:
        input_model_config = {
            "type": "DiffusersModel",
            "model_path": self.args.model_name_or_path,
        }

        # Determine data source
        if self.args.data_dir:
            load_dataset_config = {
                "type": "image_folder_dataset",
                "params": {"data_dir": self.args.data_dir},
            }
        elif self.args.data_name:
            load_dataset_config = {
                "type": "huggingface_dataset",
                "params": {
                    "data_name": self.args.data_name,
                    "split": self.args.data_split,
                    "image_column": self.args.image_column,
                },
            }
            if self.args.caption_column:
                load_dataset_config["params"]["caption_column"] = self.args.caption_column
        else:
            raise ValueError("Either --data_dir or --data_name must be provided.")

        # Build config
        config = deepcopy(TEMPLATE)
        config["data_configs"][0]["load_dataset_config"] = load_dataset_config

        # Pre-process config - auto-detect base_resolution if not specified
        base_resolution = self.args.base_resolution
        if base_resolution is None:
            # Auto-detect based on model type
            model_variant = self.args.model_variant
            if model_variant in (DiffusersModelVariant.SDXL, DiffusersModelVariant.FLUX):
                base_resolution = 1024
            elif model_variant == DiffusersModelVariant.SD:
                base_resolution = 512
            # If model_variant is "auto", leave base_resolution as None and let preprocessing use its default

        if base_resolution is not None:
            config["data_configs"][0]["pre_process_data_config"] = {
                "type": "image_lora_preprocess",
                "params": {"base_resolution": base_resolution},
            }

        # Pass config
        pass_key = ("passes", "sd_lora")
        to_replace = [
            ("input_model", input_model_config),
            ((*pass_key, "model_variant"), self.args.model_variant),
            ((*pass_key, "r"), self.args.lora_r),
            ((*pass_key, "alpha"), self.args.alpha),
            ((*pass_key, "lora_dropout"), self.args.lora_dropout),
            ((*pass_key, "dreambooth"), self.args.dreambooth),
            ((*pass_key, "instance_prompt"), self.args.instance_prompt),
            ((*pass_key, "with_prior_preservation"), self.args.with_prior_preservation),
            ((*pass_key, "class_prompt"), self.args.class_prompt),
            ((*pass_key, "class_data_dir"), self.args.class_data_dir),
            ((*pass_key, "num_class_images"), self.args.num_class_images),
            ((*pass_key, "prior_loss_weight"), self.args.prior_loss_weight),
            ((*pass_key, "merge_lora"), self.args.merge_lora),
            (
                (*pass_key, "training_args"),
                {
                    "max_train_steps": self.args.max_train_steps,
                    "learning_rate": self.args.learning_rate,
                    "train_batch_size": self.args.train_batch_size,
                    "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
                    "mixed_precision": self.args.mixed_precision,
                    "lr_scheduler": self.args.lr_scheduler,
                    "lr_warmup_steps": self.args.lr_warmup_steps,
                    "seed": self.args.seed,
                    # Flux-specific
                    "guidance_scale": self.args.guidance_scale,
                },
            ),
            ("output_dir", self.args.output_path),
            ("log_severity_level", self.args.log_level),
        ]

        if self.args.target_modules:
            to_replace.append(((*pass_key, "target_modules"), self.args.target_modules.split(",")))

        for keys, value in to_replace:
            if value is not None:
                set_nested_dict_value(config, keys, value)

        update_shared_cache_options(config, self.args)

        return config


TEMPLATE = {
    "input_model": {"type": "DiffusersModel"},
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            "accelerators": [{"device": "gpu"}],
        }
    },
    "data_configs": [
        {
            "name": "train_data",
            "type": "ImageDataContainer",
            "load_dataset_config": {},
        }
    ],
    "passes": {
        "sd_lora": {
            "type": "SDLoRA",
            "train_data_config": "train_data",
        }
    },
    "host": "local_system",
    "target": "local_system",
    "no_artifacts": True,
}
