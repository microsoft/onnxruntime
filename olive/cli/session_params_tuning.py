# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

from olive.cli.base import (
    BaseOliveCLICommand,
    add_accelerator_options,
    add_input_model_options,
    add_logging_options,
    add_save_config_file_options,
    add_shared_cache_options,
    add_telemetry_options,
    get_input_model_config,
    update_accelerator_options,
    update_shared_cache_options,
)
from olive.common.utils import set_nested_dict_value
from olive.telemetry import action


class SessionParamsTuningCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "tune-session-params",
            help=(
                "Automatically tune the session parameters for a given onnx model. "
                "Currently, for onnx model converted from huggingface model and used for "
                "generative tasks, user can simply provide the --model onnx_model_path "
                "--hf_model_name hf_model_name --device device_type to get the tuned session parameters."
            ),
        )

        add_input_model_options(sub_parser, enable_onnx=True, default_output_path="tuned-inference-settings")

        sub_parser.add_argument(
            "--cpu_cores",
            type=int,
            default=None,
            help="CPU cores used for thread tuning.",
        )
        sub_parser.add_argument(
            "--io_bind",
            action="store_true",
            help="Whether enable IOBinding Search for ONNX Runtime inference.",
        )
        sub_parser.add_argument(
            "--enable_cuda_graph",
            action="store_true",
            help="Whether enable CUDA Graph for CUDA execution provider.",
        )
        sub_parser.add_argument(
            "--execution_mode_list", type=int, nargs="*", help="Parallelism list between operators."
        )
        sub_parser.add_argument("--opt_level_list", type=int, nargs="*", help="Optimization level list for ONNX Model.")
        sub_parser.add_argument("--trt_fp16_enable", action="store_true", help="Enable TensorRT FP16 mode.")
        sub_parser.add_argument(
            "--intra_thread_num_list", type=int, nargs="*", help="List of intra thread number for test."
        )
        sub_parser.add_argument(
            "--inter_thread_num_list", type=int, nargs="*", help="List of inter thread number for test."
        )
        sub_parser.add_argument(
            "--extra_session_config",
            type=json.loads,
            default=None,
            help=(
                "Extra customized session options during tuning process. It should be a json string."
                'E.g. --extra_session_config \'{"key1": "value1", "key2": "value2"}\''
            ),
        )
        sub_parser.add_argument(
            "--disable_force_evaluate_other_eps",
            action="store_true",
            help=(
                "Whether force to evaluate all execution providers"
                " which are different with the associated execution provider."
            ),
        )
        sub_parser.add_argument(
            "--enable_profiling",
            action="store_true",
            help="Whether enable profiling for ONNX Runtime inference.",
        )

        sub_parser.add_argument(
            "--predict_with_kv_cache",
            action="store_true",
            help="Whether to use key-value cache for ORT session parameter tuning",
        )

        add_accelerator_options(sub_parser, single_provider=False)
        add_logging_options(sub_parser)
        add_save_config_file_options(sub_parser)
        add_shared_cache_options(sub_parser)
        add_telemetry_options(sub_parser)
        sub_parser.set_defaults(func=SessionParamsTuningCommand)

    def _update_pass_config(self, default_pass_config) -> dict:
        pass_config = deepcopy(default_pass_config)
        pass_config_keys = (
            "cpu_cores",
            "io_bind",
            "enable_cuda_graph",
            "execution_mode_list",
            "opt_level_list",
            "trt_fp16_enable",
            "intra_thread_num_list",
            "inter_thread_num_list",
            "extra_session_config",
        )
        args_dict = vars(self.args)
        pass_config.update({k: args_dict[k] for k in pass_config_keys if args_dict[k] is not None})
        return pass_config

    def _get_run_config(self, tempdir) -> dict:
        config = deepcopy(TEMPLATE)

        session_params_tuning_key = ("passes", "session_params_tuning")

        to_replace = [
            ("input_model", get_input_model_config(self.args)),
            (session_params_tuning_key, self._update_pass_config(config["passes"]["session_params_tuning"])),
            ("output_dir", self.args.output_path),
            ("log_severity_level", self.args.log_level),
        ]

        for k, v in to_replace:
            if v is not None:
                set_nested_dict_value(config, k, v)

        update_accelerator_options(self.args, config, single_provider=False)
        update_shared_cache_options(config, self.args)
        return config

    @action
    def run(self):
        workflow_output = self._run_workflow()

        output_path = Path(self.args.output_path).resolve()
        output_models = workflow_output.get_output_models()
        if len(output_models) < 1:
            device = workflow_output.from_device()
            print(f"Tuning for {device} failed. Please set the log_level to 1 for more detailed logs.")
        else:
            for model_output in output_models:
                acc_ep = f"{model_output.from_device().lower()}-{model_output.from_execution_provider()[:-17].lower()}"
                infer_setting_output_path = output_path / f"{acc_ep}.json"
                infer_settings = model_output.get_inference_config()
                with infer_setting_output_path.open("w") as f:
                    json.dump(infer_settings, f, indent=4)
            print(f"Inference session parameters are saved to {output_path}.")
        return workflow_output


TEMPLATE = {
    "input_model": {"type": "ONNXModel"},
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
        }
    },
    "data_configs": [
        {
            "name": "session_params_tuning_data",
            "type": "dummydatacontainer",
            "load_dataset_config": {
                "type": "dummy_dataset",
                "input_shapes": [(1, 1)],
                "input_names": ["input"],
                "max_samples": 32,
            },
        }
    ],
    "passes": {
        "session_params_tuning": {"type": "OrtSessionParamsTuning", "data_config": "session_params_tuning_data"},
    },
    "host": "local_system",
    "target": "local_system",
    "no_artifacts": True,
}
