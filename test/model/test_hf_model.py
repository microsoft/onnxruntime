# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from pathlib import Path
from unittest.mock import ANY, patch

import huggingface_hub
import pytest
import torch
import transformers

from olive.model.config.io_config import IoConfig
from olive.model.handler.hf import HfModelHandler


# pylint: disable=attribute-defined-outside-init
class TestHFModel:
    @pytest.fixture(autouse=True)
    def setup(self):
        # hf config values
        self.task = "text-generation-with-past"
        self.model_name = "katuni4ka/tiny-random-phi3"
        self.revision = "585361abfee667f3c63f8b2dc4ad58405c4e34e2"

        self.local_path = huggingface_hub.snapshot_download(self.model_name, revision=self.revision)

    @pytest.mark.parametrize("local", [True, False])
    def test_load_model(self, local):
        olive_model = HfModelHandler(
            model_path=self.local_path if local else self.model_name,
            task=self.task,
            load_kwargs={"revision": self.revision},
        )

        pytorch_model = olive_model.load_model()
        actual_class_path = f"{pytorch_model.__module__}.{pytorch_model.__class__.__name__}"
        assert actual_class_path == "transformers.models.phi3.modeling_phi3.Phi3ForCausalLM"

    @pytest.mark.parametrize("local", [True, False])
    def test_load_model_with_kwargs(self, local):
        olive_model = HfModelHandler(
            model_path=self.local_path if local else self.model_name,
            task=self.task,
            load_kwargs={"torch_dtype": "float16"},
        )
        pytorch_model = olive_model.load_model()
        assert isinstance(pytorch_model, transformers.Phi3ForCausalLM)
        assert pytorch_model.dtype == torch.float16

    @pytest.mark.parametrize("local", [True, False])
    def test_model_name_or_path(self, local):
        olive_model = HfModelHandler(model_path=self.local_path if local else self.model_name, task=self.task)
        assert olive_model.model_name_or_path == str(Path(self.local_path).resolve()) if local else self.model_name

    @pytest.mark.parametrize("local", [True, False])
    @pytest.mark.parametrize("trust_remote_code", [True, False])
    @pytest.mark.parametrize("tokenizer_exists", [True, False])
    def test_save_metadata(self, local, trust_remote_code, tokenizer_exists, tmp_path):
        olive_model = HfModelHandler(
            model_path=self.local_path if local else self.model_name,
            task=self.task,
            load_kwargs={"trust_remote_code": trust_remote_code, "revision": self.revision},
        )
        if tokenizer_exists:
            olive_model.get_hf_tokenizer().save_pretrained(tmp_path)
        saved_filepaths = olive_model.save_metadata(tmp_path)
        # transformers>=5.0.0
        assert len(saved_filepaths) == (4 if tokenizer_exists else 7)
        assert all(Path(fp).exists() for fp in saved_filepaths)
        assert isinstance(transformers.AutoConfig.from_pretrained(tmp_path), transformers.Phi3Config)
        assert isinstance(transformers.AutoTokenizer.from_pretrained(tmp_path), transformers.PreTrainedTokenizerBase)

    @pytest.mark.parametrize("local", [True, False])
    def test_save_pretrained_metadata(self, local, tmp_path):
        olive_model = HfModelHandler(
            model_path=self.local_path if local else self.model_name,
            task=self.task,
            load_kwargs={"revision": self.revision},
        )

        # modify the config and save the model
        loaded_model = olive_model.load_model()
        loaded_model.config.dummy_key = "dummy_value"
        loaded_model.save_pretrained(tmp_path)

        saved_filepaths = olive_model.save_metadata(tmp_path)
        # generation config is also saved, transformers>=5.0.0
        assert len(saved_filepaths) == 6

        with open(tmp_path / "config.json") as f:
            config = json.load(f)
        # encure already saved config is used
        assert config["dummy_key"] == "dummy_value"
        # ensure the auto_map is updated
        assert config["auto_map"] == {
            "AutoConfig": "configuration_phi3.Phi3Config",
            "AutoModelForCausalLM": "modeling_phi3.Phi3ForCausalLM",
        }


@pytest.mark.parametrize("trust_remote_code", [True, False])
def test_save_metadata_with_module_files(trust_remote_code, tmp_path):
    load_kwargs = {"trust_remote_code": trust_remote_code, "revision": "585361abfee667f3c63f8b2dc4ad58405c4e34e2"}
    olive_model = HfModelHandler(
        model_path="katuni4ka/tiny-random-phi3",
        load_kwargs=load_kwargs,
    )

    saved_filepaths = olive_model.save_metadata(tmp_path)
    assert all(Path(fp).exists() for fp in saved_filepaths)
    if trust_remote_code:
        expected_class_name = f"transformers_modules.{tmp_path.name}.configuration_phi3.Phi3Config"
    else:
        expected_class_name = "transformers.models.phi3.configuration_phi3.Phi3Config"
    config = transformers.AutoConfig.from_pretrained(tmp_path, **load_kwargs)
    assert f"{config.__module__}.{config.__class__.__name__}" == expected_class_name
    assert isinstance(
        transformers.AutoTokenizer.from_pretrained(tmp_path, **load_kwargs),
        transformers.PreTrainedTokenizerBase,
    )


class TestHFDummyInput:
    @pytest.fixture(autouse=True)
    def setup(self):
        # hf config values
        self.task = "text-classification"
        self.model_name = "hf-internal-testing/tiny-random-BertForSequenceClassification"
        self.io_config = {
            "input_names": ["input_ids", "attention_mask", "token_type_ids"],
            "input_shapes": [[1, 128], [1, 128], [1, 128]],
            "input_types": ["int64", "int64", "int64"],
            "output_names": ["output"],
            "dynamic_axes": {
                "input_ids": {"0": "batch_size", "1": "seq_length"},
                "attention_mask": {"0": "batch_size", "1": "seq_length"},
                "token_type_ids": {"0": "batch_size", "1": "seq_length"},
            },
            "dynamic_shapes": {
                "input_ids": {"0": "batch_size", "1": ["seq_length", 1, 256]},
                "attention_mask": {"0": "batch_size", "1": "seq_length"},
                "token_type_ids": {"0": "batch_size", "1": "seq_length"},
            },
        }

    def test_dummy_input_with_kv_cache(self):
        io_config = self.io_config
        io_config["kv_cache"] = True
        olive_model = HfModelHandler(model_path=self.model_name, task=self.task, io_config=io_config)
        dummy_inputs = olive_model.get_dummy_inputs()
        # len(["input_ids", "attention_mask", "token_type_ids"]) + 2 * num_hidden_layers
        assert len(dummy_inputs) == 3 + 5 * 2
        assert list(dummy_inputs["past_key_values.0.key"].shape) == [1, 4, 0, 8]

    def test_dummy_input_with_kv_cache_dict(self):
        io_config = self.io_config
        io_config["kv_cache"] = {"batch_size": 1}
        olive_model = HfModelHandler(model_path=self.model_name, task=self.task, io_config=io_config)
        dummy_inputs = olive_model.get_dummy_inputs()
        # len(["input_ids", "attention_mask", "token_type_ids"]) + 2 * num_hidden_layers
        assert len(dummy_inputs) == 3 + 5 * 2
        assert list(dummy_inputs["past_key_values.0.key"].shape) == [1, 4, 0, 8]

    def test_dynamic_shapes_is_generated_when_kv_cache_is_true(self):
        io_config = self.io_config
        io_config["kv_cache"] = True
        olive_model = HfModelHandler(model_path=self.model_name, task=self.task, io_config=io_config)
        io_config = olive_model.io_config
        assert "dynamic_shapes" in io_config
        assert "past_key_values" in io_config["dynamic_shapes"]
        assert len(io_config["dynamic_shapes"]["past_key_values"]) == 5
        assert len(io_config["dynamic_shapes"]["past_key_values"][0]) == 2
        assert io_config["dynamic_shapes"]["past_key_values"][0][0] == {0: "batch_size", 2: "past_sequence_length"}

    def test_dict_io_config(self):
        olive_model = HfModelHandler(model_path=self.model_name, task=self.task, io_config=self.io_config)
        # get io config
        io_config = olive_model.io_config
        assert io_config == IoConfig(**self.io_config).model_dump(exclude_none=True)

    @patch("olive.model.handler.mixin.hf.get_model_io_config")
    def test_hf_config_io_config(self, get_model_io_config):
        get_model_io_config.return_value = self.io_config
        olive_model = HfModelHandler(model_path=self.model_name, task=self.task)
        # get io config
        io_config = olive_model.io_config
        assert io_config == self.io_config
        get_model_io_config.assert_called_once_with(self.model_name, self.task, olive_model.load_model())

    @patch("olive.data.template.dummy_data_config_template")
    def test_input_shapes_dummy_inputs(self, dummy_data_config_template):
        olive_model = HfModelHandler(model_path=self.model_name, task=self.task, io_config=self.io_config)

        dummy_data_config_template.return_value.to_data_container.return_value.get_first_batch.return_value = 1, 0

        # get dummy inputs
        dummy_inputs = olive_model.get_dummy_inputs()

        dummy_data_config_template.assert_called_once_with(
            input_shapes=self.io_config["input_shapes"],
            input_types=self.io_config["input_types"],
            input_names=self.io_config["input_names"],
        )
        dummy_data_config_template.return_value.to_data_container.assert_called_once()
        dummy_data_config_template.return_value.to_data_container.return_value.get_first_batch.assert_called_once()
        assert dummy_inputs == 1

    @patch("olive.model.handler.mixin.hf.get_model_dummy_input")
    def test_hf_onnx_config_dummy_inputs(self, get_model_dummy_input):
        get_model_dummy_input.return_value = 1
        olive_model = HfModelHandler(model_path=self.model_name, task=self.task)
        # get dummy inputs
        dummy_inputs = olive_model.get_dummy_inputs()

        get_model_dummy_input.assert_called_once_with(self.model_name, self.task, model=ANY)
        assert dummy_inputs == 1
