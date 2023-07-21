import os

import torch


def list_checkpoint_files(checkpoint_dir, checkpoint_prefix, extension=".ort.pt"):
    ckpt_file_names = [f for f in os.listdir(checkpoint_dir) if f.startswith(checkpoint_prefix)]
    ckpt_file_names = [f for f in ckpt_file_names if f.endswith(extension)]
    ckpt_file_names = [os.path.join(checkpoint_dir, f) for f in ckpt_file_names]

    assert len(ckpt_file_names) > 0, 'No checkpoint files found with prefix "{}" in directory {}.'.format(
        checkpoint_prefix, checkpoint_dir
    )
    return ckpt_file_names


def get_checkpoint_name(prefix, is_partitioned, world_rank=None, world_size=None):
    SINGLE_CHECKPOINT_FILENAME = "{prefix}.ort.pt"  # noqa: N806
    MULTIPLE_CHECKPOINT_FILENAME = "{prefix}.ZeRO.{world_rank}.{world_size}.ort.pt"  # noqa: N806

    if is_partitioned:
        filename = MULTIPLE_CHECKPOINT_FILENAME.format(
            prefix=prefix, world_rank=world_rank, world_size=(world_size - 1)
        )
    else:
        filename = SINGLE_CHECKPOINT_FILENAME.format(prefix=prefix)

    return filename


def _split_state_dict(state_dict):
    optimizer_keys = ["Moment_1_", "Moment_2_", "Update_Count_", "Step"]
    split_sd = {"optimizer": {}, "fp32_param": {}, "fp16_param": {}}
    for k, v in state_dict.items():
        mode = "fp32_param"
        for optim_key in optimizer_keys:
            if k.startswith(optim_key):
                mode = "optimizer"
                break
        if k.endswith("_fp16"):
            mode = "fp16_param"
        split_sd[mode][k] = v
    return split_sd


class CombineZeroCheckpoint:
    def __init__(self, checkpoint_files, clean_state_dict=None):
        assert len(checkpoint_files) > 0, "No checkpoint files passed"
        self.checkpoint_files = checkpoint_files
        self.clean_state_dict = clean_state_dict
        self.world_size = int(self.checkpoint_files[0].split("ZeRO")[1].split(".")[2]) + 1
        assert len(self.checkpoint_files) == self.world_size, f"Could not find {self.world_size} files"
        self.weight_shape_map = dict()
        self.sharded_params = set()

    def _split_name(self, name: str):
        name_split = name.split("_view_")
        view_num = None
        if len(name_split) > 1:
            view_num = int(name_split[1])
        optimizer_key = ""
        mp_suffix = ""
        if name_split[0].startswith("Moment_1"):
            optimizer_key = "Moment_1_"
        elif name_split[0].startswith("Moment_2"):
            optimizer_key = "Moment_2_"
        elif name_split[0].startswith("Update_Count"):
            optimizer_key = "Update_Count_"
        elif name_split[0].endswith("_fp16"):
            mp_suffix = "_fp16"
        param_name = name_split[0]
        if optimizer_key:
            param_name = param_name.split(optimizer_key)[1]
        param_name = param_name.split("_fp16")[0]
        return param_name, optimizer_key, view_num, mp_suffix

    def _update_weight_statistics(self, name, value):
        if name not in self.weight_shape_map:
            self.weight_shape_map[name] = value.size()  # original shape of tensor

    def _reshape_tensor(self, key):
        value = self.aggregate_state_dict[key]
        weight_name, _, _, _ = self._split_name(key)
        set_size = self.weight_shape_map[weight_name]
        self.aggregate_state_dict[key] = value.reshape(set_size)

    def _aggregate(self, param_dict):
        for k, v in param_dict.items():
            weight_name, optimizer_key, view_num, mp_suffix = self._split_name(k)
            if view_num is not None:
                # parameter is sharded
                param_name = optimizer_key + weight_name + mp_suffix

                if param_name in self.aggregate_state_dict and optimizer_key not in ["Update_Count_"]:
                    self.sharded_params.add(param_name)
                    # Found a previous shard of the param, concatenate shards ordered by ranks
                    self.aggregate_state_dict[param_name] = torch.cat((self.aggregate_state_dict[param_name], v))
                else:
                    self.aggregate_state_dict[param_name] = v
            else:
                if k in self.aggregate_state_dict:
                    assert (self.aggregate_state_dict[k] == v).all(), "Unsharded params must have the same value"
                else:
                    self.aggregate_state_dict[k] = v
                self._update_weight_statistics(weight_name, v)

    def aggregate_checkpoints(self):
        checkpoint_prefix = self.checkpoint_files[0].split(".ZeRO")[0]
        self.aggregate_state_dict = dict()

        for i in range(self.world_size):
            checkpoint_name = get_checkpoint_name(checkpoint_prefix, True, i, self.world_size)
            rank_state_dict = torch.load(checkpoint_name, map_location=torch.device("cpu"))
            if "model" in rank_state_dict:
                rank_state_dict = rank_state_dict["model"]

            if self.clean_state_dict:
                rank_state_dict = self.clean_state_dict(rank_state_dict)

            rank_state_dict = _split_state_dict(rank_state_dict)
            self._aggregate(rank_state_dict["fp16_param"])
            self._aggregate(rank_state_dict["fp32_param"])
            self._aggregate(rank_state_dict["optimizer"])

        for k in self.sharded_params:
            self._reshape_tensor(k)
        return self.aggregate_state_dict
