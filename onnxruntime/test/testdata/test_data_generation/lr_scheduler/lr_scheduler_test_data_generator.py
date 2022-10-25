# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""This file is used to generate test data for LR scheduler optimizer tests in
   orttraining/orttraining/test/training_api/core/training_api_tests.cc."""

import torch
from torch.optim.lr_scheduler import LambdaLR


class SingleParameterModule(torch.nn.Module):
    """A dummy module containing only one trainable parameter."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, input1):
        """Module forward call."""
        out = self.fc1(input1)
        return out


class WarmupLinearSchedule(LambdaLR):
    """Linear warmup and then linear decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """

    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        print(f"warmup_step_count_: {self.warmup_steps }, step: {step}, total_step_count_: {self.t_total}")
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


def main():
    """Main entry."""
    num_training_steps = 100
    seed = 8888
    device = "cuda"
    batch_size, dimension_in, dimension_hidden = 2, 2, 3

    input = torch.randn(batch_size, dimension_in, device=device, dtype=torch.float32)
    save_ckpt_step = num_training_steps // 2
    num_warmup_step_data_dict = {
        "0": 0,
        "30": 30,
        # checkpoint at 50 step.
        "70": 70,
        "200": 200,
    }

    for warmup_name, num_warmup_steps in num_warmup_step_data_dict.items():
        pt_model = SingleParameterModule(dimension_in, dimension_hidden).to(device)

        import tempfile

        fp = tempfile.NamedTemporaryFile()

        adamw_optimizer = torch.optim.AdamW(pt_model.parameters(), lr=1e-3)
        scheduler = WarmupLinearSchedule(adamw_optimizer, num_warmup_steps, num_training_steps)
        data = []
        for i in range(num_training_steps):
            data.append([scheduler.last_epoch, scheduler.get_last_lr()])
            prediction = pt_model(input)
            loss = prediction.sum()
            loss.backward()
            adamw_optimizer.step()
            adamw_optimizer.zero_grad()
            scheduler.step()

            if i == save_ckpt_step:
                torch.save(
                    {
                        "optimizer": adamw_optimizer.state_dict(),
                        "lr_scheduler": scheduler.state_dict(),
                    },
                    fp.name,
                )

        import json

        json_file_name = f"warmup_linear_scheduler_warmupstep-{warmup_name}.json"
        with open(json_file_name, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        data = []
        state_dict = torch.load(fp.name)
        new_adamw_optimizer = torch.optim.AdamW(pt_model.parameters(), lr=1e-3)
        new_adamw_optimizer.load_state_dict(state_dict["optimizer"])

        new_scheduler = WarmupLinearSchedule(new_adamw_optimizer, num_warmup_steps, num_training_steps)
        new_scheduler.load_state_dict(state_dict["lr_scheduler"])
        for i in range(save_ckpt_step + 1, num_training_steps):
            data.append([new_scheduler.last_epoch, new_scheduler.get_last_lr()])
            prediction = pt_model(input)
            loss = prediction.sum()
            loss.backward()
            new_adamw_optimizer.step()
            new_adamw_optimizer.zero_grad()
            new_scheduler.step()

        import json

        json_file_name = f"warmup_linear_scheduler_warmupstep-{warmup_name}_restored.json"
        with open(json_file_name, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
