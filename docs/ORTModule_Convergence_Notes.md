# ORTModule Training Convergence Investigation

## 1. Discovering

Convergence issues can be identified by:
- Large discrepancies in core training metrics including training loss, evaluation loss, model specific AUC metrics.
- Runtime failures (for example when the loss scaler reaches the minimum, triggering an exception).

Before looking into this further, we should clarify a few things (if possible):
- If we change the seed for the baseline run, whether the metric diff is big?
  (Make sure the discrepancy is not introduced by randomness)
- What are the very first steps we see obvious divergence?
- Still reproducible once randomness is removed?
- Set same seeds
- Set the dropout ratio to 0
- Set compute to be deterministic and torch-comparable (TODO(pengwa): need a flag for this).


## 2. Collect Activation Statistics


### 2.1 Use `GlobalSubscriberManager` to collect `nn.Module` forward() outputs

<table>
<tr>
<th>Baseline</th>
<th>ORTModule</th>
</tr>
<tr>
<td>
<sub>

```python
from onnxruntime.training.utils.hooks import GlobalSubscriberManager, StatisticsSubscriber
GlobalSubscriberManager.subscribe(
    model, [StatisticsSubscriber(output_dir="pt_out", override_output_dir=True)]
)
```

</sub>
</td>
<td>
<sub>

```python
model = ORTModule(model)
from onnxruntime.training.utils.hooks import GlobalSubscriberManager, StatisticsSubscriber
GlobalSubscriberManager.subscribe(
    model, [StatisticsSubscriber(output_dir="ort_out", override_output_dir=True)]
)
```

</sub>
</td>
</tr>

<tr>
<td>

- Run training script to the steps that trigger the divergence.
- A folder named `pt_out` is created in the current working directory.
- For each step, there is a folder containing summaries for every activation tensor.

</td>
<td>


- Run training script to the steps that trigger the divergence.
- Similarly, a folder named `ort_out` is created in the current working directory.
- `StatisticsSubscriber` can be subscribed before OR after wrapping ORTModule.

</td>
</tr>
</table>


Arguments:
- output_dir: the directory in all activation statistics files will be stored.
- `start_step` [optional]: the first step that runs subscriber actions.
- `end_step` [optional]: the end step (exclusively) that runs subscriber actions.
- `override_output_dir`: whether `output_dir` can be overridden if it already exists.
- `run_on_cpu`: whether to run the subscriber actions on CPU, this should be the last resort when inserted
    inspector node affects memory peak causing the original recipe run to fail with OOM.
- `bucket_size`: the size of the bucket to split the statistic calculation.

### 2.2 Use `inspect_activation` to collect intermediate tensors in a `nn.Module` forward()

The limitation of `GlobalSubscriberManager` is, only 'nn.Module's forward output tensors will be dumped, if you want to
dump the intermediate tensors in a `nn.Module`'s forward function, refer to the following example:

```diff
+   from onnxruntime.training.utils.hooks import inspect_activation
class BloomForCausalLM(BloomPreTrainedModel):
  def __init__(self, config: BloomConfig):
    ...

  def forward(self, input_ids, ...):
    ...
    transformer_outputs = self.transformer(...)
    hidden_states = transformer_outputs[0]
    lm_logits = self.lm_head(hidden_states)
+   lm_logits = inspect_activation("lm_logits", lm_logits)
    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
+   shift_logits = inspect_activation("shift_logits", shift_logits)
    shift_labels = labels[..., 1:].contiguous()
    batch_size, seq_length, vocab_size = shift_logits.shape
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(
        shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
    )

    return loss
```

Be noted, make sure the activation name (as the first argument of `inspect_activation`) is unique, otherwise
stat file using the activation name will be overwritten by the last write. The dumped data are stored in the `output_dir`.


### 2.3 Collect on multiple ranks

`GlobalSubscriberManager` does not explicitly handle the racing condition when multiple ranks write into the same file path,
here is the example if you want to collect statistics on multiple ranks:

```python
from onnxruntime.training.utils.hooks import GlobalSubscriberManager, StatisticsSubscriber
GlobalSubscriberManager.subscribe(model, [StatisticsSubscriber(output_dir="ort_out_" + str(torch.distributed.get_rank()),
                                          override_output_dir=True)])
```

Check [StatisticsSubscriber implementation](../orttraining/orttraining/python/training/utils/hooks/_statistics_subscriber.py) for more information.

### 2.4 Run command to generate per-step summary

```bash
python -m onnxruntime.training.utils.hooks.merge_activation_summary --pt_dir pt_out --ort_dir ort_out --output_dir /tmp/output
```

### 2.5 Manually compare the generated per-step summary to find the first big diff.
