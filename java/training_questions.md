# Questions

- Should I prevent calls to OptimizerStep after ResetGrad is called? The docs say it registers a hook on TrainStep to clear the grads and it might be counterintuitive for the optimizer step to not fail.
- What is the lifetime of an OrtCheckpointState? Does it need to outlive the OrtTrainingSession it is passed into?
- Does GetLearningRate return the learning rate taking into account the scheduler, or just the custom one if set?
- How do we figure out the buffer sizing for CopyParametersToBuffer? The OrtValues need types and shapes, but we can only find the total size via GetParametersSize.
- Is GetParametersSize the amount of memory used, or the number of parameters?
- In RegisterLinearLRScheduler what is the default initial learning rate?
- How do I query the train and eval models for their input names/lengths?
- Does the save checkpoint function expect the directory to be created first, or will it create it itself?