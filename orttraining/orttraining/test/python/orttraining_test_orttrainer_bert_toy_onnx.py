
# generate sample input for our example
import inspect
import onnx
import os
import math
import pytest
import copy
import torch

from numpy.testing import assert_allclose

from onnxruntime.capi._pybind_state import set_seed
from onnxruntime.capi.ort_trainer import IODescription as Legacy_IODescription,\
                                         ModelDescription as Legacy_ModelDescription,\
                                         LossScaler as Legacy_LossScaler,\
                                         ORTTrainer as Legacy_ORTTrainer
from onnxruntime.capi.training import _utils, amp, optim, orttrainer, TrainStepInfo,\
                                      model_desc_validation as md_val,\
                                      orttrainer_options as orttrainer_options

import _test_helpers

###############################################################################
# Helper functions ############################################################
###############################################################################


# Generates a sample input for the BERT model using the model desc.
# Note: this is not a general method that can be applied to any model.
def generate_random_input_from_model_desc(desc, seed=1, device = "cuda:0"):
    torch.manual_seed(seed)
    set_seed(seed)
    dtype = torch.int64
    vocab_size = 30528
    num_classes = [vocab_size, 2, 2, vocab_size, 2]
    dims = {"batch_size":16, "seq_len":1}
    sample_input = []
    for index, input in enumerate(desc['inputs']):
        size = []
        for s in input[1]:
            if isinstance(s, (int)):
                size.append(s)
            else:
                size.append(dims[s] if s in dims else 1)
        sample_input.append(torch.randint(0,
                            num_classes[index],
                            tuple(size),
                            dtype=torch.int64).to(device))
    return sample_input

# Creates the model description dictionary with static dimensions
def bert_model_description(dynamic_shape=True):
    if dynamic_shape:
        model_desc = {'inputs': [('input_ids', ['batch_size', 'seq_len']),
                                 ('segment_ids', ['batch_size', 'seq_len'],),
                                 ('input_mask', ['batch_size', 'seq_len'],),
                                 ('masked_lm_labels', ['batch_size', 'seq_len'],),
                                 ('next_sentence_labels', ['batch_size', ],)],
                                 'outputs': [('loss', [], True)]}
    else:
        batch_size = 16
        seq_len = 1
        model_desc = {'inputs': [('input_ids', [batch_size, seq_len]),
                                ('segment_ids', [batch_size, seq_len],),
                                ('input_mask', [batch_size, seq_len],),
                                ('masked_lm_labels', [batch_size, seq_len],),
                                ('next_sentence_labels', [batch_size, ],)],
                    'outputs': [('loss', [], True)]}
    return model_desc

# A method to assign different hyper parameters for different parameter groups
def optimizer_parameters(model):
    no_decay_keys = ["bias", "gamma", "beta", "LayerNorm"]
    no_decay_param_group = []
    for initializer in model.graph.initializer:
        if any(key in initializer.name for key in no_decay_keys):
            no_decay_param_group.append(initializer.name)
    params = [{'params': no_decay_param_group, "alpha": 0.9, "beta": 0.999, "lambda_coef": 0.0, "epsilon": 1e-6}]
    return params

def load_bert_onnx_model():
    pytorch_transformer_path = os.path.join('..', '..', '..', 'onnxruntime', 'test', 'testdata')
    bert_onnx_model_path = os.path.join(pytorch_transformer_path, "bert_toy_postprocessed.onnx")
    model = onnx.load(bert_onnx_model_path)
    return model

class CustomLossScaler(amp.LossScaler):
    def __init__(self, loss_scale=float(1 << 16)):
        super().__init__(loss_scale)
        self._initial_loss_scale = loss_scale
        self.loss_scale = loss_scale

    def reset(self):
        self.loss_scale = self._initial_loss_scale

    def update(self, train_step_info):
        self.loss_scale *= 0.9
        return self.loss_scale

# LEGACY HELPER FUNCTIONS

class LegacyCustomLossScaler():
    def __init__(self, loss_scale=float(1 << 16)):
        self._initial_loss_scale = loss_scale
        self.loss_scale_ = loss_scale

    def reset(self):
        self.loss_scale_ = self._initial_loss_scale

    def update_loss_scale(self, is_all_finite):
        self.loss_scale_ *= 0.9

def legacy_model_params(use_simple_model_desc=True, device = torch.device("cuda", 0)):
    legacy_model_desc = legacy_bert_model_description()
    simple_model_desc = legacy_remove_extra_info(legacy_model_desc) if use_simple_model_desc else legacy_model_desc
    learning_rate_description = legacy_ort_trainer_learning_rate_description()
    lr = 0.001
    learning_rate = torch.tensor([lr]).to(device)
    return (simple_model_desc, learning_rate_description, learning_rate)

# in optimizer_config.params
def legacy_map_optimizer_attributes(name):
    print(name)
    no_decay_keys = ["bias", "gamma", "beta", "LayerNorm"]
    no_decay = any(no_decay_key in name for no_decay_key in no_decay_keys)
    if no_decay:
        return {"alpha": 0.9, "beta": 0.999, "lambda": 0.0, "epsilon": 1e-6}
    else:
        return {"alpha": 0.9, "beta": 0.999, "lambda": 0.01, "epsilon": 1e-6}

# Legacy lr and model desc
def legacy_ort_trainer_learning_rate_description():
    return Legacy_IODescription('Learning_Rate', [1, ], torch.float32)

def legacy_bert_model_description():
    vocab_size = 30528
    input_ids_desc = Legacy_IODescription('input_ids', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes=vocab_size)
    segment_ids_desc = Legacy_IODescription('segment_ids', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes=2)
    input_mask_desc = Legacy_IODescription('input_mask', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes=2)
    masked_lm_labels_desc = Legacy_IODescription('masked_lm_labels', ['batch', 'max_seq_len_in_batch'], torch.int64,
                                          num_classes=vocab_size)
    next_sentence_labels_desc = Legacy_IODescription('next_sentence_labels', ['batch', ], torch.int64, num_classes=2)
    loss_desc = Legacy_IODescription('loss', [], torch.float32)

    return Legacy_ModelDescription([input_ids_desc, segment_ids_desc, input_mask_desc, masked_lm_labels_desc,
                             next_sentence_labels_desc], [loss_desc])

def legacy_remove_extra_info(model_desc):
    simple_model_desc = copy.deepcopy(model_desc)
    for input_desc in simple_model_desc.inputs_:
        input_desc.dtype_ = None
        input_desc.num_classes_ = None
    for output_desc in simple_model_desc.outputs_:
        output_desc.dtype_ = None
        output_desc.num_classes_ = None
    return simple_model_desc

def constantlrscheduler_1(global_step):
    return constantlrscheduler(global_step, 1.0)

def constantlrscheduler_5(global_step):
    return constantlrscheduler(global_step, 0.5)

def constantlrscheduler(global_step, initial_lr):
    warmup = 0.5
    total_steps = 10
    lr = initial_lr
    for i in range(global_step+1):
        x = (i+1) / (total_steps+1)
        if x < warmup:
            warmup_val = x/warmup
        else:
            warmup_val =1
        lr *= warmup_val
    return lr

def cosinelrscheduler(global_step):
    initial_lr = 1.0
    warmup = 0.5
    total_steps = 10
    lr = initial_lr
    for i in range(global_step+1):
        x = (i+1) / (total_steps+1)
        if x < warmup:
            warmup_val = x/warmup
        else:
            warmup_val = 0.5 * (1.0 + math.cos(math.pi * x))
        lr *= warmup_val
    return lr

def linearlrscheduler(global_step):
    initial_lr = 1.0
    warmup = 0.5
    total_steps = 10
    lr = initial_lr
    for i in range(global_step+1):
        x = (i+1) / (total_steps+1)
        if x < warmup:
            warmup_val = x/warmup
        else:
            warmup_val = max((x - 1.0) / (warmup - 1.0), 0.0)
        lr *= warmup_val
    return lr

def polylrscheduler(global_step):
    initial_lr = 1.0
    warmup = 0.5
    total_steps = 10
    degree = 0.5
    lr = initial_lr
    for i in range(global_step+1):
        x = (i+1) / (total_steps+1)
        if x < warmup:
            warmup_val = x/warmup
        else:
            warmup_val = (1.0 - x) ** degree
        lr *= warmup_val
    return lr

def legacy_optim_params_a(name):
    return {"alpha": 0.9, "beta": 0.999, "lambda": 0.01, "epsilon": 1e-6}

def legacy_optim_params_b(name):
    params = []
    if name in params:
        return {"alpha": 0.9, "beta": 0.999, "lambda": 0.0, "epsilon": 1e-6}
    return {"alpha": 0.9, "beta": 0.999, "lambda": 0.01, "epsilon": 1e-6}

def legacy_optim_params_c(name):
    params_group = optimizer_parameters(load_bert_onnx_model())
    if name in params_group[0]['params']:
        return {"alpha": 0.9, "beta": 0.999, "lambda": 0.0, "epsilon": 1e-6}
    return {"alpha": 0.9, "beta": 0.999, "lambda": 0.01, "epsilon": 1e-6}
    

###############################################################################
# Testing starts here #########################################################
###############################################################################


@pytest.mark.parametrize("dynamic_shape", [
    (True),
    (False)
])
def testToyBERTModelSimpleTrainStep(dynamic_shape):
    model_desc = bert_model_description(dynamic_shape)
    model = load_bert_onnx_model()

    optim_config = optim.LambConfig()
    opts =  orttrainer.ORTTrainerOptions({})
    
    # Instantiate ORTTrainer
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    # Generate random sample batch using dimensions
    sample_input = generate_random_input_from_model_desc(model_desc)

    output = trainer.train_step(*sample_input)
    assert output.shape == torch.Size([]) 

@pytest.mark.parametrize("expected_losses", [
    ([10.988012313842773, 10.99226188659668, 11.090812683105469, 11.042860984802246, 10.988919258117676,
      11.105875015258789, 10.981894493103027, 11.081543922424316, 10.997451782226562, 11.10739517211914])
])
def testToyBERTDeteministicCheck(expected_losses):
    num_batches = 10

    model_desc = bert_model_description()
    model = load_bert_onnx_model()

    params = optimizer_parameters(model)
    #optim_config = optim.LambConfig(params, alpha= 0.9, beta= 0.999, lambda_coef= 0.01, epsilon= 1e-6)
    optim_config = optim.LambConfig()
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
    })
    
    torch.manual_seed(1)
    set_seed(1)
    # Instantiate ORTTrainer
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
    
    experimental_losses = []
    for i in range(num_batches):
        sample_input = generate_random_input_from_model_desc(model_desc, i)

        experimental_losses.append(trainer.train_step(*sample_input).cpu().item())

    # check that losses match with experimental losses (1e-4)
    _test_helpers.assert_model_outputs(experimental_losses, expected_losses)

@pytest.mark.parametrize("initial_lr, lr_scheduler, expected_learning_rates, expected_losses", [
    (1.0, optim.lr_scheduler.ConstantWarmupLRScheduler, [0.18181818181818182, 0.06611570247933884, 0.03606311044327573, 0.026227716686018716, 0.02384337880547156, 0.02384337880547156, 0.02384337880547156, 0.02384337880547156, 0.02384337880547156, 0.02384337880547156], [10.988012313842773, 11.637386322021484, 11.099013328552246, 11.055734634399414, 11.145816802978516, 10.974218368530273, 10.971613883972168, 11.203381538391113, 11.131250381469727, 11.017223358154297]),
    (0.5, optim.lr_scheduler.ConstantWarmupLRScheduler, [0.09090909090909091, 0.03305785123966942, 0.018031555221637866, 0.013113858343009358, 0.01192168940273578, 0.01192168940273578, 0.01192168940273578, 0.01192168940273578, 0.01192168940273578, 0.01192168940273578], [10.988012313842773, 11.310077667236328, 11.025278091430664, 10.988797187805176, 11.125761032104492, 10.958372116088867, 10.980047225952148, 11.175304412841797, 11.147686958312988, 11.10694694519043]),
    (1.0, optim.lr_scheduler.CosineWarmupLRScheduler, [0.18181818181818182, 0.06611570247933884, 0.03606311044327573, 0.026227716686018716, 0.02384337880547156, 0.010225056103441101, 0.0029887071446425494, 0.0005157600951772063, 4.093754650801759e-05, 8.291291382790071e-07], [10.988012313842773, 11.637386322021484, 11.099013328552246, 11.05573558807373, 11.145816802978516, 10.974218368530273, 10.964020729064941, 11.190014839172363, 11.16644287109375, 11.150431632995605]),
    (1.0, optim.lr_scheduler.LinearWarmupLRScheduler, [0.18181818181818182, 0.06611570247933884, 0.03606311044327573, 0.026227716686018716, 0.02384337880547156, 0.021675798914065056, 0.015764217392047315, 0.008598664032025808, 0.0031267869207366565, 0.0005685067128612105], [10.988012313842773, 11.637386322021484, 11.099013328552246, 11.05573558807373, 11.145816802978516, 10.974218368530273, 10.970070838928223, 11.198983192443848, 11.134098052978516, 11.067017555236816]),
    (1.0, optim.lr_scheduler.PolyWarmupLRScheduler, [0.18181818181818182, 0.06611570247933884, 0.03606311044327573, 0.026227716686018716, 0.02384337880547156, 0.01607520271130791, 0.009693711967693117, 0.005062375970537139, 0.0021586043667598935, 0.0006508437050332076], [10.988012313842773, 11.637386322021484, 11.099013328552246, 11.055734634399414, 11.145816802978516, 10.974217414855957, 10.96664810180664, 11.193868637084961, 11.14560604095459, 11.097070693969727])
])
def testToyBERTModelLRScheduler(initial_lr, lr_scheduler, expected_learning_rates, expected_losses):
    model_desc = bert_model_description()
    model = load_bert_onnx_model()

    total_steps = 10
    optim_config = optim.LambConfig(lr=initial_lr)
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'lr_scheduler' : lr_scheduler(total_steps=total_steps, warmup=0.5)
    })
    
    torch.manual_seed(1)
    set_seed(1)

    # Instantiate ORTTrainer
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
  
    losses = []
    learning_rates = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        losses.append(trainer.train_step(*sample_input).cpu().item())
        learning_rates.append(trainer.options.lr_scheduler.get_last_lr()[0])
    
    _test_helpers.assert_model_outputs(learning_rates, expected_learning_rates)
    _test_helpers.assert_model_outputs(losses, expected_losses)
    

@pytest.mark.parametrize("params, expected_losses", [
    # Change the hyper parameters for all parameters
    ([], [10.988012313842773, 10.99226188659668, 11.090811729431152, 11.042859077453613, 10.98891830444336, 11.10587215423584, 10.981895446777344, 11.081542015075684, 10.997452735900879, 11.107390403747559]),
    # Change the hyperparameters for a subset of hardcoded parameters
    ([{'params':["bert.embeddings.LayerNorm.weight", "bert.encoder.layer.0.attention.output.dense.weight"], "alpha": 0.9, "beta": 0.999, "lambda_coef": 0.0, "epsilon": 1e-6}], [10.988012313842773, 10.99226188659668, 11.090811729431152, 11.042859077453613, 10.98891830444336, 11.10587215423584, 10.981895446777344, 11.081542015075684, 10.997452735900879, 11.107390403747559]),
    # Change the hyperparameters for a generated set of paramers
    (optimizer_parameters(load_bert_onnx_model()), [10.988012313842773, 10.99226188659668, 11.090812683105469, 11.042858123779297, 10.98891830444336, 11.105875015258789, 10.981894493103027, 11.081543922424316, 10.997452735900879, 11.10739517211914])
])
def testToyBERTModelCustomOptimParameters(params, expected_losses):
    total_steps = 10

    model_desc = bert_model_description()
    model = load_bert_onnx_model()

    optim_config = optim.LambConfig(params, alpha= 0.9, beta= 0.999, lambda_coef= 0.01, epsilon= 1e-6)
    optim_config = optim.LambConfig()
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
    })
    
    torch.manual_seed(1)
    set_seed(1)
    # Instantiate ORTTrainer
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)

        losses.append(trainer.train_step(*sample_input).cpu().item())

    _test_helpers.assert_model_outputs(losses, expected_losses, rtol=1e-6)

# TODO: check if we need another test for explicitly setting the loss scaler
# Dynamic Loss Scaler implemented implicitly
@pytest.mark.parametrize("loss_scaler", [
    (None),
    (amp.DynamicLossScaler()),
    (CustomLossScaler())
])
def testToyBERTModelMixedPrecisionLossScaler(loss_scaler):
    total_steps = 10

    model_desc = bert_model_description()
    model = load_bert_onnx_model()

    optim_config = optim.LambConfig()
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'mixed_precision': {
            'enabled': True,
            'loss_scaler': loss_scaler
        }
    })
    
    # Instantiate ORTTrainer
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
    
    losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)

        losses.append(trainer.train_step(*sample_input).cpu().item())


###############################################################################
# Temporary tests comparing Legacy vs Experimental ORTTrainer APIs ############
###############################################################################


def testToyBERTModelLegacyExperimental():
    num_batches = 10

    model_desc = bert_model_description()
    model = load_bert_onnx_model()

    params = optimizer_parameters(model)
    #optim_config = optim.LambConfig(params, alpha= 0.9, beta= 0.999, lambda_coef= 0.01, epsilon= 1e-6)
    optim_config = optim.LambConfig()
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
    })
    
    torch.manual_seed(1)
    set_seed(1)
    # Instantiate ORTTrainer
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
    
    experimental_losses = []
    for i in range(num_batches):
        sample_input = generate_random_input_from_model_desc(model_desc, i)

        experimental_losses.append(trainer.train_step(*sample_input).cpu().item())

    # LEGACY IMPLEMENTATION
    device = torch.device("cuda", 0)
    legacy_model_desc, learning_rate_description, learning_rate = legacy_model_params() 
    torch.manual_seed(1)
    set_seed(1)
    
    legacy_trainer = Legacy_ORTTrainer(model, None, legacy_model_desc, "LambOptimizer",
                       None,
                       learning_rate_description,
                       device)
    legacy_losses = []
    for i in range(num_batches):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        legacy_sample_input = [*sample_input, learning_rate]

        legacy_losses.append(legacy_trainer.train_step(legacy_sample_input).cpu().item())

    _test_helpers.assert_model_outputs(experimental_losses, legacy_losses, True, rtol=1e-4)

@pytest.mark.parametrize("initial_lr, lr_scheduler, legacy_lr_scheduler", [
    (1.0, optim.lr_scheduler.ConstantWarmupLRScheduler, constantlrscheduler_1),
    (0.5, optim.lr_scheduler.ConstantWarmupLRScheduler, constantlrscheduler_5),
    (1.0, optim.lr_scheduler.CosineWarmupLRScheduler, cosinelrscheduler),
    (1.0, optim.lr_scheduler.LinearWarmupLRScheduler, linearlrscheduler),
    (1.0, optim.lr_scheduler.PolyWarmupLRScheduler, polylrscheduler),
])
def testToyBERTModelLRSchedulerLegacyExperimental(initial_lr, lr_scheduler, legacy_lr_scheduler):
    model_desc = bert_model_description()
    model = load_bert_onnx_model()

    total_steps = 10
    optim_config = optim.LambConfig(lr=initial_lr)
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'lr_scheduler' : lr_scheduler(total_steps=total_steps, warmup=0.5)
    })
    
    torch.manual_seed(1)
    set_seed(1)

    # Instantiate ORTTrainer
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
  
    experimental_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        experimental_losses.append(trainer.train_step(*sample_input).cpu().item())
        
        assert trainer.options.lr_scheduler.get_last_lr()[0] == legacy_lr_scheduler(i)

    # LEGACY IMPLEMENTATION
    device = torch.device("cuda", 0)
    legacy_model_desc, learning_rate_description, learning_rate = legacy_model_params() 
    torch.manual_seed(1)
    set_seed(1)
    
    legacy_trainer = Legacy_ORTTrainer(model, None, legacy_model_desc, "LambOptimizer",
                       None,
                       learning_rate_description,
                       device,
                       _use_deterministic_compute=True,
                       get_lr_this_step=legacy_lr_scheduler)
    legacy_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)

        legacy_losses.append(legacy_trainer.train_step(sample_input).cpu().item())

    _test_helpers.assert_model_outputs(experimental_losses, legacy_losses)
    print(legacy_losses)

@pytest.mark.parametrize("params, legacy_optim_map", [
    # Change the hyper parameters for all parameters
    ([], legacy_optim_params_a),
    # Change the hyperparameters for a subset of hardcoded parameters
    ([{'params':["bert.embeddings.LayerNorm.weight", "bert.encoder.layer.0.attention.output.dense.weight"], "alpha": 0.9, "beta": 0.999, "lambda_coef": 0.0, "epsilon": 1e-6}], legacy_optim_params_b),
    # Change the hyperparameters for a generated set of paramers
    (optimizer_parameters(load_bert_onnx_model()), legacy_optim_params_c)
])
def testToyBERTModelCustomOptimParametersLegacyExperimental(params, legacy_optim_map):
    total_steps = 10

    model_desc = bert_model_description()
    model = load_bert_onnx_model()

    optim_config = optim.LambConfig(params, alpha= 0.9, beta= 0.999, lambda_coef= 0.01, epsilon= 1e-6)
    optim_config = optim.LambConfig()
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
    })
    
    torch.manual_seed(1)
    set_seed(1)
    # Instantiate ORTTrainer
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    experimental_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        experimental_losses.append(trainer.train_step(*sample_input).cpu().item())
        
    # LEGACY IMPLEMENTATION
    device = torch.device("cuda", 0)
    legacy_model_desc, learning_rate_description, learning_rate = legacy_model_params() 
    torch.manual_seed(1)
    set_seed(1)
    
    legacy_trainer = Legacy_ORTTrainer(model, None, legacy_model_desc, "LambOptimizer",
                       legacy_optim_map,
                       learning_rate_description,
                       device,
                       _use_deterministic_compute=True)
    legacy_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        legacy_sample_input = [*sample_input, learning_rate]

        legacy_losses.append(legacy_trainer.train_step(legacy_sample_input).cpu().item())

    _test_helpers.assert_model_outputs(experimental_losses, legacy_losses, rtol=1e-5)
    print(legacy_losses)

@pytest.mark.parametrize("loss_scaler, legacy_loss_scaler", [
    (None, Legacy_LossScaler("ort_test_input_loss_scaler", True)),
    (amp.DynamicLossScaler(), Legacy_LossScaler("ort_test_input_loss_scaler", True)),
    (CustomLossScaler(), LegacyCustomLossScaler())
])
def testToyBERTModelMixedPrecisionLossScalerLegacyExperimental(loss_scaler, legacy_loss_scaler):
    total_steps = 10

    model_desc = bert_model_description()
    model = load_bert_onnx_model()

    optim_config = optim.LambConfig()
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'mixed_precision': {
            'enabled': True,
            'loss_scaler': loss_scaler
        }
    })
    
    # Instantiate ORTTrainer
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
    
    experimental_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        experimental_losses.append(trainer.train_step(*sample_input).cpu().item())

    # LEGACY IMPLEMENTATION
    device = torch.device("cuda", 0)
    legacy_model_desc, learning_rate_description, learning_rate = legacy_model_params() 
    torch.manual_seed(1)
    set_seed(1)
    
    legacy_trainer = Legacy_ORTTrainer(model, None, legacy_model_desc, "LambOptimizer",
                       None,
                       learning_rate_description,
                       device,
                       _use_deterministic_compute=True,
                       use_mixed_precision=True,
                       loss_scaler = legacy_loss_scaler)
    legacy_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        legacy_sample_input = [*sample_input, learning_rate]

        legacy_losses.append(legacy_trainer.train_step(legacy_sample_input).cpu().item())

    print(experimental_losses)
    print(legacy_losses)

