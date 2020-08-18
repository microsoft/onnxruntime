# generate sample input for our example
import inspect
import onnx
import os
import math
import pytest
import copy
import torch

from numpy.testing import assert_allclose

from onnxruntime import set_seed
from onnxruntime.capi.ort_trainer import IODescription as Legacy_IODescription,\
                                         ModelDescription as Legacy_ModelDescription,\
                                         LossScaler as Legacy_LossScaler,\
                                         ORTTrainer as Legacy_ORTTrainer
from onnxruntime.experimental import _utils, amp, optim, orttrainer, TrainStepInfo,\
                                      model_desc_validation as md_val,\
                                      orttrainer_options as orttrainer_options

import _test_helpers


###############################################################################
# Helper functions ############################################################
###############################################################################


def generate_random_input_from_model_desc(desc, seed=1, device = "cuda:0"):
    '''Generates a sample input for the BERT model using the model desc.'''
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
        sample_input.append(torch.randint(0, num_classes[index], tuple(size), dtype=torch.int64).to(device))
    return sample_input

# EXPERIMENTAL HELPER FUNCTIONS

def bert_model_description(dynamic_shape=True):
    '''Creates the model description dictionary with static dimensions'''
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

def optimizer_parameters(model):
    '''A method to assign different hyper parameters for different model parameter groups'''
    no_decay_keys = ["bias", "gamma", "beta", "LayerNorm"]
    no_decay_param_group = []
    for initializer in model.graph.initializer:
        if any(key in initializer.name for key in no_decay_keys):
            no_decay_param_group.append(initializer.name)
    params = [{'params': no_decay_param_group, "alpha": 0.9, "beta": 0.999, "lambda_coef": 0.0, "epsilon": 1e-6}]
    return params

def load_bert_onnx_model():
    bert_onnx_model_path = os.path.join('..', '..', '..', 'onnxruntime', 'test', 'testdata', "bert_toy_postprocessed.onnx")
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

def legacy_model_params(device = torch.device("cuda", 0)):
    legacy_model_desc = legacy_bert_model_description()
    learning_rate_description = legacy_ort_trainer_learning_rate_description()
    lr = 0.001
    learning_rate = torch.tensor([lr]).to(device)
    return (legacy_model_desc, learning_rate_description, learning_rate)

    no_decay_keys = ["bias", "gamma", "beta", "LayerNorm"]
    no_decay = any(no_decay_key in name for no_decay_key in no_decay_keys)
    if no_decay:
        return {"alpha": 0.9, "beta": 0.999, "lambda": 0.0, "epsilon": 1e-6}
    else:
        return {"alpha": 0.9, "beta": 0.999, "lambda": 0.01, "epsilon": 1e-6}

def legacy_ort_trainer_learning_rate_description():
    return Legacy_IODescription('Learning_Rate', [1, ], torch.float32)

def legacy_bert_model_description():
    vocab_size = 30528
    input_ids_desc = Legacy_IODescription('input_ids', ['batch', 'max_seq_len_in_batch'])
    segment_ids_desc = Legacy_IODescription('segment_ids', ['batch', 'max_seq_len_in_batch'])
    input_mask_desc = Legacy_IODescription('input_mask', ['batch', 'max_seq_len_in_batch'])
    masked_lm_labels_desc = Legacy_IODescription('masked_lm_labels', ['batch', 'max_seq_len_in_batch'])
    next_sentence_labels_desc = Legacy_IODescription('next_sentence_labels', ['batch', ])
    loss_desc = Legacy_IODescription('loss', [])

    return Legacy_ModelDescription([input_ids_desc, segment_ids_desc, input_mask_desc, masked_lm_labels_desc,
                             next_sentence_labels_desc], [loss_desc])

def legacy_constant_lr_scheduler_1(global_step):
    return legacy_constant_lr_scheduler(global_step, 1.0)

def legacy_constant_lr_scheduler_5(global_step):
    return legacy_constant_lr_scheduler(global_step, 0.5)

def legacy_constant_lr_scheduler(global_step, initial_lr):
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

def legacy_cosine_lr_scheduler(global_step):
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

def legacy_linear_lr_scheduler(global_step):
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

def legacy_poly_lr_scheduler(global_step):
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
    params = ['bert.embeddings.LayerNorm.bias', 'bert.embeddings.LayerNorm.weight']
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
    
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    for i in range(10):
        # Generate random sample batch using dimensions
        sample_input = generate_random_input_from_model_desc(model_desc)

        output = trainer.train_step(*sample_input)
        assert output.shape == torch.Size([]) 

@pytest.mark.parametrize("expected_losses", [
    ([10.988012313842773, 10.99226188659668, 11.090812683105469, 11.042860984802246, 10.988919258117676,
      11.105875015258789, 10.981894493103027, 11.081543922424316, 10.997451782226562, 11.10739517211914])
])
def testToyBERTDeterministicCheck(expected_losses):
    train_steps = 10
    device = 'cuda'
    seed = 1

    model_desc = bert_model_description()
    model = load_bert_onnx_model()

    params = optimizer_parameters(model)
    optim_config = optim.LambConfig()
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
    })
    
    torch.manual_seed(seed)
    set_seed(seed)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
    
    experimental_losses = []
    for i in range(train_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)

        experimental_losses.append(trainer.train_step(*sample_input).cpu().item())

    _test_helpers.assert_model_outputs(experimental_losses, expected_losses)

@pytest.mark.parametrize("initial_lr, lr_scheduler, expected_learning_rates, expected_losses", [
    (1.0, optim.lr_scheduler.ConstantWarmupLRScheduler, [0.18181818181818182, 0.06611570247933884, 0.03606311044327573, 0.026227716686018716, 0.02384337880547156,\
            0.02384337880547156, 0.02384337880547156, 0.02384337880547156, 0.02384337880547156, 0.02384337880547156],
            [10.988012313842773, 11.637386322021484, 11.099013328552246, 11.055734634399414, 11.145816802978516,\
            10.974218368530273, 10.971613883972168, 11.203381538391113, 11.131250381469727, 11.017223358154297]),
    (0.5, optim.lr_scheduler.ConstantWarmupLRScheduler, [0.09090909090909091, 0.03305785123966942, 0.018031555221637866, 0.013113858343009358, 0.01192168940273578,\
            0.01192168940273578, 0.01192168940273578, 0.01192168940273578, 0.01192168940273578, 0.01192168940273578],
            [10.988012313842773, 11.310077667236328, 11.025278091430664, 10.988797187805176, 11.125761032104492,\
            10.958372116088867, 10.980047225952148, 11.175304412841797, 11.147686958312988, 11.10694694519043]),
    (1.0, optim.lr_scheduler.CosineWarmupLRScheduler, [0.18181818181818182, 0.06611570247933884, 0.03606311044327573, 0.026227716686018716, 0.02384337880547156,\
            0.010225056103441101, 0.0029887071446425494, 0.0005157600951772063, 4.093754650801759e-05, 8.291291382790071e-07],
            [10.988012313842773, 11.637386322021484, 11.099013328552246, 11.05573558807373, 11.145816802978516,\
            10.974218368530273, 10.964020729064941, 11.190014839172363, 11.16644287109375, 11.150431632995605]),
    (1.0, optim.lr_scheduler.LinearWarmupLRScheduler, [0.18181818181818182, 0.06611570247933884, 0.03606311044327573, 0.026227716686018716, 0.02384337880547156,\
            0.021675798914065056, 0.015764217392047315, 0.008598664032025808, 0.0031267869207366565, 0.0005685067128612105],
            [10.988012313842773, 11.637386322021484, 11.099013328552246, 11.05573558807373, 11.145816802978516,\
            10.974218368530273, 10.970070838928223, 11.198983192443848, 11.134098052978516, 11.067017555236816]),
    (1.0, optim.lr_scheduler.PolyWarmupLRScheduler, [0.18181818181818182, 0.06611570247933884, 0.03606311044327573, 0.026227716686018716, 0.02384337880547156,\
            0.01607520271130791, 0.009693711967693117, 0.005062375970537139, 0.0021586043667598935, 0.0006508437050332076],
            [10.988012313842773, 11.637386322021484, 11.099013328552246, 11.055734634399414, 11.145816802978516,\
            10.974217414855957, 10.96664810180664, 11.193868637084961, 11.14560604095459, 11.097070693969727])
])
def testToyBERTModelLRScheduler(initial_lr, lr_scheduler, expected_learning_rates, expected_losses):
    model_desc = bert_model_description()
    model = load_bert_onnx_model()

    device = 'cuda'
    total_steps = 10
    seed = 1
    optim_config = optim.LambConfig(lr=initial_lr)
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
        'lr_scheduler' : lr_scheduler(total_steps=total_steps, warmup=0.5)
    })
    
    torch.manual_seed(seed)
    set_seed(seed)

    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
  
    losses = []
    learning_rates = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        losses.append(trainer.train_step(*sample_input).cpu().item())
        learning_rates.append(trainer.options.lr_scheduler.get_last_lr()[0])
    
    _test_helpers.assert_model_outputs(learning_rates, expected_learning_rates)
    _test_helpers.assert_model_outputs(losses, expected_losses, rtol=1e-6)
    

# Dynamic Loss Scaler implemented implicitly
@pytest.mark.parametrize("loss_scaler, expected_losses", [
    (None, [10.98803424835205, 10.99240493774414, 11.090575218200684, 11.042827606201172, 10.988829612731934,
        11.105679512023926, 10.981969833374023, 11.08173656463623, 10.997121810913086, 11.10731315612793]),
    (amp.DynamicLossScaler(), [10.98803424835205, 10.99240493774414, 11.090575218200684, 11.042827606201172,
        10.988829612731934, 11.105679512023926, 10.981969833374023, 11.081737518310547, 10.99714183807373, 11.107304573059082]),
    (CustomLossScaler(), [10.98803424835205, 10.99240493774414, 11.090554237365723, 11.042823791503906, 10.98877239227295,
        11.105667114257812, 10.981982231140137, 11.081765174865723, 10.997125625610352, 11.107298851013184])
])
def testToyBERTModelMixedPrecisionLossScaler(loss_scaler, expected_losses):
    total_steps = 10
    device = 'cuda'
    seed = 1

    model_desc = bert_model_description()
    model = load_bert_onnx_model()

    optim_config = optim.LambConfig()
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
        'mixed_precision': {
            'enabled': True,
            'loss_scaler': loss_scaler
        }
    })
    
    torch.manual_seed(seed)
    set_seed(seed)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
    
    losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)

        losses.append(trainer.train_step(*sample_input).cpu().item())

    _test_helpers.assert_model_outputs(losses, expected_losses, rtol=1e-5)

@pytest.mark.parametrize("gradient_accumulation_steps, expected_losses", [
    (1, [10.988012313842773, 10.99226188659668, 11.090812683105469, 11.042860984802246, 10.988919258117676,
        11.105875015258789, 10.981894493103027, 11.081543922424316, 10.997451782226562, 11.10739517211914]),
    (4, [10.988012313842773, 10.99213981628418, 11.090258598327637, 11.039335250854492, 10.986993789672852,
        11.110128402709961, 10.989538192749023, 11.072074890136719, 11.001150131225586, 11.100043296813965]),
    (7, [10.988012313842773, 10.99213981628418, 11.090258598327637, 11.039335250854492, 10.993097305297852,
        11.112862586975098, 10.996183395385742, 11.072013854980469, 11.00184154510498, 11.097928047180176])
])
def testToyBERTModelGradientAccumulation(gradient_accumulation_steps, expected_losses):
    total_steps = 10
    device = "cuda"
    seed = 1

    model_desc = bert_model_description()
    model = load_bert_onnx_model()

    optim_config = optim.LambConfig()
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
        'batch' : {
            'gradient_accumulation_steps' : gradient_accumulation_steps
        },
    })
    
    torch.manual_seed(seed)
    set_seed(seed)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
    
    losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        losses.append(trainer.train_step(*sample_input).cpu().item())
    
    _test_helpers.assert_model_outputs(losses, expected_losses)


###############################################################################
# Temporary tests comparing Legacy vs Experimental ORTTrainer APIs ############
###############################################################################


def testToyBERTModelLegacyExperimentalBasicTraining():
    train_steps = 10
    device = 'cuda'
    seed = 1

    # EXPERIMENTAL API
    model_desc = bert_model_description()
    model = load_bert_onnx_model()

    params = optimizer_parameters(model)
    optim_config = optim.LambConfig()
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
    })
   
    torch.manual_seed(seed)
    set_seed(seed)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
    
    experimental_losses = []
    for i in range(train_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)

        experimental_losses.append(trainer.train_step(*sample_input).cpu().item())

    # LEGACY IMPLEMENTATION
    device = torch.device(device)
    legacy_model_desc, learning_rate_description, learning_rate = legacy_model_params() 
    torch.manual_seed(seed)
    set_seed(seed)
    
    legacy_trainer = Legacy_ORTTrainer(model, None, legacy_model_desc, "LambOptimizer",
                       None,
                       learning_rate_description,
                       device)
    legacy_losses = []
    for i in range(train_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        legacy_sample_input = [*sample_input, learning_rate]

        legacy_losses.append(legacy_trainer.train_step(legacy_sample_input).cpu().item())

    _test_helpers.assert_model_outputs(experimental_losses, legacy_losses, True, rtol=1e-4)

@pytest.mark.parametrize("initial_lr, lr_scheduler, legacy_lr_scheduler", [
    (1.0, optim.lr_scheduler.ConstantWarmupLRScheduler, legacy_constant_lr_scheduler_1),
    (0.5, optim.lr_scheduler.ConstantWarmupLRScheduler, legacy_constant_lr_scheduler_5),
    (1.0, optim.lr_scheduler.CosineWarmupLRScheduler, legacy_cosine_lr_scheduler),
    (1.0, optim.lr_scheduler.LinearWarmupLRScheduler, legacy_linear_lr_scheduler),
    (1.0, optim.lr_scheduler.PolyWarmupLRScheduler, legacy_poly_lr_scheduler),
])
def testToyBERTModelLegacyExperimentalLRScheduler(initial_lr, lr_scheduler, legacy_lr_scheduler):
    # EXPERIMENTAL API
    model_desc = bert_model_description()
    model = load_bert_onnx_model()

    total_steps = 10
    device = 'cuda'
    seed = 1
    optim_config = optim.LambConfig(lr=initial_lr)
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
        'lr_scheduler' : lr_scheduler(total_steps=total_steps, warmup=0.5)
    })
    
    torch.manual_seed(seed)
    set_seed(seed)

    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
  
    experimental_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        experimental_losses.append(trainer.train_step(*sample_input).cpu().item())
        
        assert trainer.options.lr_scheduler.get_last_lr()[0] == legacy_lr_scheduler(i)

    # LEGACY IMPLEMENTATION
    device = torch.device(device)
    legacy_model_desc, learning_rate_description, learning_rate = legacy_model_params() 
    torch.manual_seed(seed)
    set_seed(seed)
    
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

@pytest.mark.parametrize("loss_scaler, legacy_loss_scaler", [
    (None, Legacy_LossScaler("ort_test_input_loss_scaler", True)),
    (amp.DynamicLossScaler(), Legacy_LossScaler("ort_test_input_loss_scaler", True)),
    (CustomLossScaler(), LegacyCustomLossScaler())
])
def testToyBERTModelMixedPrecisionLossScalerLegacyExperimental(loss_scaler, legacy_loss_scaler):
    total_steps = 10
    device = "cuda"
    seed = 1

    model_desc = bert_model_description()
    model = load_bert_onnx_model()

    optim_config = optim.LambConfig()
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
        'mixed_precision': {
            'enabled': True,
            'loss_scaler': loss_scaler
        }
    })
    
    torch.manual_seed(seed)
    set_seed(seed)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
    
    experimental_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        experimental_losses.append(trainer.train_step(*sample_input).cpu().item())

    # LEGACY IMPLEMENTATION
    device = torch.device(device)
    legacy_model_desc, learning_rate_description, learning_rate = legacy_model_params() 
    torch.manual_seed(seed)
    set_seed(seed)
    
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
    
    _test_helpers.assert_model_outputs(experimental_losses, legacy_losses, rtol=1e-5)


@pytest.mark.parametrize("gradient_accumulation_steps", [
    (1),
    (4),
    (7)
])
def testToyBERTModelGradientAccumulationLegacyExperimental(gradient_accumulation_steps):
    total_steps = 10
    device = "cuda"
    seed = 1

    model_desc = bert_model_description()
    model = load_bert_onnx_model()

    optim_config = optim.LambConfig()
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
        'batch' : {
            'gradient_accumulation_steps' : gradient_accumulation_steps
        },
    })
    
    torch.manual_seed(seed)
    set_seed(seed)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
    
    experimental_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        experimental_losses.append(trainer.train_step(*sample_input).cpu().item())

    # LEGACY IMPLEMENTATION
    device = torch.device(device)
    legacy_model_desc, learning_rate_description, learning_rate = legacy_model_params() 
    torch.manual_seed(seed)
    set_seed(seed)
    
    legacy_trainer = Legacy_ORTTrainer(model, None, legacy_model_desc, "LambOptimizer",
                       None,
                       learning_rate_description,
                       device,
                       _use_deterministic_compute=True,
                       gradient_accumulation_steps=gradient_accumulation_steps)
    legacy_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        legacy_sample_input = [*sample_input, learning_rate]

        legacy_losses.append(legacy_trainer.train_step(legacy_sample_input).cpu().item())
    
    _test_helpers.assert_model_outputs(experimental_losses, legacy_losses)
