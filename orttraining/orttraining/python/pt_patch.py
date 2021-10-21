import torch

from torch.onnx import symbolic_opset10
from torch.onnx import symbolic_opset12
from torch.onnx.symbolic_helper import parse_args
import torch.onnx.symbolic_helper as sym_help

@parse_args('v', 'v', 'v', 'v', 'i', 'none')
def nll_loss_10(g, self, target, weight=None, reduction='mean', ignore_index=-100):
    if not weight and not ignore_index:
        return g.op("nll_loss", self, target)
    elif ignore_index:
        ignore_index_ = g.op("Constant", value_t=torch.tensor(ignore_index, dtype=torch.int64))
        eq_ = g.op("Equal", target, ignore_index_)
        not_eq_ = g.op("Not", eq_)
        weight_ = g.op("Cast", not_eq_, to_i=1)      # FLOAT = 1;   // float
        not_eq_int64_ = g.op("Cast", not_eq_, to_i=7)   #INT64 = 7;   // int64_t
        target_ = g.op("Mul", target, not_eq_int64_)
        # if weight:
        #     weight_ = g.op("Mul", weight_, weight)
        return g.op("nll_loss", self, target_, weight_)

symbolic_opset10.nll_loss = nll_loss_10

def nll_loss_12(g, self, target, weight, reduction, ignore_index):
    # none reduction : onnx::Constant[value={0}]
    # mean reduction : onnx::Constant[value={1}]
    # sum reduction : onnx::Constant[value={2}]
    reduction = sym_help._maybe_get_const(reduction, 'i')
    reduction_vals = ['none', 'mean', 'sum']
    reduction = reduction_vals[reduction]

    # in onnx NegativeLogLikelihoodLoss specification, ignore_index is optional without default value.
    # therefore we need to set ignore_index attribute even if it is not specified (e.g. ignore_index=-100).
    ignore_index = sym_help._maybe_get_const(ignore_index, 'i')
    if weight.node().mustBeNone():
        nllloss = g.op("NegativeLogLikelihoodLoss", self, target, reduction_s=reduction, ignore_index_i=ignore_index)
    else:
        nllloss = g.op(
            "NegativeLogLikelihoodLoss", self,
            target, weight, reduction_s=reduction, ignore_index_i=ignore_index)

    return nllloss

symbolic_opset12.nll_loss = nll_loss_12
