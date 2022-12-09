import os
import numpy as np

class ShardSpec:
    def __init__(self, shard_spec, is_partial):
        self.spec = shard_spec
        self.is_partial = is_partial

    def is_valid_sharding(self, shape):
        if self.is_partial:
            return True
        if len(self.spec) != len(shape):
            return False
        
        for shard, dim in zip(self.spec, shape):
            if dim % shard != 0:
                return False
        return True
    
    def num_shard(self):
        return np.prod(self.spec)
    
    def get_shard_dims(self, shape):
        return [int(dim / shard) for shard, dim in zip(self.spec, shape)]


def get_shard_spec(num_shards, mode='allreduce'):
    use_allreduce = mode == 'allreduce'
    allreduce_shard_spec = {
                  'Attention_weight': ShardSpec([1, num_shards], False),
                  'Attention_bias': ShardSpec([num_shards], False),
                  'Attention_post_matmul': ShardSpec([num_shards, 1,], False),
                  'matmul1': ShardSpec([1, num_shards], False),
                  'gelu_bias': ShardSpec([num_shards], False),
                  'matmul2': ShardSpec([num_shards, 1], False),
            }

    allgather_shard_spec = {
                  'Attention_weight': ShardSpec([1, num_shards], False),
                  'Attention_bias': ShardSpec([num_shards], False),
                  'Attention_post_matmul': ShardSpec([1, num_shards], False),
                  'matmul1': ShardSpec([1, num_shards], False),
                  'gelu_bias': ShardSpec([num_shards], False),
                  'matmul2': ShardSpec([1, num_shards], False),
            }
    return allreduce_shard_spec if use_allreduce else allgather_shard_spec

def get_gpt2_spec(num_shards, num_layers=36, mode='allreduce'):
    name_map = {
            'transformer.h.{}.attn.c_attn.weight': 'Attention_weight',
            'transformer.h.{}.attn.c_attn.bias': 'Attention_bias',
            'transformer.h.{}.attn.c_proj.weight': 'Attention_post_matmul',
            'transformer.h.{}.mlp.c_fc.weight': 'matmul1',
            'transformer.h.{}.mlp.c_fc.bias': 'gelu_bias',
            'transformer.h.{}.mlp.c_proj.weight': 'matmul2'
            }

    res = {}
    shard_spec = get_shard_spec(num_shards, mode)
    for i in range(num_layers):
        for k, v in name_map.items():
            name = k.format(i)
            spec = shard_spec[v]
            res[name]=spec
    return res


shard = 4
megatron_shard_spec = {# layer 0
              'Attention_0_qkv_weight': ShardSpec([1, shard], False),
              'Attention_0_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1650': ShardSpec([shard, 1,], False),
              'onnx::MatMul_1651': ShardSpec([1, shard], False),
              'encoder.layer.0.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1652': ShardSpec([shard, 1], False),
              # layer 1
              'Attention_1_qkv_weight': ShardSpec([1, shard], False),
              'Attention_1_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1663': ShardSpec([shard, 1,], False),
              'onnx::MatMul_1664': ShardSpec([1, shard], False),
              'encoder.layer.1.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1665': ShardSpec([shard, 1], False),
              ## layer 2
              'Attention_2_qkv_weight': ShardSpec([1, shard], False),
              'Attention_2_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1676': ShardSpec([shard, 1,], False),
              'onnx::MatMul_1677': ShardSpec([1, shard], False),
              'encoder.layer.2.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1678': ShardSpec([shard, 1], False),
              # layer 3
              'Attention_3_qkv_weight': ShardSpec([1, shard], False),
              'Attention_3_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1689': ShardSpec([shard, 1,], False),
              'onnx::MatMul_1690': ShardSpec([1, shard], False),
              'encoder.layer.3.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1691': ShardSpec([shard, 1], False),
              # layer 4
              'Attention_4_qkv_weight': ShardSpec([1, shard], False),
              'Attention_4_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1702': ShardSpec([shard, 1,], False),
              'onnx::MatMul_1703': ShardSpec([1, shard], False),
              'encoder.layer.4.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1704': ShardSpec([shard, 1], False),
              # layer 5
              'Attention_5_qkv_weight': ShardSpec([1, shard], False),
              'Attention_5_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1715': ShardSpec([shard, 1,], False),
              'onnx::MatMul_1716': ShardSpec([1, shard], False),
              'encoder.layer.5.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1717': ShardSpec([shard, 1], False),
              # layer 6
              'Attention_6_qkv_weight': ShardSpec([1, shard], False),
              'Attention_6_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1728': ShardSpec([shard, 1,], False),
              'onnx::MatMul_1729': ShardSpec([1, shard], False),
              'encoder.layer.6.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1730': ShardSpec([shard, 1], False),
              # layer 7
              'Attention_7_qkv_weight': ShardSpec([1, shard], False),
              'Attention_7_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1741': ShardSpec([shard, 1,], False),
              'onnx::MatMul_1742': ShardSpec([1, shard], False),
              'encoder.layer.7.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1743': ShardSpec([shard, 1], False),
              # layer 8
              'Attention_8_qkv_weight': ShardSpec([1, shard], False),
              'Attention_8_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1754': ShardSpec([shard, 1,], False),
              'onnx::MatMul_1755': ShardSpec([1, shard], False),
              'encoder.layer.8.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1756': ShardSpec([shard, 1], False),
              # layer 9
              'Attention_9_qkv_weight': ShardSpec([1, shard], False),
              'Attention_9_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1767': ShardSpec([shard, 1,], False),
              'onnx::MatMul_1768': ShardSpec([1, shard], False),
              'encoder.layer.9.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1769': ShardSpec([shard, 1], False),
              # layer 10
              'Attention_10_qkv_weight': ShardSpec([1, shard], False),
              'Attention_10_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1780': ShardSpec([shard, 1,], False),
              'onnx::MatMul_1781': ShardSpec([1, shard], False),
              'encoder.layer.10.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1782': ShardSpec([shard, 1], False),
              # layer 11
              'Attention_11_qkv_weight': ShardSpec([1, shard], False),
              'Attention_11_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1793': ShardSpec([shard, 1,], False),
              'onnx::MatMul_1794': ShardSpec([1, shard], False),
              'encoder.layer.11.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1795': ShardSpec([shard, 1], False),
              }

allgather_gemm_shard_spec = {# layer 0
              'Attention_0_qkv_weight': ShardSpec([1, shard], False),
              'Attention_0_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1650': ShardSpec([1, shard], False),
              'onnx::MatMul_1651': ShardSpec([1, shard], False),
              'encoder.layer.0.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1652': ShardSpec([1, shard], False),
              # layer 1
              'Attention_1_qkv_weight': ShardSpec([1, shard], False),
              'Attention_1_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1663': ShardSpec([1, shard], False),
              'onnx::MatMul_1664': ShardSpec([1, shard], False),
              'encoder.layer.1.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1665': ShardSpec([1, shard], False),
              # layer 2
              'Attention_2_qkv_weight': ShardSpec([1, shard], False),
              'Attention_2_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1676': ShardSpec([1, shard], False),
              'onnx::MatMul_1677': ShardSpec([1, shard], False),
              'encoder.layer.2.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1678': ShardSpec([1, shard], False),
              # layer 3
              'Attention_3_qkv_weight': ShardSpec([1, shard], False),
              'Attention_3_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1689': ShardSpec([1, shard], False),
              'onnx::MatMul_1690': ShardSpec([1, shard], False),
              'encoder.layer.3.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1691': ShardSpec([1, shard], False),
              # layer 4
              'Attention_4_qkv_weight': ShardSpec([1, shard], False),
              'Attention_4_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1702': ShardSpec([1, shard], False),
              'onnx::MatMul_1703': ShardSpec([1, shard], False),
              'encoder.layer.4.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1704': ShardSpec([1, shard], False),
              # layer 5
              'Attention_5_qkv_weight': ShardSpec([1, shard], False),
              'Attention_5_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1715': ShardSpec([1, shard], False),
              'onnx::MatMul_1716': ShardSpec([1, shard], False),
              'encoder.layer.5.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1717': ShardSpec([1, shard], False),
              # layer 6
              'Attention_6_qkv_weight': ShardSpec([1, shard], False),
              'Attention_6_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1728': ShardSpec([1, shard], False),
              'onnx::MatMul_1729': ShardSpec([1, shard], False),
              'encoder.layer.6.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1730': ShardSpec([1, shard], False),
              # layer 7
              'Attention_7_qkv_weight': ShardSpec([1, shard], False),
              'Attention_7_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1741': ShardSpec([1, shard], False),
              'onnx::MatMul_1742': ShardSpec([1, shard], False),
              'encoder.layer.7.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1743': ShardSpec([1, shard], False),
              # layer 8
              'Attention_8_qkv_weight': ShardSpec([1, shard], False),
              'Attention_8_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1754': ShardSpec([1, shard], False),
              'onnx::MatMul_1755': ShardSpec([1, shard], False),
              'encoder.layer.8.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1756': ShardSpec([1, shard], False),
              # layer 9
              'Attention_9_qkv_weight': ShardSpec([1, shard], False),
              'Attention_9_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1767': ShardSpec([1, shard], False),
              'onnx::MatMul_1768': ShardSpec([1, shard], False),
              'encoder.layer.9.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1769': ShardSpec([1, shard], False),
              # layer 10
              'Attention_10_qkv_weight': ShardSpec([1, shard], False),
              'Attention_10_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1780': ShardSpec([1, shard], False),
              'onnx::MatMul_1781': ShardSpec([1, shard], False),
              'encoder.layer.10.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1782': ShardSpec([1, shard], False),
              # layer 11
              'Attention_11_qkv_weight': ShardSpec([1, shard], False),
              'Attention_11_qkv_bias': ShardSpec([shard], False),
              'onnx::MatMul_1793': ShardSpec([1, shard], False),
              'onnx::MatMul_1794': ShardSpec([1, shard], False),
              'encoder.layer.11.intermediate.dense.bias': ShardSpec([shard], False),
              'onnx::MatMul_1795': ShardSpec([1, shard], False),
              }


