import os
from collections import OrderedDict
import torch

def list_checkpoint_files(checkpoint_dir, checkpoint_prefix, extension='.ort.pt'):
    ckpt_file_names = [f for f in os.listdir(checkpoint_dir) if f.startswith(checkpoint_prefix)]
    ckpt_file_names = [f for f in ckpt_file_names if f.endswith(extension)]
    ckpt_file_names = [os.path.join(checkpoint_dir, f) for f in ckpt_file_names]
    
    assert len(ckpt_file_names) > 0, "No checkpoint files found with prefix \"{}\" in directory {}.".format(checkpoint_prefix, checkpoint_dir)
    return ckpt_file_names

def get_checkpoint_name(prefix, is_partitioned, world_rank=None, world_size=None):
    SINGLE_CHECKPOINT_FILENAME='{prefix}.ort.pt'
    MULTIPLE_CHECKPOINT_FILENAME='{prefix}.ZeRO.{world_rank}.{world_size}.ort.pt'
    
    if is_partitioned:
        filename=MULTIPLE_CHECKPOINT_FILENAME.format(prefix=prefix, world_rank=world_rank, world_size=(world_size-1))
    else:
        filename=SINGLE_CHECKPOINT_FILENAME.format(prefix=prefix)
    
    return filename

def split_state_dict(state_dict):
    optimizer_keys = ['Moment_1_', 'Moment_2_', 'Update_Count_', 'Step_']
    split_sd = {'optimizer': {}, 'fp32_param': {}, 'fp16_param': {}}
    for k,v in state_dict.items():
        mode = 'fp32_param'
        for optim_key in optimizer_keys:
            if k.startswith(optim_key):
                mode = 'optimizer'
                break
        if k.endswith('_fp16'):
            split_sd['fp16_param'][k] = v
        else:
            split_sd['fp32_param'][k] = v
    
    return split_sd

class CombineZeroCheckpoint(object):
    def __init__(self, checkpoint_files, clean_state_dict = None):

        assert len(checkpoint_files) > 0, "No checkpoint files passed."

        self.checkpoint_files = checkpoint_files
        self.clean_state_dict = clean_state_dict
        self.world_size = int(self.checkpoint_files[0].split('ZeRO')[1].split('.')[2]) +1
        print(f"World size = {self.world_size}, expecting {self.world_size} files.")        
        assert len(self.checkpoint_files) == self.world_size, "Could not find {} files".format(self.world_size)
        
        self.weight_shape_map = dict()
        self.sharded_params=set()
    
    def _is_sharded(self, name):
        if '_view_' in name:
            return True
        return False
    
    def _get_param_name(self, name):
        return name.split('_view_')[0]

    def _has_fp16_weights(self, state_dict):
        for k in state_dict.keys():
            if k.endswith('_fp16'):
                return True
        return False
    
    def _split_moment_name(self, name):
        name_split = name.split('_view_')
        if(len(name_split)>1):
            view_num = int(name_split[1])
        else:
            view_num = None
        weight_name = name_split[0].split('Moment_')[1][2:]
        moment_num = int(name_split[0].split('Moment_')[1][0])
        return moment_num, weight_name, view_num

    def _split_name(self, name):
        name_split = name.split('_view_')
        if(len(name_split)>1):
            view_num = int(name_split[1])
        else:
            view_num = None
        optimizer_key=''
        fp16_key=''
        if name_split[0].startswith('Moment_1'):
            optimizer_key='Moment_1_'
        elif name_split[0].startswith('Moment_2'):
            optimizer_key='Moment_2_'
        elif name_split[0].startswith('Update_Count'):
            optimizer_key='Update_Count_'
        elif name_split[0].endswith('_fp16'):
            fp16_key='_fp16'
        param_name=name_split[0]
        if optimizer_key is not '':
            param_name = param_name.split(optimizer_key)[1]
        param_name = param_name.split('_fp16')[0]
        return param_name, optimizer_key, view_num, fp16_key

    def _update_weight_statistics(self, name, value):
        self.weight_shape_map[name] = value.size() #original shape of tensor

    def _reshape_tensor(self, key):
        value = self.aggregate_state_dict[key]
        weight_name, _, _, _ = self._split_name(key)
        set_size = self.weight_shape_map[weight_name]    
        self.aggregate_state_dict[key] = value.reshape(set_size)

    def _reshape_tensors(self, state_dict, fp16):
        for k,v in state_dict.items():
            if k.startswith('Moment_'):
                _, weight_name, _ = self._split_moment_name(k)
                set_size = self.weight_shape_map[weight_name]    
                state_dict[k] = v.reshape(set_size)
                state_dict[weight_name] = state_dict[weight_name].reshape(set_size)
        return state_dict
  
    def aggregate_checkpoints_old(self):
        checkpoint_dir=os.path.dirname(self.checkpoint_files[0])
        checkpoint_prefix = self.checkpoint_files[0].split('.ZeRO')[0]
        self.aggregate_state_dict=dict()

        is_fp16 = False
        weight_offset = dict()
        for i in range(self.world_size):
            checkpoint_name = get_checkpoint_name(checkpoint_prefix, True, i, self.world_size)
            print("Loading state dict from: {}".format(checkpoint_name))
            rank_state_dict = torch.load(checkpoint_name, map_location=torch.device("cpu"))
            if 'model' in rank_state_dict:
                rank_state_dict = rank_state_dict['model']
            
            if self.clean_state_dict:
                rank_state_dict = self.clean_state_dict(rank_state_dict)
            
            if i==0:
                is_fp16 = self._has_fp16_weights(rank_state_dict)

            for k,v in rank_state_dict.items():
                if k.startswith('Moment_'):
                    moment_num, weight_name, view_num = self._split_moment_name(k)

                    if self._is_sharded(k):                                                
                        clean_name = 'Moment_' + str(moment_num) + '_' + weight_name
                        if clean_name in self.aggregate_state_dict:
                            # Found a previous shard of the moment, concatenate shards ordered by ranks
                            self.aggregate_state_dict[clean_name] = torch.cat((self.aggregate_state_dict[clean_name], v), 0)
                        else:
                            self.aggregate_state_dict[clean_name] = v
                    else:
                        # Moment is not sharded, add as is                        
                        self.aggregate_state_dict[k] = v

                    if is_fp16 and moment_num == 1:
                        #FP32 weights are sharded, patch together based on moments
                        if view_num == 0:
                            # This FP32 weight's first shard is present on this rank, 
                            # flatten and add the weight's first view
                            self.aggregate_state_dict[weight_name] = rank_state_dict[weight_name].view(-1)                                       
                            self._update_weight_statistics(weight_name, rank_state_dict[weight_name])
                            weight_offset[weight_name] = v.numel()
                        
                        elif view_num == 1:
                            # This FP32 weight is carryforward from previous rank
                            # Get start and end of weight slice to be updated from this rank
                            weight_start = weight_offset[weight_name]
                            weight_end = weight_start + v.numel()

                            if weight_start:
                                old_value = self.aggregate_state_dict[weight_name]
                                new_value = rank_state_dict[weight_name].view(-1)                             
                                # patch the weight together
                                self.aggregate_state_dict[weight_name] = torch.cat((old_value[:weight_start], new_value[weight_start:weight_end], old_value[weight_end:]),0)
                            
                            # update offset for next view
                            weight_offset[weight_name] = weight_end

                elif k.startswith('Update_Count'):
                    clean_name = k.split('_view_')[0]
                    # add a single copy of the 'Update_Count' tensor for current weight                    
                    if clean_name not in self.aggregate_state_dict:
                        self.aggregate_state_dict[clean_name] = v
                
                else:                    
                    if k not in self.aggregate_state_dict:
                        self.aggregate_state_dict[k] = v 
                        if not (k.endswith('_fp16') or k == 'Step'):
                            # FP32 Weight
                            self._update_weight_statistics(k,v)

        final_state_dict = self._reshape_tensors(self.aggregate_state_dict, is_fp16)
        return final_state_dict
    
    def _aggregate(self, param_dict):
        for k,v in param_dict.items():
            weight_name, optimizer_key, view_num, fp16_key = self._split_name(k)
            if view_num is not None:
                # parameter is sharded
                param_name = optimizer_key+weight_name+fp16_key
                
                if param_name in self.aggregate_state_dict and optimizer_key not in ['Update_Count_']:
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
                self._update_weight_statistics(weight_name,v)

    def aggregate_checkpoints(self):
        checkpoint_dir=os.path.dirname(self.checkpoint_files[0])
        checkpoint_prefix = self.checkpoint_files[0].split('.ZeRO')[0]
        self.aggregate_state_dict=dict()

        self.is_fp16 = False
        weight_offset = dict()
        for i in range(self.world_size):
            checkpoint_name = get_checkpoint_name(checkpoint_prefix, True, i, self.world_size)
            print("Loading state dict from: {}".format(checkpoint_name))
            rank_state_dict = torch.load(checkpoint_name, map_location=torch.device("cpu"))
            if 'model' in rank_state_dict:
                rank_state_dict = rank_state_dict['model']
            
            if self.clean_state_dict:
                rank_state_dict = self.clean_state_dict(rank_state_dict)
            
            rank_state_dict = split_state_dict(rank_state_dict)
            self._aggregate(rank_state_dict['fp16_param'])
            self._aggregate(rank_state_dict['fp32_param'])
            self._aggregate(rank_state_dict['optimizer'])

        for k in self.sharded_params:
            self._reshape_tensor(k)
        return self.aggregate_state_dict

def segregate_sd(state_dict):
    w = dict()
    w16 = dict()
    m1 = dict()
    m2 = dict()
    u = dict()
    for k,v in state_dict.items():
        if k.startswith('Moment_1'):
            m1[k] = v
        elif k.startswith('Moment_2'):
            m2[k] = v
        elif k.startswith('Update_Count'):
            u[k] = v
        elif k.endswith('_fp16'):
            w16[k] = v
        else:
            w[k] = v
    return w, w16, m1, m2, u

def test_bert_tiny():
    checkpoint_dir = '/bert_ort/aibhanda/public_ort_latest/onnxruntime/build/Linux/Debug'
    checkpoint_prefix = "ORT_checkpoint"
    checkpoint_files = sorted(list_checkpoint_files(checkpoint_dir, checkpoint_prefix))
    sd0 = torch.load(checkpoint_files[0], map_location=torch.device("cpu"))['model']
    sd1 = torch.load(checkpoint_files[1], map_location=torch.device("cpu"))['model']

    w0, w160, m10, m20, u0 = segregate_sd(sd0)
    w1, w161, m11, m21, u1 = segregate_sd(sd1)

    ckpt_agg = CombineZeroCheckpoint(checkpoint_files)
    aggregate_state_dict = ckpt_agg.aggregate_checkpoints()

    checkpoint_dir_old = '/bert_ort/aibhanda/public_ort_latest/onnxruntime/build/Linux/RelWithDebInfo'
    checkpoint_files_old = sorted(list_checkpoint_files(checkpoint_dir_old, checkpoint_prefix))
    sd0_old = torch.load(checkpoint_files_old[0], map_location=torch.device("cpu"))['model']
    sd1_old = torch.load(checkpoint_files_old[1], map_location=torch.device("cpu"))['model']
    ckpt_agg_old = CombineZeroCheckpoint(checkpoint_files_old)
    aggregate_state_dict_old = ckpt_agg_old.aggregate_checkpoints_old()

    assert(aggregate_state_dict.keys() == aggregate_state_dict_old.keys())
    allequal = {k:(aggregate_state_dict_old[k] == aggregate_state_dict[k]).all() for k in aggregate_state_dict.keys()}
    return

test_bert_tiny()