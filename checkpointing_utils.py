import os
import numpy as np
from collections import OrderedDict
import torch
import tempfile

def list_checkpoint_files(checkpoint_dir, checkpoint_prefix, extension='.ort.pt'):
    ckpt_file_names = [f for f in os.listdir(checkpoint_dir) if f.startswith(checkpoint_prefix)]
    ckpt_file_names = [f for f in ckpt_file_names if f.endswith(extension)]
    ckpt_file_names = [os.path.join(checkpoint_dir, f) for f in ckpt_file_names]
    
    assert len(ckpt_file_names) > 0, "No checkpoint files found with prefix \"{}\" in directory {}.".format(checkpoint_prefix, checkpoint_dir)
    return ckpt_file_names

def get_checkpoint_name(prefix, zero_enabled, world_rank=0, world_size=1, horizontal_parallel_size=1, pipeline_parallel_size=1):
    # data_parallel_size = world_size / horizontal_parallel_size / pipeline_parallel_size
    # need to change to this below
    data_parallel_size = int(world_size / horizontal_parallel_size / pipeline_parallel_size)
    parallellism_info = 'D.{data_parallel_size}.H.{horizontal_parallel_size}.P.{pipeline_parallel_size}'
    SINGLE_CHECKPOINT_FILENAME='{prefix}.ort.pt'
    MULTIPLE_CHECKPOINT_FILENAME='{prefix}.rank.{world_rank}.{world_size}.' + parallellism_info + '.ort.pt'
    
    is_partitioned = zero_enabled or (horizontal_parallel_size > 1) or (pipeline_parallel_size > 1)
    if is_partitioned:
        filename=MULTIPLE_CHECKPOINT_FILENAME.format(prefix=prefix, world_rank=world_rank, world_size=(world_size-1),
            data_parallel_size=data_parallel_size, 
            horizontal_parallel_size=horizontal_parallel_size, 
            pipeline_parallel_size=pipeline_parallel_size)
    else:
        filename=SINGLE_CHECKPOINT_FILENAME.format(prefix=prefix)
    
    return filename

		
def split_state_dict(state_dict):	
    optimizer_keys = ['Moment_', 'Update_Count_', 'Step']	
    split_sd = {'optimizer': {}, 'fp32_param': {}, 'fp16_param': {}}	
    for k,v in state_dict.items():	
        mode = 'fp32_param'	
        for optim_key in optimizer_keys:	
            if k.startswith(optim_key):	
                mode = 'optimizer'	
                break	
        if k.endswith('_fp16'):	
            mode = 'fp16_param'	
        split_sd[mode][k] = v
    return split_sd

def is_equal_dict(A, B):	
    try:	
        assert A.keys() == B.keys()	
        for k in A.keys():	
            assert (A[k] == B[k]).all()	
    except:	
        return False	
    return True	

def is_equal_tensor(A, B):	
    return (A == B).all()

class CombineZeroCheckpoint(object):	
    def __init__(self, checkpoint_files, clean_state_dict = None):
        self.checkpoint_files = checkpoint_files	
        self.clean_state_dict = clean_state_dict
        
        self.weight_shape_map = dict()
    
    def _is_sharded(self, name):
        if '_view_' in name:
            return True
        return False

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

    def _update_weight_statistics(self, name, value):
        self.weight_shape_map[name] = value.size() #original shape of tensor

    def _reshape_tensors(self, state_dict, fp16):
        for k,v in state_dict.items():
            if k.startswith('Moment_'):
                _, weight_name, _ = self._split_moment_name(k)
                set_size = self.weight_shape_map[weight_name]    
                state_dict[k] = v.reshape(set_size)
                state_dict[weight_name] = state_dict[weight_name].reshape(set_size)
        return state_dict
  
    def aggregate_checkpoints(self, ranks = None):	
        self.aggregate_state_dict=dict()	
        is_fp16 = False	
        weight_offset = dict()	
        if ranks == None:	
            ranks = range(len(self.checkpoint_files))	
        for i in ranks:	
            checkpoint_name = self.checkpoint_files[i]
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

from numpy import linalg as la
import numpy as np

def stat_tensor(t):
    v = t.cpu().detach().numpy()
    np.set_printoptions(threshold=np.inf)
    hists, buckets = np.histogram(v)
    print(v.shape)
    print(np.ptp(v), np.amin(v), np.amax(v))
    print(hists, buckets)
    print(la.norm(v))

class CombineMegatronCheckpoint(object):	
    def __init__(self, checkpoint_files, clean_state_dict = None):	
        self.checkpoint_files = checkpoint_files	
        self.clean_state_dict = clean_state_dict	
    	
    def _has_fp16_weights(self, state_dict):	
        for k in state_dict.keys():	
            if k.endswith('_fp16'):	
                return True	
        return False	
    	
    def _split_name(self, name):
        name_split = name.split('_rank_')	
        param_name = name_split[0]	
        is_fp16 = False	
        if(len(name_split)==2):	
            if '_fp16' in name_split[1]:	
                is_fp16 = True	
                horizontal_rank = int(name_split[1].split('_fp16')[0])	
            else:	
                horizontal_rank = int(name_split[1])	
        else:	
            horizontal_rank = None	
        row_split=True if len(param_name.split('_row'))==2 else False	
        column_split=True if len(param_name.split('_column'))==2 else False	
        param_name = param_name.split('_row')[0].split('_column')[0]	
        param_name = param_name + '_fp16' if is_fp16==True else param_name	
        return param_name, horizontal_rank, row_split, column_split	
    	
    def _aggregate(self, param_dict):	
        sharded_params=set()	
        for k,v in param_dict.items():	
            param_name, horizontal_rank, row_split, column_split = self._split_name(k)	
            assert(row_split and column_split, "Invalid case, both row and column can't be split.")	
            axis = 0 if row_split else -1 if column_split else None	
            if axis is not None: 	
                # parameter is sharded	
                sharded_params.add(param_name)	
                	
                if horizontal_rank == 0 and param_name in self.aggregate_state_dict:	
                    # delete stale initializer present in state_dict	
                    del(self.aggregate_state_dict[param_name])	
                if param_name in self.aggregate_state_dict:	
                    if not k.startswith('Update_Count'):	
                        # Found a previous shard of the param, concatenate shards ordered by ranks	
                        self.aggregate_state_dict[param_name] = torch.cat((self.aggregate_state_dict[param_name], v), axis)	
                else:	
                    self.aggregate_state_dict[param_name] = v	
            	
            else:	
                if k in sharded_params:	
                    # stale initializer that must be ignored	
                    continue	
                if k in self.aggregate_state_dict:	
                    # disabled until megatron bug is fixed	
                    # assert (self.aggregate_state_dict[k] == v).all(), "Unsharded params must have the same value"	
                    if not np.allclose(self.aggregate_state_dict[k], v, rtol=0.000001, atol=0.00001):
                        print(f"Mismatch for param :{k}")                        
                        print(stat_tensor(self.aggregate_state_dict[k]))
                        print(self.aggregate_state_dict[k])
                        print(stat_tensor(v))
                        print(v)
                else:	
                    self.aggregate_state_dict[k] = v	
    
    def aggregate_checkpoints(self, ranks=None):	
        self.aggregate_state_dict=dict()	
        if ranks == None:	
            ranks = range(len(self.checkpoint_files))	
        for i in ranks:	
            checkpoint_name = self.checkpoint_files[i]	
            print("Megatron Aggregator: Loading state dict from: {}".format(checkpoint_name))	
            rank_state_dict = torch.load(checkpoint_name, map_location=torch.device("cpu"))	
            	
            if 'model' in rank_state_dict:	
                rank_state_dict = rank_state_dict['model']	
            	
            if self.clean_state_dict:	
                rank_state_dict = self.clean_state_dict(rank_state_dict)	
            	
            rank_state_dict = split_state_dict(rank_state_dict)	
            self._aggregate(rank_state_dict['fp16_param'])	
            #need to debug	
            self._aggregate(rank_state_dict['fp32_param'])	
            self._aggregate(rank_state_dict['optimizer'])           	
        return self.aggregate_state_dict	


def compare_dp_values(aggregate_state_dict, param_dict):
    sharded_params=set()	
    for k,v in param_dict.items():		
        if k in aggregate_state_dict:	
            # disabled until megatron bug is fixed	
            # assert (self.aggregate_state_dict[k] == v).all(), "Unsharded params must have the same value"	
            if not np.allclose(aggregate_state_dict[k], v, rtol = 0.00001):
                print(f"Mismatch for param :{k}")                        
                print(stat_tensor(aggregate_state_dict[k]))
                print(stat_tensor(v))
        else:	
            aggregate_state_dict[k] = v

class CombineCheckpoint(object):	
    def __init__(self, checkpoint_files, clean_state_dict = None):	
        assert len(checkpoint_files) > 0, "No checkpoint files passed."	
        self.checkpoint_files = checkpoint_files	
        self.clean_state_dict = clean_state_dict	
        filename = os.path.basename(self.checkpoint_files[0])	
        # self.checkpoint_prefix = self.checkpoint_files[0].split('.rank')[0]	
        self.checkpoint_prefix = filename.split('.rank')[0]	
        self.world_size = int(filename.split('rank')[1].split('.')[2]) +1	
        self.D_size = int(filename.split('.D.')[1].split('.')[0])	
        self.H_size = int(filename.split('.H.')[1].split('.')[0])	
        self.P_size = int(filename.split('.P.')[1].split('.')[0])	
        print(f"World size = {self.world_size}.")        	
        assert len(self.checkpoint_files) == self.world_size, "Could not find {} files".format(self.world_size)	
        self.checkpoint_files = sorted(self.checkpoint_files, key=lambda x: int(x.split('.rank.')[-1].split(".")[0]))
    
    def get_parallellism_groups(self):	
        horizontal_parallel_size = self.H_size	
        world_size = self.world_size	
        data_parallel_size = self.D_size	
        num_data_groups = int(world_size / data_parallel_size)	
        data_groups = {}	
        for data_group_id in range(num_data_groups):	
            data_group_ranks=[]	
            for r in range(data_parallel_size):	
                data_group_ranks.append(data_group_id + horizontal_parallel_size * r)	
            print("Data Group: {} : {}".format(data_group_id, data_group_ranks))	
            data_groups[data_group_id] = data_group_ranks	
       	
        num_hori_groups = int(world_size / horizontal_parallel_size)	
        hori_groups = {}	
        for hori_group_id in range(num_hori_groups):	
            hori_group_ranks=[]	
            for r in range(horizontal_parallel_size):	
                hori_group_ranks.append(hori_group_id * horizontal_parallel_size + r)	
            print("Horizntal Group: {} : {}".format(hori_group_id, hori_group_ranks))	
            hori_groups[hori_group_id] = hori_group_ranks	
        	
        return data_groups, hori_groups	
    
    def aggregate_checkpoints(self, combine_zero, combine_megatron):	
        D_groups, H_groups = self.get_parallellism_groups()	
        save_dir = os.path.join(tempfile.gettempdir(), "ort_checkpoint_dir")	
        os.makedirs(save_dir, exist_ok = True) 	
        	
        zero_ckpt_agg = CombineZeroCheckpoint(self.checkpoint_files, self.clean_state_dict)	
        aggregate_data_checkpoint_files = []	
        aggregate_state = None
        ###################
        # if combine_zero:	
        #     for group_id in range(len(D_groups)):	
        #         aggregate_data_checkpoints = zero_ckpt_agg.aggregate_checkpoints(D_groups[group_id])	
        #         if not combine_megatron: # no need to combine other data groups	
        #             aggregate_state = aggregate_data_checkpoints	
        #             break 	
        #         filename = self.checkpoint_prefix + '.data_group.' + str(group_id) + '.ort.pt'	
        #         filepath = os.path.join(save_dir, filename)
        #         print("Saving temp file: " + filepath)
        #         torch.save(aggregate_data_checkpoints, filepath)	
        #         aggregate_data_checkpoint_files.append(filepath)
        #################
        # aggregate_data_checkpoint_files = ["/tmp/ort_checkpoint_dir2/ORT_CKPT_STEP.20_0_.data_group.0.ort.pt",
        #                                    "/tmp/ort_checkpoint_dir2/ORT_CKPT_STEP.20_0_.data_group.1.ort.pt"]

        aggregate_data_checkpoint_files = [
            "/tmp/ort_checkpoint_dir/ORT_CKPT_STEP.1000_0_.data_group.3.ort.pt",
            "/tmp/ort_checkpoint_dir/ORT_CKPT_STEP.1000_0_.data_group.0.ort.pt",
            "/tmp/ort_checkpoint_dir/ORT_CKPT_STEP.1000_0_.data_group.1.ort.pt",
            "/tmp/ort_checkpoint_dir/ORT_CKPT_STEP.1000_0_.data_group.2.ort.pt"]
        ##################
        megatron_files = self.checkpoint_files if len(aggregate_data_checkpoint_files) == 0 else aggregate_data_checkpoint_files	
        if combine_megatron:            	
            megatron_ckpt_agg = CombineMegatronCheckpoint(megatron_files, self.clean_state_dict)	
            aggregate_state = megatron_ckpt_agg.aggregate_checkpoints()	
        # remove temp files created	
        # for f in aggregate_data_checkpoint_files:	
        #     os.remove(f)	
        # os.rmdir(save_dir)
        return aggregate_state



def megatron_test_combined():	
    checkpoint_dir="/home/pengwa/dev/perf_outputs/128gpu_fix_data__batch5_acc10_mega4_zero1_fp16true_layer20_step13000000_profilefalse/ckpts/"	
    #checkpoint_dir="/home/pengwa/dev/64_outputs/64gpu_ort_fp16_mega8_zero0_2b_batch4_layer20_acc32_lr9.1875e-05_decay0.0001_bart-2b-model-dev_1607655904_7ffac2e9_decay_mode_0/ckpts/"
    #checkpoint_dir="/home/pengwa/dev/perf_outputs/16gpu_fix_data_1_6__batch5_acc10_mega2_zero1_fp16true_layer1_step130000_profilefalse/ckpts/"
    checkpoint_prefix="ORT_CKPT_STEP.1000_"	
    checkpoint_files = list_checkpoint_files(checkpoint_dir, checkpoint_prefix)	
    checkpoint_files = sorted(checkpoint_files)	

    ckpt_agg = CombineCheckpoint(checkpoint_files)	
    agg_sd = ckpt_agg.aggregate_checkpoints(combine_zero=True, combine_megatron=True)
    #torch.save(agg_sd, "/home/pengwa/dev/perf_outputs/128gpu_fix_data__batch5_acc10_mega4_zero1_fp16true_layer20_step13000000_profilefalse/aggregated_ORT_CKPT_STEP.1000.ort.pt")

megatron_test_combined()