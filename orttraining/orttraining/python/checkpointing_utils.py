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

class CombineZeroCheckpoint(object):
    def __init__(self, checkpoint_files, clean_state_dict = None):

        assert len(checkpoint_files) > 0, "No checkpoint files passed."

        self.checkpoint_files = checkpoint_files
        self.clean_state_dict = clean_state_dict
        self.world_size = int(self.checkpoint_files[0].split('ZeRO')[1].split('.')[2]) +1
        print(f"World size = {self.world_size}, expecting {self.world_size} files.")        
        assert len(self.checkpoint_files) == self.world_size, "Could not find {} files".format(self.world_size)
        
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
  
    def aggregate_checkpoints(self):
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