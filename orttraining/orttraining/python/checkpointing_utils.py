import os
from collections import OrderedDict
import torch

def list_checkpoint_files(checkpoint_dir, checkpoint_prefix, extension='.tar'):
    ckpt_file_names = [f for f in os.listdir(checkpoint_dir) if f.startswith(checkpoint_prefix)]
    ckpt_file_names = [f for f in ckpt_file_names if f.endswith(extension)]
    ckpt_file_names = [os.path.join(checkpoint_dir, f) for f in ckpt_file_names]
    
    assert len(ckpt_file_names) > 0, "No checkpoint files found with prefix \"{}\" in directory {}.".format(checkpoint_prefix, checkpoint_dir)
    return ckpt_file_names

class Combine_Zero_Checkpoint():
    def __init__(self, checkpoint_files, clean_state_dict = None):

        assert len(checkpoint_files) > 0, "No checkpoint files passed."

        self.checkpoint_files = checkpoint_files
        self.clean_state_dict = clean_state_dict
        self.world_size = int(self.checkpoint_files[0].split('ZeROrank')[1].split('.')[0].split('of')[1].strip('_')) +1        
        print(f"World size = {self.world_size}, expecting {self.world_size} files.")        
        assert len(self.checkpoint_files) == self.world_size, "Could not find {} files".format(self.world_size)
        
        self.weight_to_rank_map = OrderedDict()
        self.weight_size_map = dict()
        self.weight_shape_map = dict()
        self.weight_order = []
    
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
        self.weight_size_map[name] = value.numel() #count of elements

    def _get_fp32_weight_count(self):
        assert self.weight_size_map, "No weight size map found."
        total_count = 0
        for k,v in self.weight_size_map.items():
            total_count += v
        return total_count

    def _setup_weight_aggregation(self):
        # replicating partitioning logic for zero
        total_count = self._get_fp32_weight_count()
        assert total_count > 0, "Total count of weights is zero."
        alignment = self.world_size * 32
        padded_count = total_count + alignment - (total_count % alignment) 

        rank_start_end = OrderedDict()
        # calculate start and end for each rank
        for rank in range(self.world_size):
            rank_count = padded_count // self.world_size
            rank_start = rank * rank_count
            rank_end = rank_start + rank_count
            rank_start_end[rank] = [rank_start, rank_end]
        
        offset_dict = OrderedDict()
        for i,weight in enumerate(self.weight_order):
            if i==0:
                offset_dict[weight] = 0
            else:
                prev_weight = self.weight_order[i-1]
                offset_dict[weight] = offset_dict[prev_weight] + self.weight_size_map[prev_weight]
        
        return rank_start_end, offset_dict

    def _get_weight_boundary_in_rank(self, weight, rank_start, rank_end, offset):
        
        tensor_count = self.weight_size_map[weight]
 
        if (offset < rank_end and offset + tensor_count > rank_start):
            # parameter handled by this rank
            if (offset >= rank_start and offset + tensor_count <= rank_end):
                # parameter not partitioned, completely handled by this rank
                return(None, None)
            elif (offset < rank_start and offset + tensor_count <= rank_end):
                # parameter handled by previous and current rank
                size_for_previous_rank = rank_start - offset
                size_for_current_rank = offset + tensor_count - rank_start
                return(size_for_previous_rank, size_for_previous_rank + size_for_current_rank)
            elif (offset >= rank_start and offset + tensor_count > rank_end):
                # parameter handled by current and next rank
                size_for_current_rank = rank_end - offset
                size_for_next_rank = offset + tensor_count - rank_end
                return(0, size_for_current_rank)
            else: # parameter handled by previous, current and next rank
                size_for_previous_rank = rank_start - offset
                size_for_current_rank = rank_end - rank_start
                size_for_next_rank = offset + tensor_count - rank_end
                return(size_for_previous_rank, size_for_previous_rank + size_for_current_rank)
        else:
            # parameter not handled by this rank
            return(None, None)

    def _aggregate_weights(self):
        rank_start_end, offset_dict = self._setup_weight_aggregation()

        for weight, ranks in self.weight_to_rank_map.items():
            if len(ranks) == 1: #no aggregation required, weight present on only 1 rank        
                pass
            else:
                for i, rank in enumerate(ranks):                    
                    if i > 0: # 0'th view is saved as the weight_name itself
                        # get the boundary where weight is updated in rank
                        rank_start, rank_end = rank_start_end[rank]                        
                        offset = offset_dict[weight]
                        weight_start, weight_end = self._get_weight_boundary_in_rank(weight, rank_start, rank_end, offset)

                        if weight_start:
                            old_value = self.aggregate_state_dict[weight]                    
                            view_name = weight + '_view_' + str(i)
                            new_value = self.aggregate_state_dict[view_name]
                            del self.aggregate_state_dict[view_name]
                            
                            # patch the weight together
                            self.aggregate_state_dict[weight] = torch.cat((old_value[:weight_start], new_value[weight_start:weight_end], old_value[weight_end:]),0)                 
            # reshape the weight to original shape
            original_shape = self.weight_shape_map[weight]
            self.aggregate_state_dict[weight] = self.aggregate_state_dict[weight].reshape(original_shape)

    def _reshape_moment_tensors(self, state_dict):
        for k,v in state_dict.items():
            if k.startswith('Moment_'):
                weight_name = k.split('Moment_')[-1][2:]
                if v.size() != state_dict[weight_name].size():
                    state_dict[k] = v.resize_as_(state_dict[weight_name])
        return state_dict
  
    def aggregate_checkpoints(self):
        checkpoint_dir=os.path.dirname(self.checkpoint_files[0])
        ckpt_prefix = self.checkpoint_files[0].split('_ZeROrank_')[0]
        self.aggregate_state_dict=dict()

        is_fp16 = False
        
        for i in range(self.world_size):
            ckpt_file_name = ckpt_prefix + '_ZeROrank_' + str(i) + '_of_' + str(self.world_size-1)+'.tar'
            print("Loading Pretrained Bert state dict from: {}".format(os.path.join(checkpoint_dir, ckpt_file_name)))
            rank_state_dict = torch.load(os.path.join(checkpoint_dir, ckpt_file_name), map_location=torch.device("cpu"))
            if 'model' in rank_state_dict:
                rank_state_dict = rank_state_dict['model']
            
            if self.clean_state_dict:
                rank_state_dict = self.clean_state_dict(rank_state_dict)
            
            if i==0:
                is_fp16 = self._has_fp16_weights(rank_state_dict)

            weight_order_for_rank = []
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
                        #FP32 weights are sharded, gather info about original weight ordering from moment
                        if view_num == 1:
                            # This FP32 weight is carryforward from previous rank, should 
                            # appear first in this rank's weight ordering
                            weight_order_for_rank.insert(0,weight_name)                                                        
                            self.weight_to_rank_map[weight_name].append(i)

                            # add next copy of weight as different view                                    
                            weight_view_name = weight_name + '_view_' + str(len(self.weight_to_rank_map[weight_name])-1) 
                            self.aggregate_state_dict[weight_view_name] = rank_state_dict[weight_name].view(-1) # flatten

                        elif view_num == 0:
                            # This FP32 weight's first shard is present on this rank, 
                            # weight should appear last in this rank's weight ordering
                            weight_order_for_rank.append(weight_name)
                            self.weight_to_rank_map[weight_name] = [i]

                            # flatten and add the weight's first view
                            self.aggregate_state_dict[weight_name] = rank_state_dict[weight_name].view(-1)                                       
                            self._update_weight_statistics(weight_name, rank_state_dict[weight_name])

                        elif view_num == None:
                            # view_num is None, FP32 weight is not sharded
                            # insert weight in the middle of this rank's weight ordering
                            if len(weight_order_for_rank) == 0:
                                weight_order_for_rank.append(weight_name)
                            else:
                                weight_order_for_rank.insert(1,weight_name)

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

            if is_fp16:
                # aggregate overall weight ordering
                if len(self.weight_order) and self.weight_order[-1] == weight_order_for_rank[0]:
                    # skip first weight as it's name is already present due to previous shard
                    self.weight_order.extend(weight_order_for_rank[1:])
                else:
                    self.weight_order.extend(weight_order_for_rank)

        if is_fp16:
            # aggregate different shards of the fp32 weights        
            self._aggregate_weights()
        final_state_dict = self._reshape_moment_tensors(self.aggregate_state_dict)
        return final_state_dict