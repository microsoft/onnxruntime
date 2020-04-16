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
        self.rank_start_end = OrderedDict()
    
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
        
        # calculate start and end for each rank
        for rank in range(self.world_size):
            rank_count = padded_count // self.world_size
            rank_start = rank * rank_count
            rank_end = rank_start + rank_count
            self.rank_start_end[rank] = [rank_start, rank_end]
        
        self.offset_dict = OrderedDict()
        for i,weight in enumerate(self.weight_order):
            if i==0:
                self.offset_dict[weight] = 0
            else:
                prev_weight = self.weight_order[i-1]
                self.offset_dict[weight] = self.offset_dict[prev_weight] + self.weight_size_map[prev_weight]


    def get_wt_boundary_in_rank(self, weight, rank):
        
        tensor_count = self.weight_size_map[weight]
        offset = self.offset_dict[weight]
        assert self.rank_start_end, "Invalid call, call _setup_weight_aggregation() before this method."

        rank_start, rank_end = self.rank_start_end[rank]

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
                return(size_for_previous_rank, size_for_previous_rank + size_for_current_rank)
            else: # parameter handled by previous, current and next rank
                size_for_previous_rank = rank_start - offset
                size_for_current_rank = rank_end - rank_start
                size_for_next_rank = offset + tensor_count - rank_end
                return(size_for_previous_rank, size_for_previous_rank + size_for_current_rank)
        else:
            # parameter not handled by this rank
            return(None, None)

    def aggregate_weights(self):
        self._setup_weight_aggregation()

        for weight, ranks in self.weight_map.items():
            if len(ranks) == 1: #no aggregation required, weight present on only 1 rank        
                pass
            else:
                for i, rank in enumerate(ranks):                    
                    if i > 0:
                        # get the boundary where weight is updated in rank
                        weight_start, weight_end = self.get_wt_boundary_in_rank(weight, rank)
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

    def reshape_moment_tensors(self, state_dict):
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

        self.weight_map = OrderedDict()
        self.weight_size_map = dict()
        self.weight_shape_map = dict()
        self.weight_order = []
        self.weight_order_rank = dict()
        for i in range(self.world_size):
            ckpt_file_name = ckpt_prefix + '_ZeROrank_' + str(i) + '_of_' + str(self.world_size-1)+'.tar'
            print("Loading Pretrained Bert state dict from: {}".format(os.path.join(checkpoint_dir, ckpt_file_name)))
            bert_state_dict = torch.load(os.path.join(checkpoint_dir, ckpt_file_name), map_location=torch.device("cpu"))
            if 'model' in bert_state_dict:
                bert_state_dict = bert_state_dict['model']
            
            if self.clean_state_dict:
                bert_state_dict = self.clean_state_dict(bert_state_dict)
            
            self.weight_order_rank[i] = []
            for k,v in bert_state_dict.items():
                if k.endswith('_fp16'):
                    # fp16 weight is up to date on all ranks, just save once
                    if k not in self.aggregate_state_dict:
                        self.aggregate_state_dict[k] = v
                elif k.startswith('Moment_'):
                    if 'view' in k:                        
                        clean_name = k.split('_view_')[0]
                        if clean_name in self.aggregate_state_dict:
                            self.aggregate_state_dict[clean_name] = torch.cat((self.aggregate_state_dict[clean_name], v), 0)
                        else:
                            self.aggregate_state_dict[clean_name] = v
                    else:
                        self.aggregate_state_dict[k] = v
                elif k.startswith('Update_Count'):
                    if 'view' in k:
                        name_split = k.split('_view_')
                        ###########
                        # get original weight ordering
                        view_num = int(name_split[1])
                        weight_name = name_split[0].split('Update_Count_')[1]
                        if view_num == 1:
                            self.weight_order_rank[i].insert(0,weight_name)
                        elif view_num == 0:
                            self.weight_order_rank[i].append(weight_name)
                        ###########
                        clean_name = name_split[0]
                        if clean_name in self.aggregate_state_dict:
                            assert self.aggregate_state_dict[clean_name] == v, f'Invalid: {clean_name} values different in different zero checkpoints.'
                        else:
                            self.aggregate_state_dict[clean_name] = v
                    else:
                        weight_name =k.split('Update_Count_')[1]
                        if len(self.weight_order_rank[i]) == 0:
                            self.weight_order_rank[i].append(weight_name)
                        else:
                            self.weight_order_rank[i].insert(1,weight_name)
                        self.aggregate_state_dict[k] = v
                else:
                    # Weight        
                    if k in self.aggregate_state_dict: #weight has been seen before
                        clean_name = k + '_view_' + str(len(self.weight_map[k])) # add next copy as different view
                        self.aggregate_state_dict[clean_name] = v.view(-1) # flatten
                        self.weight_map[k].append(i)
                    else:
                        self.weight_map[k] = [i]
                        self.weight_shape_map[k] = v.size() #original shape of tensor
                        self.weight_size_map[k] = v.numel() #count of elements
                        self.aggregate_state_dict[k] = v.view(-1)

            if len(self.weight_order) and self.weight_order[-1] == self.weight_order_rank[i][0]:
                self.weight_order.extend(self.weight_order_rank[i][1:])
            else:
                self.weight_order.extend(self.weight_order_rank[i])

        # aggregate different shards of the weights        
        self.aggregate_weights()
        final_state_dict = self.reshape_moment_tensors(self.aggregate_state_dict)
        return final_state_dict
