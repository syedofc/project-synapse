# File: src/fabric_weaver.py (Refactored for specific head generation)
"""
The TaskWeaver: Generates weights for a specified target head architecture.

It takes task, device, and semantic context as input.
It is informed by the SynapseModel about the specific head architecture
(via target_head_shapes) for which it needs to generate weights.
Its internal MLP's final layer is sized to accommodate the largest potential head.
"""
import torch
import torch.nn as nn
from collections import OrderedDict

class TaskWeaver(nn.Module):
    def __init__(self, 
                 num_unique_tasks, # Number of distinct task IDs
                 task_embedding_dim, 
                 device_context_dim, 
                 semantic_context_dim, # Will be 0 if not used
                 hidden_dim, 
                 num_hidden_layers,
                 max_head_parameters): # Max number of parameters any head might require
        """
        Initializes the TaskWeaver.

        Args:
            num_unique_tasks (int): Total number of unique task IDs system handles.
            task_embedding_dim (int): Dimensionality of the task embedding.
            device_context_dim (int): Dimensionality of the device context vector.
            semantic_context_dim (int): Dimensionality of the semantic context vector.
            hidden_dim (int): Size of hidden layers in the Weaver's MLP.
            num_hidden_layers (int): Number of hidden layers in the Weaver's MLP.
            max_head_parameters (int): The total number of parameters required by the largest
                                       possible head the Weaver might need to generate weights for.
                                       This defines the size of the Weaver's output layer.
        """
        super().__init__()
        
        self.task_embedding = nn.Embedding(num_unique_tasks, task_embedding_dim)
        
        # Calculate the total input dimension to the Weaver's MLP
        current_mlp_input_dim = task_embedding_dim + device_context_dim
        if semantic_context_dim > 0:
            current_mlp_input_dim += semantic_context_dim
        
        weaver_mlp_layers = []
        weaver_mlp_layers.append(nn.Linear(current_mlp_input_dim, hidden_dim))
        weaver_mlp_layers.append(nn.ReLU())
        
        for _ in range(num_hidden_layers - 1): # num_hidden_layers includes the first one
            weaver_mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
            weaver_mlp_layers.append(nn.ReLU())
            
        # The output layer generates parameters for the largest possible head
        weaver_mlp_layers.append(nn.Linear(hidden_dim, max_head_parameters))
        
        self.mlp = nn.Sequential(*weaver_mlp_layers)
        
        self.max_head_parameters = max_head_parameters
        print(f"TaskWeaver initialized. Max output params: {max_head_parameters}. "
              f"Input to MLP dim: {current_mlp_input_dim}")

    def forward(self, task_id_tensor, device_context_vector, semantic_context_vector, target_head_shapes):
        """
        Generates weights for the specified target head architecture.

        Args:
            task_id_tensor (torch.Tensor): Tensor containing the ID(s) of the task (B, 1) or (B,).
            device_context_vector (torch.Tensor): Tensor representing device context (B, device_context_dim).
            semantic_context_vector (torch.Tensor, optional): Tensor for semantic context (B, semantic_context_dim)
                                                             or None if not used.
            target_head_shapes (OrderedDict): The shapes of the weights required by the current target head.
                                             Used to slice and reshape the output.

        Returns:
            OrderedDict or List[OrderedDict]: A dictionary (or list of dicts if batch_size > 1)
                                              of reshaped weights for the target head.
        """
        task_emb = self.task_embedding(task_id_tensor.squeeze(-1) if task_id_tensor.dim() > 1 else task_id_tensor) # B, task_emb_dim

        # Ensure context vectors are correctly batched
        batch_size = task_emb.size(0)
        
        # Device Context
        if device_context_vector.dim() == 1: # If unbatched
            device_context_vector = device_context_vector.unsqueeze(0)
        if device_context_vector.size(0) != batch_size:
            device_context_vector = device_context_vector.repeat(batch_size, 1)

        combined_input_list = [task_emb, device_context_vector]

        # Semantic Context (Optional)
        if semantic_context_vector is not None:
            if semantic_context_vector.dim() == 1: # If unbatched
                semantic_context_vector = semantic_context_vector.unsqueeze(0)
            if semantic_context_vector.size(0) != batch_size:
                semantic_context_vector = semantic_context_vector.repeat(batch_size, 1)
            combined_input_list.append(semantic_context_vector)
        
        combined_input = torch.cat(combined_input_list, dim=1)

        # Generate the flat vector of all weights (sized for the largest possible head)
        flat_weights_for_max_head_batch = self.mlp(combined_input) # (B, max_head_parameters)

        # Slice and Reshape for the current target head
        return self._slice_and_reshape_weights(flat_weights_for_max_head_batch, target_head_shapes)

    def _slice_and_reshape_weights(self, flat_weights_for_max_head_batch, target_head_shapes):
        """
        Slices the necessary portion from the full flat_weights vector (meant for the largest head)
        and reshapes it into a dictionary of tensors for the current target head.
        """
        batch_size = flat_weights_for_max_head_batch.size(0)
        
        # Calculate how many parameters the current target head actually needs
        num_params_for_current_head = sum(torch.Size(s).numel() for s in target_head_shapes.values())

        if num_params_for_current_head > self.max_head_parameters:
            raise ValueError(f"Target head requires {num_params_for_current_head} params, "
                             f"but Weaver max output is {self.max_head_parameters}.")

        # Take only the required portion from the start of the generated flat vector
        sliced_flat_weights_batch = flat_weights_for_max_head_batch[:, :num_params_for_current_head]
        
        all_reshaped_weights = []
        for i in range(batch_size):
            flat_weights_single_instance = sliced_flat_weights_batch[i]
            reshaped_weights_single = OrderedDict()
            current_pos = 0
            for name, shape in target_head_shapes.items():
                num_params_for_layer = torch.Size(shape).numel()
                reshaped_tensor = flat_weights_single_instance[current_pos : current_pos + num_params_for_layer].view(shape)
                reshaped_weights_single[name] = reshaped_tensor
                current_pos += num_params_for_layer
            all_reshaped_weights.append(reshaped_weights_single)
        
        return all_reshaped_weights[0] if batch_size == 1 else all_reshaped_weights