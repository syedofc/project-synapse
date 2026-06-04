# File: src/synapse_model.py (Adjusted Ledger Calls)
import torch
import torch.nn as nn
from collections import OrderedDict # Ensure OrderedDict is imported

from . import fabric_substrate, fabric_weaver, fabric_ledger, resource_monitor
from . import input_processors, task_heads 

class SemanticContextExtractor(nn.Module): # Keep placeholder as before
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.output_dim = config.get('output_dim', 0) 
        if self.output_dim > 0:
            print(f"SemanticContextExtractor: Initialized (output_dim={self.output_dim}). Placeholder implementation.")
        else:
            print("SemanticContextExtractor: Not enabled or output_dim is 0.")

    def forward(self, primary_input):
        if self.output_dim == 0:
            return None
        batch_size = primary_input.size(0) if torch.is_tensor(primary_input) else 1
        return torch.rand(batch_size, self.output_dim, device=self.device)


class SynapseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tasks_config_map = {task_def['id']: task_def for task_name, task_def in config['tasks'].items()}
        self.device = torch.device(config['training']['device'])

        monitor_config = config['model'].get('resource_monitor', {})
        self.monitor = resource_monitor.ResourceMonitor(
            context_dim=config['model']['weaver']['device_context_dim'],
            mode=monitor_config.get('mode', 'telemetry'),
            fallback_mode=monitor_config.get('fallback_mode', 'zeros')
        )

        self.semantic_context_dim = 0
        if config['model'].get('semantic_context_extractor', {}).get('enabled', False):
            sce_config = config['model']['semantic_context_extractor']
            self.semantic_context_extractor = SemanticContextExtractor(sce_config, self.device).to(self.device)
            self.semantic_context_dim = self.semantic_context_extractor.output_dim
        else:
            self.semantic_context_extractor = None
        
        self.substrate = fabric_substrate.NeuralSubstrate(config, self.device)

        max_params = 0
        if 'task_heads_lib' in config['model']:
            for head_conf in config['model']['task_heads_lib']:
                head_name = head_conf['name']
                shapes = self.substrate.get_head_parameter_shapes(head_name)
                current_head_params = sum(torch.Size(s).numel() for s in shapes.values())
                if current_head_params > max_params:
                    max_params = current_head_params
        if max_params == 0 and config['model']['task_heads_lib']: # only error if lib is defined but no params found
            raise ValueError("Max_head_parameters is 0. Check task_heads_lib config and get_head_parameter_shapes.")
        elif not config['model']['task_heads_lib']:
             print("Warning: task_heads_lib is empty or not defined in config. max_head_parameters set to a default small value.")
             max_params = 1000 # Default small value if no heads, should not happen in normal operation

        self.weaver = fabric_weaver.TaskWeaver(
            num_unique_tasks=config['model']['weaver']['num_unique_tasks'], # This is set by train.py
            task_embedding_dim=config['model']['weaver']['task_embedding_dim'],
            device_context_dim=config['model']['weaver']['device_context_dim'],
            semantic_context_dim=self.semantic_context_dim,
            hidden_dim=config['model']['weaver']['hidden_dim'],
            num_hidden_layers=config['model']['weaver']['num_hidden_layers'],
            max_head_parameters=max_params
        ).to(self.device)

        self.ledger = fabric_ledger.SynapticLedger(
            capacity=config['model']['ledger']['capacity']
        )
        
        self.last_used_from_cache = False
        self.last_key_components_for_cache = None # To store (task_id_scalar, dev_ctx_tensor, sem_ctx_tensor, blueprint_dict)
        self.last_weights_generated = None # To store head_weights_dict
        
        self.to(self.device)
        print(f"SynapseModel (Dynamic Orchestrator) assembled and moved to device: {self.device}")

    def _get_task_definition(self, task_id_scalar):
        if task_id_scalar not in self.tasks_config_map:
            raise ValueError(f"Task ID {task_id_scalar} not found in config['tasks']. Available: {list(self.tasks_config_map.keys())}")
        return self.tasks_config_map[task_id_scalar]

    def forward(self, inputs_dict, task_id_scalar):
        batch_size = next(iter(inputs_dict.values())).size(0) if inputs_dict else 1

        device_context_unbatched = self.monitor.get_context().to(self.device) 
        device_context_batched = device_context_unbatched.unsqueeze(0).repeat(batch_size, 1)
        
        semantic_context_for_weaver_batched = None
        semantic_context_unbatched_for_key = None
        if self.semantic_context_extractor and self.semantic_context_dim > 0:
            primary_input_key = list(inputs_dict.keys())[0] 
            if primary_input_key in inputs_dict:
                semantic_context_for_weaver_batched = self.semantic_context_extractor(inputs_dict[primary_input_key])
                if semantic_context_for_weaver_batched is not None:
                    semantic_context_unbatched_for_key = semantic_context_for_weaver_batched[0] # Use first sample's for key
            if semantic_context_for_weaver_batched is None: # Fallback if primary input missing or SCE returns None
                semantic_context_for_weaver_batched = torch.zeros(batch_size, self.semantic_context_dim, device=self.device)
        
        task_def = self._get_task_definition(task_id_scalar)
        blueprint_signals_dict = { # This is the blueprint_dict
            'input_processor_names': task_def['input_processor_names'],
            'head_name': task_def['head_name'],
            'fusion_strategy': self.config['model']['substrate'].get('fusion_type', 'concat')
        }
        
        target_head_shapes = self.substrate.get_head_parameter_shapes(blueprint_signals_dict['head_name'])

        # Create hashable tuples for ledger key components from unbatched/representative context
        dev_ctx_tuple = tuple(round(x.item(), 4) for x in device_context_unbatched) if device_context_unbatched is not None else None
        sem_ctx_tuple = tuple(round(x.item(), 4) for x in semantic_context_unbatched_for_key) if semantic_context_unbatched_for_key is not None else None
        
        blueprint_items = []
        for k_bp, v_bp in sorted(blueprint_signals_dict.items()):
            if isinstance(v_bp, list): blueprint_items.append((k_bp, tuple(sorted(v_bp))))
            else: blueprint_items.append((k_bp, v_bp))
        blueprint_tuple = tuple(blueprint_items)
        
        # Store components needed for potential caching by trainer
        self.last_key_components_for_cache = (
            task_id_scalar, 
            device_context_unbatched, # Store original tensor for store method
            semantic_context_unbatched_for_key, # Store original tensor for store method
            blueprint_signals_dict # Store original dict for store method
        )

        cached_weights = self.ledger.retrieve(task_id_scalar, dev_ctx_tuple, sem_ctx_tuple, blueprint_tuple)
        
        head_weights_to_use = None
        if cached_weights:
            head_weights_to_use = {k: v.to(self.device) for k, v in cached_weights.items()}
            self.last_used_from_cache = True
            self.last_weights_generated = head_weights_to_use 
        else:
            task_id_tensor = torch.tensor([task_id_scalar] * batch_size, device=self.device) # Batched task_id
            
            head_weights_list_or_dict = self.weaver(
                task_id_tensor, 
                device_context_batched, 
                semantic_context_for_weaver_batched, 
                target_head_shapes
            )
            
            # Weaver returns a list of dicts if batch_size > 1, or single dict if batch_size = 1
            # Substrate expects a single dict if weights are shared across batch, or per-sample logic
            # For now, assume head_weights are shared for the batch for simplicity in Substrate
            # So, we take the first set of weights if Weaver produced a batch of them.
            if isinstance(head_weights_list_or_dict, list):
                head_weights_to_use = head_weights_list_or_dict[0] 
            else:
                head_weights_to_use = head_weights_list_or_dict

            self.last_used_from_cache = False
            self.last_weights_generated = head_weights_to_use
            
        output = self.substrate.forward(inputs_dict, blueprint_signals_dict, head_weights_to_use)
        return output

    def cache_current_state(self):
        """Caches the last generated blueprint and weights if they were not from cache."""
        if not self.last_used_from_cache and \
           self.last_key_components_for_cache is not None and \
           self.last_weights_generated is not None:
            
            task_id_s, dev_ctx_t, sem_ctx_t, blueprint_d = self.last_key_components_for_cache
            
            self.ledger.store(
                task_id_scalar=task_id_s,
                device_context_tensor=dev_ctx_t, # Pass original tensor
                semantic_context_tensor=sem_ctx_t, # Pass original tensor or None
                blueprint_signals_dict=blueprint_d, # Pass original dict
                weights_dict=self.last_weights_generated
            )
            # Clear them after caching to avoid re-caching same state if no new generation occurs
            self.last_key_components_for_cache = None 
            self.last_weights_generated = None


    def get_trainable_parameters(self):
        params = list(self.weaver.parameters())
        if self.semantic_context_extractor and self.semantic_context_extractor.output_dim > 0:
            # Check if semantic_context_extractor parameters require grad
            # This assumes self.semantic_context_extractor.extractor holds the trainable model
            if hasattr(self.semantic_context_extractor, 'extractor'):
                for param in self.semantic_context_extractor.extractor.parameters():
                    if param.requires_grad:
                        params.extend(list(self.semantic_context_extractor.extractor.parameters()))
                        break
            else: # Or if SCE itself is an nn.Module with params
                 for param in self.semantic_context_extractor.parameters():
                    if param.requires_grad:
                        params.extend(list(self.semantic_context_extractor.parameters()))
                        break
        return params
