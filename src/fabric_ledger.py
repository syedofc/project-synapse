# File: src/fabric_ledger.py (Updated for richer cache keys)
"""
The SynapticLedger acts as a high-speed cache.
It now uses a more comprehensive key structure including task_id, device_context,
semantic_context (if available), and blueprint_signals.
"""
from collections import OrderedDict
import torch

class SynapticLedger:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
        print(f"SynapticLedger initialized with capacity for {capacity} configurations.")

    def _create_key(self, task_id_scalar, device_context_tuple, semantic_context_tuple, blueprint_tuple):
        """
        Creates a hashable key from all relevant components.
        
        Args:
            task_id_scalar (int): The scalar ID of the task.
            device_context_tuple (tuple): Hashable tuple of device context.
            semantic_context_tuple (tuple or None): Hashable tuple of semantic context, or None.
            blueprint_tuple (tuple): Hashable tuple of blueprint signals (e.g., from sorted dict items).
        """
        # Ensure all parts of the key are hashable and consistently ordered
        return (task_id_scalar, device_context_tuple, semantic_context_tuple, blueprint_tuple)

    def store(self, task_id_scalar, device_context_tensor, semantic_context_tensor, 
              blueprint_signals_dict, weights_dict):
        """
        Stores a weight configuration in the cache.

        Args:
            task_id_scalar (int): The scalar ID of the task.
            device_context_tensor (torch.Tensor): The original device context tensor.
            semantic_context_tensor (torch.Tensor or None): The original semantic context tensor.
            blueprint_signals_dict (dict): The blueprint signals dictionary.
            weights_dict (OrderedDict): The dictionary of weight tensors to cache.
        """
        if self.capacity == 0:
            return

        # Create hashable tuples for key components
        dev_ctx_tuple = tuple(round(x.item(), 4) for x in device_context_tensor) if device_context_tensor is not None else None
        sem_ctx_tuple = tuple(round(x.item(), 4) for x in semantic_context_tensor) if semantic_context_tensor is not None else None
        
        # Ensure blueprint_signals_dict is consistently ordered for the key
        # Convert dict items to sorted tuple of tuples: ((key1, val1), (key2, val2))
        # If vals can be lists, convert them to tuples too.
        blueprint_items = []
        for k, v in sorted(blueprint_signals_dict.items()):
            if isinstance(v, list):
                blueprint_items.append((k, tuple(sorted(v)))) # Sort lists for consistency
            else:
                blueprint_items.append((k, v))
        blueprint_tuple = tuple(blueprint_items)

        key = self._create_key(task_id_scalar, dev_ctx_tuple, sem_ctx_tuple, blueprint_tuple)
        
        detached_weights = OrderedDict([(k, v.detach().clone()) for k, v in weights_dict.items()])
        self.cache[key] = detached_weights # Store the head_weights_dict directly
        
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def retrieve(self, task_id_scalar, device_context_tuple, semantic_context_tuple, blueprint_tuple):
        """
        Retrieves a weight configuration from the cache.

        Args:
            task_id_scalar (int): The scalar ID of the task.
            device_context_tuple (tuple): Hashable tuple of device context.
            semantic_context_tuple (tuple or None): Hashable tuple of semantic context, or None.
            blueprint_tuple (tuple): Hashable tuple of blueprint signals.

        Returns:
            OrderedDict or None: The cached weights if found, otherwise None.
        """
        if self.capacity == 0:
            return None
            
        key = self._create_key(task_id_scalar, device_context_tuple, semantic_context_tuple, blueprint_tuple)
        if key not in self.cache:
            return None
        
        self.cache.move_to_end(key)
        return self.cache[key] # Returns the head_weights_dict