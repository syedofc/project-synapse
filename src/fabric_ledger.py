# File: src/fabric_ledger.py (Updated for richer cache keys)
"""
The SynapticLedger acts as a high-speed cache.
It now uses a more comprehensive key structure including task_id, device_context,
semantic_context (if available), and blueprint_signals.
"""
from collections import OrderedDict


def _freeze_nested(value):
    if isinstance(value, dict):
        return tuple((key, _freeze_nested(nested)) for key, nested in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_nested(item) for item in value)
    return value


def _freeze_context_vector(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().view(-1)
        return tuple(round(item.item(), 4) for item in value)
    if isinstance(value, (list, tuple)):
        return tuple(round(float(item), 4) for item in value)
    return value

class SynapticLedger:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
        print(f"SynapticLedger initialized with capacity for {capacity} configurations.")

    def _create_key(self, task_id_scalar, device_context, semantic_context, blueprint_signals):
        """
        Creates a hashable key from all relevant components.
        
        Args:
            task_id_scalar (int): The scalar ID of the task.
            device_context: Tensor or hashable representation of device context.
            semantic_context: Tensor or hashable representation of semantic context.
            blueprint_signals: Blueprint dictionary or already-frozen representation.
        """
        device_context_tuple = _freeze_context_vector(device_context)
        semantic_context_tuple = _freeze_context_vector(semantic_context)
        blueprint_tuple = _freeze_nested(blueprint_signals)
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

        key = self._create_key(
            task_id_scalar,
            device_context_tensor,
            semantic_context_tensor,
            blueprint_signals_dict,
        )
        
        detached_weights = OrderedDict(
            (k, v.detach().cpu().clone()) for k, v in weights_dict.items()
        )
        self.cache[key] = detached_weights

        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def retrieve(self, task_id_scalar, device_context, semantic_context, blueprint_signals):
        """
        Retrieves a weight configuration from the cache.

        Args:
            task_id_scalar (int): The scalar ID of the task.
            device_context: Tensor or hashable representation of device context.
            semantic_context: Tensor or hashable representation of semantic context.
            blueprint_signals: Blueprint dictionary or already-frozen representation.

        Returns:
            OrderedDict or None: The cached weights if found, otherwise None.
        """
        if self.capacity == 0:
            return None
            
        key = self._create_key(task_id_scalar, device_context, semantic_context, blueprint_signals)
        if key not in self.cache:
            return None
        
        self.cache.move_to_end(key)
        return OrderedDict((k, v.clone()) for k, v in self.cache[key].items())
