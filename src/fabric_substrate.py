# File: src/fabric_substrate.py (Refactored for Dynamic Assembly)
"""
The NeuralSubstrate: A Dynamic Assembler of AI Components.

This version of the Substrate doesn't have a fixed architecture. Instead, it
holds a library of pre-defined Input Processors and Task Heads.
Based on a 'blueprint' (signals from the TaskWeaver), it selects the appropriate
components, assembles them, populates the selected head with Weaver-generated
weights, and performs the forward pass.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# We will import our new modules. Make sure they are in the same src directory
# or adjust sys.path if running scripts from a different location.
from . import input_processors 
from . import task_heads

class NeuralSubstrate(nn.Module):
    def __init__(self, config, device):
        """
        Initializes the NeuralSubstrate as a dynamic assembler.

        Args:
            config (dict): The main configuration dictionary, expected to contain
                           'model.input_processors_lib' and 'model.task_heads_lib'.
            device (torch.device): The device to move modules to.
        """
        super().__init__()
        self.config = config
        self.device = device

        # --- Instantiate all available Input Processors ---
        self.input_processor_modules = nn.ModuleDict()
        if 'input_processors_lib' not in config['model']:
            raise ValueError("Config missing 'model.input_processors_lib'")
        for proc_conf in config['model']['input_processors_lib']:
            name = proc_conf['name']
            proc_type = proc_conf['type']
            if proc_type == "ImageProcessor":
                self.input_processor_modules[name] = input_processors.ImageProcessor(
                    backbone_name=proc_conf['backbone_name'],
                    pretrained=proc_conf.get('pretrained', True),
                    freeze_backbone=proc_conf.get('freeze_backbone', True),
                    output_flattened_features=proc_conf.get('output_flattened_features', False) # Important for heads
                ).to(self.device)
            elif proc_type == "LiDARBEVProcessor": # Example
                 self.input_processor_modules[name] = input_processors.LiDARBEVProcessor(
                    input_bev_channels=proc_conf['input_bev_channels'],
                    output_feature_channels=proc_conf['output_feature_channels'],
                    num_conv_layers=proc_conf.get('num_conv_layers',3)
                ).to(self.device)
            elif proc_type == "IdentityProcessor":
                self.input_processor_modules[name] = input_processors.IdentityProcessor(
                    output_features_dim=proc_conf['output_features_dim']
                ).to(self.device)
            # Add other processor types here (e.g., TextProcessor)
            else:
                raise ValueError(f"Unknown input_processor type: {proc_type}")
        print(f"NeuralSubstrate: Initialized {len(self.input_processor_modules)} input processors.")

        # --- Instantiate all available Task Heads ---
        self.task_head_modules = nn.ModuleDict()
        if 'task_heads_lib' not in config['model']:
            raise ValueError("Config missing 'model.task_heads_lib'")
        for head_conf in config['model']['task_heads_lib']:
            name = head_conf['name']
            head_type = head_conf['type']
            # These dimensions will be derived based on chosen backbone and task in practice
            # The config for heads should specify how to get these (e.g., from backbone output_dim)
            # For now, we pass placeholder dims if not available, or expect them in head_conf
            
            # This part needs careful linking with backbone output dims and task-specific class numbers
            # For now, the head's __init__ must take what it needs based on its config.
            # The SynapseModel will ensure the config provides these details when heads are defined.
            
            if head_type == "ClassificationHead":
                self.task_head_modules[name] = task_heads.ClassificationHead(
                    input_features_dim=head_conf['input_features_dim'], # Must be known (e.g., output of a specific backbone)
                    num_classes=head_conf['num_classes'],
                    num_hidden_layers=head_conf.get('num_hidden_layers_in_head', 0),
                    hidden_dim=head_conf.get('head_hidden_dim', None)
                ).to(self.device)
            elif head_type == "SegmentationHead":
                self.task_head_modules[name] = task_heads.SegmentationHead(
                    input_feature_channels=head_conf['input_feature_channels'], # Must be known
                    num_classes=head_conf['num_classes'],
                    intermediate_channels=head_conf.get('intermediate_channels', 256),
                    upsample_factor=head_conf.get('upsample_factor', 2)
                ).to(self.device)
            # Add other head types here
            else:
                raise ValueError(f"Unknown task_head type: {head_type}")
        print(f"NeuralSubstrate: Initialized {len(self.task_head_modules)} task heads.")
        
        # TODO: Implement more sophisticated fusion modules if needed
        # For now, simple concatenation will be handled in the forward pass if multiple features

    def _resolve_input_key(self, processor_name, inputs_dict, input_key_map):
        if input_key_map:
            if processor_name not in input_key_map:
                raise ValueError(
                    f"Processor '{processor_name}' is missing an explicit input mapping."
                )
            input_key = input_key_map[processor_name]
            if input_key not in inputs_dict:
                raise ValueError(
                    f"Input key '{input_key}' for processor '{processor_name}' not found in inputs_dict."
                )
            return input_key

        if processor_name in inputs_dict:
            return processor_name

        if len(inputs_dict) == 1:
            return next(iter(inputs_dict.keys()))

        raise ValueError(
            "Multiple inputs were provided without an explicit 'input_key_map'. "
            "Please define processor-to-input bindings in the task config."
        )

    def _slice_inputs_for_sample(self, inputs_dict, sample_idx):
        sample_inputs = {}
        for key, value in inputs_dict.items():
            if torch.is_tensor(value):
                sample_inputs[key] = value[sample_idx : sample_idx + 1]
            else:
                sample_inputs[key] = value
        return sample_inputs

    def _forward_single(self, inputs_dict, blueprint_signals, head_weights_dict):
        processed_features_list = []
        if not isinstance(blueprint_signals.get('input_processor_names'), list) or \
           not blueprint_signals['input_processor_names']:
            raise ValueError("Blueprint must specify at least one 'input_processor_names' as a list.")

        input_key_map = blueprint_signals.get("input_key_map", {})

        for processor_name in blueprint_signals['input_processor_names']:
            if processor_name not in self.input_processor_modules:
                raise ValueError(f"Input processor '{processor_name}' not found in blueprint.")

            input_key_for_processor = self._resolve_input_key(
                processor_name,
                inputs_dict,
                input_key_map,
            )

            processor = self.input_processor_modules[processor_name]
            raw_input = inputs_dict[input_key_for_processor]
            features = processor(raw_input)
            processed_features_list.append(features)

        if len(processed_features_list) == 0:
            raise ValueError("No features processed from input processors.")
        elif len(processed_features_list) == 1:
            fused_features = processed_features_list[0]
        else:
            fusion_strategy = blueprint_signals.get('fusion_strategy', 'concat')
            if fusion_strategy == 'concat':
                flat_features_list = []
                for feat in processed_features_list:
                    if feat.dim() > 2:
                        flat_features_list.append(torch.flatten(feat, 1))
                    else:
                        flat_features_list.append(feat)
                fused_features = torch.cat(flat_features_list, dim=1)
            else:
                raise ValueError(f"Unsupported fusion strategy: {fusion_strategy}")

        head_name = blueprint_signals.get('head_name')
        if not head_name or head_name not in self.task_head_modules:
            raise ValueError(f"Head '{head_name}' not found or specified in blueprint.")

        selected_head = self.task_head_modules[head_name]
        return selected_head(fused_features, head_weights_dict)

    def get_head_parameter_shapes(self, head_name):
        """
        Gets the parameter shapes for a specific, named head.
        The TaskWeaver uses this to know what weights to generate.

        Args:
            head_name (str): The name of the head (must match a key in self.task_head_modules).

        Returns:
            OrderedDict: Parameter names and their shapes for the specified head.
        """
        if head_name not in self.task_head_modules:
            raise ValueError(f"Head '{head_name}' not found in available task heads.")
        return self.task_head_modules[head_name].get_parameter_shapes()

    def forward(self, inputs_dict, blueprint_signals, head_weights_dict):
        """
        Dynamically assembles and executes the forward pass.

        Args:
            inputs_dict (dict): Dictionary of input tensors, keyed by modality name
                                (e.g., {'camera_image': tensor, 'lidar_bev': tensor}).
            blueprint_signals (dict): Signals from the TaskWeaver, e.g.:
                {
                    'input_processor_names': ['ProcessorName1', 'ProcessorName2'], # Which processors to use
                    'head_name': 'HeadName',                                    # Which head to use
                    'fusion_strategy': 'concat' (or 'none', 'attention', etc.) # How to combine processor outputs
                }
            head_weights_dict (OrderedDict): Weights for the selected head, generated by TaskWeaver.

        Returns:
            torch.Tensor: The final output of the assembled network.
        """
        if isinstance(head_weights_dict, list):
            if not inputs_dict:
                raise ValueError("Per-sample head weights require a non-empty inputs_dict.")

            first_tensor = next(
                (value for value in inputs_dict.values() if torch.is_tensor(value)),
                None,
            )
            if first_tensor is None:
                raise ValueError("Could not infer batch size because inputs_dict has no tensor values.")

            batch_size = first_tensor.size(0)
            if len(head_weights_dict) != batch_size:
                raise ValueError(
                    f"Received {len(head_weights_dict)} per-sample weight sets for batch_size={batch_size}."
                )

            sample_outputs = []
            for sample_idx, sample_weights in enumerate(head_weights_dict):
                sample_inputs = self._slice_inputs_for_sample(inputs_dict, sample_idx)
                sample_outputs.append(
                    self._forward_single(sample_inputs, blueprint_signals, sample_weights)
                )
            return torch.cat(sample_outputs, dim=0)

        return self._forward_single(inputs_dict, blueprint_signals, head_weights_dict)
