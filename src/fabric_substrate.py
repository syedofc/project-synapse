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
        
        # --- 1. Select and Run Input Processor(s) ---
        processed_features_list = []
        if not isinstance(blueprint_signals.get('input_processor_names'), list) or \
           not blueprint_signals['input_processor_names']:
            raise ValueError("Blueprint must specify at least one 'input_processor_names' as a list.")

        for i, processor_name in enumerate(blueprint_signals['input_processor_names']):
            if processor_name not in self.input_processor_modules:
                raise ValueError(f"Input processor '{processor_name}' not found in blueprint.")
            
            # Determine the correct input key for this processor
            # This assumes a convention, e.g., if processor_name is "A2D2_Camera_MobileNetV2_FeatExtractor",
            # it expects an input from inputs_dict['camera_image'] or similar.
            # For simplicity, let's assume inputs_dict keys match processor names or a defined mapping.
            # A more robust system would have this mapping in the task config.
            # For now, if only one processor, use the first key in inputs_dict.
            # If multiple, this needs careful handling.
            
            # Simple heuristic for now:
            input_key_for_processor = list(inputs_dict.keys())[i] if len(inputs_dict.keys()) > i else list(inputs_dict.keys())[0]
            if input_key_for_processor not in inputs_dict:
                 raise ValueError(f"Input data for key '{input_key_for_processor}' (for processor '{processor_name}') not found in inputs_dict.")

            processor = self.input_processor_modules[processor_name]
            raw_input = inputs_dict[input_key_for_processor]
            features = processor(raw_input)
            processed_features_list.append(features)

        # --- 2. Fuse Features (if multi-modal) ---
        if len(processed_features_list) == 0:
            raise ValueError("No features processed from input processors.")
        elif len(processed_features_list) == 1:
            fused_features = processed_features_list[0]
        else:
            # Multi-modal: Implement fusion strategy based on blueprint_signals['fusion_strategy']
            fusion_strategy = blueprint_signals.get('fusion_strategy', 'concat')
            if fusion_strategy == 'concat':
                # Ensure features can be concatenated (e.g., all flat vectors, or feature maps of same H,W)
                # This might require processors to output compatible shapes or further processing here.
                # For now, simple flatten and concat if they are maps of different spatial sizes
                flat_features_list = []
                for feat in processed_features_list:
                    if feat.dim() > 2: # If it's a feature map BxCxHxW
                        flat_features_list.append(torch.flatten(feat, 1))
                    else: # Already a vector BxFeatures
                        flat_features_list.append(feat)
                fused_features = torch.cat(flat_features_list, dim=1)
            # elif fusion_strategy == 'attention':
            #     # TODO: Implement attention-based fusion module
            #     pass
            else:
                raise ValueError(f"Unsupported fusion strategy: {fusion_strategy}")
        
        # --- 3. Select Head and Apply Weights ---
        head_name = blueprint_signals.get('head_name')
        if not head_name or head_name not in self.task_head_modules:
            raise ValueError(f"Head '{head_name}' not found or specified in blueprint.")
        
        selected_head = self.task_head_modules[head_name]
        
        # The selected_head's forward method now takes the features and the generated weights
        output = selected_head(fused_features, head_weights_dict)
            
        return output
