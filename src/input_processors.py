# File: src/input_processors.py (Corrected with necessary imports)
"""
Input Processors for SynapseOS.
"""
import torch
import torch.nn as nn # <<< THIS IMPORT WAS LIKELY MISSING OR MISPLACED
from torchvision import models

class _SimpleDepthCNN(nn.Module):
    """A very simple CNN for processing depth maps."""
    def __init__(self, input_channels=1, output_channels=256):
        super().__init__()
        self.output_features_dim = output_channels

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, output_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        # print(f"Initialized _SimpleDepthCNN: input_channels={input_channels}, output_channels={output_channels}") # Optional print

    def forward(self, x):
        return self.conv_layers(x)


class ImageProcessor(nn.Module):
    def __init__(self, backbone_name="resnet50", pretrained=True, freeze_backbone=True, 
                 output_flattened_features=False, input_channels=3):
        super().__init__()
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.output_flattened_features = output_flattened_features
        self.is_custom_cnn = False

        if backbone_name == "custom_depth_cnn":
            self.is_custom_cnn = True
            # Configurable output_channels for the custom CNN can be set in YAML for this processor
            # For example, add 'custom_cnn_output_channels: 256' to its definition
            custom_cnn_output_channels = 256 # Default, can be overridden by config later if needed
            self.backbone = _SimpleDepthCNN(input_channels=input_channels, output_channels=custom_cnn_output_channels)
            self.output_features_dim = custom_cnn_output_channels
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            # print(f"ImageProcessor: Initialized custom_depth_cnn. Output feature channels: {self.output_features_dim}") # Optional

        elif hasattr(models, backbone_name):
            if input_channels != 3 and pretrained:
                print(f"Warning: Backbone '{backbone_name}' is typically pretrained on 3-channel RGB images. "
                      f"You specified input_channels={input_channels}. Pretrained weights for the first layer will be lost if modified.")
            
            temp_backbone = getattr(models, backbone_name)(weights='DEFAULT' if pretrained else None) # Use new weights API
            
            # Modify the first convolutional layer if input_channels is not 3
            first_conv_modified = False
            if input_channels != 3:
                if hasattr(temp_backbone, 'conv1') and isinstance(temp_backbone.conv1, nn.Conv2d): # ResNet family
                    original_first_layer = temp_backbone.conv1
                    temp_backbone.conv1 = nn.Conv2d(input_channels, 
                                                original_first_layer.out_channels, 
                                                kernel_size=original_first_layer.kernel_size, 
                                                stride=original_first_layer.stride, 
                                                padding=original_first_layer.padding, 
                                                bias=original_first_layer.bias is not None)
                    first_conv_modified = True
                elif hasattr(temp_backbone, 'features') and isinstance(temp_backbone.features, nn.Sequential) and \
                     len(temp_backbone.features) > 0 and isinstance(temp_backbone.features[0], nn.Conv2d): # MobileNetV2, EfficientNet
                    original_first_layer = temp_backbone.features[0]
                    temp_backbone.features[0] = nn.Conv2d(input_channels, 
                                                        original_first_layer.out_channels, 
                                                        kernel_size=original_first_layer.kernel_size, 
                                                        stride=original_first_layer.stride, 
                                                        padding=original_first_layer.padding, 
                                                        bias=original_first_layer.bias is not None)
                    first_conv_modified = True
                
                if first_conv_modified and pretrained:
                     print(f"  First conv layer of {backbone_name} re-initialized for {input_channels} channels. Original pretrained weights for this layer are not used.")
                elif not first_conv_modified:
                    print(f"Warning: Could not automatically adapt first conv layer of {backbone_name} for {input_channels} channels. Using original first layer.")

            self.backbone = temp_backbone

            if hasattr(self.backbone, 'fc'): 
                self.output_features_dim = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
            elif hasattr(self.backbone, 'classifier') and isinstance(self.backbone.classifier, nn.Sequential) and len(self.backbone.classifier) > 0:
                last_linear_layer = None
                for layer_idx in range(len(self.backbone.classifier) -1, -1, -1):
                    layer = self.backbone.classifier[layer_idx]
                    if isinstance(layer, nn.Linear):
                        last_linear_layer = layer
                        break
                if last_linear_layer:
                    self.output_features_dim = last_linear_layer.in_features
                    self.backbone.classifier = nn.Identity() 
                else: 
                    # If classifier has no linear layer (e.g. just Dropout), remove it and get features before it
                    self.backbone.classifier = nn.Identity() 
                    # Determine output features by passing a dummy input
                    # This requires knowing the expected input shape (e.g., 224x224)
                    # Use a generic shape; actual image size might vary but features before global pool are usually consistent in channels
                    dummy_input = torch.randn(1, input_channels, 224, 224) 
                    with torch.no_grad():
                        dummy_output_map = self.backbone(dummy_input) # Output is usually (B, C, H_feat, W_feat)
                    self.output_features_dim = dummy_output_map.shape[1] # Number of channels
            elif hasattr(self.backbone, 'head') and isinstance(self.backbone.head, nn.Linear): # For ViT models (e.g. vit_b_16)
                self.output_features_dim = self.backbone.head.in_features
                self.backbone.head = nn.Identity()
            elif hasattr(self.backbone, 'heads') and hasattr(self.backbone.heads, 'head') and isinstance(self.backbone.heads.head, nn.Linear): # For newer ViT models
                self.output_features_dim = self.backbone.heads.head.in_features
                self.backbone.heads.head = nn.Identity()
            else:
                raise ValueError(f"Unsupported backbone structure for {backbone_name} for head removal. Output features dim couldn't be determined.")

            if self.freeze_backbone and pretrained:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            # print(f"ImageProcessor: Loaded '{backbone_name}' (input_channels={input_channels}, pretrained={pretrained}, frozen={freeze_backbone}). Output features/channels: {self.output_features_dim}") # Optional

        else:
            raise ValueError(f"Backbone '{backbone_name}' not handled by ImageProcessor (not 'custom_depth_cnn' and not in torchvision.models).")

    def forward(self, x):
        features = self.backbone(x) 
        
        if self.output_flattened_features:
            if features.dim() > 2 and not self.is_custom_cnn: 
                 pool = nn.AdaptiveAvgPool2d((1,1))
                 features = pool(features)
            features = torch.flatten(features, 1) 
        return features

class LiDARBEVProcessor(nn.Module):
    def __init__(self, input_bev_channels, output_feature_channels=256, num_conv_layers=3):
        super().__init__()
        self.output_features_dim = output_feature_channels 
        layers = []
        current_channels = input_bev_channels
        for i in range(num_conv_layers):
            out_c = output_feature_channels // (2**(num_conv_layers - 1 - i)) if i < num_conv_layers -1 else output_feature_channels
            layers.append(nn.Conv2d(current_channels, out_c, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            current_channels = out_c
        self.processor = nn.Sequential(*layers)
        # print(f"LiDARBEVProcessor initialized. Output feature channels: {self.output_features_dim}") # Optional

    def forward(self, x_bev):
        features = self.processor(x_bev)
        return features

class IdentityProcessor(nn.Module):
    def __init__(self, output_features_dim, **kwargs): 
        super().__init__()
        self.output_features_dim = output_features_dim
        # print(f"IdentityProcessor initialized. Expecting and outputting features of dim: {output_features_dim}") # Optional

    def forward(self, x):
        if x.dim() > 2 and x.shape[1] != self.output_features_dim : # If input is BxCxHxW, needs flattening
            # This case assumes the feature dim is the flattened dim.
            # If output_features_dim is meant to be channel dim for a map, this needs adjustment.
            # For pre-extracted features that are already vectors (B, Features), this won't trigger.
             x = torch.flatten(x,1)
        if x.shape[-1] != self.output_features_dim:
            # This can happen if output_features_dim from config doesn't match actual loaded feature dim
            print(f"Warning: IdentityProcessor input feature dim {x.shape[-1]} != configured output_features_dim {self.output_features_dim}. Passing through input as is.")
        return x