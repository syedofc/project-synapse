# File: data/a2d2_loader.py
"""
A2D2 Dataset Loader for Camera-based Semantic Segmentation.
"""
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import json
import torchvision.transforms as T

def extract_timestamp_from_a2d2_filename(filepath):
    """
    Extracts timestamp from A2D2 filenames like:
    YYYYMMDDhhmmss_camera_cam_front_center_000001617.png
    YYYYMMDDhhmmss_label_cam_front_center_000001617.png
    Returns 'YYYYMMDDhhmmss_000001617' for a specific camera view,
    or just the frame ID '000001617' if we only care about matching within a view.
    For simplicity, let's match on the frame ID part.
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]
    # Example: 20180807145028_camera_frontcenter_000001617
    # We want the last part: 000001617
    return basename.split('_')[-1]

class A2D2SemanticSegmentationDataset(Dataset):
    def __init__(self, root_dir, camera_view, image_transform=None, mask_transform=None, color_to_class_id_map=None):
        self.root_dir = root_dir
        self.camera_view = camera_view # e.g., "cam_front_center"
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.color_to_class_id_map = color_to_class_id_map

        if self.color_to_class_id_map is None or not self.color_to_class_id_map:
            raise ValueError("[A2D2SemanticSegmentationDataset]: color_to_class_id_map is essential.")

        # Glob for scene folders (YYYYMMDD_hhmmss)
        scene_folders = glob.glob(os.path.join(self.root_dir, "20*")) # Assumes scenes start with "20"
        
        self.image_files = []
        self.label_files = [] # Semantic segmentation label images

        for scene_path in scene_folders:
            # Path patterns based on A2D2 documentation for semantic segmentation
            current_image_files = sorted(glob.glob(os.path.join(scene_path, "camera", self.camera_view, "*.png")))
            current_label_files = sorted(glob.glob(os.path.join(scene_path, "label", self.camera_view, "*.png")))
            
            self.image_files.extend(current_image_files)
            self.label_files.extend(current_label_files)

        # Pair images and labels by extracted timestamp/frame ID
        image_map = {extract_timestamp_from_a2d2_filename(f): f for f in self.image_files}
        label_map = {extract_timestamp_from_a2d2_filename(f): f for f in self.label_files}

        common_keys = set(image_map.keys()) & set(label_map.keys())
        self.valid_pairs = []
        for key in sorted(list(common_keys)):
            self.valid_pairs.append((image_map[key], label_map[key]))

        print(f"[A2D2SemSeg] Found {len(self.image_files)} total camera images in view '{self.camera_view}'.")
        print(f"[A2D2SemSeg] Found {len(self.label_files)} total label images in view '{self.camera_view}'.")
        print(f"[A2D2SemSeg] Found {len(self.valid_pairs)} matched image-label pairs.")
        if not self.valid_pairs:
            print(f"  WARNING: No pairs found. Check root_dir '{self.root_dir}', camera_view '{self.camera_view}', and A2D2 structure.")

    def __len__(self):
        return len(self.valid_pairs)

    def _convert_rgb_mask_to_class_ids(self, mask_image_pil):
        mask_np_rgb = np.array(mask_image_pil)
        h, w, c = mask_np_rgb.shape
        if c != 3:
            raise ValueError(f"Expected RGB mask (3 channels), got {c} channels for mask.")
            
        class_id_mask = torch.full((h, w), -1, dtype=torch.int64) # -1 for ignore_index
        for color_rgb_tuple, class_id in self.color_to_class_id_map.items():
            matches = (mask_np_rgb[:, :, 0] == color_rgb_tuple[0]) & \
                      (mask_np_rgb[:, :, 1] == color_rgb_tuple[1]) & \
                      (mask_np_rgb[:, :, 2] == color_rgb_tuple[2])
            class_id_mask[matches] = class_id
        return class_id_mask

    def __getitem__(self, idx):
        img_path, label_path = self.valid_pairs[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            label_pil_image = Image.open(label_path).convert("RGB") # Ensure label is also RGB before mapping
        except FileNotFoundError:
            print(f"Error: File not found. Image: {img_path} or Mask: {label_path}")
            # Fallback to return dummy data of expected type/shape if a transform is defined
            # This needs to be more robust based on how transforms are defined
            dummy_img_size = self.image_transform.transforms[0].size if self.image_transform and hasattr(self.image_transform, 'transforms') and isinstance(self.image_transform.transforms[0], T.Resize) else (224,224)
            return {'camera_image': torch.zeros(3, *dummy_img_size)}, torch.zeros(*dummy_img_size, dtype=torch.long)

        target_segmentation_mask = self._convert_rgb_mask_to_class_ids(label_pil_image)

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            # For torchvision transforms like Resize, mask needs to be PIL or C,H,W
            # Our target_segmentation_mask is H,W. Add channel dim, transform, then remove.
            target_segmentation_mask = self.mask_transform(T.ToPILImage()(target_segmentation_mask))


        inputs_dict = {'camera_image': image}
        target = target_segmentation_mask # This is now a LongTensor of class IDs, HxW
        
        return inputs_dict, target


def get_a2d2_loaders(task_config, batch_size, num_workers=4, data_path=None):
    """
    Creates and returns A2D2 DataLoaders for camera-based semantic segmentation.
    'data_path' should be the root of the A2D2 dataset scenes.
    'task_config' is the 'dataset_specific_params' from the main YAML.
    """
    if data_path is None: # data_path is the root_dir for A2D2SemanticSegmentationDataset
        raise ValueError("data_path (A2D2 root directory) must be provided for get_a2d2_loaders")

    img_size_hw = task_config.get('img_size_a2d2_camera', [384, 608]) # H, W
    num_classes = task_config.get('num_classes')
    camera_view = task_config.get('camera_view', 'cam_front_center') # e.g. from YAML

    a2d2_color_map = None
    color_map_json_path = task_config.get('color_map_json_path')

    if color_map_json_path and os.path.exists(color_map_json_path):
        try:
            with open(color_map_json_path, 'r') as f:
                raw_map = json.load(f)
                a2d2_color_map = {tuple(map(int, k.split(','))): int(v) for k,v in raw_map.items()}
            print(f"Successfully loaded A2D2 color map from {color_map_json_path}")
        except Exception as e:
            print(f"ERROR loading or parsing color map from {color_map_json_path}: {e}.")
            raise ValueError(f"A2D2 color_map_json_path '{color_map_json_path}' is invalid. Cannot proceed.")
    else:
        raise ValueError(f"A2D2 color_map_json_path '{color_map_json_path}' is missing or invalid. Cannot proceed.")

    if num_classes is None:
        num_classes = len(set(a2d2_color_map.values()))
        print(f"Inferred num_classes for A2D2 from color_map: {num_classes}")
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError(f"Number of classes for A2D2 segmentation is invalid ({num_classes}). Check config and color map.")

    image_transform = T.Compose([
        T.Resize(img_size_hw), 
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = T.Compose([
        T.Resize(img_size_hw, interpolation=T.InterpolationMode.NEAREST),
        # Output from PILToTensor will be (1, H, W), squeeze to (H,W)
        T.PILToTensor(),
        T.Lambda(lambda x: x.squeeze().long()) 
    ])
    
    # Assuming data_path is the root where scene folders like '20180807_145028' are located
    full_dataset = A2D2SemanticSegmentationDataset(
        root_dir=data_path,
        camera_view=camera_view,
        image_transform=image_transform,
        mask_transform=mask_transform,
        color_to_class_id_map=a2d2_color_map
    )

    if len(full_dataset) == 0:
        print(f"ERROR: A2D2SemanticSegmentationDataset loaded 0 samples from {data_path} for view {camera_view}.")
        return None, None, {}, 0

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(task_config.get('seed', config.get('seed', 42) if 'config' in globals() else 42))
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)
    
    print(f"A2D2 SemSeg ({camera_view}) full dataset: {len(full_dataset)}. Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    # Determine input_dims_dict from a sample
    # For this loader, input is always 'camera_image'
    sample_input_dict, _ = full_dataset[0]
    input_dims_dict = {'camera_image': tuple(sample_input_dict['camera_image'].shape)} 
    
    return train_loader, val_loader, input_dims_dict, num_classes

if __name__ == '__main__':
    # Example usage for testing this file directly
    print("Running a2d2_loader.py directly for testing semantic segmentation...")
    
    # Create a dummy color map json for testing if one doesn't exist
    dummy_color_map_file = "dummy_a2d2_seg_color_map.json"
    if not os.path.exists(dummy_color_map_file):
        print(f"Creating dummy color map: {dummy_color_map_file}")
        # These colors and IDs are PURELY EXAMPLES. Use A2D2's actual scheme.
        dummy_map_content = { 
            "128,0,128": 1,  # Example: Car (Purple)
            "224,224,192": 2 # Example: Road (Beige)
            # Add an entry for class 0, often background/unlabeled
            # "0,0,0": 0 
        }
        with open(dummy_color_map_file, 'w') as f:
            json.dump(dummy_map_content, f)

    sample_a2d2_task_config = {
        'camera_view': 'cam_front_center', # Specify which camera view
        'img_size_a2d2_camera': [192, 320], # Smaller size for quick test H,W
        'num_classes': None, # Will be inferred from color_map if map is good
        'color_map_json_path': dummy_color_map_file, # Path to your actual color map JSON
        'seed': 42
    }
    
    # This is the root path where your A2D2 scene folders (e.g., "20180807_145028") are.
    # The loader will look for "camera/cam_front_center" and "label/cam_front_center" inside these scene folders.
    a2d2_dataset_root = "/home/icps/Downloads/Data/a2d2_camera_lidar_semantic_bboxes/camera_lidar_semantic" # ADJUST THIS PATH

    if not os.path.isdir(a2d2_dataset_root):
        print(f"ERROR: A2D2 dataset root path does not exist: {a2d2_dataset_root}")
        print("Please adjust 'a2d2_dataset_root' in the __main__ block of a2d2_loader.py for testing.")
    else:
        train_loader, val_loader, input_dims, num_classes_out = get_a2d2_loaders(
            data_path=a2d2_dataset_root,
            task_config=sample_a2d2_task_config,
            batch_size=2,
            num_workers=0
        )

        if train_loader:
            print(f"\nSuccessfully created A2D2 (Semantic Segmentation) train_loader with {len(train_loader)} batches.")
            print(f"Input Dims: {input_dims}, Num Classes: {num_classes_out}")
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                print(f"\nBatch {batch_idx + 1}:")
                print(f"  Inputs ('camera_image') shape: {inputs['camera_image'].shape}") # B, C, H, W
                print(f"  Targets (segmentation mask) shape: {targets.shape}") # B, H, W
                print(f"  Targets data type: {targets.dtype}")
                print(f"  Targets unique values: {torch.unique(targets)}") # Should be class IDs
                if batch_idx == 0: # Inspect only the first batch
                    break
        else:
            print("Failed to create A2D2 Semantic Segmentation DataLoaders in __main__ test.")