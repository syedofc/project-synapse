# File: data/coco_loader.py
"""
COCO Dataset Loader for Object Detection.
"""
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T
from PIL import Image

# COCO class IDs (91 classes including background/stuff, but typically 80 object classes are used)
# We will map these to a continuous range 0-79 if using the 80 common objects.
# This mapping can be handled based on the specific needs of the detection head.
# For now, the loader will return the original COCO category IDs.
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
# Mapping from COCO original category IDs to a continuous range (0-79 for 80 common objects)
# This is often needed for model heads.
# coco_categories = datasets.CocoDetection(root=..., annFile=...).coco.cats # gets category info
# coco_id_to_continuous_id = {cat_id: i for i, cat_id in enumerate(coco_categories.keys())}
# continuous_id_to_coco_id = {i: cat_id for i, cat_id in enumerate(coco_categories.keys())}

class CocoDetectionDataset(datasets.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        super().__init__(root, annFile, transform=transform, target_transform=target_transform)
        # Filter out images with no annotations, if necessary for your training
        # self.ids = [img_id for img_id in self.ids if len(self.coco.getAnnIds(imgIds=img_id)) > 0]
        # print(f"COCO: Kept {len(self.ids)} images with annotations.")


    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_anns = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        # Prepare targets for detection
        boxes = []
        labels = [] # COCO category IDs
        areas = []
        iscrowd = []

        for ann in coco_anns:
            if ann.get('iscrowd', 0) == 0: # Ignore crowd annotations for now
                # Bounding box [x,y,width,height]
                bbox_xywh = ann['bbox']
                # Convert to [x_min, y_min, x_max, y_max]
                boxes.append([bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]])
                labels.append(ann['category_id'])
                areas.append(ann['area'])
                iscrowd.append(ann['iscrowd'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels # These are original COCO category IDs
        target["image_id"] = torch.tensor([img_id])
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms is not None: # Standard Torchvision transforms apply to img and target
            img, target = self.transforms(img, target)
            
        # Our SynapseOS loader expects inputs_dict, target_for_loss
        # For COCO detection, the target itself is a dictionary.
        # The 'img' is the primary input.
        inputs_dict = {'camera_image': img} 
        
        return inputs_dict, target


def get_coco_detection_collate_fn(batch):
    """
    Collate function for object detection.
    Input `batch` is a list of tuples: [(inputs_dict_0, target_0), (inputs_dict_1, target_1), ...]
    where inputs_dict_0 = {'camera_image': img_tensor_0}
    and   target_0 = {'boxes': boxes_tensor_0, 'labels': labels_tensor_0, ...}
    """
    # Separate inputs and targets
    input_dicts = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Batch images (they should be stackable after transforms)
    batched_images = torch.stack([d['camera_image'] for d in input_dicts], dim=0)
    
    batched_inputs_dict = {'camera_image': batched_images}
    
    # Targets remain a list of dictionaries, as number of boxes varies per image
    return batched_inputs_dict, targets


def get_coco_loaders(data_path, ann_file_name_train, ann_file_name_val, 
                     batch_size, num_workers=4, img_size=None): # img_size can be [H,W] or int
    """
    Creates and returns COCO DataLoaders for object detection.

    Args:
        data_path (str): Path to the COCO image directory (e.g., .../coco/images/).
        ann_file_name_train (str): Name of the training annotations JSON file (e.g., "instances_train2017.json").
        ann_file_name_val (str): Name of the validation annotations JSON file (e.g., "instances_val2017.json").
        batch_size (int): Batch size.
        num_workers (int): Number of workers.
        img_size (int or list/tuple): Target image size. If int, resizes to (img_size, img_size).
                                      If list/tuple [H,W], resizes to that.

    Returns:
        tuple: train_loader, val_loader, input_dim_tuple, num_coco_classes (typically 80 or 91)
    """
    # Standard COCO transforms for object detection
    # Note: Albumentations is often preferred for detection transforms, but torchvision is simpler to start.
    # Detection models usually handle resizing internally or expect specific input sizes.
    # For now, a simple ToTensor and Normalize. Resizing can be part of the model or pre-applied.
    
    image_transforms = []
    if img_size is not None:
        if isinstance(img_size, int):
            image_transforms.append(T.Resize((img_size, img_size)))
        elif isinstance(img_size, (list, tuple)) and len(img_size) == 2:
            image_transforms.append(T.Resize(list(img_size))) # H, W
        else:
            print(f"Warning: Invalid img_size {img_size} for COCO. Using original image size.")
            
    image_transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # For CocoDetection, transforms are applied to (image, target) pair
    # We need a wrapper for standard image transforms if they don't handle targets.
    class CocoTransforms:
        def __init__(self, img_transforms):
            self.img_transforms = T.Compose(img_transforms)

        def __call__(self, image, target):
            image = self.img_transforms(image)
            return image, target

    composed_transforms = CocoTransforms(image_transforms)

    # It's common for COCO datasets to have subfolders like 'train2017', 'val2017' inside 'data_path'
    # And annotations in an 'annotations' folder parallel to 'images'.
    # The paths below assume `data_path` points to the directory containing `train2017`, `val2017`
    # and `ann_file_name_train` is just the filename.
    
    train_img_dir = os.path.join(data_path, "train2017") # Common COCO structure
    train_ann_file = os.path.join(data_path, "../annotations", ann_file_name_train) # Common structure

    val_img_dir = os.path.join(data_path, "val2017")
    val_ann_file = os.path.join(data_path, "../annotations", ann_file_name_val)

    print(f"COCO Train: Img dir '{train_img_dir}', Ann file '{train_ann_file}'")
    print(f"COCO Val: Img dir '{val_img_dir}', Ann file '{val_ann_file}'")

    if not os.path.isdir(train_img_dir) or not os.path.isfile(train_ann_file):
        print(f"ERROR: COCO training data/annotations not found at specified paths.")
        print(f"  Checked img_dir: {train_img_dir}")
        print(f"  Checked ann_file: {train_ann_file}")
        # Return None to indicate failure
        return None, None, None, 0 
    
    if not os.path.isdir(val_img_dir) or not os.path.isfile(val_ann_file):
        print(f"ERROR: COCO validation data/annotations not found at specified paths.")
        # Optionally, can proceed without validation set for initial testing
        print(f"  Checked img_dir: {val_img_dir}")
        print(f"  Checked ann_file: {val_ann_file}")
        return None, None, None, 0


    train_dataset = CocoDetectionDataset(root=train_img_dir, annFile=train_ann_file, transforms=composed_transforms)
    val_dataset = CocoDetectionDataset(root=val_img_dir, annFile=val_ann_file, transforms=composed_transforms)

    # Get number of classes (typically 80 for objects, or 91 including 'stuff' categories)
    # This depends on how your detection head is structured.
    # For simplicity, let's use the number of categories in the annotation file.
    # coco.cats gives a dict {cat_id: category_info}. Some cat_ids might be skipped.
    num_coco_classes = len(train_dataset.coco.cats) 
    # A common practice is to map these to a continuous range 0-79 for 80 object classes.
    # The actual number of classes your model predicts depends on this mapping.
    # For SynapseOS, the head definition in YAML will specify the output num_classes.
    # We should return the number of *valid object categories* we want to detect.
    # Let's assume we use the 80 common object classes.
    # The COCO_INSTANCE_CATEGORY_NAMES list has 91 entries, but many are "N/A" or background.
    # A standard set is 80 object classes.
    num_detection_classes = 80 # Standard number of COCO object classes to detect
    print(f"COCO dataset: Using {num_detection_classes} object classes for detection.")


    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, collate_fn=get_coco_detection_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True, collate_fn=get_coco_detection_collate_fn
    )

    print(f"COCO data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Determine input_dim_tuple based on transforms
    c, h, w = 3, img_size if isinstance(img_size, int) else img_size[0], img_size if isinstance(img_size, int) else img_size[1]
    if img_size is None: # If no resize, get from a sample image (more complex)
        # For simplicity, assume img_size is always provided for COCO
        # sample_img_dict, _ = train_dataset[0]
        # c, h, w = sample_img_dict['camera_image'].shape
        raise ValueError("img_size must be provided for COCO loader to determine input_dim_tuple.")
        
    input_dim_tuple = (c, h, w)
    
    return train_loader, val_loader, input_dim_tuple, num_detection_classes

if __name__ == '__main__':
    print("Running coco_loader.py directly for testing object detection loading...")
    
    # --- !!! IMPORTANT: USER CONFIGURATION REQUIRED BELOW !!! ---
    # Adjust these paths to your actual COCO dataset locations
    # COCO_IMG_ROOT should point to the directory containing 'train2017', 'val2017' image folders
    COCO_IMG_ROOT = "/home/icps/Downloads/Data/coco2017/val2017" 
    # COCO_ANN_ROOT should point to the directory containing 'instances_train2017.json', 'instances_val2017.json'
    COCO_ANN_ROOT = "/home/icps/Downloads/Data/coco2017/annotations"
    
    ANN_FILE_TRAIN = "instances_train2017.json"
    ANN_FILE_VAL = "instances_val2017.json"

    TARGET_IMG_SIZE = [480, 640] # Example [H, W]
    # --- END OF USER CONFIGURATION ---

    # Construct full paths for annotations
    full_train_ann_path = os.path.join(COCO_ANN_ROOT, ANN_FILE_TRAIN)
    full_val_ann_path = os.path.join(COCO_ANN_ROOT, ANN_FILE_VAL)

    # Check if paths exist before calling
    if not os.path.isdir(COCO_IMG_ROOT) or not os.path.isfile(full_train_ann_path):
        print(f"ERROR: COCO image root '{COCO_IMG_ROOT}' or train annotation file '{full_train_ann_path}' not found.")
        print("Please adjust paths in the __main__ block of coco_loader.py for testing.")
    else:
        # For the standalone test, we pass annotation file names directly
        # The get_coco_loaders expects data_path (for images) and then annotation file *names*
        # It constructs the full annotation path internally relative to data_path + "../annotations"
        # Let's adjust the test call to match how train.py will call it, assuming a slightly different structure.

        # Re-adjusting for how train.py would call it using 'data_path_key' and 'ann_file_key' from config
        # For standalone test, we'll define what the config would provide:
        coco_config_data_path = COCO_IMG_ROOT # This would be config['data'][task_def['dataset_specific_params']['data_path_key']]
        coco_config_ann_train_name = ANN_FILE_TRAIN # This would be config['data'][task_def['dataset_specific_params']['ann_file_train_key']]
        coco_config_ann_val_name = ANN_FILE_VAL   # This would be config['data'][task_def['dataset_specific_params']['ann_file_val_key']]

        print(f"Test Call to get_coco_loaders:")
        print(f"  Image Root (data_path): {coco_config_data_path}")
        print(f"  Train Ann Filename: {coco_config_ann_train_name}")
        print(f"  Val Ann Filename: {coco_config_ann_val_name}")


        train_loader, val_loader, input_dims, num_classes_out = get_coco_loaders(
            data_path=coco_config_data_path, # Path to image dir (e.g., .../coco/images)
            ann_file_name_train=coco_config_ann_train_name, # e.g., instances_train2017.json
            ann_file_name_val=coco_config_ann_val_name,     # e.g., instances_val2017.json
            batch_size=2,
            num_workers=0,
            img_size=TARGET_IMG_SIZE
        )

        if train_loader:
            print(f"\nSuccessfully created COCO train_loader with {len(train_loader)} batches.")
            print(f"Input Dims: {input_dims}, Num Detection Classes: {num_classes_out}")
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                print(f"\nBatch {batch_idx + 1}:")
                cam_images = inputs['camera_image']
                print(f"  Input 'camera_image' shape: {cam_images.shape}") # B, C, H, W
                print(f"  Targets in batch: {len(targets)}") # Should be batch_size (list of dicts)
                
                if targets:
                    first_target = targets[0]
                    print(f"  First sample's target keys: {first_target.keys()}")
                    print(f"    Boxes shape: {first_target['boxes'].shape}") # (NumObjects, 4)
                    print(f"    Labels shape: {first_target['labels'].shape}") # (NumObjects,)
                    print(f"    Labels: {first_target['labels']}")
                
                if batch_idx == 0: # Inspect only the first batch
                    # Optional: Visualize first image if matplotlib is imported
                    # if cam_images.shape[0] > 0:
                    #     imshow(cam_images[0], title="Sample COCO Image") # Assuming imshow is defined
                    break
        else:
            print("Failed to create COCO DataLoaders in __main__ test. Check paths and annotation files.")