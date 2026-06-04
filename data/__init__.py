# File: data/__init__.py
from .cifar100_loader import get_cifar100_loaders
from .newsgroups20_loader import get_20newsgroups_loaders
from .imagenet_loader import get_imagenet_loaders
from .a2d2_loader import get_a2d2_loaders
from .coco_loader import get_coco_loaders
from .imagenet_feature_loader import get_imagenet_feature_loaders # <<< ADD THIS LINE
from .toy_loader import get_toy_feature_loaders
from .digits_loader import get_digits_loaders
