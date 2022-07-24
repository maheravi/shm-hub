# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco

from .face import build as build_face
from .crack import build as build_crack
from .crack_instance import build as build_crack_instance

from .coco_playground import build as build_playground


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'face':
        return build_face(image_set, args)
    if args.dataset_file == 'crack':
        return build_crack(image_set, args)
    if args.dataset_file == 'crack_instance':
        return build_crack_instance(image_set, args)
    if args.dataset_file == 'coco_playground':
        return build_playground(image_set, args)

    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
