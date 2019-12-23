import os
import json
import numpy as np
import torch
import xmltodict
from glob import glob
from torch.utils import data
import skimage.io as skio
from PIL import Image
from torchvision import transforms
import albumentations
from albumentations import augmentations


def parse_annotations(path):
    with open(path, 'r') as f:
        text_data = f.read()
        dict_data = xmltodict.parse(text_data)

    filename = dict_data['annotation']['filename']
    bbox = dict_data['annotation']['object']['bndbox']
    bbox = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
    class_name = dict_data['annotation']['object']['name']

    return (filename, bbox, class_name)


class PetDataset(data.Dataset):

    def __init__(self, data_path, image_size=(300, 300), use_aug=False, to_tensors=False):
        super(PetDataset, self).__init__()
        self.image_size = image_size
        self.use_aug = use_aug
        self.to_tensors = to_tensors
        self.data_path = data_path
        self.dataset = []

        annotation_paths = glob(os.path.join(self.data_path, "annotations/xmls/*.xml"))
        for path in annotation_paths:
            try:
                filename, bbox, class_name = parse_annotations(path)
                self.dataset.append([filename, bbox, class_name])
            except (KeyError, TypeError):
                print(path)

        self.classes = {
            'object_id': 1,
            'background_id': 2,
            'object_edge_id': 3
        }
        self.num_classes = 2

    def __getitem__(self, idx):
        # get annotations
        filename, bbox, class_name = self.dataset[idx]
        # make paths for images
        mask_path = os.path.join(self.data_path, "annotations/trimaps", filename.split('.')[0] + '.png')
        image_path = os.path.join(self.data_path, "images", filename)
        # open images
        image = skio.imread(image_path)
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, repeats=3, axis=2)
        mask = skio.imread(mask_path)
        # convert mask
        mask[mask == self.classes['background_id']] = 0
        mask[mask != 0] = 1
        # make transform
        min_size = max(image.shape[0], image.shape[1])
        composition = [
            augmentations.transforms.PadIfNeeded(min_height=min_size, min_width=min_size, value=(128, 128, 128)),
            augmentations.transforms.Resize(height=self.image_size[1], width=self.image_size[0], always_apply=True)
        ]

        if self.use_aug:
            composition.extend([
                augmentations.transforms.Blur(p=0.3),
                augmentations.transforms.VerticalFlip(p=0.5),
                augmentations.transforms.HorizontalFlip(p=0.5),
                augmentations.transforms.RandomFog(p=0.1),
                augmentations.transforms.ISONoise(p=0.1),
                augmentations.transforms.JpegCompression(p=0.3),
                augmentations.transforms.HueSaturationValue(p=0.3)
            ])

        aug_compose = albumentations.core.composition.Compose(
            composition
        )
        augmented = aug_compose(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        # transform to tensors
        if self.to_tensors:
            mask = torch.from_numpy(mask).long()
            mask = torch.nn.functional.one_hot(mask, self.num_classes).permute(2, 0, 1).float()
            image = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])(image)
        sample = {'image': image, 'mask': mask}

        return sample

    def __len__(self):
        return len(self.dataset)
