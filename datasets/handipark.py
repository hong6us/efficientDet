import argparse
import os
import torch.utils.data as data
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import json
import sys
import cv2


from datasets.voc0712 import VOCAnnotationTransform
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from datasets import (Augmenter, Normalizer,
                      Resizer, VOCDetection, collater, detection_collate,
                      get_augumentation)
from models.efficientdet import EfficientDet
from utils import EFFICIENTDET, get_state_dict


HandiPark_CLASSES = (  # always index 0
    'sign-h','lamp','fh')

class HandiParkDetection(data.Dataset):
    """HandiPark Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to HaniPark folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
    """

    def __init__(self, root,
                 image_set='',
                 transform=None, target_transform=VOCAnnotationTransform(dict(zip(HandiPark_CLASSES,range(len(HandiPark_CLASSES)))))):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self._annopath = os.path.join(root,image_set, 'annotation')
        self._imgpath = os.path.join(root,image_set, 'image')
        self.ids = [fname[:-4] for fname in os.listdir(self._annopath)]
        self.ids = [basename for basename in self.ids if basename + ".jpg" in os.listdir(self._imgpath)]
   

    def __getitem__(self, index):
        img_id = self.ids[index]
        annofile = os.path.join(self._annopath, img_id+".xml")
        imgfile = os.path.join(self._imgpath, img_id+".jpg")
        target = ET.parse(annofile).getroot()
        img = cv2.imread(imgfile)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        target = np.array(target)
        sample = {'img': img, 'annot': target}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

        # bbox = target[:, :4]
        # labels = target[:, 4]

        # if self.transform is not None:
        #     annotation = {'image': img, 'bboxes': bbox, 'category_id': labels}
        #     augmentation = self.transform(**annotation)
        #     img = augmentation['image']
        #     bbox = augmentation['bboxes']
        #     labels = augmentation['category_id']
        # return {'image': img, 'bboxes': bbox, 'category_id': labels}

    def __len__(self):
        return len(self.ids)

    def num_classes(self):
        return len(HandiPark_CLASSES)

    def label_to_name(self, label):
        return HandiPark_CLASSES[label]

    def load_annotations(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        gt = np.array(gt)
        return gt
