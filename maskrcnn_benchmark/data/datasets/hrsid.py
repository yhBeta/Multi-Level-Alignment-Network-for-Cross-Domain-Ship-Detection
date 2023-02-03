import os

import torch
import torchvision
import torch.utils.data
from PIL import Image
import sys
import cv2 as cv
import xmltodict

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from maskrcnn_benchmark.structures.bounding_box import BoxList


class HRSIDDataset(torchvision.datasets.coco.CocoDetection):
    CLASSES = (
        "__background__ ",
        "ship",
    )

    def __init__(self, data_dir, transforms=None, is_source= False):
        # super(HRSIDDataset, self).__init__(os.path.join(data_dir, "JPEGImages"), os.path.join(data_dir, "annotations", "train_test2017.json"))
        super(HRSIDDataset, self).__init__(os.path.join(data_dir, "JPEGImages"),
                                           os.path.join(data_dir, "annotations", "test2017.json"))
        # super(HRSIDDataset, self).__init__(os.path.join(data_dir, "JPEGImages"), os.path.join(data_dir, "annotations", "inshore.json"))
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        # ids = []
        # for img_id in self.ids:
        #     ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        #     anno = self.coco.loadAnns(ann_ids)
        #     ids.append(img_id)
        # self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms
        self.is_source = is_source

    def __getitem__(self, index):
        img, anno = super(HRSIDDataset, self).__getitem__(index)
        # img.save('datasets/HRSID_inshore/' + self.coco.imgs[index]['file_name'])

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        # masks = [obj["segmentation"] for obj in anno]
        # masks = SegmentationMask(masks, img.size)
        # target.add_field("masks", masks)

        domain_labels = torch.ones_like(classes, dtype=torch.uint8) if self.is_source else torch.zeros_like(classes,
                                                                                                            dtype=torch.uint8)
        target.add_field("is_source", domain_labels)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

    def get_groundtruth(self, index):
        img, anno = super(HRSIDDataset, self).__getitem__(index)
        anno = [obj for obj in anno if obj["iscrowd"] == 0]
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")
        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)
        target.add_field("difficult", torch.zeros_like(target.extra_fields["labels"]))
        domain_labels = torch.ones_like(classes, dtype=torch.uint8) if self.is_source else torch.zeros_like(classes,
                                                                                                            dtype=torch.uint8)
        target.add_field("is_source", domain_labels)
        return target

    def map_class_id_to_class_name(self, class_id):
        return HRSIDDataset.CLASSES[class_id]
