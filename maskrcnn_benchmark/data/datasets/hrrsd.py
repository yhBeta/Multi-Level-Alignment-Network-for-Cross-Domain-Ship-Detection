import os

import torch
import torch.utils.data
from PIL import Image
import sys
import cv2 as cv
import xmltodict

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from maskrcnn_benchmark.structures.bounding_box import BoxList


class HRRSDDataset(torch.utils.data.Dataset):
    CLASSES = (
        "__background__ ",
        "ship",
    )

    def __init__(self, data_dir, transforms=None, is_source= False):
        self.root = data_dir
        self.transforms = transforms
        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self.ids = []
        cls = HRRSDDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        for x in os.listdir(os.path.join(self.root, "JPEGImages")):
            anno = ET.parse(self._annopath % x.split('.')[0]).getroot()
            anno = self._preprocess_annotation(anno)
            if anno["boxes"].shape != torch.Size([0]):
                self.ids.append(x.split('.')[0])
        self.transforms = transforms
        self.is_source = is_source

    def __getitem__(self, index):
        img_id = self.ids[index]
        # print('SSDD2016', img_id)
        img = Image.open(self._imgpath % img_id).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)
        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        domain_labels = torch.ones_like(anno["labels"], dtype=torch.uint8) \
            if self.is_source else torch.zeros_like(anno["labels"], dtype=torch.uint8)
        # is_gt = torch.ones_like(anno["labels"], dtype=torch.uint8) \
        #     if (int(img_id) < 10) else torch.zeros_like(anno["labels"], dtype=torch.uint8)
        # is_gt = torch.zeros_like(anno["labels"], dtype=torch.uint8)
        target.add_field("is_source", domain_labels)
        # target.add_field("is_gt", is_gt)
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1

        for obj in target.iter("object"):
            # difficult = int(obj.find("difficult").text) == 1
            difficult = 0
            # if not self.keep_difficult and difficult:
            #     continue
            name = obj.find("name").text.lower().strip()
            if name != 'ship':
                continue
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text,
                bb.find("ymin").text,
                bb.find("xmax").text,
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return HRRSDDataset.CLASSES[class_id]
