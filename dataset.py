import os

from pycocotools.coco import COCO
import numpy as np
from PIL import Image

from util import get_id_cat_map

class COCODataset():
    def __init__(self, annotations_path, img_dir):
        self.coco = COCO(annotations_path)
        self.image_infos = self.coco.loadImgs(self.coco.getImgIds())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.id_cat_map = get_id_cat_map(self.cats)

        self.img_dir = img_dir

    def getCats(self):
        return self.cats

    def __len__(self):
        return len(self.image_infos)

    def __getitem__(self, idx):
        image_info_slice = self.image_infos[idx]
        if isinstance(idx, int):
            image_info_slice = [image_info_slice]
        
        samples = []
        for image_info in image_info_slice:
            img_path = os.path.join(self.img_dir, image_info['file_name'])
            image = Image.open(img_path)
            
            label_ids = self.coco.getAnnIds(image_info['id'])
            labels = self.coco.loadAnns(label_ids)
            
            # Add category name to every label dict
            for label in labels:
                label['category_name'] = self.id_cat_map[label['category_id']]

            sample = {'image' : image, 'image_info' : image_info, 'labels' : labels}
            samples.append(sample)
        
        if isinstance(idx, int): return samples[0]
        return samples

    def get_by_id(self, img_id):
        for i, info in enumerate(self.image_infos):
            if info['id'] == img_id:
                return self.__getitem__(i)