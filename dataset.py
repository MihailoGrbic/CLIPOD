import os
import json

import numpy as np
from PIL import Image

class Dataset():
    def __init__(self, annotations_path, img_dir, transform=None, target_transform=None):
        f = open(annotations_path)
        self.annotations = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations['images'])

    #TODO add random shuffle
    def __getitem__(self, idx):
        elems = self.annotations['images'][idx]
        if not isinstance(elems, list):
            elems = [elems]
        
        samples = []
        for elem in elems:
            img_path = os.path.join(self.img_dir, elem['file_name'])
            image = Image.open(img_path)
            image_id = elem['id']

            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            sample = {'image' : image, 'id' : image_id}
            samples.append(sample)
            
        return samples