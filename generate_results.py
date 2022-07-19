import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

from dataset import ZeroShotDataset

def get_cat_id_map(cats):
    result = {}
    for elem in cats:
        result[elem['name']] = elem['id']
    return result

dataset_path = 'C:\\datasets\\COCO\\'
train_dir = 'train2017'
val_dir = 'val2017'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # Load the pretrained model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device).eval()

    # Load the annotations and COCO API object
    ann_path = '{}\\annotations\\instances_{}.json'.format(dataset_path, val_dir)
    val_img_dir = '{}\\{}\\'.format(dataset_path, val_dir)
    coco = COCO(ann_path)

    cats = coco.loadCats(coco.getCatIds())
    cat_map = get_cat_id_map(cats)
    cat_names = [cat['name'] for cat in cats]

    val_dataset = ZeroShotDataset(ann_path, val_img_dir)

    with torch.no_grad():
        for elem in tqdm(val_dataset[0]):
            image, image_id = elem["image"], elem["id"]
            image[:, :, 0:370] = 0
            image[:, :, 500:640] = 0
            image[:, 350:425, :] = 0
            image[:, 0:60, :] = 0
            #print(image.shape)
            #inputs = processor(text=cat_names, images=image, return_tensors="pt", padding=True)
            inputs = processor(text=['person', 'background'], images=image, return_tensors="pt", padding=True)
            # Move inputs to GPU
            for key, val in inputs.items(): inputs[key] = val.to(device)

            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

            probs = probs.cpu().numpy().flatten()
            print(probs)
            # for i, name in enumerate(cat_names):
            #     print(name + "  " + str(probs[i]))

            image = torch.moveaxis(image, 0, -1).cpu().numpy()
            plt.imshow(image)
            plt.show()
