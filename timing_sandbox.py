import time

import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

from dataset import COCODataset
from util import get_cat_id_map

np.random.seed(3141592)

dataset_path = 'C:\\datasets\\COCO\\'
val_dir = 'val2017'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # Load the pretrained model
    model_name = "openai/clip-vit-base-patch32"
    # model_name = "openai/clip-vit-base-patch16"
    # model_name = "openai/clip-vit-large-patch14"
    # model_name = "openai/clip-vit-large-patch14-336"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.to(device).eval()

    # Load the annotations and COCO API object
    val_ann_path = f'{dataset_path}\\annotations\\instances_{val_dir}.json'
    val_img_dir = f'{dataset_path}\\{val_dir}\\'

    val_dataset = COCODataset(val_ann_path, val_img_dir)

    cats = val_dataset.getCats()
    cat_map = get_cat_id_map(cats)

    cat_names = ["A photo of a " + cat['name'] for cat in cats]

    t1 = time.time()
    images = []
    for elem in val_dataset[0:50]:
        image, labels = elem["image"], elem["labels"]
        #image = image.resize((224,224))
        images.append(image)

    t2 = time.time()
    inputs = processor(text=cat_names, images=images, return_tensors="pt", padding=True)
    t3 = time.time()

    # Move inputs to GPU
    for key, val in inputs.items():
        inputs[key] = val.to(device)

    t4 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        probs = probs.cpu().numpy()[0, :].tolist()
    t5 = time.time()

    print(t2 - t1)
    print(t3 - t2)
    print(t5 - t4)

    # probs, cat_names = (list(t) for t in zip(*sorted(zip(probs, cat_names))))
    # for i, name in enumerate(cat_names):
    #     print(name + "  " + str(probs[i]))

