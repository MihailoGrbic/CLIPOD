import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

from dataset import Dataset

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
    # model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device).eval()

    # Load the annotations and COCO API object
    ann_path = '{}\\annotations\\instances_{}.json'.format(dataset_path, val_dir)
    val_img_dir = '{}\\{}\\'.format(dataset_path, val_dir)
    coco = COCO(ann_path)

    cats = coco.loadCats(coco.getCatIds())
    cat_map = get_cat_id_map(cats)
    #cat_names = [cat['name'] for cat in cats]
    cat_names = ["a photo of a " + cat['name'] for cat in cats]

    val_dataset = Dataset(ann_path, val_img_dir)

    with torch.no_grad():
        # for elem in tqdm(val_dataset[:40]):
        for elem in tqdm(val_dataset[0]):
            #3, 10, 11, 21
            image, image_id = elem["image"], elem["id"]

            w, h = image.size
            #image = image.crop((200, 200, 300, 300))  #21
            #image = image.crop((270, 280, 425, 427))  #10
            #image = image.crop((315, 280, 370, 427))  #10
            #image = image.crop((370, 70, 480, 350))  #3

            # image = image.crop((2*w/3, h/3, 3 * w/3, 2*h/3))
            #image = image.crop((w/3, 2*h/3, 2 * w/3, 3*h/3))
            # image[:, :, 0:370] = 0
            # image[:, :, 500:640] = 0
            # image[:, 350:425, :] = 0
            # image[:, 0:60, :] = 0
            #print(image.shape)
            
            #cat_names[0] = "a photo of a person"
            # cat_names = ['a photo of a person', 'a photo of a airplane', 'a photo of a dog', 'a photo of a bear', 
            #             'a photo of a baseball glove', 'a photo of a bed', 'a photo of a fork']
            # cat_names = ['person', 'airplane', 'dog', 'bear', 
            #             'baseball glove', 'bed', 'fork']
            # cat_names = ['a photo of a person', 'a photo of a motorcycle']
            # cat_names = ['a photo of a person', 'a photo of a skateboard']
            # cat_names = ['a photo of a person', 'a photo of a oven']
            # cat_names = ['a photo of a person', 'background']
            # print(cat_names)

            inputs = processor(text=cat_names, images=image, return_tensors="pt", padding=True)

            #inputs = processor(text=['person', 'other'], images=image, return_tensors="pt", padding=True)
            # Move inputs to GPU
            for key, val in inputs.items(): inputs[key] = val.to(device)

            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

            probs = probs.cpu().numpy().flatten()
            # print(probs)
            
            for i, name in enumerate(cat_names):
                print(name + "  " + str(probs[i]))

            # image = torch.moveaxis(image, 0, -1).cpu().numpy()
            plt.imshow(image)
            plt.show()
