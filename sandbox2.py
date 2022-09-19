from PIL import Image
from PIL import ImageFilter
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

from dataset import COCODataset
from configs.detection_configs import text_strat_configs

np.random.seed(3141592)

def get_cat_id_map(cats):
    result = {}
    for elem in cats:
        result[elem['name']] = elem['id']
    return result


dataset_path = 'C:\\datasets\\COCO\\'
train_dir = 'train2017'
val_dir = 'val2017'
train_dir = 'train2017'
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
    train_ann_path = f'{dataset_path}\\annotations\\instances_{train_dir}.json'
    val_img_dir = f'{dataset_path}\\{val_dir}\\'
    train_img_dir = f'{dataset_path}\\{train_dir}\\'

    val_dataset = COCODataset(val_ann_path, val_img_dir)
    train_dataset = COCODataset(train_ann_path, train_img_dir)

    cats = val_dataset.getCats()
    cat_map = get_cat_id_map(cats)

    #cat_names = [cat['name'] for cat in cats]
    cat_names = ["A photo of a " + cat['name'] for cat in cats]

    with torch.no_grad():
        for elem in tqdm(val_dataset[4:5]):
        # for elem in tqdm([train_dataset.get_by_id(456438)]):
            #0, 3, 4, 7, 21
            image, labels = elem["image"], elem["labels"]

            w, h = image.size

            # The JPG adversarial attack
            # image = image.crop((130, 31, 259, 141))  #4 never a car
            # image = image.filter(ImageFilter.BLUR)
            #image2 = Image.open("examples/img/not_a_car.jpg")

            # with BytesIO() as f:
            #     image.save(f, format='JPEG')
            #     f.seek(0)
            #     image = Image.open(f)
            #     image.load()

            # image = image.crop((140, 40, 249, 131))  #4 never a car
            # image = image.crop((147, 68, 240, 115))  #4 car

            # 4 bench with negative text
            # image = image.crop(([2, 136, 196, 269]))  #supposedly 0.64837
            # image = image.crop(([2, 100, 225, 218]))  # supposedly 0.9532
            # image = image.crop(([0, 129, 213, 258]))  #supposedly 0.830736

            # 4 bycicle
            # image = image.crop((274, 232, 438, 307))  # with promising cats 0.9961785078048706
            image = image.crop((16, 57, 640, 327))  # all cats 0.9889393
            
            # image = image.crop((200, 0, 500, 350))  # ideal 0.97616
            
            # image = image.crop((186, 16, 338, 83))  #4 bycicle
            # image = image.crop((141, 26, 323, 178))  #4 bycicle
            # image = image.crop((190, 0, 500, 350))  #4 bycicle 9.7443e-01
            # image = image.crop((220, 0, 500, 350))  #4 bycicle 9.7444e-01
            # whole image bycicle 9.6786e-01

            # promising cats
            # cat_names = ['skateboard', 'parking meter', 'motorcycle', 'bicycle', 'hair drier', 'bird', 'bench',
            #              'fire hydrant', 'traffic light', 'car', 'mouse', 'dog', 'stop sign', 'person']
            # cat_names = ["A photo of a " + cat for cat in cat_names]

            # negative text
            # cat_names = ['a photo of a bench']
            # cat_names.extend(text_strat_configs['negative_text']['detection_text_settings']['negative_text'])

            # cat_names = ['a photo of a car', 'a photo of a bicycle']

            #images = [image for i in range(200)]
            images = [image]
            # for i in range(200):
            #     new_bbox = ([0, 0, image.width, image.height] + np.random.randint(-20, 20, (1, 4))).tolist()[0]
            #     images.append(image.crop(new_bbox))

            inputs = processor(text=cat_names, images=images, return_tensors="pt", padding=True)

            # Move inputs to GPU
            for key, val in inputs.items():
                inputs[key] = val.to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            probs = probs.cpu().numpy()
            print(probs)
            probs = probs[0, :].tolist()
            probs, cat_names = (list(t) for t in zip(*sorted(zip(probs, cat_names))))
            for i, name in enumerate(cat_names):
                print(name + "  " + str(probs[i]))

            # image = torch.moveaxis(image, 0, -1).cpu().numpy()
            plt.subplot(1,3,1)
            plt.imshow(image)
            plt.subplot(1,3,2)
            plt.imshow(image2)
            plt.subplot(1,3,3)
            dif = np.array(image, dtype=float) - np.array(image2, dtype=float)
            dif = (dif - dif.min()) / (dif.max() - dif.min())
            plt.imshow(dif)
            plt.show()
            # image = image.save("examples/img/not_a_car.jpg")

            # plt.imshow(image2)
            # plt.show()

            # plt.imshow(image3)
            # plt.show()
