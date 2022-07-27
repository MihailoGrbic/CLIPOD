import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import torch
from tqdm import tqdm

from dataset import Dataset
from object_detector import CLIPObjectDetector
from util import get_cat_id_map


dataset_path = 'C:\\datasets\\COCO\\'
train_dir = 'train2017'
val_dir = 'val2017'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', dest='config', type=str, help='Path to config file')
    args = parser.parse_args()

    # Initialize the object detector
    detector = CLIPObjectDetector("openai/clip-vit-base-patch32", device, batch_size=32)

    # Load the annotations and COCO API object
    ann_path = '{}\\annotations\\instances_{}.json'.format(dataset_path, val_dir)
    val_img_dir = '{}\\{}\\'.format(dataset_path, val_dir)
    coco = COCO(ann_path)

    # Get all category names
    cats = coco.loadCats(coco.getCatIds())
    cat_id_map = get_cat_id_map(cats)
    cat_names = [cat['name'] for cat in cats]

    val_dataset = Dataset(ann_path, val_img_dir)
    
    # === Main Loop ===
    with torch.no_grad():
        #for elem in tqdm(val_dataset[:10]):
        for elem in val_dataset[:100]:
            image, image_id = elem["image"], elem["id"]

            detections = detector.detect(image, cat_names)
            
            print(detections)
            # plt.imshow(image)
            # plt.show()
            # if len(detections) > 20:
            #     plt.imshow(image)
            #     plt.show()
            
            
