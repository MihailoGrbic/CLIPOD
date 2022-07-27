import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import COCODataset
from object_detector import CLIPObjectDetector
from object_detector_config import CLIPObjectDetectorConfig
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from tqdm import tqdm

from configs.init_detection_configs import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_path = 'C:\\datasets\\COCO\\'
val_dir = 'val2017'

if __name__ == "__main__":
    # Initialize the object detector
    detector = CLIPObjectDetector("openai/clip-vit-base-patch32", device, config=config_single_high)

    # Load the annotations and COCO API object
    ann_path = '{}\\annotations\\instances_{}.json'.format(dataset_path, val_dir)
    val_img_dir = '{}\\{}\\'.format(dataset_path, val_dir)
    
    val_dataset = COCODataset(ann_path, val_img_dir)

    # Get all category names
    cats = val_dataset.getCats()
    cat_names = [cat['name'] for cat in cats]
    
    # === Main Loop ===
    with torch.no_grad():
        all_predictions = []
        all_labels = []
        all_probs = []
        for elem in tqdm(val_dataset[:100]):
            image, labels = elem["image"], elem["labels"]

            positive_cats, _, probs = detector.init_detection(image, cat_names, return_probs=True)
            all_probs.append(np.amax(probs, axis=0))

            label_list = [label['category_name'] for label in labels]
            all_labels.append(tuple(label_list))
            all_predictions.append(tuple(positive_cats))

            # print(positive_cats)
            # plt.imshow(image)
            # plt.show()

    all_probs = np.array(all_probs)
    prediction_counts = [len(prediction) for prediction in all_predictions]
    avg_objs = sum(prediction_counts) / len(prediction_counts)
    print("Average number of unique objects detected per image: " + str(avg_objs))

    mlb = MultiLabelBinarizer()
    tr_labels = mlb.fit_transform(all_labels)
    tr_predictions = mlb.transform(all_predictions)
    
    print(classification_report(tr_labels, tr_predictions, target_names=mlb.classes_))