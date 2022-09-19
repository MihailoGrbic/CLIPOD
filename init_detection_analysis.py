import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, roc_curve, auc
from tqdm import tqdm

from configs.init_detection_configs import base_config, init_detection_configs
from dataset import COCODataset
from object_detector import CLIPObjectDetector
from util import write_to_json

random.seed(3141592)
np.random.seed(3141592)

dataset_path = 'C:\\datasets\\COCO\\'
val_dir = 'val2017'

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', required=True, type=str,
                        help="Key name of the config dict in init_detection_configs")
    args = parser.parse_args()

    # Initialize the object detector
    config = base_config
    if args.config_name not in init_detection_configs.keys():
        raise ValueError(
            f"Invalid config name: {args.config_name}. Check in configs/init_detection_configs for valid config names.")
    else:
        config.update(init_detection_configs[args.config_name])
    detector = CLIPObjectDetector("openai/clip-vit-base-patch32", config=config)

    # Load the annotations and COCO API object
    ann_path = f'{dataset_path}\\annotations\\instances_{val_dir}.json'
    val_img_dir = f'{dataset_path}\\{val_dir}\\'

    val_dataset = COCODataset(ann_path, val_img_dir)

    # Get all category names
    cats = val_dataset.getCats()
    cat_names = [cat['name'] for cat in cats]
    cat_names = sorted(cat_names)

    # === Main Loop ===
    all_predictions = []
    all_labels = []
    all_probs = []
    for elem in tqdm(val_dataset):
        image, labels = elem["image"], elem["labels"]

        positive_cats, _, probs = detector.init_detection(image, cat_names, return_probs=True)
        all_probs.append(np.amax(probs, axis=0))

        label_list = [label['category_name'] for label in labels]
        all_labels.append(tuple(label_list))
        all_predictions.append(tuple(positive_cats))

        # print(positive_cats)
        # plt.imshow(image)
        # plt.show()

    # Count average number of promising classes
    prediction_counts = [len(prediction) for prediction in all_predictions]
    avg_cats = sum(prediction_counts) / len(prediction_counts)
    print("Average number of classes detected per image: " + str(avg_cats))

    # Create classification report
    mlb = MultiLabelBinarizer()
    mlb.fit([cat_names])
    tr_labels = mlb.transform(all_labels)
    tr_predictions = mlb.transform(all_predictions)

    print(classification_report(tr_labels, tr_predictions, target_names=mlb.classes_))
    report = classification_report(tr_labels, tr_predictions, target_names=mlb.classes_, output_dict=True)
    report['avg_det_cats'] = avg_cats
    report['config'] = config

    # Calculate ROC curve (won't be correct if using methods with repeat)
    all_probs = np.array(all_probs)
    fpr, tpr, thresholds = roc_curve(tr_labels.ravel(), all_probs.ravel())
    auc_score = auc(fpr, tpr)
    roc_report = {
        'auc': auc_score,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
    }
    write_to_json(roc_report, f'./results/init_detection_roc/{args.config_name}.json')

    plt.plot(fpr, tpr, color="darkorange")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Stopa lazno pozitivinih")
    plt.ylabel("Stopa stvarno pozitivnih")
    plt.show()

    # write_to_json(report, f'./results/init_detection/{args.config_name}.json')

