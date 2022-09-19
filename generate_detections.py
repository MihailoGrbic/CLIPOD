import argparse
from genericpath import isfile
import json
import os.path
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from tqdm import tqdm


from configs.detection_configs import base_config, text_strat_configs, detection_strat_configs
from dataset import COCODataset
from object_detector import CLIPObjectDetector
from util import get_cat_id_map, write_to_json, load_json

random.seed(3141592)
np.random.seed(3141592)
torch.manual_seed(3141592)

dataset_path = 'C:\\datasets\\COCO\\'
train_dir = 'train2017'
val_dir = 'val2017'

cmap = mpl.colormaps['plasma']

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-tsc', '--text_strat_config', required=True, type=str,
                        help="Key name of the config dict in detection_text_configs")
    parser.add_argument('-dsc', '--detection_strat_config', required=True, type=str,
                        help="Key name of the config dict in detection_strat_configs")
    parser.add_argument('--draw', action='store_true',
                        help="If true draws the image and each detected bbox")
    args = parser.parse_args()

    # Initialize the object detector
    config = base_config
    if args.text_strat_config not in text_strat_configs.keys():
        raise ValueError(
            f"Invalid config name: {args.text_strat_config}. Check in configs/detection_configs.py for valid config names.")
    elif args.detection_strat_config not in detection_strat_configs.keys():
        raise ValueError(
            f"Invalid config name: {args.detection_strat_config}. Check in configs/detection_configs.py for valid config names.")
    else:
        config.update(text_strat_configs[args.text_strat_config])
        config.update(detection_strat_configs[args.detection_strat_config])
    detector = CLIPObjectDetector("openai/clip-vit-large-patch14-336", config=config)

    # Load the annotations and COCO API object
    ann_path = f'{dataset_path}\\annotations\\instances_{val_dir}.json'
    val_img_dir = f'{dataset_path}\\{val_dir}\\'
    #results_json_path = f'results\\detection\\{args.text_strat_config}-{args.detection_strat_config}.json'
    results_json_path = f'results\\detection\\test.json'
    val_dataset = COCODataset(ann_path, val_img_dir)

    # Get all category names
    cats = val_dataset.getCats()
    cat_names = [cat['name'] for cat in cats]
    cat_id_map = get_cat_id_map(cats)

    results = {
        'image_ids' : [],
        'detections' : [],
        'config' : config,
    }
    if os.path.isfile(results_json_path):
        results = load_json(results_json_path)

    # === Main Loop ===
    for img_idx, elem in tqdm(enumerate(val_dataset[4:5])):
        image, labels = elem["image"], elem["labels"]
        if len(labels) == 0: continue
        if labels[0]['image_id'] in results['image_ids']: continue

        detections = detector.detect(image, cat_names)

        for det in detections:
            det['category_id'] = cat_id_map[det['cat_name']]
            det['image_id'] = labels[0]['image_id']
        
        # Postprocess detections
        filtered_detections = []
        threshold = 0.8341
        for det in detections:
            if det['score'] >= threshold:
                det['score'] = 0.5 + 0.5*(det['score'] - threshold)/(1 - threshold)
                filtered_detections.append(det)
        detections = filtered_detections


        results['detections'].extend(detections)
        results['image_ids'].append(labels[0]['image_id'])

        if args.draw:
            print(f"Detected {len(detections)} objects on image.")
            fig, ax = plt.subplots()
            ax.imshow(image)
            
            for det in detections:
                left, top, right, bottom = det['bbox']
                width = right - left
                height = bottom - top

                cmap_index = int(det['category_id'] / cats[-1]['id'] * (len(cmap.colors)-1))
                rect = patches.Rectangle((left, top), width, height, linewidth=2,
                                         edgecolor=cmap.colors[cmap_index], facecolor='none')
                ax.add_patch(rect)
                ax.text(left, top, f"{det['cat_name']} {det['score']:.3f}",
                        fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
            
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                        hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.show()
            fig.savefig(f'results/images/l14-336', bbox_inches='tight', pad_inches=0)

        # write_to_json(results, results_json_path)