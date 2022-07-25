import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import torch

from dataset import Dataset
from object_detector import CLIPObjectDetector
from util import dotdict

device = torch.device("cuda:0" if torch.cuda.is_available() else "+")
dataset_path = 'C:\\datasets\\COCO\\'
val_dir = 'val2017'

if __name__ == "__main__":
    config_single = dotdict({
        'init_strat' : 'single',
        'init_threshold' : 1e-2,
    })
    config_segments = dotdict({
        'init_strat' : 'segments',
        'num_segments' : 5,
        'init_threshold' : 1e-2,
    })
    config_single_repeat = dotdict({
        'init_strat' : 'single',
        'repeat_wo_best' : True,
        'init_threshold' : 1e-2,
    })
    config_segments_repeat = dotdict({
        'init_strat' : 'segments',
        'num_segments' : 5,
        'repeat_wo_best' : True,
        'init_threshold' : 1e-1,
    })

    # Initialize the object detector
    detector = CLIPObjectDetector("openai/clip-vit-base-patch32", config=config_single)

    # Load the annotations and COCO API object
    ann_path = '{}\\annotations\\instances_{}.json'.format(dataset_path, val_dir)
    val_img_dir = '{}\\{}\\'.format(dataset_path, val_dir)
    coco = COCO(ann_path)

    # Get all category names
    cats = coco.loadCats(coco.getCatIds())
    cat_names = [cat['name'] for cat in cats]

    val_dataset = Dataset(ann_path, val_img_dir)
    
    # === Main Loop ===
    with torch.no_grad():
        #for elem in tqdm(val_dataset[:10]):
        for elem in val_dataset[:200]:
            image, image_id = elem["image"], elem["id"]

            detections = detector.init_detection(image, cat_names)
            print(detections[0])

            plt.imshow(image)
            plt.show()
            