import matplotlib.pyplot as plt
from pycocotools.coco import COCO

from util import get_cat_id_map

dataset_path = 'C:\\datasets\\COCO\\'
val_dir = 'val2017'

if __name__ == "__main__":
    # Load the annotations and COCO API object
    ann_path = '{}\\annotations\\instances_{}.json'.format(dataset_path, val_dir)
    val_img_dir = '{}\\{}\\'.format(dataset_path, val_dir)
    coco = COCO(ann_path)

    # Number of objects per image
    img_ids = coco.getImgIds()
    
    obj_counts = {}
    for img_id in img_ids:
        obj_counts.append(len(coco.getAnnIds(imgIds=img_id)))

    avg_objs = sum(obj_counts) / len(obj_counts)
    print("Total number of pictures in the dataset: " + str(len(img_ids)))
    print("Total number of objects in the dataset: " + str(sum(obj_counts)))
    print("Average number of objects per image in the dataset: " + str(avg_objs))
    
    plt.hist(obj_counts, bins=40)
    plt.show()