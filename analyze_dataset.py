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
    
    obj_counts = []
    unique_obj_counts = []
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        

        anns = coco.loadAnns(ann_ids)
        cats_present = set()
        for ann in anns:
            cats_present.add(ann['category_id'])
        unique_obj_counts.append(len(cats_present))
        obj_counts.append(len(ann_ids))
        

    avg_objs = sum(obj_counts) / len(obj_counts)
    avg_uniq_objs = sum(unique_obj_counts) / len(unique_obj_counts)
    print("Total number of pictures in the dataset: " + str(len(img_ids)))
    print("Total number of objects in the dataset: " + str(sum(obj_counts)))
    print("Average number of total objects per image in the dataset: " + str(avg_objs))
    print("Average number of unique objects per image in the dataset: " + str(avg_uniq_objs))

    plt.hist(obj_counts, bins=40)
    plt.title("Total objects per image histogram")
    plt.show()

    plt.hist(unique_obj_counts, bins=15)
    plt.title("Unique objects per image histogram")
    plt.show()
