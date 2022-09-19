import matplotlib.pyplot as plt
from pycocotools.coco import COCO

dataset_path = 'C:\\datasets\\COCO\\'
val_dir = 'val2017'

if __name__ == "__main__":
    # Load the annotations and COCO API object
    ann_path = f'{dataset_path}\\annotations\\instances_{val_dir}.json'
    val_img_dir = f'{dataset_path}\\{val_dir}\\'
    coco = COCO(ann_path)

    # Number of objects per image
    img_ids = coco.getImgIds()
    
    obj_counts = []
    unique_obj_counts = []
    for img_id in img_ids:
        label_ids = coco.getAnnIds(imgIds=img_id)
        labels = coco.loadAnns(label_ids)

        cats_present = set()
        for label in labels:
            cats_present.add(label['category_id'])
        unique_obj_counts.append(len(cats_present))
        obj_counts.append(len(label_ids))
        

    avg_objs = sum(obj_counts) / len(obj_counts)
    avg_uniq_objs = sum(unique_obj_counts) / len(unique_obj_counts)
    print("Total number of pictures in the dataset: " + str(len(img_ids)))
    print("Total number of objects in the dataset: " + str(sum(obj_counts)))
    print("Average number of objects per image in the dataset: " + str(avg_objs))
    print("Average number of unique objects per image in the dataset: " + str(avg_uniq_objs))

    plt.hist(obj_counts, bins=40)
    plt.title("Total objects per image histogram")
    plt.show()

    plt.hist(unique_obj_counts, bins=15)
    plt.title("Unique objects per image histogram")
    plt.show()
