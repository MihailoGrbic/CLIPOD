import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataset import COCODataset

dataset_path = 'C:\\datasets\\COCO\\'
train_dir = 'train2017'
val_dir = 'val2017'

cmap = mpl.colormaps['viridis']

if __name__ == "__main__":
    # Load the annotations and COCO API object
    ann_path = f'{dataset_path}\\annotations\\instances_{val_dir}.json'
    val_img_dir = f'{dataset_path}\\{val_dir}\\'

    val_dataset = COCODataset(ann_path, val_img_dir)
    cats = val_dataset.getCats()
    image = val_dataset[1568]['image']

    detections = [
        {   
            'cat_name': 'horse',
            'category_id': 19,
            'bbox': [205, 190, 489, 352],
            'score': 0.884
        },
        {   
            'cat_name': 'person',
            'category_id': 0,
            'bbox': [327, 116, 394, 281],
            'score': 0.923
        },
        {   
            'cat_name': 'dog',
            'category_id': 8,
            'bbox': [103, 287, 201, 343],
            'score': 0.751
        }
    ]

    fig, ax = plt.subplots()
    ax.imshow(image)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    
    for i, det in enumerate(detections):
        left, top, right, bottom = det['bbox']
        width = right - left
        height = bottom - top

        cmap_index = int(det['category_id'] / cats[-1]['id'] * (len(cmap.colors)-1))
        rect = patches.Rectangle((left, top), width, height, linewidth=2,
                                    edgecolor=cmap.colors[cmap_index], facecolor='none')
        ax.add_patch(rect)
        ax.text(left, top, f"{det['cat_name']} {det['score']:.3f}",
                fontsize=11, bbox=dict(facecolor='white', alpha=0.5))

    plt.axis('off')
    plt.show()
    fig.savefig('examples/img/OD_example.png', bbox_inches='tight', pad_inches=0)
