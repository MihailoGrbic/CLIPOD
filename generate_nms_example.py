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
    image = val_dataset[4]['image']

    detections = [
        {   
            'cat_name': 'bicycle',
            'category_id': 2,
            'bbox': [188, 10, 500, 385],
            'score': 0.861
        },
        {   
            'cat_name': 'bicycle',
            'category_id': 2,
            'bbox': [170, 23, 510, 360],
            'score': 0.968
        },
        {   
            'cat_name': 'bicycle',
            'category_id': 2,
            'bbox': [200, 35, 495, 378],
            'score': 0.718
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
    fig.savefig('examples/img/NMS_example.png', bbox_inches='tight', pad_inches=0)
