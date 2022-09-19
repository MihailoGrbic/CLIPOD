import sys

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

sys.path.append('.')
sys.path.append('..')
from util import get_cat_id_map

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
np.random.seed(3141592)

dataDir='C:\\datasets\\COCO'
dataType='train2017'
annFile=f'{dataDir}\\annotations\\instances_{dataType}.json'

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
names = [cat['name'] for cat in cats]
print(f'COCO categories: \n{" ".join(names)}\n')

names = set([cat['supercategory'] for cat in cats])
print(f'COCO supercategories: \n{" ".join(names)}')

cat_id_map = get_cat_id_map(cats)
# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['cat', 'dog'])
imgIds = coco.getImgIds(catIds=catIds)
# imgIds = coco.getImgIds(imgIds=[324158])

# imgIds = coco.getImgIds()
img_indexes = np.random.randint(0, len(imgIds), 100)

for img_index in img_indexes:
    img = coco.loadImgs(imgIds[img_index])[0]
    print(imgIds[img_index])
    print(img['file_name'])
    # load and display image
    I = io.imread('%s/%s/%s'%(dataDir,dataType,img['file_name']))
    # plt.axis('off')
    # plt.imshow(I)
    # plt.show()

    # load and display instance annotations
    plt.imshow(I); plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'])
    anns = coco.loadAnns(annIds)

    print("Categories present in this image: ")
    present_cats = coco.loadCats([ann['category_id'] for ann in anns])
    print([cat['name'] for cat in present_cats])
    print()

    #coco.showAnns(anns, draw_bbox=True)
    plt.show()
