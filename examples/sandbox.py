from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

#initialize COCO ground truth api
dataDir='C:\\datasets\\COCO'
dataType='val2017'
annFile='{}\\annotations\\instances_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)

with open(annFile, 'r') as f:
    data = json.load(f)

print()
print()
# cats = coco.loadCats(coco.getCatIds())
# print(cats)
# names=[cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(names)))
