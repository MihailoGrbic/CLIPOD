from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

#initialize COCO ground truth api
dataDir='C:\\datasets\\COCO'
dataType='val2017'
annFile=f'{dataDir}\\annotations\\instances_{dataType}.json'
coco=COCO(annFile)

annIds = coco.getAnnIds()
anns = coco.loadAnns(annIds)

result = []
for ann in anns:
    res_dict = {
        'image_id' : ann['image_id'],
        'category_id' : ann['category_id'],
        'bbox' : ann['bbox'],
        'score' : 1.0
    }
    result.append(res_dict)

with open('results\\instances_val2017_perfect_bbox100_results.json', 'w') as f:
    json.dump(result, f)

