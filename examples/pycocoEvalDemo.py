import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

print('Running demo for bbox results.')

#initialize COCO ground truth api
dataType='val2017'
annFile=f'C:\\datasets\\COCO\\annotations\\instances_{dataType}.json'
cocoGt=COCO(annFile)

#initialize COCO detections api
resFile=f'results\\instances_{dataType}_perfect_bbox100_results.json'
cocoDt=cocoGt.loadRes(resFile)

imgIds=sorted(cocoGt.getImgIds())
imgIds=imgIds[0:1000]

# running evaluation
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()