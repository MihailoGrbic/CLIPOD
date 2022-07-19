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
annFile='C:\\datasets\\COCO\\annotations\\instances_{}.json'.format(dataType)
cocoGt=COCO(annFile)

#initialize COCO detections api
resFile='results\\instances_%s_perfect_bbox100_results.json'
resFile = resFile%(dataType)
cocoDt=cocoGt.loadRes(resFile)

imgIds=sorted(cocoGt.getImgIds())
imgIds=imgIds[0:100]
imgId = imgIds[np.random.randint(100)]

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()