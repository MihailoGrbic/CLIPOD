import argparse

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

from util import load_json

IMAGE_NUM = 100

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-rp', '--report_path', required=True, type=str,
                        help="Path to the report json")
    parser.add_argument('-davg', '--desired_avg_det', default=10.5, type=float,
                        help="Desired average number of objects detected per image. \
                            Used to calculate postprocessing threshold.")
    args = parser.parse_args()

    # initialize COCO ground truth api
    ann_path = 'C:\\datasets\\COCO\\annotations\\instances_val2017.json'
    coco_gt = COCO(ann_path)

    # initialize COCO detections api
    report = load_json(args.report_path)
    
    # Select a subset of images and only take detections for those images
    image_ids = report['image_ids'][:IMAGE_NUM]
    all_detections = []
    for det in report['detections']:
        if det['image_id'] in image_ids:
            all_detections.append(det)

    # Calculate the postprocessing threshold based on the desired number of average detections per image
    all_detections = sorted(all_detections, key=lambda d: d['score'], reverse=True)
    desired_det_num = int(len(image_ids) * args.desired_avg_det)
    threshold = all_detections[desired_det_num]['score']

    print(f"Using threshold {threshold:.4} to make {args.desired_avg_det} detections per image on average.")

    # Postprocess detection
    final_detections = []
    for det in all_detections:
        assert det['image_id'] in image_ids
        if det['score'] >= threshold:
            det['score'] = 0.5 + 0.5*(det['score'] - threshold)/(1 - threshold)
            final_detections.append(det)
    avg_dets = len(final_detections) / len(image_ids)


    coco_dt = coco_gt.loadRes(final_detections)
    # running evaluation
    cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')
    cocoEval.params.imgIds = report['image_ids']
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
