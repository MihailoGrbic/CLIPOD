from collections.abc import Iterable

from util import load_json

report = load_json('./results/init_detection/config_4_segments_repeat_high.json')

cls_recall = []
for key in report.keys():
    if isinstance(report[key], Iterable) and 'recall' in report[key]:
        if key not in ['micro avg', 'macro avg', 'weighted avg', 'samples avg']:
            cls_recall.append((key, report[key]['recall'], report[key]['support']))

cls_recall = sorted(cls_recall, key=lambda tup: tup[1])

for elem in cls_recall[:10]:
    print(elem)