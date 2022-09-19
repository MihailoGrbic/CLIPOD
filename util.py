import json

def get_cat_id_map(cats):
    result = {}
    for elem in cats:
        result[elem['name']] = elem['id']
    return result

def get_id_cat_map(cats):
    result = {}
    for elem in cats:
        result[elem['id']] = elem['name']
    return result

def write_to_json(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def iou(bbox1, bbox2):
    left1, top1, right1, bottom1 = bbox1
    left2, top2, right2, bottom2 = bbox2

    # Find intersection edges
    int_left = max(left1, left2)
    int_top = max(top1, top2)
    int_right = min(right1, right2)
    int_bottom = min(bottom1, bottom2)
    int_w = max(0, int_right - int_left)
    int_h = max(0, int_bottom - int_top)

    bbox1_area = (right1 - left1) * (bottom1 - top1)
    bbox2_area = (right2 - left2) * (bottom2 - top2)
    intersection_area = int_w * int_h
    union_area = bbox1_area + bbox2_area - intersection_area

    return intersection_area / union_area

def fix_bbox(bbox, width, height, min_dim):
    bbox[0] = max(0,min(width, bbox[0]))
    bbox[1] = max(0,min(height, bbox[1]))
    bbox[2] = max(0,min(width, bbox[2]))
    bbox[3] = max(0,min(height, bbox[3]))
    bbox[2] += max(0, min_dim - (bbox[2] - bbox[0]))
    bbox[3] += max(0, min_dim - (bbox[3] - bbox[1]))
    return bbox

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__