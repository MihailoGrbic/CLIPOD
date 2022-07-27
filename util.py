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

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__