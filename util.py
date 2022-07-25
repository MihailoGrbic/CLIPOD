def get_cat_id_map(cats):
    result = {}
    for elem in cats:
        result[elem['name']] = elem['id']
    return result
