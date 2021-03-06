from pathlib import Path
import logging
import numpy as np
from collections import defaultdict


def parse_image_lists_with_intrinsics(paths):
    results = []
    files = list(Path(paths.parent).glob(paths.name))
    assert len(files) > 0

    for lfile in files:
        with open(lfile, 'r') as f:
            raw_data = f.readlines()

        logging.info(f'Importing {len(raw_data)} queries in {lfile.name}')
        for data in raw_data:
            data = data.strip('\n').split(' ')
            name, camera_model, width, height = data[:4]
            params = np.array(data[4:], float)
            info = (camera_model, int(width), int(height), params)
            results.append((name, info))

    assert len(results) > 0
    return results


def parse_retrieval(path):
    retrieval = defaultdict(list)
    with open(path, 'r') as f:
        for p in f.read().rstrip('\n').split('\n'):
            if p.find('#') != -1:
                continue
#             if len(p.split(',')) == 3:
#             q, r, score = p.split(',')
#             q = q.lstrip('testing/')
#             r = r.lstrip('training/')
#             retrieval[q.rstrip('.png')].append({
#                 r.rstrip('.png') : 1.0
#             })
            if len(p.split(' ')) == 2:
                q, r = p.split(' ')
                retrieval[q.rstrip('.png')].append({
                    r.rstrip('.png') : 1.0
                })
            elif len(p.split(' ')) == 3:
                q, r, score = p.split(' ')
                retrieval[q.rstrip('.png')].append({
                    r.rstrip('.png') : score
                })
            elif len(p.split(' ')) == 4:
                q, r, L2_metric, score = p.split(' ')
                retrieval[q.rstrip('.png')].append({
                    r.rstrip('.png') : score
                })
    return retrieval


def names_to_pair(name0, name1):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))
