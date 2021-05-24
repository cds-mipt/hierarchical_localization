import argparse
import json
import sqlite3
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import os
from pathlib import Path
import logging

from .utils.read_write_model import Camera, Image, Point3D, CAMERA_MODEL_NAMES
from .utils.read_write_model import write_model


def recover_database_images_and_ids(database_path):
    images = {}
    cameras = {}
    i = 0
    for filename in sorted(os.listdir(database_path)):
        if filename.find('.png') != -1:
            images[filename[:filename.rfind('.')]] = i
            cameras[filename[:filename.rfind('.')]] = i
            i += 1
    logging.info(
        f'Found {len(images)} images and {len(cameras)} cameras in database.')
    return images, cameras

def quaternion_to_rotation_matrix(qvec):
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    R = np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])
    return R


def camera_center_to_translation(c, qvec):
    R = quaternion_to_rotation_matrix(qvec)
    return (-1) * np.matmul(R, c) 


def read_data(path_to_db_images, data_dataset, image_ids, camera_ids, skip_points=False):

#     if points3D_file == '':
#         skip_points = True
#     else:
#         with open(points3D_file) as json_file:
#             points_3D_file = json.load(json_file)
    cameras = {}
    params = [320, 320, 128, 128]
    params = [float(p) for p in params]
    for name, camera_id in camera_ids.items():
        camera_model = CAMERA_MODEL_NAMES["PINHOLE"]
        camera = Camera(id=camera_id, model="PINHOLE",
                        width=256, height=256, params=params)
        cameras[camera_id] = camera

    data_file = open(data_dataset, 'r')
    
    list_of_images = list(Path(path_to_db_images).glob('**/*database*.png'))
    num_images = len(list_of_images)

#     assert num_images == len(cameras)

    logging.info(f'Reading {num_images} images...')
    image_data = []
    i = 0
    while i < num_images:
        line = data_file.readline()
        if line == '\n':
            continue
        data = line.strip('\n').split(' ')
        image_data.append(data)
        i += 1

    num_points = 0

    if skip_points:
        logging.info(f'Skipping points.')
        num_points = 0
    else:
        logging.info(f'Reading {num_points} points...')
    points3D = {}
    image_idx_to_keypoints = defaultdict(list)
    i = 0
    pbar = tqdm(total=num_points, unit='pts')
    while i < num_points:
        line = nvm_f.readline()
        if line == '\n':
            continue

        data = line.strip('\n').split(' ')
        x, y, z, r, g, b, num_observations = data[:7]
        obs_image_ids, point2D_idxs = [], []
        for j in range(int(num_observations)):
            s = 7 + 4*j
            img_index, kp_index, kx, ky = data[s:s+4]
            image_idx_to_keypoints[int(img_index)].append(
                (int(kp_index), float(kx), float(ky), i))
            db_image_id = image_idx_to_db_image_id[int(img_index)]
            obs_image_ids.append(db_image_id)
            point2D_idxs.append(kp_index)

        point = Point3D(
            id=i,
            xyz=np.array([x, y, z], float),
            rgb=np.array([r, g, b], int),
            error=1.,  # fake
            image_ids=np.array(obs_image_ids, int),
            point2D_idxs=np.array(point2D_idxs, int))
        points3D[i] = point

        i += 1
        pbar.update(1)
    pbar.close()

    logging.info('Parsing image data...')
    images = {}
    for i, data in enumerate(image_data):
        # Skip the focal length. Skip the distortion and terminal 0.
        name, qw, qx, qy, qz, cx, cy, cz = data
        qvec = np.array([qw, qx, qy, qz], float)
        c = np.array([cx, cy, cz], float)
        t = camera_center_to_translation(c, qvec)

        if i in image_idx_to_keypoints:
            # NVM only stores triangulated 2D keypoints: add dummy ones
            keypoints = image_idx_to_keypoints[i]
            point2D_idxs = np.array([d[0] for d in keypoints])
            tri_xys = np.array([[x, y] for _, x, y, _ in keypoints])
            tri_ids = np.array([i for _, _, _, i in keypoints])

            num_2Dpoints = max(point2D_idxs) + 1
            xys = np.zeros((num_2Dpoints, 2), float)
            point3D_ids = np.full(num_2Dpoints, -1, int)
            xys[point2D_idxs] = tri_xys
            point3D_ids[point2D_idxs] = tri_ids
        else:
            xys = np.zeros((0, 2), float)
            point3D_ids = np.full(0, -1, int)

        image_id = image_ids[name]
        image = Image(
            id=image_id,
            qvec=qvec,
            tvec=t,
            camera_id=camera_ids[name],
            name=name,
            xys=xys,
            point3D_ids=point3D_ids)
        images[image_id] = image

    return cameras, images, points3D


def main(path_to_db_images, data_dataset, output, skip_points=True):

    image_ids, camera_ids = recover_database_images_and_ids(path_to_db_images)

    model = read_data(
        path_to_db_images, data_dataset, image_ids, camera_ids, skip_points=skip_points)

    logging.info('Writing the COLMAP model...')
    output.mkdir(exist_ok=True, parents=True)
    write_model(*model, path=str(output), ext='.txt')
    logging.info('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_db_images', required=True, type=Path)
    parser.add_argument('--data_dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--skip_points', action='store_true')
    args = parser.parse_args()
    main(**args.__dict__)
