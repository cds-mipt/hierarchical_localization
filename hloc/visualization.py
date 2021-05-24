import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import random
import cv2
import numpy as np
import pickle

from .utils.read_write_model import read_images_binary, read_points3d_binary
from .utils.viz import plot_images, plot_keypoints, plot_matches, cm_RdGn
from .utils.parsers import names_to_pair
from scipy.spatial.transform import Rotation as R

import matplotlib
# from habitat.utils.visualizations import maps
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import os
import ast
from tqdm import tqdm


def read_image(path):
    appendix = str(path).split('_')
    directory = str(path)[:str(path).rfind('/')]
    subdir = appendix[-4][appendix[-4].rfind('/'):]+'_'+appendix[-3]
    subdir = subdir[1:]
    filename = str(path).split('/')[-1] + '.png'
#     path = directory + '/' + subdir + '/' +  filename
    path = directory + '/' +  filename
    path = Path(path)
    assert path.exists(), path
    image = cv2.imread(str(path))
    if len(image.shape) == 3:
        image = image[:, :, ::-1]
    return image


def visualize_sfm_2d(sfm_model, image_dir, color_by='visibility',
                     selected=[], n=1, seed=0, dpi=75):
    assert sfm_model.exists()
    assert image_dir.exists()

    images = read_images_binary(sfm_model / 'images.bin')
    if color_by in ['track_length', 'depth']:
        points3D = read_points3d_binary(sfm_model / 'points3D.bin')

    if not selected:
        image_ids = list(images.keys())
        selected = random.Random(seed).sample(image_ids, n)

    for i in selected:
        name = images[i].name
        image = read_image(image_dir / name)
        keypoints = images[i].xys
        visible = images[i].point3D_ids != -1

        if color_by == 'visibility':
            color = [(0, 0, 1) if v else (1, 0, 0) for v in visible]
            text = f'visible: {np.count_nonzero(visible)}/{len(visible)}'
        elif color_by == 'track_length':
            tl = np.array([len(points3D[j].image_ids) if j != -1 else 1
                           for j in images[i].point3D_ids])
            max_, med_ = np.max(tl), np.median(tl[tl > 1])
            tl = np.log(tl)
            color = cm.jet(tl / tl.max()).tolist()
            text = f'max/median track length: {max_}/{med_}'
        elif color_by == 'depth':
            p3ids = images[i].point3D_ids
            p3D = np.array([points3D[j].xyz for j in p3ids if j != -1])
            z = (images[i].qvec2rotmat() @ p3D.T)[-1] + images[i].tvec[-1]
            z -= z.min()
            color = cm.jet(z / np.percentile(z, 99.9))
            text = f'visible: {np.count_nonzero(visible)}/{len(visible)}'
            keypoints = keypoints[visible]
        else:
            raise NotImplementedError(f'Coloring not implemented: {color_by}.')

        plot_images([image], dpi=dpi)
        plot_keypoints([keypoints], colors=[color], ps=4)
        fig = plt.gcf()
        fig.text(
            0.01, 0.99, text, transform=fig.axes[0].transAxes,
            fontsize=10, va='top', ha='left', color='k',
            bbox=dict(fc=(1, 1, 1, 0.5), edgecolor=(0, 0, 0, 0)))
        fig.text(
            0.01, 0.01, name, transform=fig.axes[0].transAxes,
            fontsize=5, va='bottom', ha='left', color='w')


def visualize_loc(results, image_dir, sfm_model=None, top_k_db=2,
                  selected=[], n=1, seed=0, prefix=None, dpi=75):
    assert image_dir.exists()
    
    hdf5_dir = '/datasets/Habitat/1LXtFkjw3qL_point0/hdf5'
    output_dir_for_images = '/datasets/Habitat/Hierarchical_Localization_outputs/results'
    os.makedirs(output_dir_for_images, exist_ok=True)
    

    with open(str(results)+'_logs.pkl', 'rb') as f:
        logs = pickle.load(f)

    if not selected:
        queries = list(logs['loc'].keys())
        if prefix:
            queries = [q for q in queries if q.startswith(prefix)]
        selected = random.Random(seed).sample(queries, n)

    is_sfm = sfm_model is not None
    if is_sfm:
        assert sfm_model.exists()
        images = read_images_binary(sfm_model / 'images.bin')
        points3D = read_points3d_binary(sfm_model / 'points3D.bin')
    num = 1
    for q in selected:
        
        hdf5_filename = '_'.join(q.split('_')[:2]) + '.hdf5'
        print(hdf5_filename)
        hdf5_dataset_path = os.path.join(hdf5_dir, hdf5_filename)
        hdf5_file = h5py.File(hdf5_dataset_path, 'r')
        num_image_in_hdf5 = int(q.split('_')[-1])
        t_vec_gt = hdf5_file['gps'][num_image_in_hdf5]
        gt_quat_wxyz = hdf5_file['quat'][num_image_in_hdf5]
        gt_quat_xyzw = [gt_quat_wxyz[1], gt_quat_wxyz[2], gt_quat_wxyz[3], gt_quat_wxyz[0]]
        r = R.from_quat(gt_quat_xyzw)
        euler_x_gt, euler_y_gt, euler_z_gt = r.as_euler('xyz', degrees=True)
        
        q_image = read_image(image_dir / q)
        
        loc = logs['loc'][q]
        if not(loc['PnP_ret']['success']):
            continue
        inliers = np.array(loc['PnP_ret']['inliers'])
        mkp_q = loc['keypoints_query']
        t_vec_query = loc['PnP_ret']['tvec']
        quat_vec_query = loc['PnP_ret']['qvec']
        r = R.from_quat(list(quat_vec_query[1:])+[quat_vec_query[0]])
        euler_x_query, euler_y_query, euler_z_query = r.as_euler('xyz', degrees=True)

        n = len(loc['db'])
        if is_sfm:
            # for each pair of query keypoint and its matched 3D point,
            # we need to find its corresponding keypoint in each database image
            # that observes it. We also count the number of inliers in each.
            kp_idxs, kp_to_3D_to_db = loc['keypoint_index_to_db']
            counts = np.zeros(n)
            dbs_kp_q_db = [[] for _ in range(n)]
            inliers_dbs = [[] for _ in range(n)]
            for i, (inl, (p3D_id, db_idxs)) in enumerate(zip(inliers,
                                                             kp_to_3D_to_db)):
                p3D = points3D[p3D_id]
                for db_idx in db_idxs:
                    counts[db_idx] += inl
                    kp_db = p3D.point2D_idxs[
                        p3D.image_ids == loc['db'][db_idx]][0]
                    dbs_kp_q_db[db_idx].append((i, kp_db))
                    inliers_dbs[db_idx].append(inl)
        else:
            # for inloc the database keypoints are already in the logs
            assert 'keypoints_db' in loc
            assert 'indices_db' in loc
            counts = np.array([
                np.sum(loc['indices_db'][inliers] == i) for i in range(n)])

        # display the database images with the most inlier matches
        db_sort = np.argsort(-counts)
        for db_idx in db_sort[:top_k_db]:
            if is_sfm:
                db = images[loc['db'][db_idx]]
                db_name = db.name
                db_kp_q_db = np.array(dbs_kp_q_db[db_idx])
                kp_q = mkp_q[db_kp_q_db[:, 0]]
                kp_db = db.xys[db_kp_q_db[:, 1]]
                inliers_db = inliers_dbs[db_idx]
            else:
                db_name = loc['db'][db_idx]
                kp_q = mkp_q[loc['indices_db'] == db_idx]
                kp_db = loc['keypoints_db'][loc['indices_db'] == db_idx]
                inliers_db = inliers[loc['indices_db'] == db_idx]

            num_image_in_hdf5 = int(db_name.split('_')[-1])
            hdf5_filename = '_'.join(db_name.split('_')[:2]) + '.hdf5'
            hdf5_dataset_path = os.path.join(hdf5_dir, hdf5_filename)
            hdf5_file = h5py.File(hdf5_dataset_path, 'r')
            t_vec_db = hdf5_file['gps_base'][num_image_in_hdf5]
            db_quat_wxyz = hdf5_file['quat_base'][num_image_in_hdf5]
            db_quat_xyzw = [db_quat_wxyz[1], db_quat_wxyz[2], db_quat_wxyz[3], db_quat_wxyz[0]]
            r = R.from_quat(db_quat_xyzw)
            euler_x_db, euler_y_db, euler_z_db = r.as_euler('xyz', degrees=True)
            
            db_image = read_image(image_dir / db_name)
            color = cm_RdGn(inliers_db).tolist()
            text = f'inliers: {sum(inliers_db)}/{len(inliers_db)}'

            plot_images([q_image, db_image], dpi=dpi)
            plot_matches(kp_q, kp_db, color, a=0.1)
            fig = plt.gcf()
            fig.text(
                0.01, 0.99, text, transform=fig.axes[0].transAxes,
                fontsize=15, va='top', ha='left', color='k',
                bbox=dict(fc=(1, 1, 1, 0.5), edgecolor=(0, 0, 0, 0)))
            fig.text(
                0.01, 0.01, 'QUERY, {}\nQ: x={:.2f}, y={:.2f}, z={:.2f}, yaw={:.1f}, pitch={:.1f}, roll={:.1f}\nGT: x={:.2f}, y={:.2f}, z={:.2f}, yaw={:.1f}, pitch={:.1f}, roll={:.1f}'.format(q.split('/')[-1], t_vec_query[0], t_vec_query[1], t_vec_query[2], euler_x_query, euler_y_query, euler_z_query, t_vec_gt[0], t_vec_gt[1], t_vec_gt[2], euler_x_gt, euler_y_gt, euler_z_gt), transform=fig.axes[0].transAxes,
                fontsize=10, va='bottom', ha='left', color='royalblue')
            fig.text(
                0.01, 0.01, 'DATABASE, top-{}, {}\nx={:.2f}, y={:.2f}, z={:.2f}, yaw={:.1f}, pitch={:.1f}, roll={:.1f}'.format((((num+1) % 2)+1), db_name.split('/')[-1], t_vec_db[0], t_vec_db[1], t_vec_db[2], euler_x_db, euler_y_db, euler_z_db), transform=fig.axes[1].transAxes,
                fontsize=10, va='bottom', ha='left', color='green')
            plt.savefig("{}/{}.png".format(output_dir_for_images, num), dpi=dpi)
            num = num + 1
            break
                
def visualize_matches(feature_filename, match_filename, image_dir, image_filename_1, image_filename_2, dpi=300):
    
    image_1 = read_image(image_filename_1)
    image_2 = read_image(image_filename_2)
    plot_images([image_1, image_2], dpi=dpi)
    
    feature_file = h5py.File(str(feature_filename), 'r')
    match_file = h5py.File(str(match_filename), 'r')
#     print(len(match_file['1LXtFkjw3qL_point1-around_0000.png_1LXtFkjw3qL_point1-base_0000.png']['matching_scores0'].__array__()))
#     print(match_file['1LXtFkjw3qL_point1-around_0000.png_1LXtFkjw3qL_point1-base_0000.png']['matches0'].__array__())
    
    names_in_filename_rel_to_dir =str(Path(image_filename_1).relative_to(image_dir)).split('/')
#     keypoints_image_1 = feature_file[str(names_in_filename_rel_to_dir[0])][str(names_in_filename_rel_to_dir[1])]['keypoints'].__array__()
    keypoints_image_1 = feature_file[str(names_in_filename_rel_to_dir[0]).rstrip('.png')]['keypoints'].__array__()
    names_in_filename_rel_to_dir =str(Path(image_filename_2).relative_to(image_dir)).split('/')
#     keypoints_image_2 = feature_file[str(names_in_filename_rel_to_dir[0])][str(names_in_filename_rel_to_dir[1])]['keypoints'].__array__()
    keypoints_image_2 = feature_file[str(names_in_filename_rel_to_dir[0]).rstrip('.png')]['keypoints'].__array__()
    
    pair = names_to_pair(str(Path(image_filename_1).relative_to(image_dir)).rstrip('.png'), str(Path(image_filename_2).relative_to(image_dir)).rstrip('.png'))
    m = match_file[pair]['matches0'].__array__()
    v = (m > -1)
    matched_keypoints_image_1, matched_keypoints_image_2 = keypoints_image_1[v], keypoints_image_2[m[v]]
    
    plot_matches(matched_keypoints_image_1, matched_keypoints_image_2, color=None, a=0.1)
