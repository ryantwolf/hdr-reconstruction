import numpy as np
import subprocess
import open3d as o3d
import os
import torch
import cv2
from PIL import Image
from PIL.ExifTags import TAGS

def load_from_match_list_file(fpath):
    with open(fpath, "r") as f:
        raw_matches = [line.strip().split(" ") for line in f.readlines()]
    img_names_all = sorted(list(set([img_name for pairs in raw_matches for img_name in pairs])))
    return img_names_all, raw_matches

def get_img_meta(img_path):
    # read the image data using PIL
    image = Image.open(img_path)
    # extract EXIF data
    exifdata = image.getexif()
    # iterating over all EXIF data fields
    for tag_id in exifdata:
        # get the tag name, instead of human unreadable tag id
        tag = TAGS.get(tag_id, tag_id)
        data = exifdata.get(tag_id)
        # decode bytes 
        if isinstance(data, bytes):
            data = data.decode()
        print(f"{tag:25}: {data}")

def estimate_relative_pose(kpts0, kpts1, K0, K1, thresh=1, conf=0.99999):
    if len(kpts0) < 5:
        return None, None
    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)
    assert E is not None
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret, E

def run_geometric_verification(db_fpath, match_list_fpath, colmap_path="", hide_output=False, type="pairs"):
    if hide_output:
        pipe = subprocess.DEVNULL
    else:
        pipe = None
    logfile_name = os.path.join('D:\HDR_Surface_Reconstruction\my_data\Bracketed_Rubik', 'colmap_output.txt')
    logfile = open(logfile_name, 'w')
    
    feat_output = subprocess.check_output(['COLMAP.bat', "matches_importer",
                     "--database_path", db_fpath,
                     "--match_list_path", match_list_fpath,
                     "--match_type", type],
                     universal_newlines=True)

    logfile.write(feat_output)


def run_mapper(db_fpath, img_path, out_path, colmap_path="", hide_output=False):
    if hide_output:
        pipe = subprocess.DEVNULL
    else:
        pipe = None
    logfile_name = os.path.join('D:\HDR_Surface_Reconstruction\my_data\Bracketed_Rubik', 'colmap_output.txt')
    logfile = open(logfile_name, 'w')
    feat_output = subprocess.check_output(['COLMAP.bat', "mapper",
                     "--database_path", db_fpath,
                     "--image_path", img_path,
                     "--output_path", out_path,
                    #  "--Mapper.tri_merge_max_reproj_error", "3",
                    #  "--Mapper.filter_max_reproj_error", "2",
                     "--Mapper.init_min_num_inliers", "5",
                     "--Mapper.min_num_matches", "5",
                     "--Mapper.abs_pose_min_num_inliers", "5",
                     "--Mapper.abs_pose_min_inlier_ratio", "0.1",
                     "--Mapper.ba_refine_focal_length", "0",
                     "--Mapper.ba_refine_principal_point", "0",
                     "--Mapper.ba_refine_extra_params", "0"],
                     universal_newlines=True)

    logfile.write(feat_output)

if __name__ == '__main__':
    base_path = r'D:\HDR Surface Reconstruction\my_data\Bracketed Rubik'
    img_path = os.path.join(base_path, 'IMG_2014.jpg')
    get_img_meta(img_path)