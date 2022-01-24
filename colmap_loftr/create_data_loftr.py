import cv2
import kornia as K
import kornia.feature as KF
from kornia_moons.feature import *
import numpy as np
import torch
import os
import subprocess
from tqdm import tqdm
import matplotlib.pyplot as plt

from colmap_from_matches_loftr import run_colmap_matches_loftr
from colmap_database import COLMAPDatabase
from pose_utils import gen_poses

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running with device: {device}')

# $ DATASET_PATH=/path/to/dataset

# $ colmap feature_extractor \
#    --database_path $DATASET_PATH/database.db \
#    --image_path $DATASET_PATH/images
def run_colmap(out_dir, image_dir, database_path):
    """
    Runs the COLMAP feature extractor on the images in the given directory.
    """
    
    logfile_name = os.path.join(out_dir, 'colmap_output.txt')
    logfile = open(logfile_name, 'a')
    
    # Run the feature mapper just to get the camera parameters
    feature_extractor_args = [
        'COLMAP.bat', 'feature_extractor', 
            '--database_path', database_path, 
            '--image_path', image_dir,
            '--ImageReader.single_camera', '1',
            # '--SiftExtraction.use_gpu', '0',
    ]
    feat_output = ( subprocess.check_output(feature_extractor_args, universal_newlines=True) )
    logfile.write(feat_output)
    print('Features extracted')

def clean_database(database_path):
    """
    Get rid of all the tables in the database except the cameras table.
    """
    database = COLMAPDatabase.connect(database_path)
    database.clean_tables()
    database.commit()
    database.close()

def load_torch_image(fname):
    img = K.image_to_tensor(cv2.imread(fname), False).float() /255.
    img = K.color.bgr_to_rgb(img)
    img = img.to(device)
    return img

def get_keypoints_loftr(img1, img2):
    matcher = KF.LoFTR(pretrained='outdoor')
    matcher.to(device)

    input_dict = {"image0": K.color.rgb_to_grayscale(img1), # LofTR works on grayscale images only 
                "image1": K.color.rgb_to_grayscale(img2)}

    with torch.no_grad():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()

    if mkpts0.shape[0] >= 10 and mkpts1.shape[0] >= 10:   
        H, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
        inliers = inliers > 0
        inliers = inliers.squeeze()

        mkpts0 = mkpts0[inliers]
        mkpts1 = mkpts1[inliers]

    return mkpts0, mkpts1

def write_matches_file(fname, img_name_pairs):
    with open(fname, 'w') as f:
        for img1, img2 in img_name_pairs:
            f.write(f'{img1} {img2}\n')

def old_run_colmap_mapper(basedir):
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'a')

    sparse_dir = os.path.join(basedir, 'sparse')
    if not os.path.exists(sparse_dir):
        os.makedirs(sparse_dir)

    mapper_args = [
        'COLMAP.bat', 'mapper',
            '--database_path', os.path.join(basedir, 'database.db'),
            '--image_path', os.path.join(basedir, 'images'),
            '--output_path', sparse_dir, # --export_path changed to --output_path in colmap 3.6
            '--Mapper.num_threads', '16',
            '--Mapper.init_min_tri_angle', '4',
            '--Mapper.multiple_models', '0',
            '--Mapper.extract_colors', '0',
    ]

    map_output = ( subprocess.check_output(mapper_args, universal_newlines=True) )
    logfile.write(map_output)
    logfile.close()
    print('Sparse map created')

def create_pairwise_keypoints_loftr(image_dir):
    """
    Creates a pairwise keypoint file for each image pair in the given directory
    and adds them to a COLMAP database.
    """
    # Go through each image in the directory
    img_names = os.listdir(image_dir)
    pairs = []
    data_all = {}
    data_all['unary'] = {}
    data_all['pair'] = {}

    for i in tqdm(range(len(img_names))):
        img1 = load_torch_image(os.path.join(image_dir, img_names[i]))
        data_all['unary'][img_names[i]] = 0 # Need this field for an assertion later
        for j in range(i + 1, len(img_names)):
            img2 = load_torch_image(os.path.join(image_dir, img_names[j]))
            pairs.append((img_names[i], img_names[j]))
            key_points1, key_points2 = get_keypoints_loftr(img1, img2)
            combined = np.concatenate((key_points1, key_points2), axis=1)
            data_all['pair'][(img_names[i], img_names[j])] = {}
            data_all['pair'][(img_names[i], img_names[j])]['kpts_loftr'] = combined    

    return data_all, pairs

if __name__ == '__main__':
    basedir = 'D:\HDR_Surface_Reconstruction\my_data\Bracketed_Rubik'
    out_dir = os.path.join(basedir, 'loftr_-3_0')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    image_dir = os.path.join(basedir, r'images_640_480_-3_0')
    database_path = os.path.join(out_dir, 'database.db')
    data_all_path = os.path.join(out_dir, 'data_all.pkl')

    # If the database file doesn't exist
    if not os.path.exists(database_path):
        print('Running COLMAP')
        run_colmap(out_dir, image_dir, database_path)
        print('Cleaning database')
        clean_database(database_path)

    # If the data_all file doesn't exist
    if not os.path.exists(data_all_path):
        print('Creating pairwise keypoints')
        data_all, pairs = create_pairwise_keypoints_loftr(image_dir)
        # Write the data_all file
        with open(data_all_path, 'wb') as f:
            torch.save(data_all, f)
        # Write the matches file
        matches_file = os.path.join(out_dir, 'match_list.txt')
        write_matches_file(matches_file, pairs)
    else:
        with open(data_all_path, 'rb') as f:
            data_all = torch.load(f)
        matches_file = os.path.join(out_dir, 'match_list.txt')
    
    if not os.path.exists(os.path.join(out_dir, 'sparse')):
        run_colmap_matches_loftr(out_dir, image_dir, data_all)
        print('Created COLMAP binaries')

    print('Generating poses')
    gen_poses(out_dir)
    print('Done')