import os
import subprocess

def run_dense_colmap(basedir, image_dir, sparse_dir):

    undistorter_args = ['COLMAP.bat', 'image_undistorter',
        '--image_path', image_dir,
        '--input_path', sparse_dir,
        '--output_path', os.path.join(basedir, 'dense'),
        '--output_type', 'COLMAP',
        '--max_image_size', '2000'
    ]

    undistorter_output = ( subprocess.check_output(undistorter_args, universal_newlines=True) )
    # Write output to log file
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'a')
    logfile.write(undistorter_output)

    """
    colmap patch_match_stereo \
        --workspace_path $DATASET_PATH/dense \
        --workspace_format COLMAP \
        --PatchMatchStereo.geom_consistency true
    """
    patch_match_stereo_args = ['COLMAP.bat', 'patch_match_stereo',
        '--workspace_path', os.path.join(basedir, 'dense'),
        '--workspace_format', 'COLMAP',
        '--PatchMatchStereo.geom_consistency', 'true',
    ]

    patch_match_stereo_output = ( subprocess.check_output(patch_match_stereo_args, universal_newlines=True) )
    logfile.write(patch_match_stereo_output)

    """
    $ colmap stereo_fusion \
        --workspace_path $DATASET_PATH/dense \
        --workspace_format COLMAP \
        --input_type geometric \
        --output_path $DATASET_PATH/dense/fused.ply
    """
    stereo_fusion_args = ['COLMAP.bat', 'stereo_fusion', 
        '--workspace_path', os.path.join(basedir, 'dense'),
        '--workspace_format', 'COLMAP',
        '--input_type', 'geometric',
        '--output_path', os.path.join(basedir, 'dense/fused.ply'),
    ]

    stereo_fusion_output = ( subprocess.check_output(stereo_fusion_args, universal_newlines=True) )
    logfile.write(stereo_fusion_output)

    """
    $ colmap poisson_mesher \
        --input_path $DATASET_PATH/dense/fused.ply \
        --output_path $DATASET_PATH/dense/meshed-poisson.ply
    """
    poisson_mesher_args = ['COLMAP.bat', 'poisson_mesher',
        '--input_path', os.path.join(basedir, 'dense/fused.ply'),
        '--output_path', os.path.join(basedir, 'dense/meshed-poisson.ply'),
    ]

    poisson_mesher_output = ( subprocess.check_output(poisson_mesher_args, universal_newlines=True) )
    logfile.write(poisson_mesher_output)

    """
    $ colmap delaunay_mesher \
        --input_path $DATASET_PATH/dense \
        --output_path $DATASET_PATH/dense/meshed-delaunay.ply
    """
    delaunay_mesher_args = ['COLMAP.bat', 'delaunay_mesher',
        '--input_path', os.path.join(basedir, 'dense'),
        '--output_path', os.path.join(basedir, 'dense/meshed-delaunay.ply'),
    ]

    delaunay_mesher_output = ( subprocess.check_output(delaunay_mesher_args, universal_newlines=True) )
    logfile.write(delaunay_mesher_output)
    logfile.close()
    
    print('Dense construction completed')

if __name__ == '__main__':
    base = r'D:\HDR_Surface_Reconstruction\my_data\Bracketed_Rubik\loftr'
    image_dir = r'D:\HDR_Surface_Reconstruction\my_data\Bracketed_Rubik\images'
    sparse_dir = os.path.join(base, r'sparse\0')
    dense_dir = os.path.join(base, 'dense')
    if not os.path.exists(dense_dir):
        os.makedirs(dense_dir)
    run_dense_colmap(base, image_dir, sparse_dir)