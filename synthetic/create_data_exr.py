from tonemapper import tonemap_images
from mask_maker import make_masks
from matrix_converter import convert_blender_json_npz
import os

def create_dirs(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'mask'), exist_ok=True)

    return os.path.join(out_dir, "image"), os.path.join(out_dir, "mask")

if __name__ =='__main__':
    exr_dir = r'/media/ryan/DATA/HDR_Surface_Reconstruction/my_data/v100_drums_val'
    exposure_levels = [3]
    exposure_string = '_'.join(str(x) for x in exposure_levels)
    out_dir = r'/media/ryan/DATA/HDR_Surface_Reconstruction/my_data/v100_drums_val_' + exposure_string
    image_dir, mask_dir = create_dirs(out_dir)
    tonemap_images(exr_dir, image_dir, exposure_levels)
    # make_masks(image_dir, mask_dir)
    convert_blender_json_npz(os.path.join(exr_dir, 'transforms.json'), os.path.join(out_dir, 'cameras_sphere.npz'))