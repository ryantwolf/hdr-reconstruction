from tonemapper import tonemap_images
from mask_maker import make_masks
import os

def create_dirs(out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(os.path.join(out_dir, "image")):
        os.mkdir(os.path.join(out_dir, "image"))
    if not os.path.exists(os.path.join(out_dir, "mask")):
        os.mkdir(os.path.join(out_dir, "mask"))

    return os.path.join(out_dir, "image"), os.path.join(out_dir, "mask")

if __name__ =='__main__':
    exr_dir = r'/media/ryan/DATA/HDR_Surface_Reconstruction/my_data/view70_samples512_no_display_openexr_P'
    exposure_levels = [0, 3]
    exposure_string = '_'.join(str(x) for x in exposure_levels)
    out_dir = r'/media/ryan/DATA/HDR_Surface_Reconstruction/my_data/hotdog_tonemapped_' + exposure_string
    image_dir, mask_dir = create_dirs(out_dir)
    tonemap_images(exr_dir, image_dir, exposure_levels)
    make_masks(exr_dir, mask_dir)