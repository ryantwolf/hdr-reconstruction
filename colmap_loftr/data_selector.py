import os
import shutil
import numpy as np

def mass_rename(image_dir):
    """
    Renames all the n images in image_dir to numbers 0 to n-1
    """
    n = len(os.listdir(image_dir))
    for i, image_name in enumerate(sorted(os.listdir(image_dir))):
        # Get the file extension
        ext = os.path.splitext(image_name)[1]
        # Rename the file
        os.rename(os.path.join(image_dir, image_name), os.path.join(image_dir, str(i) + ext))

def copy_images_with_exposure(image_dir, output_dir, exposure_levels):
    """
    Copies images from image_dir to output_dir, where all the images in
    image_dir are numbered from 0 and there are 7 levels of exposure

    Args:
        image_dir: The directory containing the images
        output_dir: The directory to copy the images to
        exposure_levels: A list of desired exposure levels to copy where each level ranges from -3 to 3 (in stops)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ev_7 = np.array(exposure_levels, dtype=int) + 3
    files = os.listdir(image_dir)
    for ev in ev_7:
        for i in range(ev, len(files), 7):
            shutil.copyfile(os.path.join(image_dir, f'{i}.JPG'), os.path.join(output_dir, f'{i}.JPG'))

if __name__ == '__main__':
    base_dir = r'D:\HDR_Surface_Reconstruction\my_data\Bracketed_Rubik'
    # mass_rename(os.path.join(base_dir, 'images_640_480'))
    copy_images_with_exposure(os.path.join(base_dir, 'images_640_480'), os.path.join(base_dir, 'images_640_480_-3_0'), [-3, 0])