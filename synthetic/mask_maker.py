import cv2
import numpy as np
import os

def make_masks(image_dir, mask_dir):
    # Go through each image in image_dir
    for image_name in os.listdir(image_dir):
        if image_name.endswith(".png"):
            # Read in the image
            image = cv2.imread(os.path.join(image_dir, image_name))
            # Create a mask, where any pixel in image that is not black is 255 in mask
            mask = np.where(image != [0, 0, 0], 255, 0)
            # Save the mask
            cv2.imwrite(os.path.join(mask_dir, image_name), mask)

if __name__ == '__main__':
    image_dir = r'/media/ryan/DATA/HDR_Surface_Reconstruction/my_data/hotdog_tonemapped_0/image'
    mask_dir = r'/media/ryan/DATA/HDR_Surface_Reconstruction/my_data/hotdog_tonemapped_0/mask'
    make_masks(image_dir, mask_dir)