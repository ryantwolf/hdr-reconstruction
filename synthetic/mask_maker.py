import cv2
import numpy as np
import os

def make_masks(image_dir, mask_dir):
    # Go through each image in image_dir
    for image_name in os.listdir(image_dir):
        if image_name.endswith(".png"):
            # Read in the image
            image = cv2.imread(os.path.join(image_dir, image_name))
            r = image[:,:,0]
            g = image[:,:,1]
            b = image[:,:,2]
            # Make a mask
            mask = np.zeros(r.shape)
            has_red = r > 5
            has_green = g > 5
            has_blue = b > 5
            has_color = np.logical_or(has_red, has_green, has_blue)
            mask[has_color] = 255

            # Blur the mask
            # for i in range(100):
            #     mask = cv2.GaussianBlur(mask, (9,9), 0)
            #     new_white = mask > (255 / 2)
            #     mask[new_white] = 255
            #     mask[~new_white] = 0

            # Save the mask
            cv2.imwrite(os.path.join(mask_dir, image_name), mask)

def make_masks_from_alpha(image_dir, mask_dir):
    # Go through each image in image_dir
    for image_name in os.listdir(image_dir):
        if image_name.endswith(".png"):
            # Read in the image
            image = cv2.imread(os.path.join(image_dir, image_name), cv2.IMREAD_UNCHANGED)
            # Make a mask
            mask = np.zeros(image.shape[:2])
            mask[image[:,:,3] > 5] = 255

            # Blur the mask
            # for i in range(500):
            #     mask = cv2.GaussianBlur(mask, (9,9), 0)
            #     new_white = mask > (255 / 3)
            #     mask[new_white] = 255
            #     mask[~new_white] = 0

            # Save the mask
            cv2.imwrite(os.path.join(mask_dir, image_name), mask)

if __name__ == '__main__':
    image_dir = r'/media/ryan/DATA/HDR_Surface_Reconstruction/my_data/v100_drums_val_mask/'
    mask_dir = r'/media/ryan/DATA/HDR_Surface_Reconstruction/my_data/v100_drums_val_mask/mask'
    os.makedirs(mask_dir, exist_ok=True)
    make_masks_from_alpha(image_dir, mask_dir)