import cv2
import os
import numpy as np

def make_masks(input_dir, output_dir):
    images = [name for name in os.listdir(input_dir) if name.endswith('.png')]
    for image_name in images:
        image = cv2.imread(os.path.join(input_dir, image_name), cv2.IMREAD_UNCHANGED)
        mask = np.zeros(image.shape)
        mask[image[:, :, 3] > 255 * 5/9] = 255
        for i in range(1):
            mask = cv2.GaussianBlur(mask, (9,9), 0)
            new_white = mask > (255 * 5/9)
            mask[new_white] = 255
            mask[~new_white] = 0
        cv2.imwrite(os.path.join(output_dir, image_name), mask[:, :, :3])


if __name__ == '__main__':
    image_dir = '/home/ryan/Repos/NeuS/public_data/v100_rubik_0/filtered_images'
    mask_dir = '/home/ryan/Repos/NeuS/public_data/v100_rubik_0/filtered_masks'
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    make_masks(image_dir, mask_dir)