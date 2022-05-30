import os
import cv2

def resize_dir(image_dir, output_dir, width, height):
    """
    Resizes all the images in the image_dir to the specified resolution
    """
    img_names = os.listdir(image_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for img_name in img_names:
        img = cv2.imread(os.path.join(image_dir, img_name))
        img = cv2.resize(img, (width, height))
        cv2.imwrite(os.path.join(output_dir, img_name), img)

if __name__ == '__main__':
    # Resize the bracketed rubik images
    # base_path = r'D:\HDR_Surface_Reconstruction\my_data\Bracketed_Rubik\images'
    # output_path = r'D:\HDR_Surface_Reconstruction\my_data\Bracketed_Rubik\images_640_480'
    base_path = '/media/ryan/DATA/HDR_Surface_Reconstruction/my_data/v100_rubik_-1/full_res'
    output_path = '/media/ryan/DATA/HDR_Surface_Reconstruction/my_data/v100_rubik_-1/images_640_480'
    resize_dir(base_path, output_path, 640, 480)