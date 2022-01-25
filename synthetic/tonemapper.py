import os
import cv2
import numpy as np

def tonemap_images(image_dir, out_dir, exposure_levels, gamma=0.5):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    ev_ind = 0
    exposure_map = []
    for f in sorted(os.listdir(image_dir)):
        if f.endswith(".exr"):
            img_path = os.path.join(image_dir, f)
            out_path = os.path.join(out_dir, f)

            img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            img /= np.max(img)
            img_out = np.clip((img * (2 ** exposure_levels[ev_ind]) ) ** gamma, 0, 1)
            file = out_path.replace(".exr", "_exp{}.png".format(exposure_levels[ev_ind]))
            exposure_map.append(exposure_levels[ev_ind])
            cv2.imwrite(file, img_out * 255)
            ev_ind = (ev_ind + 1) % len(exposure_levels)

    np.save(os.path.join(out_dir, "exposure_levels.npy"), np.array(exposure_map))

if __name__ == '__main__':
    image_dir = r'D:\HDR_Surface_Reconstruction\my_data\view70_samples512_no_display_openexr_P'
    out_dir = r'D:\HDR_Surface_Reconstruction\my_data\hotdog_tonemapped_0'
    exposure_levels = [0]
    tonemap_images(image_dir, out_dir, exposure_levels)
            