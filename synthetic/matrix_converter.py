import json
import numpy as np
import cv2

def load_blender_json(json_path, out_path):
    with open(json_path, 'r') as f:
        camera_dict = json.load(f)
    camera_list = sorted(list(camera_dict.keys()))[2:]
    print(camera_list)
    fact = 0.5
    scale = np.array([[fact, 0, 0, 0], [0, fact, 0, 0], [0, 0, fact, 0], [0, 0, 0, 1]])
    idx = 0
    out_camera_dict = {}
    for i in camera_list:
        e = np.array(camera_dict[i])
        e = np.concatenate((e, np.array([[0, 0, 0, 1]])))
        out_camera_dict[f'world_mat_{idx}'] = e
        out_camera_dict[f'world_mat_inv_{idx}'] = np.linalg.inv(e)
        out_camera_dict[f'scale_mat_{idx}'] = scale
        out_camera_dict[f'scale_mat_inv_{idx}'] = np.linalg.inv(scale)
        idx += 1
        
    np.savez(out_path, **out_camera_dict)

if __name__ == '__main__':
    json_path = r'C:\Users\ryant\Repos\hdr-reconstruction\synthetic\transforms.json'
    out_path = r'C:\Users\ryant\Repos\hdr-reconstruction\synthetic\cameras_sphere_hotdog_0.npz'
    load_blender_json(json_path, out_path)
