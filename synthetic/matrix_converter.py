import json
import numpy as np

def convert_blender_json_npz(json_path, out_path):
    with open(json_path, 'r') as f:
        camera_dict = json.load(f)
    camera_list = sorted(list(camera_dict.keys()))[2:]
    print(camera_list)
    fact = 4 / 2
    # fact = 0.6583 / 2 # hotdog
    # fact = 2
    scale = np.array([[fact, 0, 0, 0], [0, fact, 0, 0], [0, 0, fact, 0], [0, 0, 0, 1]])
    print(scale)
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
    json_path = r'/media/ryan/DATA/HDR_Surface_Reconstruction/my_data/new_cam/transforms.json'
    out_path = r'/media/ryan/DATA/HDR_Surface_Reconstruction/my_data/new_cam/cameras_sphere_hotdog.npz'
    convert_blender_json_npz(json_path, out_path)
