from nuscenes import NuScenes
import pickle
import os
import numpy as np
import json
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud

CUSTOM_SPLIT = [
    "scene-0008", "scene-0009", "scene-0019", "scene-0029", "scene-0032", "scene-0042",
    "scene-0045", "scene-0049", "scene-0052", "scene-0054", "scene-0056", "scene-0066",
    "scene-0067", "scene-0073", "scene-0131", "scene-0152", "scene-0166", "scene-0168",
    "scene-0183", "scene-0190", "scene-0194", "scene-0208", "scene-0210", "scene-0211",
    "scene-0241", "scene-0243", "scene-0248", "scene-0259", "scene-0260", "scene-0261",
    "scene-0287", "scene-0292", "scene-0297", "scene-0305", "scene-0306", "scene-0350",
    "scene-0352", "scene-0358", "scene-0361", "scene-0365", "scene-0368", "scene-0377",
    "scene-0388", "scene-0391", "scene-0395", "scene-0413", "scene-0427", "scene-0428",
    "scene-0438", "scene-0444", "scene-0452", "scene-0453", "scene-0459", "scene-0463",
    "scene-0464", "scene-0475", "scene-0513", "scene-0533", "scene-0544", "scene-0575",
    "scene-0587", "scene-0589", "scene-0642", "scene-0652", "scene-0658", "scene-0669",
    "scene-0678", "scene-0687", "scene-0701", "scene-0703", "scene-0706", "scene-0710",
    "scene-0715", "scene-0726", "scene-0735", "scene-0740", "scene-0758", "scene-0786",
    "scene-0790", "scene-0804", "scene-0806", "scene-0847", "scene-0856", "scene-0868",
    "scene-0882", "scene-0897", "scene-0899", "scene-0976", "scene-0996", "scene-1012",
    "scene-1015", "scene-1016", "scene-1018", "scene-1020", "scene-1024", "scene-1044",
    "scene-1058", "scene-1094", "scene-1098", "scene-1107",
]


data_path = "/home/PJLAB/liuyouquan/nuScenes/"
nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)
# imageset = "/home/PJLAB/liuyouquan/nuScenes/nuscenes_infos_val.pkl"
#############train
# phase_scenes = create_splits_scenes()['val']
phase_scenes = CUSTOM_SPLIT
# phase_scenes = list(set(create_splits_scenes()["train"]) - set(CUSTOM_SPLIT))
skip_counter = 0
list_keyframes = []
for scene_idx in range(len(nusc.scene)):
    scene = nusc.scene[scene_idx]
    if scene["name"] in phase_scenes:
        skip_counter += 1
        if skip_counter % 1 == 0:
            current_sample_token = scene["first_sample_token"]
            # Loop to get all successive keyframes
            list_data = []
            while current_sample_token != "":
                current_sample = nusc.get("sample", current_sample_token)
                list_data.append(current_sample["data"])
                current_sample_token = current_sample["next"]
            list_keyframes.extend(list_data)

b = json.dumps(list_keyframes)
f2 = open('./list_keyframes_verifying.json','w')
f2.write(b)
f2.close()

camera_list = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]

save_dict = {}
for idx in range(len(list_keyframes)):
    lk = list_keyframes[idx]
    pointsensor = nusc.get("sample_data", lk["LIDAR_TOP"])
    sub_pcl_path = pointsensor["filename"]
    labels_filename = nusc.get("lidarseg", lk["LIDAR_TOP"])["filename"].replace("lidarseg/", "lidarseg2/")

    print(sub_pcl_path)
    print(labels_filename)

    cam_dict = {}
    for i, camera_name in enumerate(camera_list):
        ap_list = {}
        cam = nusc.get("sample_data", lk[camera_name])
        cam_sub_path = cam["filename"]     # todo
        ap_list["camera_name"] = cam_sub_path

        ap_list['token'] = cam['token']

        cs_record = nusc.get(
            "calibrated_sensor", pointsensor["calibrated_sensor_token"])
        ap_list["cs_record"] = cs_record

        poserecord = nusc.get("ego_pose", pointsensor["ego_pose_token"])
        ap_list["poserecord"] = poserecord

        poserecord_ = nusc.get("ego_pose", cam["ego_pose_token"])
        ap_list["poserecord_"] = poserecord_

        cs_record_ = nusc.get(
            "calibrated_sensor", cam["calibrated_sensor_token"]
        )
        ap_list["cs_record_"] = cs_record_

        cam_dict[camera_name] = ap_list

    # save_dict[lk["LIDAR_TOP"]] = [sub_pcl_path, cam_dict]
    save_dict[lk["LIDAR_TOP"]] = {"lidar_name": sub_pcl_path,
                                  "labels_name": labels_filename,
                                  "cam_info": cam_dict}


b1 = json.dumps(save_dict)
f = open('./save_dict_verifying.json','w')
f.write(b1)
f.close()

















# print(scene)

# '''
# with open(imageset, 'rb') as f:
#     data = pickle.load(f)
# nusc_infos = data['infos']
# nusc_train = {}
# for index in range(len(nusc_infos)):
#     info = nusc_infos[index]
#     lidar_path = info['lidar_path'][16:]
#     print(lidar_path)
#     print('='*80)
#     lidar_sd_token = nusc.get('sample', info['token'])['data']['LIDAR_TOP']
#     lidarseg_labels_filename = os.path.join("s3://liuyouquan/nuScenes",
#                                             nusc.get('lidarseg', lidar_sd_token)['filename'])
#     print(lidarseg_labels_filename)
#     nusc_train[lidar_path] = lidarseg_labels_filename
#     points = np.fromfile(os.path.join(data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
#
# # b = json.dumps(nusc_train)
# # f2 = open('./nusc_val.json','w')
# # f2.write(b)
# # f2.close()


# read
# f = open('./nusc_val.json','r')
# content = f.read()
# a = json.loads(content)
# print(a)
# print(len(a))
# f.close()
