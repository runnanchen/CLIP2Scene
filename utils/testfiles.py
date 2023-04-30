import os
import copy
import torch
import numpy as np
from PIL import Image
# import MinkowskiEngine as ME
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
from torchsparse.utils.quantize import sparse_quantize
import json
from petrel_client.client import Client
import cv2
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


def minkunet_collate_pair_fn(list_data):
    """
    Collate function adapted for creating batches with MinkowskiEngine.
    """
    (
        coords,
        feats,
        images,
        pairing_points,
        pairing_images,
        inverse_indexes,
        superpixels,
    ) = list(zip(*list_data))
    batch_n_points, batch_n_pairings = [], []

    offset = 0
    for batch_id in range(len(coords)):

        # Move batchids to the beginning
        coords[batch_id][:, -1] = batch_id
        pairing_points[batch_id][:] += offset
        pairing_images[batch_id][:, 0] += batch_id * images[0].shape[0]

        batch_n_points.append(coords[batch_id].shape[0])
        batch_n_pairings.append(pairing_points[batch_id].shape[0])
        offset += coords[batch_id].shape[0]

    # Concatenate all lists
    coords_batch = torch.cat(coords, 0).int()
    print(coords_batch.size())
    pairing_points = torch.tensor(np.concatenate(pairing_points))
    pairing_images = torch.tensor(np.concatenate(pairing_images))
    feats_batch = torch.cat(feats, 0).float()
    images_batch = torch.cat(images, 0).float()
    superpixels_batch = torch.tensor(np.concatenate(superpixels))
    return {
        "sinput_C": coords_batch,
        "sinput_F": feats_batch,
        "input_I": images_batch,
        "pairing_points": pairing_points,
        "pairing_images": pairing_images,
        "batch_n_pairings": batch_n_pairings,
        "inverse_indexes": inverse_indexes,
        "superpixels": superpixels_batch,
    }


class NuScenesMatchDataset(Dataset):
    """
    Dataset matching a 3D points cloud and an image using projection.
    """

    def __init__(
        self,
        # phase,
        # config,
        shuffle=False,
        cloud_transforms=None,
        mixed_transforms=None,
        **kwargs,
    ):
        # self.phase = phase
        self.shuffle = shuffle
        self.cloud_transforms = cloud_transforms
        self.mixed_transforms = mixed_transforms
        self.cylinder = True
        self.voxel_size = 0.1
        # self.voxel_size = config["voxel_size"]
        # self.cylinder = config["cylindrical_coordinates"]
        # self.superpixels_type = config["superpixels_type"]
        # self.bilinear_decoder = config["decoder"] == "bilinear"

        if "cached_nuscenes" in kwargs:
            self.nusc = kwargs["cached_nuscenes"]
        else:
            self.nusc = NuScenes(
                version="v1.0-trainval", dataroot="s3://dataset/nuScenes/", verbose=False
            )



        # a skip ratio can be used to reduce the dataset size and accelerate experiments
        try:
            skip_ratio = 1
        except KeyError:
            skip_ratio = 1
        skip_counter = 0

        self.dataroot = "s3://liuyouquan/nuScenes"  #todo

        # self.dataroot = "s3://dataset/nuScenes"
        self.client = Client('~/.petreloss.conf')
        # print(phase)
        # if phase == "train":
        #     f = open('./list_keyframes_train.json', 'r')
        #     content = f.read()
        #     self.list_keyframes = json.loads(content)
        #
        #     f1 = open('./save_dict_train.json', 'r')
        #     content1 = f1.read()
        #     self.frames_corrs_info = json.loads(content1)
        #
        # elif phase == "val":
        #     f = open('./list_keyframes_val.json', 'r')
        #     content = f.read()
        #     self.list_keyframes = json.loads(content)
        #
        #     f1 = open('./save_dict_val.json', 'r')
        #     content1 = f1.read()
        #     self.frames_corrs_info = json.loads(content1)
        #
        # elif phase == "parametrizing":
        #     with open('./list_keyframes_parametrizing.json', 'r') as f:
        #         self.list_keyframes = json.load(f)
        #
        #     f1 = open('./save_dict_train.json', 'r')
        #     content = f1.read()
        #     self.frames_corrs_info = json.loads(content)
        #     f1.close()
        #     # phase_scenes = list(
        #     #     set(create_splits_scenes()["train"]) - set(CUSTOM_SPLIT)
        #     # )
        # elif phase == "verifying":
        #     phase_scenes = CUSTOM_SPLIT

        with open('./list_keyframes_parametrizing.json', 'r') as f:
            self.list_keyframes = json.load(f)

        f1 = open('./save_dict_train.json', 'r')
        content = f1.read()
        self.frames_corrs_info = json.loads(content)
        f1.close()
        # print(data1[key_["LIDAR_TOP"]])

        # pcl_path = os.path.join("s3://liuyouquan/nuScenes/", data1[key_["LIDAR_TOP"]][0].replace("samples", ""))
        # pcl_path = "s3://liuyouquan/nuScenes/" + data1[key_["LIDAR_TOP"]][0].replace("samples", "")


        # f = open('./list_keyframes_parametrizing.json', 'r')
        # content = f.read()
        # self.list_keyframes = json.loads(content)
        #
        # f1 = open('./save_dict_parametrizing.json', 'r')
        # content1 = f1.read()
        # self.frames_corrs_info = json.loads(content1)


        # phase_scenes = list(
        # print(self.list_keyframes)
        # print(type(self.list_keyframes))

        # create a list of camera & lidar scans
        # for scene_idx in range(len(self.nusc.scene)):
        #     scene = self.nusc.scene[scene_idx]
        #     if scene["name"] in phase_scenes:
        #         skip_counter += 1
        #         if skip_counter % skip_ratio == 0:
        #             self.create_list_of_scans(scene)

    # def create_list_of_scans(self, scene):
    #     # Get first and last keyframe in the scene
    #     current_sample_token = scene["first_sample_token"]
    #     # Loop to get all successive keyframes
    #     list_data = []
    #     while current_sample_token != "":
    #         current_sample = self.nusc.get("sample", current_sample_token) #TODO
    #         list_data.append(current_sample["data"])
    #         current_sample_token = current_sample["next"]
    #
    #     # Add new scans in the list
    #     self.list_keyframes.extend(list_data)

    def map_pointcloud_to_image(self, data, min_dist: float = 1.0):
        """
        Given a lidar token and camera sample_data token, load pointcloud and map it to
        the image plane. Code adapted from nuscenes-devkit
        https://github.com/nutonomy/nuscenes-devkit.
        :param min_dist: Distance from the camera below which points are discarded.
        """
        # pointsensor = self.nusc.get("sample_data", data["LIDAR_TOP"])
        key_ = data["LIDAR_TOP"]
        pcl_path = "s3://liuyouquan/nuScenes" + self.frames_corrs_info[key_][0].replace("samples", "")
        # print(pcl_path)
        # pcl_path = os.path.join("s3://liuyouquan/nuScenes/", self.frames_corrs_info[key_][0].replace("samples",""))
        # print(pcl_path)

        # try:
        #     pc_original = LidarPointCloud.from_file(pcl_path)
        #     # print("pcl_path: ", pcl_path)
        #     pc_ref = pc_original.points
        # except Exception as e:
        #     print("pcl_path: ", pcl_path)



        images = []
        superpixels = []
        pairing_points = np.empty(0, dtype=np.int64)
        pairing_images = np.empty((0, 3), dtype=np.int64)
        camera_list = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ]
        if self.shuffle:
            np.random.shuffle(camera_list)

        tot = 0

        camera_info = self.frames_corrs_info[key_][1]
        for i, camera_name in enumerate(camera_list):
            # pc = copy.deepcopy(pc_original)
            # cam = self.nusc.get("sample_data", data[camera_name])  #todo
            camera_path = camera_info[camera_name]["camera_name"]

            # print(pc_ref.shape)
            # import pdb
            # pdb.set_trace()

            # camera_path = "samples/CAM_FRONT/n008-2018-07-27-12-07-38-0400__CAM_FRONT__1532707811012460.jpg"

            try:
                img_bytes = self.client.get(self.dataroot + "/" + camera_path, update_cache=True)
                assert img_bytes is not None
                # print(camera_path)
            except Exception as e:
                tot += 1
                print(camera_path)
                continue

            return tot

            # img_bytes = self.client.get("s3://dataset/nuScenes/samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883530412470.jpg", update_cache=True)
            # assert  img_bytes is not  None
            img_mem_view = memoryview(img_bytes)
            buffer = np.frombuffer(img_mem_view, np.uint8)
            im = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            # cv2.imwrite("ttt.jpg", im)

            # im = im.reshape(im_shape)
            im = np.array(im)

            # import pdb
            # pdb.set_trace()

            # print(im.shape)
            # print(im.shape)

            # sp = Image.open(
            #     f"superpixels/nuscenes/"
            #     f"superpixels_{self.superpixels_type}/{camera_info[camera_name]['token']}.png"
            # )
            # superpixels.append(np.array(sp))

            # Points live in the point sensor frame. So they need to be transformed via
            # global to the image plane.
            # First step: transform the pointcloud to the ego vehicle frame for the
            # timestamp of the sweep.
            # cs_record = self.nusc.get(
            #     "calibrated_sensor", pointsensor["calibrated_sensor_token"]
            # )
            cs_record = camera_info[camera_name]["cs_record"]
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
            pc.translate(np.array(cs_record["translation"]))

            # Second step: transform from ego to the global frame.
            # poserecord = self.nusc.get("ego_pose", pointsensor["ego_pose_token"])
            poserecord = camera_info[camera_name]["poserecord"]
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
            pc.translate(np.array(poserecord["translation"]))

            # Third step: transform from global into the ego vehicle frame for the
            # timestamp of the image.
            # poserecord = self.nusc.get("ego_pose", cam["ego_pose_token"])
            poserecord = camera_info[camera_name]["poserecord_"]
            pc.translate(-np.array(poserecord["translation"]))
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

            # Fourth step: transform from ego into the camera.
            # cs_record = self.nusc.get(
            #     "calibrated_sensor", cam["calibrated_sensor_token"]
            # )
            cs_record = camera_info[camera_name]["cs_record_"]
            pc.translate(-np.array(cs_record["translation"]))
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

            # Fifth step: actually take a "picture" of the point cloud.
            # Grab the depths (camera frame z axis points away from the camera).
            depths = pc.points[2, :]

            # Take the actual picture
            # (matrix multiplication with camera-matrix + renormalization).
            points = view_points(
                pc.points[:3, :],
                np.array(cs_record["camera_intrinsic"]),
                normalize=True,
            )

            # Remove points that are either outside or behind the camera.
            # Also make sure points are at least 1m in front of the camera to avoid
            # seeing the lidar points on the camera
            # casing for non-keyframes which are slightly out of sync.


            points = points[:2].T
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > min_dist)
            mask = np.logical_and(mask, points[:, 0] > 0)
            mask = np.logical_and(mask, points[:, 0] < im.shape[1] - 1)
            mask = np.logical_and(mask, points[:, 1] > 0)
            mask = np.logical_and(mask, points[:, 1] < im.shape[0] - 1)
            matching_points = np.where(mask)[0]
            #
            matching_pixels = np.round(
                np.flip(points[matching_points], axis=1)
            ).astype(np.int64)
            images.append(im / 255)
            pairing_points = np.concatenate((pairing_points, matching_points))
            pairing_images = np.concatenate(
                (
                    pairing_images,
                    np.concatenate(
                        (
                            np.ones((matching_pixels.shape[0], 1), dtype=np.int64) * i,
                            matching_pixels,
                        ),
                        axis=1,
                    ),
                )
            )
        # return tot
        return pc_ref.T, images, pairing_points, pairing_images

    def __len__(self):
        return len(self.list_keyframes)

    def getitem(self, idx):
        # tot = self.map_pointcloud_to_image(self.list_keyframes[idx])
        # return tot


        (
            pc,
            images,
            pairing_points,
            pairing_images,
        ) = self.map_pointcloud_to_image(self.list_keyframes[idx])
        # superpixels = torch.tensor(superpixels)

        intensity = torch.tensor(pc[:, 3:])
        pc = torch.tensor(pc[:, :3])

        # print(images)
        # import pdb
        # pdb.set_trace()
        #
        images = torch.tensor(np.array(images, dtype=np.float32).transpose(0, 3, 1, 2))

        # if self.cloud_transforms:
        #     pc = self.cloud_transforms(pc)
        # if self.mixed_transforms:
        #     (
        #         pc,
        #         intensity,
        #         images,
        #         pairing_points,
        #         pairing_images,
        #         superpixels,
        #     ) = self.mixed_transforms(
        #         pc, intensity, images, pairing_points, pairing_images
        #     )

        if self.cylinder:
            # Transform to cylinder coordinate and scale for voxel size
            x, y, z = pc.T
            rho = torch.sqrt(x ** 2 + y ** 2) / self.voxel_size
            phi = torch.atan2(y, x) * 180 / np.pi  # corresponds to a split each 1°
            z = z / self.voxel_size
            coords_aug = torch.cat((rho[:, None], phi[:, None], z[:, None]), 1)
        else:
            coords_aug = pc / self.voxel_size
        #
        # # Voxelization with MinkowskiEngine
        discrete_coords, indexes, inverse_indexes = sparse_quantize(
            coords_aug.contiguous().numpy(), return_index=True, return_inverse=True
        )

        discrete_coords, indexes, inverse_indexes = torch.from_numpy(discrete_coords), torch.from_numpy(indexes), torch.from_numpy(inverse_indexes)

        # # indexes here are the indexes of points kept after the voxelization
        pairing_points = inverse_indexes[pairing_points]
        #
        unique_feats = intensity[indexes]
        #
        discrete_coords = torch.cat(
            (
                discrete_coords,
                torch.zeros(discrete_coords.shape[0], 1, dtype=torch.int32),
            ),
            1,
        )
        # return

        return (
            discrete_coords,
            unique_feats,
            images,
            pairing_points,
            pairing_images,
            inverse_indexes,
        )


Dataset = NuScenesMatchDataset()
print("len: ", len(Dataset))
sum_t = 0
for i in range(len(Dataset)):
# for i in range(100):
    print(i)
    tot = Dataset.getitem(i)
    # sum_t += tot
print("sum_t", sum_t)