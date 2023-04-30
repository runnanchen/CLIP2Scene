
import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset
# from MinkowskiEngine.utils import sparse_quantize
from utils.transforms import make_transforms_clouds
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
import cv2
import copy

TRAIN_SET = {0, 1, 2, 3, 4, 5, 6, 7, 9, 10}
VALIDATION_SET = {8}
TEST_SET = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20}



def kitti_collate_pair_fn(list_data):
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
    # print(coords_batch.size())
    pairing_points = torch.tensor(np.concatenate(pairing_points))
    pairing_images = torch.tensor(np.concatenate(pairing_images))
    feats_batch = torch.cat(feats, 0).float()
    images_batch = torch.cat(images, 0).float()
    return {
        "sinput_C": coords_batch,
        "sinput_F": feats_batch,
        "input_I": images_batch,
        "pairing_points": pairing_points,
        "pairing_images": pairing_images,
        "batch_n_pairings": batch_n_pairings,
        "inverse_indexes": inverse_indexes,
    }


class KittiMatchDataset(Dataset):
    """
    Dataset returning a lidar scene and associated labels.
    Note that superpixels fonctionality have been removed.
    """

    def __init__(
        self,
        phase,
        config,
        shuffle=False,
        cloud_transforms=None,
        mixed_transforms=None,
        **kwargs,
    ):
        self.phase = phase
        self.shuffle = shuffle
        self.cloud_transforms = cloud_transforms
        self.mixed_transforms = mixed_transforms
        self.voxel_size = config["voxel_size"]
        self.cylinder = config["cylindrical_coordinates"]
        self.superpixels_type = config["superpixels_type"]
        self.bilinear_decoder = config["decoder"] == "bilinear"

        # a skip ratio can be used to reduce the dataset size
        # and accelerate experiments
        skip_ratio = config["dataset_skip_step"]

        if phase in ("train", "parametrizing"):
            phase_set = TRAIN_SET
        elif phase in ("val", "verifying"):
            phase_set = VALIDATION_SET
        elif phase == "test":
            phase_set = TEST_SET

        self.list_files = []
        for num in phase_set:
            directory = next(
                os.walk(
                    f"/mnt/lustre/share_data/liuyouquan/semantickitti/sequences/{num:0>2d}/velodyne"
                )
            )
            self.list_files.extend(
                map(
                    lambda x: f"/mnt/lustre/share_data/liuyouquan/semantickitti/sequences/"
                    f"{num:0>2d}/velodyne/" + x,
                    directory[2],
                )
            )
        self.list_files = sorted(self.list_files)[::skip_ratio]

        # labels' names lookup table
        self.eval_labels = {
            0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5, 30: 6, 31: 7,
            32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13, 51: 14, 52: 0, 60: 9, 70: 15,
            71: 16, 72: 17, 80: 18, 81: 19, 99: 0, 252: 1, 253: 7, 254: 6, 255: 8,
            256: 5, 257: 5, 258: 4, 259: 5,
        }

    def select_points_in_frustum(self, points_2d, x1, y1, x2, y2):
        """
        Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
        :param points_2d: point cloud projected into 2D
        :param points_3d: point cloud
        :param x1: left bound
        :param y1: upper bound
        :param x2: right bound
        :param y2: lower bound
        :return: points (2D and 3D) that are in the frustum
        """
        keep_ind = (points_2d[:, 0] > x1) * \
                   (points_2d[:, 1] > y1) * \
                   (points_2d[:, 0] < x2) * \
                   (points_2d[:, 1] < y2)

        return keep_ind

    def read_calib(self, calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
        calib_out['Tr'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)

        return calib_out

    def map_pointcloud_to_image(self, ann_info, min_dist: float = 1.0):
        """
        Given a lidar token and camera sample_data token, load pointcloud and map it to
        the image plane. Code adapted from nuscenes-devkit
        https://github.com/nutonomy/nuscenes-devkit.
        :param min_dist: Distance from the camera below which points are discarded.
        """
        # pointsensor = self.nusc.get("sample_data", data["LIDAR_TOP"])

        points = np.fromfile(ann_info, dtype=np.float32).reshape((-1, 4))
        pc_ref = copy.deepcopy(points)

        path_splits = ann_info.split('/')
        calib_path = os.path.join("/mnt/lustre/share_data/liuyouquan/semantickitti/sequences",path_splits[-3], "calib.txt")
        image_path = os.path.join("/mnt/lustre/share_data/chenrunnan/dataset/sequences/",path_splits[-3],"image_2", path_splits[-1].replace("bin", "png"))

        image = cv2.imread(image_path)
        image = cv2.resize(image, (1241, 376), interpolation=cv2.INTER_LINEAR)

        calib = self.read_calib(calib_path)
        proj_matrix = calib['P2'] @ calib['Tr']
        proj_matrix = proj_matrix.astype(np.float32)

        # project points into image
        keep_idx = points[:, 0] > 0  # only keep point in front of the vehicle
        points_hcoords = np.concatenate([points[:, :3], np.ones([len(points), 1], dtype=np.float32)], axis=1)
        img_points = (proj_matrix @ points_hcoords.T).T
        matching_pixel = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points

        # print(img_points)
        keep_idx_img_pts = self.select_points_in_frustum(matching_pixel, 0, 0, 1241, 376)
        # print(keep_idx)
        keep_idx = keep_idx_img_pts & keep_idx
        # print(sum(keep_idx))
        # print("+"*90)
        matching_pixel = matching_pixel[keep_idx]
        # cv2.namedWindow('win', cv2.WINDOW_NORMAL)
        # for i in range(len(matching_pixel)):
        #     cv2.circle(image, (int(matching_pixel[i][0]), int(matching_pixel[i][1])), 1, (255, 255, 0), -1)

        # cv2.imwrite('./vis.png',image)
        # points_h = points[keep_idx]

        pairing_points = np.where(keep_idx==True)[0]

        pairing_images = np.concatenate(
                            (
                                np.zeros((matching_pixel.shape[0], 1), dtype=np.int64),
                                matching_pixel,
                            ),
                            axis=1,
                        )

        assert pairing_images.shape[1] == 3

        images = [image / 255]

        return pc_ref, images, pairing_points, pairing_images


    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        lidar_file = self.list_files[idx]
        (
            pc,
            images,
            pairing_points,
            pairing_images,
        ) = self.map_pointcloud_to_image(lidar_file)

        # points = np.fromfile(lidar_file, dtype=np.float32).reshape((-1, 4))
        # get the points (4th coordinate is the point intensity)

        intensity = torch.tensor(pc[:, 3:] + 1.)
        pc = torch.tensor(pc[:, :3])

        # print("pairing_points size: ", pairing_points.shape)
        # print("pairing_images size: ", pairing_images.shape)
        # print("images size: ", images[0].shape)
        # print("pc size: ", pc.shape)

        # images size: (900, 1600, 3)
        # pc size: torch.Size([34688, 3])
        # pairing_points size: (22585,)
        # pairing_images size: (22585, 3)

        images = torch.tensor(np.array(images, dtype=np.float32).transpose(0, 3, 1, 2))

        # apply the transforms (augmentation)
        if self.cloud_transforms:
            pc = self.cloud_transforms(pc)

        if self.mixed_transforms:
            (
                pc,
                intensity,
                images,
                pairing_points,
                pairing_images,
            ) = self.mixed_transforms(
                pc, intensity, images, pairing_points, pairing_images
            )

        if self.cylinder:
            # Transform to cylinder coordinate and scale for voxel size
            x, y, z = pc.T
            rho = torch.sqrt(x ** 2 + y ** 2) / self.voxel_size
            # corresponds to a split each 1°
            phi = torch.atan2(y, x) * 180 / np.pi
            z = z / self.voxel_size
            coords_aug = torch.cat((rho[:, None], phi[:, None], z[:, None]), 1)
        else:
            coords_aug = pc / self.voxel_size

        # Voxelization
        # discrete_coords, indexes, inverse_indexes = sparse_quantize(
        #     coords_aug, return_index=True, return_inverse=True
        # )

        discrete_coords, indexes, inverse_indexes = sparse_quantize(coords_aug.numpy(),
                                                    return_index=True,
                                                    return_inverse=True)

        discrete_coords, indexes, inverse_indexes = torch.from_numpy(discrete_coords), torch.from_numpy(indexes), torch.from_numpy(inverse_indexes)

        # indexes here are the indexes of points kept after the voxelization
        pairing_points = inverse_indexes[pairing_points]

        unique_feats = intensity[indexes]

        discrete_coords = torch.cat(
            (
                discrete_coords,
                torch.zeros(discrete_coords.shape[0], 1, dtype=torch.int32),
            ),
            1,
        )

        return (
            discrete_coords,
            unique_feats,
            images,
            pairing_points,
            pairing_images,
            inverse_indexes,
        )
