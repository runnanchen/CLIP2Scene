import os
import copy
import torch
import numpy as np
from PIL import Image
import MinkowskiEngine as ME
from torch.utils.data import Dataset
# import pc_utils
from plyfile import PlyData, PlyElement
import math
# from pc_utils import write_ply_rgb
import sys
sys.path.append("..")
# from MinkowskiEngine.utils import sparse_quantize

import imageio
import cv2
import random

def write_ply_rgb(points, colors, filename, text=True):
    """ input: Nx3, Nx3 write points and colors to filename as PLY format. """
    num_points = len(points)
    assert len(colors) == num_points

    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    colors = [(colors[i, 0], colors[i, 1], colors[i, 2]) for i in range(colors.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    color = np.array(colors, dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(num_points, vertex.dtype.descr + color.dtype.descr)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    for prop in color.dtype.names:
        vertex_all[prop] = color[prop]

    el = PlyElement.describe(vertex_all, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)


def scannet_collate_pair_fn(batch):

    (
        coords,
        feats,
        labels,
        imgs,
        pairing_points,
        pairing_images,
        inverse_indexes,
        scan_names,
    ) = list(zip(*batch))

    offset_point = 0
    offset_image = 0

    for batch_id in range(len(coords)):
        pairing_points[batch_id][:] += offset_point
        offset_point += coords[batch_id].shape[0]

        pairing_images[batch_id][:, 0] += offset_image
        offset_image += imgs[batch_id].shape[0]

    coords = ME.utils.batched_coordinates(coords, dtype=torch.float32)
    feats = torch.cat(feats, dim=0)
    imgs = torch.cat(imgs, dim=0)

    pairing_points = torch.cat(pairing_points, dim=0)
    pairing_images = torch.cat(pairing_images, dim=0)

    return {
        "sinput_C": coords,
        "sinput_F": feats,
        "input_I": imgs,
        "pairing_points": pairing_points,
        "pairing_images": pairing_images,
        "inverse_indexes": inverse_indexes,
    }

class scannet_Dataset(Dataset):
    def __init__(self, phase, config, shuffle = True, cloud_transforms = None, mixed_transforms = None):

        self.scannet_root_dir = config['dataRoot_scannet']
        if phase == 'train':
            self.scannet_file_list = self.read_files(config['train_file'])
        else:
            self.scannet_file_list = self.read_files(config['val_file'])

        self.mixed_transforms = mixed_transforms

        self.voxel_size = config['voxel_size']
        self.phase = phase
        self.config = config
        self.imageDim = (640, 480)
        # self.imageDim = (224, 416)
        self.cloud_transforms = cloud_transforms
        self.maxImages = 8

    def read_files(self, file):
        f = open(file)
        lines = f.readlines()
        name_list = [line.split('.')[0] for line in lines]
        f.close()
        return name_list

    def __len__(self):
        return len(self.scannet_file_list)

    def read_pose_file(self, fname):
        posemat = np.asarray([[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in
                              (x.split(" ") for x in open(fname).read().splitlines())])
        return posemat

    def read_intrinsic_file(self, fname):
        intrinsic = np.asarray([[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in
                                (x.split(" ") for x in open(fname).read().splitlines())])
        return intrinsic

    def read_txt(self, path):
        # Read txt file into lines.
        with open(path) as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]
        return lines

    def computeLinking(self, camera_to_world, coords, depth, link_proj_threshold, intrinsic_color, intrinsic_depth, imageDim):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :intrinsic_depth: 4 x 4
        :intrinsic_color: 4 x 4, not used currently
        :return: linking, N x 3 format, (H,W,mask)
        """

        # print("imageDim ", imageDim)

        intrinsic = intrinsic_depth
        link = np.zeros((3, coords.shape[0]), dtype=float)
        coordsNew = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T #4 x N
        assert coordsNew.shape[0] == 4, "[!] Shape error"

        world_to_camera = np.linalg.inv(camera_to_world) # 4 x 4
        p = np.matmul(world_to_camera, coordsNew) # 4 x N
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]

        pi = p
        inside_mask = (pi[0] >= 0) * (pi[1] >= 0) * (pi[0] <= imageDim[1] - 1) * (pi[1] <= imageDim[0]-1)

        occlusion_mask = np.abs(depth[np.round(pi[1][inside_mask]).astype(np.int), np.round(pi[0][inside_mask]).astype(np.int)] - p[2][inside_mask]) <= link_proj_threshold

        inside_mask[inside_mask == True] = occlusion_mask
        link[0][inside_mask] = pi[1][inside_mask]
        link[1][inside_mask] = pi[0][inside_mask]
        link[2][inside_mask] = 1
        return link.T

    def __getitem__(self, idx):
        path = os.path.join(self.scannet_root_dir, self.scannet_file_list[idx], self.scannet_file_list[idx]+"_new_semantic.npy")
        data = torch.from_numpy(np.load(path))
        coords, feats, labels = data[:, :3], data[:, 3: 6], data[:, 9:]
        sceneName = self.scannet_file_list[idx]

        feats = feats / 127.5 - 1

        frame_names = []
        imgs = []
        links = []

        intrinsic_color = self.read_intrinsic_file(os.path.join(self.config['dataRoot_images'], sceneName, 'intrinsics_color.txt'))
        intrinsic_depth = self.read_intrinsic_file(os.path.join(self.config['dataRoot_images'], sceneName, 'intrinsics_depth.txt'))

        for framename in os.listdir(os.path.join(self.config['dataRoot_images'], sceneName, 'color')):
            frame_names.append(framename.split('.')[0])

        pairing_points = []
        pairing_images = []

        frame_names = random.sample(frame_names, min(self.maxImages, len(frame_names)))


        for i, frameid in enumerate(frame_names):
            f = os.path.join(self.config['dataRoot_images'], sceneName, 'color', frameid + '.jpg')
            img = imageio.imread(f) / 255
            img = cv2.resize(img, self.imageDim)
            depth = imageio.imread(f.replace('color', 'depth').replace('.jpg', '.png')) / 1000.0  # convert to meter
            posePath = f.replace('color', 'pose').replace('.jpg', '.txt')
            pose = self.read_pose_file(posePath)

            link = self.computeLinking(pose, coords, depth, 0.05, intrinsic_color, intrinsic_depth, depth.shape)

            pairing_point = torch.from_numpy(np.argwhere(link[:, 2] == 1)).squeeze()
            pairing_points.append(pairing_point)

            link = torch.from_numpy(link).int()
            imgs.append(torch.from_numpy(img.transpose((2, 0, 1))))

            pairing_image = link[pairing_point, :2]
            pairing_images.append(torch.cat((torch.ones(pairing_point.shape[0], 1) * i,
                                            pairing_image), dim=1))


        imgs = torch.stack(imgs)
        pairing_points = torch.cat(pairing_points, dim=0).numpy()
        pairing_images = torch.cat(pairing_images, dim=0).numpy()

        if self.cloud_transforms:
            coords = self.cloud_transforms(coords.float())

        if self.mixed_transforms:
            (
                coords_b,
                feats_b,
                imgs_b,
                pairing_points_b,
                pairing_images_b,
            ) = self.mixed_transforms(
                coords, feats, imgs, pairing_points, pairing_images
            )

        coords, feats, imgs, pairing_points, pairing_images = coords_b, feats_b, imgs_b, torch.from_numpy(pairing_points_b),\
            torch.from_numpy(pairing_images_b)

        coords = (coords - coords.mean(0)) / self.voxel_size

        discrete_coords, indexes, inverse_indexes = ME.utils.sparse_quantize(
            coords.contiguous(), return_index=True, return_inverse=True
        )
        # indexes here are the indexes of points kept after the voxelization
        pairing_points = inverse_indexes[pairing_points]

        feats = feats[indexes]
        assert pairing_points.shape[0] == pairing_images.shape[0]

        packages = (discrete_coords, feats, labels, imgs, pairing_points, pairing_images, inverse_indexes, self.scannet_file_list[idx])
        return packages
