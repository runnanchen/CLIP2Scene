import torch
import torch.nn as nn


def compute_chamfer_distance(p1, p2):
    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[bn, N, D]
    :param p2: size[bn, M, D]
    :param debug: whether need to output debug info
    :return: sum of Chamfer Distance of two point sets
    '''

    diff = p1[:, :, None, :] - p2[:, None, :, :]
    dist = torch.sum(diff*diff,  dim=3)
    dist1 = dist
    dist2 = torch.transpose(dist, 1, 2)

    dist_min1, _ = torch.min(dist1, dim=2)
    dist_min2, _ = torch.min(dist2, dim=2)

    return dist_min1, dist_min2


class ComputeCDLoss(nn.Module):
    def __init__(self):
        super(ComputeCDLoss, self).__init__()

    def forward(self, recon_points, gt_points):

        dist1, dist2 = compute_chamfer_distance(recon_points, gt_points)

        loss = (torch.sum(dist1) + torch.sum(dist2)) / (recon_points.shape[0] + 1E-6)
        # print(loss)
        return loss