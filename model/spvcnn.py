import torchsparse
import torchsparse.nn as spnn
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from torch import nn
from torchsparse import PointTensor
from torchsparse import SparseTensor
from torchsparse.nn.utils import fapply
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .range_utils import resample_grid_stacked
import torch
from torch.nn import functional as F1
# import range_utils.nn.functional as rnf
import torch
import torchsparse.nn.functional as F
from torchsparse import PointTensor, SparseTensor
from torchsparse.nn.utils import get_kernel_offsets
import os


# z: PointTensor
# return: SparseTensor
def initial_voxelize(z, init_res, after_res):
    new_float_coord = torch.cat(
        [(z.C[:, :3] * init_res) / after_res, z.C[:, -1].view(-1, 1)], 1)

    pc_hash = F.sphash(torch.floor(new_float_coord).int())
    sparse_hash = torch.unique(pc_hash)
    idx_query = F.sphashquery(pc_hash, sparse_hash)
    counts = F.spcount(idx_query.int(), len(sparse_hash))

    inserted_coords = F.spvoxelize(torch.floor(new_float_coord), idx_query,
                                   counts)
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = F.spvoxelize(z.F, idx_query, counts)

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts
    z.C = new_float_coord

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: SparseTensor
def point_to_voxel(x, z):
    if z.additional_features is None or z.additional_features.get(
            'idx_query') is None or z.additional_features['idx_query'].get(
                x.s) is None:
        pc_hash = F.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1))
        sparse_hash = F.sphash(x.C)
        idx_query = F.sphashquery(pc_hash, sparse_hash)
        counts = F.spcount(idx_query.int(), x.C.shape[0])
        z.additional_features['idx_query'][x.s] = idx_query
        z.additional_features['counts'][x.s] = counts
    else:
        idx_query = z.additional_features['idx_query'][x.s]
        counts = z.additional_features['counts'][x.s]

    inserted_feat = F.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor.cmaps = x.cmaps
    new_tensor.kmaps = x.kmaps

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: PointTensor
def voxel_to_point(x, z, nearest=False):
    if z.idx_query is None or z.weights is None or z.idx_query.get(
            x.s) is None or z.weights.get(x.s) is None:
        off = get_kernel_offsets(2, x.s, 1, device=z.F.device)
        old_hash = F.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1), off)
        pc_hash = F.sphash(x.C.to(z.F.device))
        idx_query = F.sphashquery(old_hash, pc_hash)
        weights = F.calc_ti_weights(z.C, idx_query,
                                    scale=x.s[0]).transpose(0, 1).contiguous()
        idx_query = idx_query.transpose(0, 1).contiguous()
        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1
        new_feat = F.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights

    else:
        new_feat = F.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features

    return new_tensor

save_ceph = False
if save_ceph:
    from petrel_client.client import Client
    ceph_client = Client()

__all__ = ['SPVCNN']


class SyncBatchNorm(nn.SyncBatchNorm):

    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class BasicConvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            SyncBatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True),
            SyncBatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            SyncBatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation,
                        stride=1),
            SyncBatchNorm(outc),
        )

        if inc == outc * self.expansion and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(inc, outc * self.expansion, kernel_size=1, dilation=1,
                            stride=stride),
                SyncBatchNorm(outc * self.expansion),
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, 1, bias=False),
            SyncBatchNorm(outc),
            spnn.Conv3d(outc, outc, ks, stride, bias=False, dilation=dilation),
            SyncBatchNorm(outc),
            spnn.Conv3d(outc, outc * self.expansion, 1, bias=False),
            SyncBatchNorm(outc * self.expansion)
        )

        if inc == outc * self.expansion and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(inc, outc * self.expansion, kernel_size=1, dilation=1,
                            stride=stride),
                SyncBatchNorm(outc * self.expansion),
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out




class BaseSegmentor(nn.Module):
    def __init__(self, model_cfg, num_class):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_class = num_class
        # self.dataset = dataset
        # self.class_names = dataset.class_names

    def load_params(self, model_state_disk, strict=False):
        my_model_dict = self.state_dict()
        part_load = {}
        for k in model_state_disk.keys():
            value = model_state_disk[k]
            if k.startswith("module."):
                k = k[len("module."):]
            if k in my_model_dict and my_model_dict[k].shape == value.shape:
                part_load[k] = value

        return self.load_state_dict(part_load, strict=strict)

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        model_state_disk = torch.load(filename, map_location=loc_type)
        if 'model_state' in model_state_disk:
            model_state_disk = model_state_disk['model_state']
        msg = self.load_params(model_state_disk)
        logger.info(f"==> Done {msg}")

    def forward(self, batch_dict):
        raise NotImplementedError

class SPVCNN(nn.Module):

    def _make_layer(self, block, out_channels, num_block, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels))
        return layers

    # (self, in_channels, out_channels, config, D=3):
    # def __init__(self, model_cfg, num_class, dataset=None):
    def __init__(self, in_channels, num_class, config):
        super().__init__()
        self.name = "spvcnn"
        self.in_feature_dim = in_channels
        self.num_class = num_class
        self.config = config

        # Default is MinkUNet50
        # self.num_layer = model_cfg.get('NUM_LAYER', [2, 3, 4, 6, 2, 2, 2, 2])
        # [2, 3, 4, 6, 2, 2, 2, 2]
        self.num_layer = [2, 2, 2, 2, 2, 2, 2, 2]
        # self.num_layer = [2, 3, 4, 6, 2, 2, 2, 2]
        self.block = ResidualBlock
        # self.block = {
        #     'ResBlock': ResidualBlock,
        #     'Bottleneck': Bottleneck,
        # }[model_cfg.get('BLOCK', 'Bottleneck')]
        cr = 1
        # cs = model_cfg.get('PLANES', [32, 32, 64, 128, 256, 256, 128, 96, 96])
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]

        self.pres = 0.05
        self.vres = 0.05

        self.stem = nn.Sequential(
            spnn.Conv3d(self.in_feature_dim, cs[0], kernel_size=3, stride=1),
            SyncBatchNorm(cs[0]), spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            SyncBatchNorm(cs[0]), spnn.ReLU(True))

        self.in_channels = cs[0]
        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(self.in_channels, self.in_channels, ks=2, stride=2, dilation=1),
            *self._make_layer(self.block, cs[1], self.num_layer[0]),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(self.in_channels, self.in_channels, ks=2, stride=2, dilation=1),
            *self._make_layer(self.block, cs[2], self.num_layer[1]),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(self.in_channels, self.in_channels, ks=2, stride=2, dilation=1),
            *self._make_layer(self.block, cs[3], self.num_layer[2]),
        )
        
        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(self.in_channels, self.in_channels, ks=2, stride=2, dilation=1),
            *self._make_layer(self.block, cs[4], self.num_layer[3]),
        )

        self.up1 = [BasicDeconvolutionBlock(self.in_channels, cs[5], ks=2, stride=2)]
        self.in_channels = cs[5] + cs[3] * self.block.expansion
        self.up1.append(nn.Sequential(*self._make_layer(self.block, cs[5], self.num_layer[4])))
        self.up1 = nn.ModuleList(self.up1)

        self.up2 = [BasicDeconvolutionBlock(self.in_channels, cs[6], ks=2, stride=2)]
        self.in_channels = cs[6] + cs[2] * self.block.expansion
        self.up2.append(nn.Sequential(*self._make_layer(self.block, cs[6], self.num_layer[5])))
        self.up2 = nn.ModuleList(self.up2)

        self.up3 = [BasicDeconvolutionBlock(self.in_channels, cs[7], ks=2, stride=2)]
        self.in_channels = cs[7] + cs[1] * self.block.expansion
        self.up3.append(nn.Sequential(*self._make_layer(self.block, cs[7], self.num_layer[6])))
        self.up3 = nn.ModuleList(self.up3)

        self.up4 = [BasicDeconvolutionBlock(self.in_channels, cs[8], ks=2, stride=2)]
        self.in_channels = cs[8] + cs[0]
        self.up4.append(nn.Sequential(*self._make_layer(self.block, cs[8], self.num_layer[7])))
        self.up4 = nn.ModuleList(self.up4)

        # self.multi_scale = self.model_cfg.get('MULTI_SCALE', 'concat')
        self.multi_scale = 'concat'
        if self.multi_scale == 'concat':
            self.classifier = nn.Sequential(nn.Linear((cs[4] + cs[6] + cs[8]) * self.block.expansion, self.num_class))
        elif self.multi_scale == 'sum':
            raise Exception('obsolete')
            self.l1 = nn.Linear(cs[4] * self.block.expansion, cs[8] * self.block.expansion)
            self.l2 = nn.Linear(cs[6] * self.block.expansion, cs[8] * self.block.expansion)
            self.classifier = nn.Sequential(nn.Linear(cs[8] * self.block.expansion + (23 if self.concatattheend else 0), self.num_class))
        elif self.multi_scale == 'se':
            raise Exception('obsolete')
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.attn = nn.Sequential(
                nn.Linear((cs[4] + cs[6] + cs[8])  * self.block.expansion + (23 if self.concatattheend else 0), cs[8] * self.block.expansion, bias=False),
                nn.ReLU(True),
                nn.Linear(cs[8] * self.block.expansion, (cs[4] + cs[6] + cs[8])  * self.block.expansion + (23 if self.concatattheend else 0), bias=False),
                nn.Sigmoid(),
            )
            self.classifier = nn.Sequential(nn.Linear((cs[4] + cs[6] + cs[8]) * self.block.expansion + (23 if self.concatattheend else 0), self.num_class))
        else:
            self.classifier = nn.Sequential(nn.Linear(cs[8] * self.block.expansion + (23 if self.concatattheend else 0), self.num_class))

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[4] * self.block.expansion),
                nn.SyncBatchNorm(cs[4] * self.block.expansion),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[4] * self.block.expansion, cs[6] * self.block.expansion),
                nn.SyncBatchNorm(cs[6] * self.block.expansion),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[6] * self.block.expansion, cs[8] * self.block.expansion),
                nn.SyncBatchNorm(cs[8] * self.block.expansion),
                nn.ReLU(True),
            )
        ])

        self.weight_initialization()

        dropout_p = 0.0  #model_cfg.get('DROPOUT_P', 0.3)
        self.dropout = nn.Dropout(dropout_p, True)

        self.text_embeddings_path = self.config['text_embeddings_path']
        text_categories = self.config['text_categories']
        if self.text_embeddings_path is None:
            self.text_embeddings = nn.Parameter(torch.zeros(text_categories, 512))
            nn.init.normal_(self.text_embeddings, mean=0.0, std=0.01)
        else:
            self.register_buffer('text_embeddings', torch.randn(text_categories, 512))
            loaded = torch.load(self.text_embeddings_path, map_location='cuda')
            self.text_embeddings[:, :] = loaded[:, :]
        self.text_embeddings = torch.cat((self.text_embeddings[0, :].unsqueeze(0)*0, self.text_embeddings), dim=0)

        self.point_mapping_local = nn.Linear(480, 512)

        self.point_mapping_global = nn.Linear(480, 512)
        self.point_mapping_global_random = nn.Linear(480, 512)



    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # def forward(self, x):
    def forward(self, batch_dict, return_logit=False, return_tta=False): 
        """, previous_memory=[None, None, None, None], previous_offset=None, return_memory=False):"""

        x = batch_dict
        z = PointTensor(x.F, x.C.float())
        x0 = initial_voxelize(z, self.pres, self.vres)
        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        z1 = voxel_to_point(x4, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

        y1 = point_to_voxel(x4, z1)
        y1.F = self.dropout(y1.F)
        y1 = self.up1[0](y1)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)
        z2 = voxel_to_point(y2, z1)
        z2.F = z2.F + self.point_transforms[1](z1.F)

        y3 = point_to_voxel(y2, z2)
        y3.F = self.dropout(y3.F)
        y3 = self.up3[0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.point_transforms[2](z2.F)

        if self.multi_scale == 'concat':
            feat = torch.cat([z1.F, z2.F, z3.F], dim=1)
            if self.config['mode'] == 'pretrain':
                point_local = self.point_mapping_local(feat)
                point_global = self.point_mapping_global(feat)
                return point_local, point_global

            elif self.config['mode'] == 'finetune':
                out = self.classifier(feat)
                return out
            elif self.config['mode'] == 'source_free':

                feat = self.point_mapping_global(feat)
                out = F1.conv1d(feat.unsqueeze(-1), self.text_embeddings[:, :, None]).squeeze()

                return out
            elif self.config['mode'] == 'zero_shot':

                feat = self.point_mapping_global(feat)
                out = F1.conv1d(feat.unsqueeze(-1), self.text_embeddings[:, :, None]).squeeze()

                return out

        elif self.multi_scale == 'sum':
            out = self.classifier(self.l1(z1.F) + self.l2(z2.F) + z3.F)
        elif self.multi_scale == 'se':
            attn = torch.cat([z1.F, z2.F, z3.F], dim=1)
            attn = self.pool(attn.permute(1, 0)).permute(1, 0)
            attn = self.attn(attn)
            out = self.classifier(torch.cat([z1.F, z2.F, z3.F], dim=1) * attn)
        else:
            out = self.classifier(z3.F)

        return out

    def forward_ensemble(self, batch_dict):
        return self.forward(batch_dict, ensemble=True)
