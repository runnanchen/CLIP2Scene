import os
import re
import torch
import numpy as np
import torch.optim as optim
import MinkowskiEngine as ME
import pytorch_lightning as pl
from utils.chamfer_distance import ComputeCDLoss
from pretrain.criterion import NCELoss, DistillKL, semantic_NCELoss
from pytorch_lightning.utilities import rank_zero_only
from torchsparse import SparseTensor as spvcnn_SparseTensor
from torch import nn
import torch.nn.functional as F
import random
import numba as nb


@nb.jit()
def nb_pack(counts):
    return [np.array(list(range(i))) for i in counts]


class LightningPretrain(pl.LightningModule):
    def __init__(self, model_points, model_images, model_fusion, config):
        super().__init__()
        self.model_points = model_points
        self.model_images = model_images
        self.model_fusion = model_fusion
        self._config = config
        self.losses = config["losses"]
        self.train_losses = []
        self.val_losses = []
        self.num_matches = config["num_matches"]
        self.batch_size = config["batch_size"]
        self.num_epochs = config["num_epochs"]
        self.superpixel_size = config["superpixel_size"]
        self.epoch = 0
        self.cot = 0
        self.CE = nn.CrossEntropyLoss()
        self.CD_loss = ComputeCDLoss()
        self.KLloss = DistillKL(T=1)
        if config["resume_path"] is not None:
            self.epoch = int(
                re.search(r"(?<=epoch=)[0-9]+", config["resume_path"])[0]
            )
        self.criterion = NCELoss(temperature=config["NCE_temperature"])
        self.sem_NCE = semantic_NCELoss(temperature=config["NCE_temperature"])


        self.working_dir = os.path.join(config["working_dir"], config["datetime"])
        if os.environ.get("LOCAL_RANK", 0) == 0:
            os.makedirs(self.working_dir, exist_ok=True)

        self.text_embeddings_path = config['text_embeddings_path']
        text_categories = config['text_categories']
        if self.text_embeddings_path is None:
            self.text_embeddings = nn.Parameter(torch.zeros(text_categories, 512))
            nn.init.normal_(self.text_embeddings, mean=0.0, std=0.01)
        else:
            self.register_buffer('text_embeddings', torch.randn(text_categories, 512))
            loaded = torch.load(self.text_embeddings_path, map_location='cuda')
            self.text_embeddings[:, :] = loaded[:, :]

        self.saved = False
        self.max_size = 8
    def get_in_field(self, coords, feats):
        in_field = ME.TensorField(coordinates=coords.float(), features=feats.int(),
                                  # coordinate_map_key=A.coordiante_map_key, coordinate_manager=A.coordinate_manager,
                                  quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                  minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                                  # minkowski_algorithm=ME.MinkowskiAlgorithm.MEMORY_EFFICIENT,
                                  # device=self.config.device,
                                  ).float()
        return in_field


    def configure_optimizers(self):
        optimizer = optim.SGD(
            list(self.model_points.parameters()) + list(self.model_images.parameters()) + list(self.model_fusion.parameters()),
            lr=self._config["lr"],
            momentum=self._config["sgd_momentum"],
            dampening=self._config["sgd_dampening"],
            weight_decay=self._config["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)
        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def training_step(self, batch, batch_idx):
        self.model_points.train()

        sinput_C = batch["sinput_C"]
        sinput_F = batch["sinput_F"]

        if self._config['dataset'] == "nuscenes":
            sweepIds = batch["sweepIds"]

        if self._config['max_sweeps'] > 1:
            for sweepid in range(1, self._config['max_sweeps']):
                sweepInd = sweepIds == sweepid
                sinput_C[sweepInd, -1] = sinput_C[sweepInd, -1] + self._config['batch_size'] * sweepid

        if self._config['dataset'] == "scannet":
            sparse_input = ME.SparseTensor(sinput_F.float(), coordinates=sinput_C.int())
        else:
            sparse_input = spvcnn_SparseTensor(sinput_F, sinput_C)

        output_points = self.model_points(sparse_input)
        output_images = self.model_images(batch["input_I"].float())

        del batch["sinput_F"]
        del batch["sinput_C"]
        del batch["input_I"]
        del sparse_input
        # each loss is applied independtly on each GPU
        losses = [
            getattr(self, loss)(batch, output_points, output_images)
            for loss in self.losses
        ]
        loss = torch.mean(torch.stack(losses))
        torch.cuda.empty_cache()
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
        )

        if not self.saved:

            if self.epoch == 10:
                self.save()
                self.saved = True

        self.train_losses.append(loss.detach().cpu())
        return loss


    def scannet_loss(self, batch, output_points, output_images):
        # output_images.shape: torch.Size([96, 64, 224, 416])
        # output_points.shape: torch.Size([225648, 64])
        # pairing_points.shape: torch.Size([214155])
        # pairing_images.shape: torch.Size([214155, 3])
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]

        image_feats, image_pred = output_images
        point_feats_a, point_feats_b = output_points

        # global
        point_logists = F.conv1d(point_feats_a.unsqueeze(-1), self.text_embeddings[:, :, None]).squeeze()
        k_logists = point_logists[pairing_points]
        m_pred = tuple(pairing_images.T.long())
        q_pred = image_pred[m_pred]

        # switchable training strategy
        if self.epoch >= 10:
            rd = random.randint(1, 10)
            if rd > 5: q_pred = k_logists.argmax(dim=1)

        loss_semantic = self.CE(k_logists, q_pred)

        point_feats_b = point_feats_b[pairing_points]
        image_feats = image_feats.permute(0, 2, 3, 1)[m_pred]
        loss_spatial = torch.mean(1 - F.cosine_similarity(image_feats, point_feats_b, dim=1))

        return loss_semantic + loss_spatial


    def feature_packaging(self, image_global_allpoints, point_global_allpoints, inverse_indexes_merged, image_pred):
        uni_feature = torch.cat((image_global_allpoints, point_global_allpoints, image_pred.unsqueeze(-1)), dim=1)
        max_inverse_indexes = inverse_indexes_merged.max()
        feature_packages = torch.zeros((max_inverse_indexes + 1) * self.max_size, uni_feature.shape[1]).cuda()

        sorted_inverse_indexes, sorted_indices = torch.sort(inverse_indexes_merged)
        uni_feature = uni_feature[sorted_indices]
        _, counts = torch.unique(sorted_inverse_indexes, return_counts=True)

        offset = nb_pack(counts.detach().cpu().numpy())
        offset = torch.from_numpy(np.concatenate(offset, axis=0)).cuda()
        valid_index = offset < self.max_size

        offset = offset[valid_index]
        sorted_inverse_indexes = sorted_inverse_indexes[valid_index]
        uni_feature = uni_feature[valid_index]

        index = sorted_inverse_indexes * self.max_size + offset
        feature_packages[index] = uni_feature
        feature_packages = feature_packages.view((max_inverse_indexes + 1), self.max_size, uni_feature.shape[1])

        return feature_packages

    def loss_nuscenes(self, batch, output_points, output_images):
        # output_images.shape: torch.Size([96, 64, 224, 416])
        # output_points.shape: torch.Size([225648, 64])

        # pairing_points.shape: torch.Size([214155])
        # pairing_images.shape: torch.Size([214155, 3])
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        inverse_indexes_group = batch["inverse_indexes_group"]
        inverse_indexes_merged = batch['inverse_indexes_merged']

        image_global, image_pred = output_images
        point_local, point_global = output_points

        point_local = point_local[inverse_indexes_group]
        point_local_allpoints = point_local[pairing_points]

        point_global = point_global[inverse_indexes_group]
        point_global_allpoints = point_global[pairing_points]
        inverse_indexes_merged = inverse_indexes_merged[pairing_points]

        m_pred = tuple(pairing_images.T.long())
        image_global_allpoints = image_global.permute(0, 2, 3, 1)[m_pred]
        image_pred = image_pred[m_pred]

        feature_packages = self.feature_packaging(image_global_allpoints, point_local_allpoints, inverse_indexes_merged, image_pred)

        super_nodes_points, inner_products, pixel_pred = self.model_fusion(feature_packages)
        super_nodes_logit = F.conv1d(point_global_allpoints.unsqueeze(-1), self.text_embeddings[:, :, None]).squeeze()
        loss_semantic = 0

        # Switchable Self-training Strategy
        if self.epoch > 10:
            index_set = set(np.array(list(range(inverse_indexes_group.shape[0]))))
            pairing_set = set(pairing_points.detach().long().cpu().numpy())
            index_set_rest = list(index_set - pairing_set)
            point_global_rest = point_global[index_set_rest]
            point_global_logits = F.conv1d(point_global_rest.unsqueeze(-1), self.text_embeddings[:, :, None]).squeeze()
            point_global_pred = point_global_logits.argmax(dim=1)
            loss_semantic += self.CE(point_global_logits, point_global_pred)

            rd = random.randint(1, 10)
            if rd > 5: image_pred = super_nodes_logit.argmax(dim=1)

        loss_semantic = self.CE(super_nodes_logit, image_pred)
        loss_spatial_temporal = torch.mean(1 - inner_products)

        return loss_semantic + loss_spatial_temporal

    def loss(self, batch, output_points, output_images):
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        idx = np.random.choice(pairing_points.shape[0], self.num_matches, replace=False)
        k = output_points[pairing_points[idx]]
        m = tuple(pairing_images[idx].T.long())
        q = output_images.permute(0, 2, 3, 1)[m]

        return self.criterion(k, q)

    def loss_superpixels_average(self, batch, output_points, output_images):
        # compute a superpoints to superpixels loss using superpixels
        torch.cuda.empty_cache()  # This method is extremely memory intensive
        superpixels = batch["superpixels"]
        pairing_images = batch["pairing_images"]
        pairing_points = batch["pairing_points"]

        superpixels = (
            torch.arange(
                0,
                output_images.shape[0] * self.superpixel_size,
                self.superpixel_size,
                device=self.device,
            )[:, None, None] + superpixels
        )
        m = tuple(pairing_images.cpu().T.long())

        superpixels_I = superpixels.flatten()
        idx_P = torch.arange(pairing_points.shape[0], device=superpixels.device)
        total_pixels = superpixels_I.shape[0]
        idx_I = torch.arange(total_pixels, device=superpixels.device)

        with torch.no_grad():
            one_hot_P = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels[m], idx_P
                ), dim=0),
                torch.ones(pairing_points.shape[0], device=superpixels.device),
                (superpixels.shape[0] * self.superpixel_size, pairing_points.shape[0])
            )

            one_hot_I = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels_I, idx_I
                ), dim=0),
                torch.ones(total_pixels, device=superpixels.device),
                (superpixels.shape[0] * self.superpixel_size, total_pixels)
            )

        k = one_hot_P @ output_points[pairing_points]
        k = k / (torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6)
        q = one_hot_I @ output_images.permute(0, 2, 3, 1).flatten(0, 2)
        q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)

        mask = torch.where(k[:, 0] != 0)
        k = k[mask]
        q = q[mask]

        return self.criterion(k, q)

    def training_epoch_end(self, outputs):
        self.epoch += 1
        if self.epoch == self.num_epochs:
            self.save()
        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):

        sinput_C = batch["sinput_C"]
        sinput_F = batch["sinput_F"]

        if self._config['dataset'] == "scannet":
            sparse_input = ME.SparseTensor(sinput_F.float(), coordinates=sinput_C.int())
        else:
            sparse_input = spvcnn_SparseTensor(sinput_F, sinput_C)

        output_points = self.model_points(sparse_input)

        self.model_images.eval()
        output_images = self.model_images(batch["input_I"])

        losses = [
            getattr(self, loss)(batch, output_points, output_images)
            for loss in self.losses
        ]
        loss = torch.mean(torch.stack(losses))
        self.val_losses.append(loss.detach().cpu())

        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size
        )
        return loss

    @rank_zero_only
    def save(self):
        path = os.path.join(self.working_dir, "model.pt")
        torch.save(
            {
                "model_points": self.model_points.state_dict(),
                "model_images": self.model_images.state_dict(),
                "model_fusion": self.model_fusion.state_dict(),
                "epoch": self.epoch,
                "config": self._config,
            },
            path,
        )
