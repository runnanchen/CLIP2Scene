import os
import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import model_urls
from model.modules.resnet_encoder import resnet_encoders
import model.modules.dino.vision_transformer as dino_vit
class fusionNet(nn.Module):
    """
    Dilated ResNet Feature Extractor
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_embeddings_path = self.config['text_embeddings_path']
        text_categories = self.config['text_categories']
        if self.text_embeddings_path is None:
            self.text_embeddings = nn.Parameter(torch.zeros(text_categories, 512))
            nn.init.normal_(self.text_embeddings, mean=0.0, std=0.01)
        else:
            self.register_buffer('text_embeddings', torch.randn(text_categories, 512))
            loaded = torch.load(self.text_embeddings_path, map_location='cuda')
            self.text_embeddings[:, :] = loaded[:, :]

        self.img_size = (224, 416)

        self.t = 1


    def forward(self, feature_packages):
        # feature_packages size: voxelSize * 8 * 1537
        # pixel_feature, point_feature, text_embedding, pred = feature_packages[:, :, :512], feature_packages[:, :, 512:1024], feature_packages[:, :, 1024:1536], feature_packages[:, :, -1]
        pixel_feature, point_feature, pred = feature_packages[:, :, :512], feature_packages[:, :, 512:1024], feature_packages[:, :, -1]

        pixel_pred = pred[:, 0].long()
        text_embedding = self.text_embeddings[pixel_pred].unsqueeze(1)

        pixel_point_feature = point_feature
        pixel_point_attention = torch.sum(pixel_point_feature * text_embedding, dim=2)

        index_point_sum = torch.sum(pixel_point_attention, dim=1) != 0
        pixel_point_attention = pixel_point_attention[index_point_sum] / self.t
        pixel_point_feature = pixel_point_feature[index_point_sum]
        pixel_pred = pixel_pred[index_point_sum]

        attention_union_sparse = pixel_point_attention.to_sparse()
        attention_union_dense = torch.sparse.softmax(attention_union_sparse, dim=1).to_dense()

        fusion_feature = torch.sum(attention_union_dense.unsqueeze(-1) * pixel_point_feature, dim=1)
        inner_products = torch.sigmoid(torch.sum(fusion_feature.unsqueeze(1) * pixel_point_feature, dim=2))

        return fusion_feature, inner_products, pixel_pred
