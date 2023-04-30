import torch.nn as nn
import torch.nn.functional as F
import clip


class ClipFeatureExtractor(nn.Module):
    """
    DINO Vision Transformer Feature Extractor.
    """
    def __init__(self, config, preprocessing=None):
        super(ClipFeatureExtractor, self).__init__()

        self.encoder, preprocess = clip.load("ViT-B/32", device="cuda")

        for param in self.encoder.parameters():
            param.requires_grad = False

        # self.decoder = nn.Sequential(
        #     nn.Conv2d(embed_dim, config["model_n_out"], 1),
        #     nn.Upsample(scale_factor=patch_size, mode="bilinear", align_corners=True),
        # )
        self.preprocessing = preprocess
        self.normalize_feature = config["normalize_features"]

    def forward(self, x):
        if self.preprocessing:
            x = self.preprocessing(x)
        batch_size, _, height, width = x.size()

        print(x.size())

        x = self.encoder(x)
        # the output of x should be [batch_size x (1 + f_height * f_width) x self.embed_dim]

        x = self.decoder(x)
        if self.normalize_feature:
            x = F.normalize(x, p=2, dim=1)
        return x
