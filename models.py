"""Multi-modal (RGB+Flow+Depth) teacher and RGB-only KD student, both MobileNetV3-Large."""
import torch
import torch.nn as nn
import torchvision.models as tvm


def _adapt_first_conv(conv3: nn.Conv2d, new_in_ch: int) -> nn.Conv2d:
    """Rebuild a pretrained 3-channel first conv for a different input channel count,
    averaging the pretrained RGB weights across the input dimension and replicating them."""
    assert isinstance(conv3, nn.Conv2d) and conv3.in_channels == 3
    if new_in_ch not in (1, 2, 3):
        raise ValueError("new_in_ch must be 1, 2 or 3")

    new_conv = nn.Conv2d(
        in_channels=new_in_ch, out_channels=conv3.out_channels,
        kernel_size=conv3.kernel_size, stride=conv3.stride, padding=conv3.padding,
        dilation=conv3.dilation, groups=conv3.groups, bias=(conv3.bias is not None),
        padding_mode=conv3.padding_mode,
    )
    with torch.no_grad():
        if new_in_ch == 3:
            new_conv.weight.copy_(conv3.weight)
        else:
            mean_w = conv3.weight.mean(dim=1, keepdim=True)
            for c in range(new_in_ch):
                new_conv.weight[:, c:c + 1] = mean_w
        if conv3.bias is not None:
            new_conv.bias.copy_(conv3.bias)
    return new_conv


def _mobilenet_v3_large_backbone(pretrained: bool) -> nn.Module:
    weights = tvm.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    return tvm.mobilenet_v3_large(weights=weights)


class TripleBranchMobileNetV3LImgFlowDepth(nn.Module):
    """Multi-modal FacePAD teacher: independent RGB / Flow / Depth MobileNetV3-Large
    branches, pooled features concatenated, classified by a 2-layer FC head.

    Input: [B, 1, 3 + flow_channels + depth_channels, H, W] (RGB | Flow | Depth channels, in that order).
    """

    def __init__(self,
                 num_classes: int = 2,
                 flow_channels: int = 3,
                 depth_channels: int = 1,
                 pretrained: bool = True,
                 mlp_hidden: int = 768,
                 dropout: float = 0.2):
        super().__init__()
        assert flow_channels in (2, 3)
        assert depth_channels in (1, 3)

        self.rgb_branch = _mobilenet_v3_large_backbone(pretrained)
        rgb_feat_dim = self.rgb_branch.classifier[0].in_features
        self.rgb_branch.classifier = nn.Identity()

        self.flow_branch = _mobilenet_v3_large_backbone(pretrained)
        self.flow_branch.features[0][0] = _adapt_first_conv(self.flow_branch.features[0][0], flow_channels)
        flow_feat_dim = self.flow_branch.classifier[0].in_features
        self.flow_branch.classifier = nn.Identity()

        self.depth_branch = _mobilenet_v3_large_backbone(pretrained)
        self.depth_branch.features[0][0] = _adapt_first_conv(self.depth_branch.features[0][0], depth_channels)
        depth_feat_dim = self.depth_branch.classifier[0].in_features
        self.depth_branch.classifier = nn.Identity()

        concat_dim = rgb_feat_dim + flow_feat_dim + depth_feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes),
        )

        self.flow_channels = flow_channels
        self.depth_channels = depth_channels

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        B, T, C, H, W = x.shape
        assert T == 1, "expects single-frame input (num_frames=1)"
        expected_C = 3 + self.flow_channels + self.depth_channels
        assert C == expected_C, f"Expected C={expected_C}, got {C}"
        x = x.squeeze(1)

        c_rgb, c_flow = 3, self.flow_channels
        rgb = x[:, :c_rgb]
        flow = x[:, c_rgb:c_rgb + c_flow]
        depth = x[:, c_rgb + c_flow:c_rgb + c_flow + self.depth_channels]

        fused = torch.cat([self.rgb_branch(rgb), self.flow_branch(flow), self.depth_branch(depth)], dim=1)
        logits = self.classifier(fused)
        if return_feat:
            return logits, fused
        return logits


class StudentMobileNetV3(nn.Module):
    """RGB-only MobileNetV3-Large KD student. `projector_dim=0` (default) keeps the
    backbone's native 960-dim features; a positive value inserts a linear projector
    (used by feature-space KD variants)."""

    def __init__(self,
                 num_classes: int = 2,
                 projector_dim: int = 0,
                 classifier_hidden: int = 512,
                 dropout: float = 0.2,
                 pretrained: bool = True):
        super().__init__()
        self.backbone = _mobilenet_v3_large_backbone(pretrained)
        feature_dim = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()

        if projector_dim and projector_dim > 0 and projector_dim != feature_dim:
            self.projector = nn.Linear(feature_dim, projector_dim, bias=False)
            feat_dim = projector_dim
        else:
            self.projector = nn.Identity()
            feat_dim = feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        assert x.ndim == 5 and x.size(1) == 1 and x.size(2) == 3, \
            f"Expected [B,1,3,H,W] RGB input, got {tuple(x.shape)}"
        x = x.squeeze(1)
        feat = self.projector(self.backbone(x))
        logits = self.classifier(feat)
        if return_feat:
            return logits, feat
        return logits
