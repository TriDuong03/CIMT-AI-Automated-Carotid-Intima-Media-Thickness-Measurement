import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.basic_unetplusplus import BasicUNetPlusPlus


class ASPP(nn.Module):
    def __init__(self, channels: int, rates: tuple = (1, 2, 4, 6)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            for r in rates
        ])
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(channels * (len(rates) + 1), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        feats = [b(x) for b in self.branches]
        g = F.interpolate(self.global_branch(x), size=(h, w),
                          mode="bilinear", align_corners=False)
        feats.append(g)
        return self.project(torch.cat(feats, dim=1))


class DilatedUNetPlusPlus(nn.Module):
    """
    Wrapper BasicUNetPlusPlus + ASPP tại bottleneck qua forward hook.
    Hook chặn output của conv_4_0 → thay bằng ASPP(output).
    Toàn bộ forward/dense connections của MONAI chạy bình thường.
    """
    def __init__(self, aspp_rates: tuple = (1, 2, 4, 6), **kwargs):
        super().__init__()
        self.backbone = BasicUNetPlusPlus(**kwargs)

        bottleneck_ch = kwargs.get("features", (32, 32, 64, 128, 256, 32))[4]
        self.aspp = ASPP(channels=bottleneck_ch, rates=aspp_rates)

        # Hook trả về giá trị → PyTorch tự động thay output của conv_4_0
        self._hook = self.backbone.conv_4_0.register_forward_hook(
            self._bottleneck_hook
        )

    def _bottleneck_hook(self, module, input, output):
        return self.aspp(output)   # output cũ bị thay hoàn toàn

    def forward(self, x: torch.Tensor):
        return self.backbone(x)   # MONAI xử lý toàn bộ, hook tự kích hoạt

    def remove_hook(self):
        """Gọi khi cần debug / export ONNX không dùng ASPP."""
        self._hook.remove()

model = DilatedUNetPlusPlus(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    features=(32, 32, 64, 128, 256, 32),
    aspp_rates=(1, 2, 4, 8),
)
