import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- Building blocks ---------
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, g=1, name=None):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False),
            nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.9),
            nn.ReLU(inplace=True),
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, g=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False),
            nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.9),
        )


class InvertedResidual(nn.Module):
    """MobileNetV2-style IR block with expansion, depthwise, and projection + residual."""
    def __init__(self, ch, expand_ratio=2):
        super().__init__()
        hid = ch * expand_ratio
        self.expand = ConvBNReLU(ch, hid, k=1, s=1, p=0, g=1)
        self.dw     = ConvBNReLU(hid, hid, k=3, s=1, p=1, g=hid)
        self.proj   = ConvBN(hid, ch, k=1, s=1, p=0, g=1)

    def forward(self, x):
        y = self.expand(x)
        y = self.dw(y)
        y = self.proj(y)
        y = x + y
        return F.relu(y, inplace=True)


class DeconvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=2, p=1, op=1):
        super().__init__(
            nn.ConvTranspose2d(in_ch, out_ch, k, stride=s, padding=p, output_padding=op, bias=False),
            nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.9),
            nn.ReLU(inplace=True),
        )


# --------- Model ---------
class MobileUNetTiny(nn.Module):
    """
    Input:  (1, 3, 112, 112)
    Output: (1, 3, 112, 112)

    Rough correspondences to your profile:
      - conv2d/0: 3x3 s2 -> (8, 56, 56)
      - conv2d/3: 3x3 s1 8->4 at 56x56
      - 28x28 & 14x14 stages use multiple 1x1 and depthwise 3x3 (IR blocks)
      - Decoder uses ConvTranspose(3x3, s2) with skip connections at 56/28/14
      - Final 56->112 upsample is a deconv 4->3 (mirrors Conv__283-like line)
    """
    def __init__(self):
        super().__init__()

        # Encoder
        self.stem = ConvBNReLU(3, 8, k=3, s=2, p=1)          # 112 -> 56
        self.reduce56 = ConvBNReLU(8, 4, k=3, s=1, p=1)      # keep 56, reduce ch: 8->4

        self.dw56_28 = ConvBNReLU(4, 4, k=3, s=2, p=1, g=4)  # depthwise 56 -> 28
        self.pw28    = ConvBNReLU(4, 16, k=1, s=1, p=0)      # pointwise 4 -> 16

        self.ir28_0 = InvertedResidual(16, expand_ratio=2)
        self.ir28_1 = InvertedResidual(16, expand_ratio=2)

        self.dw28_14 = ConvBNReLU(16, 16, k=3, s=2, p=1, g=16) # 28 -> 14
        self.pw14    = ConvBNReLU(16, 32, k=1, s=1, p=0)

        self.ir14_0 = InvertedResidual(32, expand_ratio=2)
        self.ir14_1 = InvertedResidual(32, expand_ratio=2)
        self.ir14_2 = InvertedResidual(32, expand_ratio=2)

        self.dw14_7 = ConvBNReLU(32, 32, k=3, s=2, p=1, g=32) # 14 -> 7
        self.pw7    = ConvBNReLU(32, 64, k=1, s=1, p=0)

        # Bottleneck 1x1s (heavy channel mixing near low-res)
        self.mix1 = ConvBNReLU(64, 96, k=1, s=1, p=0)
        self.mix2 = ConvBNReLU(96, 64, k=1, s=1, p=0)

        # Decoder
        self.up7_14 = DeconvBNReLU(64, 32, k=3, s=2, p=1, op=1) # 7 -> 14
        self.fuse14 = ConvBNReLU(64, 32, k=1, s=1, p=0)         # concat(32+32) -> 32

        self.up14_28 = DeconvBNReLU(32, 16, k=3, s=2, p=1, op=1) # 14 -> 28
        self.fuse28  = ConvBNReLU(32, 16, k=1, s=1, p=0)         # concat(16+16) -> 16

        self.up28_56 = DeconvBNReLU(16, 8, k=3, s=2, p=1, op=1)  # 28 -> 56
        self.fuse56_to4 = ConvBNReLU(16, 4, k=1, s=1, p=0)       # concat(8+8) -> 4

        # Final 56->112 upsample 4->3
        self.up56_112_rgb = nn.ConvTranspose2d(
            4, 3,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=True,
        )

    def forward(self, x):
        # Encoder
        x56 = self.stem(x)           # (8,56,56)
        x56_narrow = self.reduce56(x56)  # (4,56,56)

        x28 = self.dw56_28(x56_narrow)
        x28 = self.pw28(x28)         # (16,28,28)

        x28 = self.ir28_0(x28)
        x28 = self.ir28_1(x28)

        x14 = self.dw28_14(x28)
        x14 = self.pw14(x14)         # (32,14,14)

        x14 = self.ir14_0(x14)
        x14 = self.ir14_1(x14)
        x14 = self.ir14_2(x14)

        x7 = self.dw14_7(x14)
        x7 = self.pw7(x7)            # (64,7,7)

        x7 = self.mix1(x7)           # (96,7,7)
        x7 = self.mix2(x7)           # (64,7,7)

        # Decoder with UNet-style skip connections
        d14 = self.up7_14(x7)                     # (32,14,14)
        d14 = torch.cat([d14, x14], dim=1)        # (64,14,14)
        d14 = self.fuse14(d14)                    # (32,14,14)

        d28 = self.up14_28(d14)                   # (16,28,28)
        d28 = torch.cat([d28, x28], dim=1)        # (32,28,28)
        d28 = self.fuse28(d28)                    # (16,28,28)

        d56 = self.up28_56(d28)                   # (8,56,56)
        d56 = torch.cat([d56, x56], dim=1)        # (16,56,56)
        d56 = self.fuse56_to4(d56)                # (4,56,56)

        out = self.up56_112_rgb(d56)              # (3,112,112)
        return out


def export_onnx(save_path: str = "auto_model_close_to_profile.onnx") -> None:
    model = MobileUNetTiny().eval()
    dummy = torch.randn(1, 3, 112, 112, dtype=torch.float32)

    # Dry run to ensure it executes
    with torch.no_grad():
        _ = model(dummy)

    torch.onnx.export(
        model,
        dummy,
        save_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes=None,  # fixed batch = 1
    )
    print(f"Exported ONNX to: {save_path}")


if __name__ == "__main__":
    export_onnx()
