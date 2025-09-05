import torch
import torch.nn as nn
import torchvision.models as models
import yaml
from torchvision.models import (
    ShuffleNet_V2_X0_5_Weights,
    ShuffleNet_V2_X1_0_Weights,
    ShuffleNet_V2_X1_5_Weights,
    ShuffleNet_V2_X2_0_Weights,
    MobileNet_V2_Weights,
    MobileNet_V3_Small_Weights,
    MobileNet_V3_Large_Weights,
)

class GAP_FC_Head(nn.Module):
    def __init__(self, in_features, num_classes):
        super(GAP_FC_Head, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.gap(x)             # [B, C, 1, 1]
        x = torch.flatten(x, 1)     # [B, C]
        x = self.fc(x)              # [B, num_classes]
        return x

def build_backbone(backbone_type: str, pretrained: bool, weights_name: str):
    if backbone_type.startswith("shufflenet_v2"):
        weights_dict = {
            "shufflenet_v2_x0_5": ShuffleNet_V2_X0_5_Weights,
            "shufflenet_v2_x1_0": ShuffleNet_V2_X1_0_Weights,
            "shufflenet_v2_x1_5": ShuffleNet_V2_X1_5_Weights,
            "shufflenet_v2_x2_0": ShuffleNet_V2_X2_0_Weights,
        }
        if backbone_type not in weights_dict:
            raise ValueError(f"Unsupported ShuffleNetV2 type: {backbone_type}")

        weights_cls = weights_dict[backbone_type]
        weights = weights_cls[weights_name] if pretrained else None
        model = getattr(models, backbone_type)(weights=weights)

        backbone = nn.Sequential(
            model.conv1,
            model.maxpool,
            model.stage2,
            model.stage3,
            model.stage4,
            model.conv5,
        )
        out_dim = 1024
        if backbone_type == "shufflenet_v2_x2_0":
            out_dim = 2048

    elif backbone_type == "mobilenet_v2":
        weights = MobileNet_V2_Weights[weights_name] if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        backbone = model.features
        out_dim = 1280

    elif backbone_type == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights[weights_name] if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        backbone = model.features
        out_dim = 576

    elif backbone_type == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights[weights_name] if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        backbone = model.features
        out_dim = 960

    else:
        raise ValueError(f"Unsupported backbone: {backbone_type}")

    return backbone, out_dim

class Classifier(nn.Module):
    def __init__(self, cfg):
        super(Classifier, self).__init__()

        backbone_cfg = cfg["model"]["backbone"]
        head_cfg = cfg["model"]["head"]

        self.backbone, out_dim = build_backbone(
            backbone_type=backbone_cfg["type"],
            pretrained=backbone_cfg["pretrained"],
            weights_name=backbone_cfg["weights"],
        )

        if head_cfg["type"] == "gap_fc":
            self.head = GAP_FC_Head(out_dim, head_cfg["num_classes"])
        else:
            raise ValueError(f"Unsupported head: {head_cfg['type']}")

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

def load_model(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    model = Classifier(cfg)
    return model

if __name__ == "__main__":
    model = load_model("configs/config_train.yaml")
    print(model)
    from torchsummary import summary
    summary(model, input_size=(3, 224, 224))
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)