import torch
import torchvision


class ResNetEncoder(torch.nn.Module):
    """Image encoder with resnet backbone.

    Attributes:
        resnet_backbone: Resnet backbone architecture. One of 18, 34, 50.
        latent_dim: Output shape of encoder.
        in_channels: Number of input channels. E.g. 3 for RGB, 1 for grayscale.
    """

    def __init__(
        self,
        resnet_backbone: int = 18,
        latent_dim: int = 512,
        in_channels: int = 1,
    ):
        super().__init__()

        if resnet_backbone == 18:
            backbone = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            )
        elif resnet_backbone == 34:
            backbone = torchvision.models.resnet34(
                weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1
            )
        elif resnet_backbone == 50:
            backbone = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            )

        # Customize resnet to new input channel dimensions.
        if in_channels != 3:
            backbone.conv1 = torch.nn.Conv2d(
                in_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )

        if resnet_backbone == 50:
            backbone.fc = torch.nn.Linear(
                in_features=2048, out_features=latent_dim, bias=True
            )
        else:
            backbone.fc = torch.nn.Linear(
                in_features=512, out_features=latent_dim, bias=True
            )

        self.encoder = backbone

    def forward(self, batch):
        return self.encoder(batch)


class EfficientNetB1(torch.nn.Module):
    """Image encoder with a efficientnet_b1 backbone.

    Attributes:
        latent_dim: Output shape of encoder.
        in_channels: Number of input channels. E.g. 3 for RGB, 1 for grayscale.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        in_channels: int = 1,
    ):
        super().__init__()

        backbone = torchvision.models.efficientnet_b1(
            weights=torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V2,
        )

        if in_channels != 3:
            backbone.features[0][0] = torch.nn.Conv2d(
                in_channels,
                32,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            )

        backbone.classifier[1] = torch.nn.Linear(
            in_features=1280, out_features=latent_dim, bias=False
        )
        self.model = backbone

    def forward(self, batch):
        return self.model(batch)


class SimpleEncoder(torch.nn.Module):
    """Simple image encoder with 3 conv layers.

    Attributes:
        latent_dim: Output shape of encoder.
    """

    def __init__(
        self,
        latent_dim: int = 512,
    ):
        super().__init__()
        self.fc = torch.nn.Linear(256, 256)
        self.fc_pt_sc = torch.nn.Linear(256, latent_dim)

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, 3, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(6, 16, 3, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 16, 3, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
        )

    def forward(self, X):
        X = self.features(X)
        X = X.view(-1, X.shape[1] * X.shape[2] * X.shape[3])
        X = torch.nn.functional.relu(self.fc(X))
        return self.fc_pt_sc(X)
