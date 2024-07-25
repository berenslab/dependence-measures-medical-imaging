import torch


class LinearClassifier(torch.nn.Module):
    """Linear classification head with no activation function.

    Attributes:
        z_shape: Latent space dimension.
        c_shape: Output/class dimension.
    """

    def __init__(
        self,
        z_shape: int = 512,
        c_shape: int = 2,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(z_shape, c_shape)

    def forward(self, z):
        return self.linear(z)


class TwoLayerLinearClassifier(torch.nn.Module):
    """Two linear classification head with no activation function.

    Should be equivalent to `LinearClassifier`.

    Attributes:
        z_shape: Latent space dimension.
        c_shape: Output/class dimension.
    """

    def __init__(
        self,
        z_shape: int = 512,
        c_shape: int = 2,
    ):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(z_shape, z_shape // 2),
            torch.nn.Linear(z_shape // 2, c_shape),
        )

    def forward(self, z):
        return self.layers(z)
