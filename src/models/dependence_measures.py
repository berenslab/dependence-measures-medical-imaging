import numpy as np
import torch


class MIEstimator(torch.nn.Module):
    """Lower bound mutual information (MI) estimator.

    Reference: https://arxiv.org/abs/1801.04062

    Attributes:
        feature_dim: Dimension of input feature tensor to estimate MI from.
    """

    def __init__(
        self,
        feature_dim: int,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.device = "cuda"

        # Create statistics network.
        self.stats_network = statistics_network(self.feature_dim)
        self.register_buffer("running_exp", torch.tensor(float("nan")))

    def forward(self, x, y):
        batch_size = x.shape[0]
        xy = torch.cat(
            [x.repeat_interleave(batch_size, dim=0), y.tile(batch_size, 1)], -1
        )
        stats = self.stats_network(xy).reshape(batch_size, batch_size)

        diag = torch.diagonal(stats).mean()
        logmeanexp = self.logmeanexp_off_diagonal(stats)
        mi_estimate = diag - logmeanexp
        return mi_estimate

    def logmeanexp_off_diagonal(self, x):
        batch_size = x.shape[0]
        off_diag = x - torch.diag(np.inf * torch.ones(batch_size).to(self.device))
        logsumexp = torch.logsumexp(off_diag, dim=0)
        logsumexp = logsumexp - torch.log(torch.tensor(batch_size)).to(
            self.device
        )  # numerically stabilize
        return logsumexp.mean()


class statistics_network(torch.nn.Module):
    """Statistice neural network for mutual information estimator `MIEstimator`

    Attributes:
        in_feature: Input feature dimension:
    """

    def __init__(
        self,
        in_feature: int,
    ):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_feature, 400, bias=False),
            torch.nn.BatchNorm1d(400),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(400, 400, bias=False),
            torch.nn.BatchNorm1d(400),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(400, 400, bias=False),
            torch.nn.BatchNorm1d(400),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(400, 1),
        )

    def forward(self, x):
        return self.layers(x)


def distance_correlation(x: torch.Tensor, y: torch.Tensor):
    """Calculate the empirical distance correlation as described in [2].
    This statistic describes the dependence between `x` and `y`, which are
    random vectors of arbitrary length. The statistics' values range between 0
    (implies independence) and 1 (implies complete dependence).

    Args:
        x: Tensor of shape (batch-size, x_dimensions).
        y: Tensor of shape (batch-size, y_dimensions).

    Returns:
        The empirical distance correlation between `x` and `y`

    References:
        [1] https://en.wikipedia.org/wiki/Distance_correlation
        [2] Szekely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007)
            "Measuring and testing dependence by correlation of distances".
            Annals of Statistics, Vol. 35 No. 6, pp. 2769-2794.
    """
    # Euclidean distance between vectors.
    a = torch.cdist(x, x, p=2)  # N x N
    b = torch.cdist(y, y, p=2)  # N x N

    a_row_means = a.mean(axis=0, keepdims=True)
    b_row_means = b.mean(axis=0, keepdims=True)
    a_col_means = a.mean(axis=1, keepdims=True)
    b_col_means = b.mean(axis=1, keepdims=True)
    a_mean = a.mean()
    b_mean = b.mean()

    # Empirical distance matrices.
    A = a - a_row_means - a_col_means + a_mean
    B = b - b_row_means - b_col_means + b_mean

    # Empirical distance covariance.
    dcov = torch.mean(A * B)

    # Empirical distance variances.
    dvar_x = torch.mean(A * A)
    dvar_y = torch.mean(B * B)

    return torch.sqrt(dcov / torch.sqrt(dvar_x * dvar_y))


class GRLayer(torch.autograd.Function):
    """Gradient reversal layer.

    Acts as am identity function in the forward pass and
    inverts the gradient during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.scale, None


class AdvClassifierLinear(torch.nn.Module):
    """Linear adversarial Classification head with gradient reversal layer (GRL).

    Reference for GRL:
        paper: https://arxiv.org/abs/1505.07818
        example implementation: Adversarial classifier: https://github.com/NaJaeMin92/pytorch-DANN

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

    def forward(self, z, alpha):
        reversed_input = GRLayer.apply(z, alpha)
        x = self.layers(reversed_input)
        return x


class AdvClassifier(torch.nn.Module):
    """Nonlinear adversarial Classification head with gradient reversal layer (GRL).

    Reference for GRL:
        paper: https://arxiv.org/abs/1505.07818
        example implementation: Adversarial classifier: https://github.com/NaJaeMin92/pytorch-DANN

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
            torch.nn.Linear(z_shape, z_shape * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(z_shape * 2, z_shape * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(z_shape * 2, c_shape),
        )

    def forward(self, z, alpha):
        reversed_input = GRLayer.apply(z, alpha)
        x = self.layers(reversed_input)
        return x
