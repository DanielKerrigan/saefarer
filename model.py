"""
This code is from
https://github.com/openai/sparse_autoencoder/
https://github.com/jbloomAus/SAELens
https://github.com/neelnanda-io/1L-Sparse-Autoencoder
"""

from typing import Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def LN(
    x: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std


class Autoencoder(nn.Module):
    """Sparse autoencoder

    Implements:
        latents = activation(encoder(x - pre_bias) + latent_bias)
        recons = decoder(latents) + pre_bias
    """

    def __init__(
        self, n_latents: int, n_inputs: int, k: int, normalize: bool = False
    ) -> None:
        """
        :param n_latents: dimension of the autoencoder latent
        :param n_inputs: dimensionality of the original data (e.g residual stream, number of MLP hidden units)
        :param k: number of neurons in the hidden layer of the autoencoder that fire at a given time
        :param normalize: make the input data have a mean of 0 and standard deviation of 1
        """
        super().__init__()

        self.dtype = torch.float32
        self.device = torch.device("cpu")

        self.b_dec = nn.Parameter(torch.zeros(n_inputs))
        self.b_enc = nn.Parameter(torch.zeros(n_latents))

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(n_latents, n_inputs, dtype=self.dtype, device=self.device)
            )
        )
        self.set_decoder_norm_to_unit_norm()

        self.W_enc = nn.Parameter(self.W_dec.t().clone())

        self.activation = TopK(k)

        self.normalize = normalize

    def encode_pre_act(
        self, x: torch.Tensor, latent_slice: slice = slice(None)
    ) -> torch.Tensor:
        """
        :param x: input data (shape: [batch, n_inputs])
        :param latent_slice: slice of latents to compute
            Example: latent_slice = slice(0, 10) to compute only the first 10 latents.
        :return: autoencoder latents before activation (shape: [batch, n_latents])
        """
        x = x - self.b_dec
        latents_pre_act = x @ self.W_enc + self.b_enc
        return latents_pre_act

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.normalize:
            return x, dict()
        x, mu, std = LN(x)
        return x, dict(mu=mu, std=std)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return: autoencoder latents (shape: [batch, n_latents])
        """
        x, info = self.preprocess(x)
        return self.activation(self.encode_pre_act(x)), info

    def decode(
        self, latents: torch.Tensor, info: dict[str, Any] | None = None
    ) -> torch.Tensor:
        """
        :param latents: autoencoder latents (shape: [batch, n_latents])
        :return: reconstructed data (shape: [batch, n_inputs])
        """
        recontructed = latents * self.W_dec + self.b_dec

        if self.normalize:
            assert info is not None
            recontructed = recontructed * info["std"] + info["mu"]

        return recontructed

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return:  autoencoder latents pre activation (shape: [batch, n_latents])
                  autoencoder latents (shape: [batch, n_latents])
                  reconstructed data (shape: [batch, n_inputs])
        """
        x, info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x)
        latents = self.activation(latents_pre_act)
        recons = self.decode(latents, info)

        mse_loss = F.mse_loss(recons, x, reduction="sum")

        return mse_loss

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec = self.W_dec / self.W_dec.norm(dim=1, keepdim=True)

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_in) shape
        """
        assert self.W_dec.grad is not None  # keep pyright happy

        # parallel_component = (self.W_dec.grad * self.W_dec.data).sum(
        #     dim=1, keepdim=True
        # )
        # self.W_dec.grad -= parallel_component * self.W_dec.data

        parallel_component = torch.einsum(
            "fd, fd -> f",
            self.W_dec.grad,
            self.W_dec.data,
        )

        self.W_dec.grad -= torch.einsum(
            "f, fd -> fd",
            parallel_component,
            self.W_dec.data,
        )

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> "Autoencoder":
        n_latents, d_model = state_dict["encoder.weight"].shape

        # Retrieve activation
        activation_class_name = "TopK"
        activation_class = TopK
        normalize = (
            activation_class_name == "TopK"
        )  # NOTE: hacky way to determine if normalization is enabled
        activation_state_dict = state_dict.pop("activation_state_dict", {})
        if hasattr(activation_class, "from_state_dict"):
            activation = activation_class.from_state_dict(
                activation_state_dict, strict=strict
            )
        else:
            activation = activation_class()
            if hasattr(activation, "load_state_dict"):
                activation.load_state_dict(activation_state_dict, strict=strict)

        autoencoder = cls(
            n_latents, d_model, activation=activation, normalize=normalize
        )
        # Load remaining state dict
        autoencoder.load_state_dict(state_dict, strict=strict)
        return autoencoder

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        sd = super().state_dict(destination, prefix, keep_vars)
        sd[prefix + "activation"] = self.activation.__class__.__name__
        if hasattr(self.activation, "state_dict"):
            sd[prefix + "activation_state_dict"] = self.activation.state_dict()
        return sd


class TopK(nn.Module):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self.postact_fn = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        # make all other values 0
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict.update(
            {
                prefix + "k": self.k,
            }
        )
        return state_dict

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> "TopK":
        k = state_dict["k"]
        return cls(k=k)
