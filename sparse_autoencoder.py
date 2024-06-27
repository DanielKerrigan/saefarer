"""
Sparse autoencoder model.

This code is from
https://github.com/openai/sparse_autoencoder/
https://github.com/jbloomAus/SAELens
https://github.com/neelnanda-io/1L-Sparse-Autoencoder
"""

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


def LN(
    x: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std


class SAE(nn.Module):
    """Sparse autoencoder

    Implements:
        latents = activation(encoder(x - pre_bias) + latent_bias)
        recons = decoder(latents) + pre_bias
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()

        self.dtype = getattr(torch, cfg.dtype)
        self.device = torch.device(cfg.device)

        self.cfg = cfg

        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae))

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    cfg.d_sae,
                    cfg.d_in,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
        )
        self.set_decoder_norm_to_unit_norm()

        self.W_enc = nn.Parameter(self.W_dec.t().clone())

        self.activation = TopK(cfg.k)

    def encode_pre_act(self, x: torch.Tensor) -> torch.Tensor:
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
        """Mean center, standard"""
        if not self.cfg.normalize:
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
        """
        Set decoder weights to have unit norm.
        """
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True)

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_in) shape
        """
        assert self.W_dec.grad is not None  # keep pyright happy

        # The below code is equivalent to
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

    def save(self, path):
        """Save model to path."""
        torch.save([self.cfg, self.state_dict()], path)

    @classmethod
    def load(cls, path):
        """Load model from path."""
        config, state = torch.load(path)
        model = cls(config)
        model.load_state_dict(state)
        return model


class TopK(nn.Module):
    """TopK activation."""

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self.postact_fn = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        # make all other values 0
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result
