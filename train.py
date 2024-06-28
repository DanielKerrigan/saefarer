"""Script for training a SAE."""

import torch
import torch.nn.functional as F
import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from activations_store import ActivationsStore
from config import Config
from sparse_autoencoder import SAE


def mse(output, target):
    """MSE loss"""
    return F.mse_loss(output, target, reduction="mean")


def normalized_mse(output, target):
    """Normalized MSE loss"""
    target_mu = target.mean(dim=0)
    target_mu_reshaped = target_mu.unsqueeze(0).broadcast_to(target.shape)
    loss = mse(output, target) / mse(target_mu_reshaped, target)
    return loss


def get_mse_coef(activaitons):
    """Get coefficient for MSE loss"""
    return 1 / ((activaitons.mean(dim=0) - activaitons) ** 2).mean()


def main():
    """Train the SAE"""

    print("Loading model")

    model = AutoModelForCausalLM.from_pretrained(
        "./saved_models/roneneldan/TinyStories-1M"
    )

    print("Loading tokenizer")

    tokenizer = AutoTokenizer.from_pretrained("./saved_models/EleutherAI/gpt-neo-125M")

    print("Loading dataset")

    dataset = load_from_disk("saved_datasets/tinystories")

    cfg = Config(
        device="cpu",
        dtype="float32",
        # dimensions
        d_sae=256,
        d_in=64,
        # loss functions
        k=4,
        aux_k=32,
        aux_k_coef=1 / 32,
        dead_tokens_threshold=10_000_000,
        hidden_state_index=-2,
        normalize=False,
        # batch sizes
        lm_sequence_length=256,
        lm_batch_size_sequences=32,
        n_batches_in_store=20,
        sae_batch_size_tokens=4096,
        # tokenization
        prepend_bos_token=True,
        # adam
        lr=3e-4,
        beta1=0.9,
        beta2=0.999,
        eps=6.25e-10,
    )

    sae = SAE(cfg)

    print("Initializing activations store")

    store = ActivationsStore(model, tokenizer, dataset["train"], cfg)

    print("Calculating MSE coefficient.")

    sample_activations = store.next()
    mse_coef = get_mse_coef(sample_activations).item()

    print(f"MSE coefficient = {mse_coef}")

    num_tokens = 100_000_000
    num_batches = num_tokens // cfg.sae_batch_size_tokens

    optimizer = torch.optim.Adam(
        sae.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps
    )

    print("Beginning training.")

    for i in tqdm.trange(num_batches):
        x = store.next()
        recons, aux_recons = sae(x)

        mse_loss = mse(recons, x)

        aux_loss = normalized_mse(
            aux_recons, x - recons.detach() + sae.b_dec.detach()
        ).nan_to_num(0)

        loss = mse_coef * mse_loss + cfg.aux_k_coef * aux_loss
        loss.backward()

        sae.set_decoder_norm_to_unit_norm()
        sae.remove_gradient_parallel_to_decoder_directions()

        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            print(
                loss.item(),
                mse_coef * mse_loss.item(),
                cfg.aux_k_coef * aux_loss.item(),
            )

    print("Finished training.")

    sae.save("sae.pt")

    print("Saved model.")


if __name__ == "__main__":
    main()
