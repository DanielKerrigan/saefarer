"""Script for training a SAE."""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparse_autoencoder.config import Config
from sparse_autoencoder.training import train


def main():
    """Train the SAE"""

    cfg = Config(
        device="cuda",
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
        model_sequence_length=256,
        model_batch_size_sequences=16,
        n_batches_in_store=64,
        sae_batch_size_tokens=4096,
        # tokenization
        prepend_bos_token=True,
        # adam
        lr=3e-4,
        beta1=0.9,
        beta2=0.999,
        eps=6.25e-10,
        # training
        total_training_tokens=100_000_000,
    )

    print("Loading model")

    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
    model.to(cfg.device)

    print("Loading tokenizer")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    print("Loading dataset")

    dataset = load_dataset("roneneldan/TinyStories", split="train")

    train(cfg, model, tokenizer, dataset, "sae.pt")


if __name__ == "__main__":
    main()
