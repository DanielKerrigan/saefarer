from pathlib import Path

from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification

from saefarer.training import TrainingConfig, train
from saefarer.utils import get_default_device


def main():
    """Train the SAE"""

    dataset = load_from_disk("stocktwits-crypto_tokenized/train")

    cfg = TrainingConfig(
        device=get_default_device(),
        dtype="float32",
        # dataset
        dataset_column="input_ids",
        attn_mask_column="attention_mask",
        # dimensions
        d_in=768,
        expansion_factor=4,
        # loss functions
        k=4,
        aux_k=512,
        aux_k_coef=1 / 32,
        dead_tokens_threshold=10_000_000,
        hidden_state_index=10,
        normalize=False,
        # batch sizes
        model_sequence_length=128,
        model_batch_size_sequences=32,
        n_batches_in_store=64,
        sae_batch_size_tokens=4096,
        # adam
        lr=3e-4,
        beta1=0.9,
        beta2=0.999,
        eps=6.25e-10,
        # training
        total_training_tokens=100_000_000,
        # logging
        logger="jsonl",
        log_batch_freq=500,
        # checkpointing
        checkpoint_batch_freq=10_000,
    )

    model = AutoModelForSequenceClassification.from_pretrained("ElKulako/cryptobert")
    model.to(cfg.device)

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train(
        cfg=cfg,
        model=model,
        dataset=dataset,  # type: ignore
        save_path="sae.pt",
        log_dir="logs",
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    main()
