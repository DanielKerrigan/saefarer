import argparse
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification

from saefarer.training import TrainingConfig, train
from saefarer.utils import get_default_device


def main(
    dataset_path,
    expansion_factor,
    k,
    hidden_state_index,
    model_batch_size_sequences,
    n_batches_in_store,
    sae_batch_size_tokens,
    total_training_tokens,
    logger,
):
    """Train the SAE"""

    dataset = load_from_disk(dataset_path)

    cfg = TrainingConfig(
        device=get_default_device(),
        dtype="float32",
        # dataset
        token_ids_column="input_ids",
        attn_mask_column="attention_mask",
        # dimensions
        d_in=768,
        expansion_factor=expansion_factor,
        # loss functions
        k=k,
        aux_k=512,
        aux_k_coef=1 / 32,
        dead_tokens_threshold=10_000_000,
        hidden_state_index=hidden_state_index,
        normalize=False,
        # batch sizes
        model_sequence_length=128,
        model_batch_size_sequences=model_batch_size_sequences,
        n_batches_in_store=n_batches_in_store,
        sae_batch_size_tokens=sae_batch_size_tokens,
        # adam
        lr=3e-4,
        beta1=0.9,
        beta2=0.999,
        eps=6.25e-10,
        # training
        total_training_tokens=total_training_tokens,
        # logging
        logger=logger,
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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="stocktwits-crypto_tokenized/train",
    )

    parser.add_argument(
        "--expansion_factor",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--k",
        type=int,
        help="k",
        default=4,
    )

    parser.add_argument(
        "--hidden_state_index",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--model_batch_size_sequences",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--n_batches_in_store",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--sae_batch_size_tokens",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--total_training_tokens",
        type=int,
        default=100_000_000,
    )

    parser.add_argument(
        "--logger",
        type=str,
        default="jsonl",
    )

    args = parser.parse_args()

    main(
        args.dataset,
        args.expansion_factor,
        args.k,
        args.hidden_state_index,
        args.model_batch_size_sequences,
        args.n_batches_in_store,
        args.sae_batch_size_tokens,
        args.total_training_tokens,
        args.logger,
    )
