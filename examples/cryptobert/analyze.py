import argparse

from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    RobertaTokenizerFast,
)

from saefarer.adapters.tokenizers import HuggingFaceRobertaTokenizerAdapter
from saefarer.analysis import AnalysisConfig, analyze
from saefarer.sae import SAE
from saefarer.utils import get_default_device


def main(
    dataset_path,
    sae_path,
    db_path,
    device,
    num_features,
    model_batch_size_sequences,
    feature_batch_size,
    total_analysis_tokens,
):
    """Analyze the SAE"""

    print("Loading dataset")

    dataset = load_from_disk(dataset_path)

    print("Loading model and tokenizer")

    model_name = "ElKulako/cryptobert"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer: "RobertaTokenizerFast" = AutoTokenizer.from_pretrained(
        model_name, use_fast=True
    )  # type: ignore
    sf_tokenizer = HuggingFaceRobertaTokenizerAdapter(tokenizer)

    print("Creating config")

    device = get_default_device() if device == "auto" else device
    feature_indices = [] if num_features == -1 else list(range(num_features))

    cfg = AnalysisConfig(
        device=device,
        token_ids_column="input_ids",
        attn_mask_column="attention_mask",
        labels=[label for _, label in sorted(model.config.id2label.items())],
        model_batch_size_sequences=model_batch_size_sequences,
        model_sequence_length=128,
        feature_batch_size=feature_batch_size,
        total_analysis_tokens=total_analysis_tokens,
        feature_indices=feature_indices,
        n_example_sequences=10,
        n_context_tokens=5,
    )

    print(f"Using device {cfg.device}")

    print("Loading SAE")

    sae = SAE.load(sae_path, cfg.device)

    print("Starting analysis")

    analyze(
        cfg=cfg,
        model=model,
        dataset=dataset,  # type: ignore
        sae=sae,
        tokenizer=sf_tokenizer,
        output_path=db_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # inputs

    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset path",
        default="stocktwits-crypto_tokenized/train",
    )

    parser.add_argument("--sae", type=str, help="SAE path", default="sae.pt")

    parser.add_argument(
        "--output",
        type=str,
        default="analysis.db",
        help="Output database file path",
    )

    # config

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )

    parser.add_argument(
        "--num_features",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--model_batch_size_sequences",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--feature_batch_size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--total_analysis_tokens",
        type=int,
        default=10_000_000,
    )

    args = parser.parse_args()

    main(
        args.dataset,
        args.sae,
        args.output,
        args.device,
        args.num_features,
        args.model_batch_size_sequences,
        args.feature_batch_size,
        args.total_analysis_tokens,
    )
