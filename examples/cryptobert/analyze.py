import argparse

from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from saefarer.analysis import AnalysisConfig, analyze
from saefarer.sae import SAE
from saefarer.utils import get_default_device


def main(dataset_path, sae_path, db_path):
    """Analyze the SAE"""

    print("Loading dataset")

    dataset = load_from_disk(dataset_path)

    print("Loading model and tokenizer")

    model_name = "ElKulako/cryptobert"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    print("Creating config")

    cfg = AnalysisConfig(
        device=get_default_device(),
        tokens_column="input_ids",
        attn_mask_column="attention_mask",
        labels=[label for _, label in sorted(model.config.id2label.items())],
        model_batch_size_sequences=32,
        model_sequence_length=128,
        feature_batch_size=128,
        total_analysis_tokens=10_000_000,
        feature_indices=[],
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
        tokenizer=tokenizer,  # type: ignore
        output_path=db_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Dataset path",
        default="stocktwits-crypto_tokenized/train",
    )
    parser.add_argument("-s", "--sae", type=str, help="SAE path", default="sae.pt")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="analysis.db",
        help="Output database file path",
    )
    args = parser.parse_args()
    main(args.dataset, args.sae, args.output)
