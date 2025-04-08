import argparse

from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from saefarer.analysis import AnalysisConfig, analyze
from saefarer.sae import SAE
from saefarer.utils import get_default_device


def main(sae_path, db_path):
    """Analyze the SAE"""

    dataset = load_from_disk("stocktwits-crypto_tokenized/train")

    model_name = "ElKulako/cryptobert"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    cfg = AnalysisConfig(
        device=get_default_device(),
        tokens_column="input_ids",
        attn_mask_column="attention_mask",
        labels=[label for _, label in sorted(model.config.id2label.items())],
        model_batch_size_sequences=32,
        model_sequence_length=128,
        feature_batch_size=8,
        total_analysis_tokens=1_000_000,
        feature_indices=list(range(8)),
        n_example_sequences=10,
        n_context_tokens=5,
    )

    sae = SAE.load(sae_path, cfg.device)

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
    parser.add_argument("-s", "--sae", type=str, help="SAE file path", default="sae.pt")
    parser.add_argument(
        "-d",
        "--db",
        type=str,
        default="analysis.db",
        help="Output database file path",
    )
    args = parser.parse_args()
    main(args.sae, args.db)
