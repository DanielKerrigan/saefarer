from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from saefarer.analyzing import analyze
from saefarer.config import AnalysisConfig
from saefarer.model import SAE


def main():
    """Analyze the SAE"""

    cfg = AnalysisConfig(
        device="cuda",
        dataset_column="input_ids",
        attn_mask_column="attention_mask",
        model_batch_size_sequences=32,
        model_sequence_length=128,
        feature_batch_size=64,
        total_analysis_tokens=10_000_000,
        feature_indices=list(range(64)),
        n_example_sequences=10,
        n_context_tokens=5,
    )

    dataset = load_from_disk("stocktwits-crypto_tokenized/train")

    model_name = "ElKulako/cryptobert"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    sae = SAE.load("sae.pt", cfg.device)

    output_path = "analysis.db"

    analyze(
        cfg=cfg,
        model=model,
        dataset=dataset,  # type: ignore
        sae=sae,
        tokenizer=tokenizer,  # type: ignore
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
